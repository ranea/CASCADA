"""Represent bit-vector functions and their representation in static single assignment (SSA) form.

.. autosummary::
   :nosignatures:

    BvFunction
    RoundBasedFunction
    SSAReturn
    SSA
    get_random_bvfunction
"""
import collections
import functools
import warnings

from cascada.bitvector import context
from cascada.bitvector import core
from cascada.bitvector import operation


zip = functools.partial(zip, strict=True)


class BvFunction(object):
    """Represent fixed-width bit-vector functions.

    A `BvFunction` takes fixed-width `Constant` operands and return a
    tuple of fixed-width `Constant`.

    Similar to `Operation`, `BvFunction` is evaluated
    using the operator ``()`` and provides *Automatic Constant Conversion*.

    .. note::
        `BvFunction` only supports `Constant` operands and
        must return a tuple of `Constant` values
        (as opposed to `Operation` that accepts `Term` and scalar operands
        and returns a single `Term`).

        `BvFunction` does not support returned values that independent
        of the inputs.

    This class is not meant to be instantiated but to provide a base class for
    bit-vector functions with fixed operands. To define a bit-vector function,
    subclass `BvFunction`, set the class attributes `input_widths` and
    `output_widths` and implement `eval` with bit-vector operations.

        >>> from cascada.bitvector.ssa import BvFunction
        >>> class MyFoo(BvFunction):
        ...     input_widths, output_widths = [8, 8], [8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y):  return x ^ y, x
        >>> MyFoo.get_name()
        'MyFoo'
        >>> MyFoo(1, 1)  # automatic conversion from int to Constant
        (0x00, 0x01)

    Attributes:
        input_widths: a list containing the widths of the inputs
        output_widths: a list containing the widths of the outputs

    """
    input_widths = None
    output_widths = None
    _logger = None

    def __new__(cls, *args, **options):
        if len(cls.input_widths) != len(args):
            raise ValueError(f"expected {len(cls.input_widths)} inputs, not {len(args)} = len({args})")

        cls._logger = []

        newargs = []
        for arg, width in zip(args, cls.input_widths):
            newargs.append(core.bitvectify(arg, width))
        args = newargs

        simplify = options.pop("simplify", context.Simplification.current_context)
        symbolic_inputs = options.pop("symbolic_inputs", False)
        ct_inputs = all(isinstance(arg, core.Constant) for arg in args)

        if ct_inputs or symbolic_inputs:
            if not ct_inputs and simplify and context.Memoization.current_context is None:
                warnings.warn(f"very slow evaluation of {cls.get_name()} (with symbolic inputs, "
                              f"with the Simplification context and without the "
                              f"Memoization context)")
            with context.Simplification(simplify):
                result = cls.eval(*args)
        else:
            raise TypeError(f"expected bit-vector Constant arguments, not {args}")

        if isinstance(result, (int, core.Term)):
            raise ValueError(f"{cls.__name__} must return a tuple of bit-vectors, "
                             f"not a single {type(result).__name__} object")
        elif not isinstance(result, collections.abc.Sequence) or len(cls.output_widths) != len(result):
            raise ValueError(f"expected a {len(cls.output_widths)}-length tuple "
                             f"of bit-vectors returned, not {len(result)}-length tuple {result}")

        if symbolic_inputs:
            if any(isinstance(v, core.Constant) for v in result):
                raise ValueError(f"if symbolic_inputs, expected no Constant values "
                                 f"in the returned tuple, not {result}")
        else:
            if not all(isinstance(v, core.Constant) for v in result):
                raise ValueError(f"expected a tuple of Constant values returned, not {result}")

        output = []
        for r, width in zip(result, cls.output_widths):
            output.append(core.bitvectify(r, width))

        cls._logger = tuple(cls._logger)

        return tuple(output)

    @classmethod
    def get_name(cls):
        """Return the name of the function (by default ``cls.__name__``)."""
        return cls.__name__

    @classmethod
    def vrepr(cls):
        """Return an executable string representation.

        This method returns a string so that ``eval(cls.vrepr())``
        returns a new `BvFunction` object with the same content.
        """
        return cls.__name__

    @classmethod
    def eval(cls, *args):
        """Evaluate the function (internal method)."""
        raise NotImplementedError("subclasses need to override this method")

    # noinspection PyArgumentList
    @classmethod
    def to_ssa(cls, input_names, id_prefix, decompose_sec_ops=False, **ssa_options):
        """Return the `SSA` object of the bit-vector function.

        Args:
            input_names: the names for the input variables
            id_prefix: the prefix to denote the intermediate variables
            decompose_sec_ops: if ``decompose_sec_ops`` is ``True``,
                `SecondaryOperation` objects within the bit-vector function are
                replaced by their decompositions into  `PrimaryOperation` objects.
            ssa_options: options passed to the ``__init__`` method of `SSA`.

        """
        assert len(input_names) == len(cls.input_widths), \
            f"len({input_names}) != len({cls.input_widths})"
        for name in input_names:
            if name.startswith(id_prefix):
                raise ValueError(f"input_name {name} starts with prefix {id_prefix}")

        if len(set(input_names)) < len(input_names):
            warnings.warn(f"found duplicated names in input_names {input_names}")

        input_vars = []
        for name, width in zip(input_names, cls.input_widths):
            input_vars.append(core.Variable(name, width))
        input_vars = tuple(input_vars)

        table = context.MemoizationTable(id_prefix=id_prefix)

        with context.Memoization(table), context.SecondaryOperationEvaluation(decompose_sec_ops):
            # noinspection PyArgumentList
            output_vars = list(cls(*input_vars, symbolic_inputs=True))

        ##debugging
        # print(f"\n{cls.__name__}(input_vars) = {cls(*input_vars, symbolic_inputs=True, simplify=False)}")
        # print("Memoization | table.table:", table.table)

        if getattr(cls, "_ignore_replace_multiuse_vars", False) is True:
            ssa_options["replace_multiuse_vars"] = False

        return SSA(input_vars=input_vars, output_vars=output_vars,
                   assignments=table.table, **ssa_options)

    @classmethod
    def log_msg(cls, format_string, format_field_objects=None):
        """Log a message during the evaluation of the bit-vector function.

        This method is meant to be used within `eval` to log the values
        of internal variables. At the beggining of each evaluation,
        messages logged from the last evaluation are automatically removed.

            >>> from cascada.bitvector.ssa import BvFunction
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):
            ...         cls.log_msg('input variables = [{}, {}]', [x, y])
            ...         return x ^ y, x
            >>> MyFoo(0, 1)
            (0x01, 0x00)
            >>> MyFoo.get_formatted_logged_msgs()
            ['input variables = [0x00, 0x01]']

        Args:
            format_string: a Python `format string
                 <https://docs.python.org/3/library/string.html#formatstrings>`_
            format_field_objects: an optional list with the objects that will
                replace the format fields in the format_string (i.e.,
                the arguments associated to ``format_string.format()``)

        """
        if cls._logger is None:
            cls._logger = []
        assert isinstance(format_string, str)
        if format_field_objects is None:
            format_field_objects = []
        elif isinstance(format_field_objects, core.Term):
            format_field_objects = [format_field_objects]
        assert isinstance(format_string, collections.abc.Sequence)
        cls._logger.append((format_string, format_field_objects))

    @classmethod
    def get_formatted_logged_msgs(cls):
        """Return the list of logged messages.

        If the bit-vector function includes `log_msg` calls in its ``eval``,
        this method return the list of messages logged in the last evaluation
        with the format field objects applied. Otherwise, an exception is raised.

        See also `log_msg`.
        """
        if cls._logger is None:
            raise ValueError("eval must be called before get_formatted_logged_msgs")
        list_msgs = []
        for format_string, format_field_objects in cls._logger:
            list_msgs.append(format_string.format(*format_field_objects))
        return list_msgs

    # dotprinting last method
    @classmethod
    def dotprinting(cls, input_names, repeat=True, vrepr_label=False, **kwargs):
        """Return the DOT description of the expression tree of the function.

        See also `printing.dotprinting`.

        Args:
            input_names: the names for the input variables
            repeat: whether to use different nodes for common subexpressions
                (default True)
            vrepr_label: wheter to use the verbose representation (`Term.vrepr`)
                to label the nodes (default False)
            kwargs: additional arguments passed to `printing.dotprinting`

        ::

            >>> from cascada.bitvector.ssa import BvFunction
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x):  return x, x ^ 1
            >>> print(MyFoo.dotprinting(["x"], repeat=False))  # doctest:+NORMALIZE_WHITESPACE
            digraph{
            # Graph style
            "ordering"="out"
            "rankdir"="TD"
            #########
            # Nodes #
            #########
            "Tuple(Variable('x', width=8), BvXor(Variable('x', width=8), Constant(0b00000001, width=8)))"
                ["color"="grey", "label"="Tuple", "shape"="ellipse"];
            "Variable('x', width=8)" ["color"="aquamarine3", "label"="x", "shape"="box"];
            "BvXor(Variable('x', width=8), Constant(0b00000001, width=8))"
                ["color"="black", "label"="BvXor", "shape"="ellipse"];
            "Constant(0b00000001, width=8)" ["color"="blue1", "label"="0x01", "shape"="box"];
            <BLANKLINE>
            #########
            # Edges #
            #########
            "Tuple(Variable('x', width=8), BvXor(Variable('x', width=8), Constant(0b00000001, width=8)))" ->
                "Variable('x', width=8)";
            "Tuple(Variable('x', width=8), BvXor(Variable('x', width=8), Constant(0b00000001, width=8)))" ->
                "BvXor(Variable('x', width=8), Constant(0b00000001, width=8))";
            "BvXor(Variable('x', width=8), Constant(0b00000001, width=8))" -> "Variable('x', width=8)";
            "BvXor(Variable('x', width=8), Constant(0b00000001, width=8))" -> "Constant(0b00000001, width=8)";
            }

        """
        from cascada.bitvector.printing import dotprinting
        from sympy import Tuple

        assert len(input_names) == len(cls.input_widths), \
            f"len({input_names}) != len({cls.input_widths})"

        input_vars = []
        for name, width in zip(input_names, cls.input_widths):
            input_vars.append(core.Variable(name, width))
        input_vars = tuple(input_vars)

        outputs = cls(*input_vars, symbolic_inputs=True, simplify=False)
        return dotprinting(Tuple(*outputs), repeat=repeat, vrepr_label=vrepr_label, **kwargs)


class RoundBasedFunction(BvFunction):
    """Represent round-based fixed-width bit-vector functions.

    A `RoundBasedFunction` is a `BvFunction` that can be decomposed
    into *rounds*, where in each round some subroutine is computed
    (not necessarily the same). Moreoever, the next round not only
    can use the outputs of the previous round but also any of the
    previous variables computed.

    In practice, the ``eval`` method of `RoundBasedFunction`
    depends on the class attribute `num_rounds`, which can be changed by
    calling the method `set_num_rounds`.

    This class is not meant to be instantiated but to provide a base class for
    round-based bit-vector functions with fixed operands. To define such a
    function, subclass `RoundBasedFunction`, set the class attributes
    `input_widths` and ``output_widths`` and implement ``eval`` with bit-vector
    operations (similar as `BvFunction`). Moreover, set the class attribute
    `num_rounds` and implement `set_num_rounds`.

        >>> from cascada.bitvector.ssa import RoundBasedFunction
        >>> class MyFoo(RoundBasedFunction):
        ...     input_widths, output_widths, num_rounds = [8, 8], [8, 8], 1
        ...     @classmethod
        ...     def eval(cls, x, y):
        ...         for _ in range(cls.num_rounds):  x, y = y, x + 1
        ...         return x, y
        ...     @classmethod
        ...     def set_num_rounds(cls, new_num_rounds):  cls.num_rounds = new_num_rounds
        >>> MyFoo.get_name()
        'MyFoo_1R'
        >>> MyFoo(0, 0)
        (0x00, 0x01)
        >>> MyFoo.set_num_rounds(2)
        >>> MyFoo.get_name()
        'MyFoo_2R'
        >>> MyFoo(0, 0)
        (0x01, 0x01)

    Attributes:
        num_rounds: the number of rounds

    """
    num_rounds = None
    _rounds_outputs = None  # private attribute as _logger

    def __new__(cls, *args, **options):
        cls._rounds_outputs = []
        result = super().__new__(cls, *args, **options)
        if len(cls._rounds_outputs) not in [0, cls.num_rounds]:
            raise ValueError("add_round_outputs must be called 0 or num_rounds times")
        cls._rounds_outputs = tuple(cls._rounds_outputs)
        return result

    @classmethod
    def get_name(cls):
        """Return the class name and the current number of rounds."""
        return f"{super().get_name()}_{cls.num_rounds}R"

    @classmethod
    def to_ssa(cls, input_names, id_prefix, decompose_sec_ops=False, **ssa_options):
        """Return the `SSA` object of the round-based bit-vector function.

        This method calls `BvFunction.to_ssa` with the same argument list,
        and stores the round outputs in the `SSA` object if `add_round_outputs`
        calls were added in ``eval``.
        """
        my_ssa = super().to_ssa(input_names, id_prefix, decompose_sec_ops=decompose_sec_ops, **ssa_options)
        if cls._rounds_outputs is not None:
            my_ssa._rounds_outputs = cls._rounds_outputs[:]
        return my_ssa

    @classmethod
    def set_num_rounds(cls, new_num_rounds):
        """Set `RoundBasedFunction.num_rounds` and update `input_widths` and ``output_widths`` if necessary."""
        raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def set_num_rounds_and_return(cls, new_num_rounds):
        """Call `set_num_rounds` and return ``cls``."""
        cls.set_num_rounds(new_num_rounds)
        return cls

    @classmethod
    def vrepr(cls):
        """Return an executable string representation.

        This method returns a string so that ``eval(cls.vrepr())``
        returns a new `RoundBasedFunction` object with the same content.
        """
        return f"{super().vrepr()}.set_num_rounds_and_return({cls.num_rounds})"

    @classmethod
    def add_round_outputs(cls, *args):
        """Store the current round outputs in the evaluation of the round-based function.

        This method is meant to be used within ``eval`` to store the outputs
        of each round. At the beggining of each evaluation, round ouputs stored
        in the last evaluation are automatically removed.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.ssa import RoundBasedFunction
            >>> class MyFoo(RoundBasedFunction):
            ...     input_widths, output_widths, num_rounds = [8, 8], [8, 8], 1
            ...     @classmethod
            ...     def eval(cls, x, y):
            ...         for _ in range(cls.num_rounds):
            ...             x, y = y, x + 1
            ...             cls.add_round_outputs(x, y)
            ...         return x, y
            ...     @classmethod
            ...     def set_num_rounds(cls, new_num_rounds):  cls.num_rounds = new_num_rounds
            >>> MyFoo.set_num_rounds(3)
            >>> MyFoo(0, 0)
            (0x01, 0x02)
            >>> MyFoo.get_rounds_outputs()
            ((0x00, 0x01), (0x01, 0x01), (0x01, 0x02))
            >>> MyFoo(Variable("x", 8), Variable("y", 8), symbolic_inputs=True)
            (y + 0x01, x + 0x02)
            >>> MyFoo.get_rounds_outputs()
            ((y, x + 0x01), (x + 0x01, y + 0x01), (y + 0x01, x + 0x02))

        Args:
            args: a list of bit-vectors representing the outputs of the current round.

        """
        if len(args) == 1 and isinstance(args[0], collections.abc.Sequence):
            args = args[0]
        if not all(isinstance(bv, core.Term) for bv in args):
            raise ValueError("the arguments of add_round_outputs must be a bit-vectors")
        if cls._rounds_outputs is None:
            cls._rounds_outputs = []
        cls._rounds_outputs.append(tuple(args))

    @classmethod
    def get_rounds_outputs(cls):
        """Return the list of round outputs obtained in the last evaluation.

        See also `add_round_outputs`.
        """
        if cls._rounds_outputs is None:
            raise ValueError("eval must be called before get_rounds_outputs")
        return cls._rounds_outputs


class SSAReturn(operation.BvIdentity):
    """Subclass of `BvIdentity` used to denote the returned variables in `SSA`.

    Note that both `BvIdentity` and `SSAReturn` have the same string representation
    (but different `Term.vrepr`).

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import BvIdentity
        >>> from cascada.bitvector.ssa import SSAReturn
        >>> BvIdentity(Constant(0x1, 4)), SSAReturn(Constant(0x1, 4))
        (0x1, 0x1)
        >>> BvIdentity(Variable("x", 8)), SSAReturn(Variable("x", 8))
        (Id(x), Id(x))
        >>> BvIdentity(Variable("x", 8)).vrepr(), SSAReturn(Variable("x", 8)).vrepr()
        ("BvIdentity(Variable('x', width=8))", "SSAReturn(Variable('x', width=8))")

    """
    pass


class SSA(object):
    """Represent a `BvFunction` as a static single assignment program.

    An `SSA` object represents a decomposition of a `BvFunction`
    into a list of *simple* assignments :math:`x_{i+1} \leftarrow f_i(x_i)`,
    similar as the method `memoization_table` for `Operation` objects.
    Recall a *simple* assignment is given by the output `Variable`
    :math:`x_{i+1}` representing the output of the given
    `Operation` :math:`f_i(x_i)`, where :math:`x_i` is an input or
    previous output `Variable`, a `Constant`, a scalar, or a list of them
    (but not an `Operation` object).

    An `SSA` object is defined for

    -  A list of `Variable` objects representing the inputs of the bit-vector function.
    -  A list of `Variable` objects representing the outputs of the function.
    -  The list of of assignments as a `collections.OrderedDict` mapping
       each intermediate `Variable` :math:`x_{i+1}` to the `Operation` :math:`f_i(x_i)`.

    The assignments might contain variables not defined in previous assignments
    nor contained in the input variables.
    These variables are called *external variables* and are stored
    in the attribute `external_vars`.

    Apart from these three initialization argument,
    if the initialization argument ``replace_multiuse_vars`` is set
    to ``True``, variables appearing two or more times
    in the list of assignments (as arguments of `Operation` objects
    excluding `BvIdentity`) are replaced by new variables (ending in
    ``__d`` where ``d`` is the occurrence index) that are used only once.
    In this case, the mapping from the old multiuse variables
    to the new singleuse variables is stored in ``multiuse_var2singleuse_vars``.
    In other words, new variables are created to uniquely label each
    branch of the forking subroutine (see `LinearModelFreeBranch`).

    .. note::

        For example, with ``replace_multiuse_vars=True`` the assignment list
        ``(a0, x + 1), (a1, x & 1)`` is replaced by
        ``(x__0, Id(x)), (x__1, Id(x)) (a0, x__0 + 1), (a1, x__1 & 1)``.

    `SSA` objects can be easily created from `BvFunction.to_ssa`.

    .. note::
        When initializing the SSA object, external variables are
        automatically obtained, redundant assignments are automatically
        removed (and a warning is printed in this case), and
        partial operations are automatically created from
        operations with constant or scalar operands.

        Moreover, additional assignments with the operations `BvIdentity`
        and `SSAReturn` might be added at the end.
        These last assignemnts are added to ensure that each output
        variable of the `SSA` has a unique name (ending in ``_out``)
        and to ensure that the output variables of the last assignments
        are actually the output variables of the `SSA`.
        If these assignments are already present in the given
        list of assigments no additional ones are added.

    ::

        >>> from cascada.bitvector.ssa import BvFunction
        >>> class MyFoo(BvFunction):
        ...     input_widths, output_widths = [8, 8], [8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y):  return x ^ y, x
        >>> MyFoo.to_ssa(["x", "y"], "a")  # doctest:+NORMALIZE_WHITESPACE
        SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
            assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])
        >>> ssa = MyFoo.to_ssa(["x", "y"], "a", replace_multiuse_vars=True)
        >>> ssa  # doctest:+NORMALIZE_WHITESPACE
        SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
            assignments=[(x__0, Id(x)), (x__1, Id(x)), (a0, x__0 ^ y), (a0_out, Id(a0)), (x_out, Id(x__1))])
        >>> ssa.multiuse_var2singleuse_vars, ssa.singleuse_var2multiuse_var
        (OrderedDict([(x, [x__0, x__1])]), OrderedDict([(x__0, x), (x__1, x)]))
        >>> last_assignments = list(ssa.assignments.items())[-len(ssa.output_vars):]
        >>> for var, expr in last_assignments: print(var.vrepr(), ", ", expr.vrepr(), sep="")
        Variable('a0_out', width=8), SSAReturn(Variable('a0', width=8))
        Variable('x_out', width=8), SSAReturn(Variable('x__1', width=8))

    Attributes:
        input_vars: a list of `Variable` objects representing the inputs.
        output_vars: a list of `Variable` objects representing the outputs.
        assignments: a `collections.OrderedDict` containing the assignments
            (`Variable`, `Operation`) of the SSA program.
        external_vars: a list of `Variable` objects which appear in the
            assignments but are not defined in previous assignments
            nor contained in the input variables.
        multiuse_var2singleuse_vars: an empty dictionary if the initialization
            argument ``replace_multiuse_vars`` is set to ``False``, otherwise
            a `collections.OrderedDict` mapping the old multiuse variables
            to the new singleused variables.
        singleuse_var2multiuse_var: the inverse mapping of
            ``multiuse_var2singleuse_vars``.

    .. Implementation details::
        This class does not decompose secondary operations into
        primary ones. This is implemented in `BvFunction.to_ssa`.

        The list of assignments is not stored as a `MemoizationTable`
        (a bidict.OrderedBidict()) since duplicated operation objects
        might be contained in the list of assignments.

    """

    def __init__(self, input_vars, output_vars, assignments, replace_multiuse_vars=False):
        assert all(isinstance(v, core.Variable) for v in input_vars), \
            f"expected a list of Variable for the input_vars, not {input_vars}"
        assert all(isinstance(v, core.Variable) for v in output_vars), \
            f"expected a list of Variable for the output_vars, not {output_vars}"

        if len(set(input_vars)) < len(input_vars):
            warnings.warn(f"found duplicated vars in input_vars {input_vars}")
        if len(set(output_vars)) < len(output_vars):
            warnings.warn(f"found duplicated vars in output_vars {output_vars}")

        input_vars = input_vars[:]
        output_vars = output_vars[:]

        if not isinstance(assignments, collections.OrderedDict):
            assignments = collections.OrderedDict(assignments)
        else:
            assignments = assignments.copy()

        def is_simple_assignment(my_outvar, my_expr, raise_exception):
            if not isinstance(my_expr, operation.Operation):
                if raise_exception:
                    raise ValueError(f"found invalid assignment (expr is not an Operation): {my_outvar} <- {expr}")
                else:
                    return False
            if isinstance(my_expr, operation.BvIdentity) and my_expr.args[0] == my_outvar:
                if raise_exception:
                    raise ValueError(f"redundant identity operations {my_outvar} <- {my_expr} not supported")
                else:
                    return False
            for my_arg in my_expr.args:
                if isinstance(my_arg, operation.Operation) or \
                        not isinstance(my_arg, (int, core.Constant, core.Variable)):
                    if raise_exception:
                        raise ValueError(f"found not simple assignment {my_outvar} <- {my_expr}")
                    else:
                        return False
            return True

        for outvar, expr in assignments.items():
            is_simple_assignment(outvar, expr, raise_exception=True)

        if not self._are_last_assignments_valid(assignments, output_vars):
            # _make_last_assignments_valid before _replace_multiuse_vars
            # as the first call might create branches
            self._make_last_assignments_valid(assignments, output_vars, input_vars)

        self._old_outvar_assignments = tuple(assignments.keys())

        arg2outvars = collections.OrderedDict()  # list of outvar where arg appears as operand
        set_in_out_vars = set(input_vars) | set(output_vars)

        def find_partial_and_redundant_assignments(ignore_arg2outvars):
            if not ignore_arg2outvars:
                arg2outvars.clear()
            to_delete = []
            vars_needed = set(output_vars)
            for my_outvar in reversed(assignments):
                # find partial operations
                my_expr = assignments[my_outvar]
                if any(isinstance(arg, (int, core.Constant)) for arg in my_expr.args):
                    fixed_args = []
                    new_args = []
                    for i, arg in enumerate(my_expr.args):
                        if isinstance(arg, (int, core.Constant)):
                            fixed_args.append(arg)
                        else:
                            fixed_args.append(None)
                            new_args.append(arg)
                    expr_op = operation.make_partial_operation(type(my_expr), tuple(fixed_args))
                    my_expr = expr_op(*new_args)
                    assignments[my_outvar] = my_expr
                # find non-redundant assignments
                if my_outvar in vars_needed:
                    for arg in my_expr.args:  # expr.atoms() not needed due to simple assignments
                        assert isinstance(arg, core.Variable)
                        vars_needed.add(arg)
                        if not ignore_arg2outvars and arg not in set_in_out_vars and arg in assignments:
                            if isinstance(my_expr, operation.BvIdentity) or \
                                    isinstance(assignments[arg], operation.BvIdentity):
                                # added 2 occurrences to ensure it is not used when simplified
                                arg2outvars[arg] = [None, None]
                            else:
                                arg2outvars[arg] = arg2outvars.get(arg, []) + [my_outvar]
                else:
                    to_delete.append(my_outvar)

            if to_delete:
                warnings.warn("removing redundant assignments {}".format(
                    [(my_outvar, assignments[my_outvar]) for my_outvar in to_delete]))
                for my_outvar in to_delete:
                    del assignments[my_outvar]

        if not context.Simplification.current_context:
            find_partial_and_redundant_assignments(ignore_arg2outvars=True)
        else:
            def assignments_xreplace(my_dict):
                # assume keys() in my_dict are not in assignments
                my_outvar2ct_expr = collections.OrderedDict()
                for my_outvar in assignments:
                    my_expr = assignments[my_outvar].xreplace(my_dict)
                    assignments[my_outvar] = my_expr
                    # vars are not removed from SSA to avoid side effects
                    if isinstance(my_expr, core.Variable):
                        assignments[my_outvar] = operation.BvIdentity(my_expr)
                    if isinstance(my_expr, core.Constant):
                        my_outvar2ct_expr[my_outvar] = my_expr
                if my_outvar2ct_expr:
                    for my_outvar in my_outvar2ct_expr.keys():
                        del assignments[my_outvar]
                    assignments_xreplace(my_outvar2ct_expr)

            while True:
                find_partial_and_redundant_assignments(ignore_arg2outvars=False)
                # only variables appearing once in assignments
                arg2outvars = collections.OrderedDict(
                    [(arg, outvars[0]) for arg, outvars in arg2outvars.items() if len(outvars) == 1]
                )
                # print(f"\nbefore simplifying\n\targ2outvars = {arg2outvars}\n\tassignments: {assignments}")

                assignments_modified = False
                for arg, outvar in tuple(arg2outvars.items()):
                    # outvar can also be a previous arg its assignment
                    # could have been deleted (in del assignments[arg])
                    if outvar not in assignments:
                        continue
                    new_expr = assignments[outvar].xreplace({arg: assignments[arg]})
                    if isinstance(new_expr, core.Variable):
                        # avoid replacing vars
                        new_expr = operation.BvIdentity(new_expr)
                    if isinstance(new_expr, core.Constant) or \
                            is_simple_assignment(outvar, new_expr, raise_exception=False):
                        warnings.warn(f"assignment ({outvar}, {assignments[outvar]}) was simplified to "
                                      f"{new_expr} using {arg} <- {assignments[arg]}")
                        assignments_modified = True
                        del assignments[arg]
                        if isinstance(new_expr, core.Constant):
                            del assignments[outvar]
                            assignments_xreplace({outvar: new_expr})
                        else:
                            assignments[outvar] = new_expr
                if not assignments_modified:
                    break

        if replace_multiuse_vars:
            aux = self._replace_multiuse_vars(assignments, output_vars, input_vars)
            assignments, multiuse_var2singleuse_vars, singleuse_var2multiuse_var = aux
        else:
            multiuse_var2singleuse_vars = collections.OrderedDict()
            singleuse_var2multiuse_var = collections.OrderedDict()

        self._are_last_assignments_valid(assignments, output_vars, ignore_exception=False)

        external_vars = []
        input_vars_set = set(input_vars)
        input_vars_used_set = set()
        for outvar, expr in assignments.items():
            for arg in expr.args:
                if isinstance(arg, operation.Operation) or not isinstance(arg, (core.Constant, core.Variable)):
                    raise ValueError(f"assignment {outvar} <- {expr} not properly decomposed")
                if isinstance(arg, core.Variable):
                    if arg in input_vars_set:
                        input_vars_used_set.add(arg)
                    elif arg not in assignments:  # arg not an outvar
                        assert arg not in output_vars
                        if arg not in external_vars:  # avoid duplicates
                            external_vars.append(arg)

        input_vars_not_used = tuple(input_vars_set.difference(input_vars_used_set))
        if input_vars_not_used:
            warnings.warn("found unused input vars {}".format(input_vars_not_used))

        self.input_vars = tuple(input_vars)
        self.output_vars = tuple(output_vars)
        self.assignments = assignments

        self.external_vars = tuple(external_vars)
        self.multiuse_var2singleuse_vars = multiuse_var2singleuse_vars
        self.singleuse_var2multiuse_var = singleuse_var2multiuse_var

        self._replace_multiuse_vars_bool = replace_multiuse_vars
        self._input_vars_not_used = sorted(input_vars_not_used, key=lambda x: self.input_vars.index(x))

    @staticmethod
    def _replace_multiuse_vars(assignments, output_vars, input_vars):
        assert isinstance(assignments, collections.OrderedDict)

        multiuse_var2singleuse_vars = collections.OrderedDict()
        singleuse_var2multiuse_var = collections.OrderedDict()

        # 1st iteration: find multiuse vars (variables used 2+ times)

        var2occurrences = collections.OrderedDict()
        var2assigns_with_var = collections.OrderedDict()
        for var_i, expr_i in assignments.items():
            for var_j in expr_i.args:
                if isinstance(var_j, core.Variable):
                    var2occurrences[var_j] = var2occurrences.get(var_j, 0) + 1
                    var2assigns_with_var[var_j] = var2assigns_with_var.get(var_j, []) + [(var_i, expr_i)]

        # vars v where all the occurrences are * <- BvIdentity(v) (and not SSAReturn) do not need to be replaced
        # (but they need to be stored in multi2single and single2multi)
        for var_j, assigns in var2assigns_with_var.items():
            if len(assigns) >= 2 and all(type(e) == operation.BvIdentity for _, e in assigns):
                del var2occurrences[var_j]
                # see below definition of var_j and var_j__k
                multiuse_var2singleuse_vars[var_j] = []
                for var_i, expr_i in assigns:
                    assert expr_i.args[0] == var_j
                    var_j__k = var_i
                    assert var_j__k not in multiuse_var2singleuse_vars
                    assert var_j__k not in singleuse_var2multiuse_var
                    multiuse_var2singleuse_vars[var_j].append(var_j__k)
                    singleuse_var2multiuse_var[var_j__k] = var_j

        multiuse_vars = set()
        for var_j, occurrences in var2occurrences.items():
            if occurrences >= 2:
                assert var_j not in output_vars, f"{var_j} in {output_vars}"
                multiuse_vars.add(var_j)

        # var2occurrences = out_var + external vars
        all_vars = list(input_vars) + list(output_vars) + list(var2occurrences)
        branch_infix = "__"
        while True:
            for v in all_vars:
                if branch_infix in str(v):
                    branch_infix += "_"
                    break
            else:
                break

        # 2nd iteration: replace multiuse variables by new unique variables
        #                and add new assignments to define the new variables
        new_assignments = []
        for var_i, expr_i in assignments.items():
            # var_i no need to replace, only the vars (the args) in expr_i
            new_args_expr_i = []

            for var_j in expr_i.args:
                if not isinstance(var_j, core.Variable) or var_j not in multiuse_vars:
                    new_args_expr_i.append(var_j)
                else:
                    # var_j is a branched variable that appears in expr_i
                    # var_j__k is the new variable replacing (in expr_i) the k-th global occurrence var_j

                    if var_j not in multiuse_var2singleuse_vars:  # first occurrence of var_j in ssa
                        multiuse_var2singleuse_vars[var_j] = []
                        # we define all var_j__k variables consecutively for readability
                        # with at least 2 underscores (branch_infix) to avoid collisions with MemoizationTable
                        for k in range(var2occurrences[var_j]):
                            var_j__k = core.Variable(f"{var_j.name}{branch_infix}{k}", var_j.width)
                            if var_j__k in all_vars:
                                raise ValueError(f"new variable var_j{branch_infix}k={var_j__k} already in SSA, "
                                                 f"input_vars={input_vars}, output_vars={output_vars},\n"
                                                 f"assignments={assignments}")
                            new_assignments.append((var_j__k, operation.BvIdentity(var_j)))

                    name_var_j__k = "{}{}{}".format(var_j.name, branch_infix, len(multiuse_var2singleuse_vars[var_j]))
                    var_j__k = core.Variable(name_var_j__k, var_j.width)
                    new_args_expr_i.append(var_j__k)

                    assert var_j__k not in multiuse_var2singleuse_vars
                    assert var_j__k not in singleuse_var2multiuse_var
                    multiuse_var2singleuse_vars[var_j].append(var_j__k)
                    singleuse_var2multiuse_var[var_j__k] = var_j

            if new_args_expr_i != expr_i.args:
                expr_i = type(expr_i)(*new_args_expr_i)
            new_assignments.append((var_i, expr_i))

        new_assignments = collections.OrderedDict(new_assignments)

        return new_assignments, multiuse_var2singleuse_vars, singleuse_var2multiuse_var

    @staticmethod
    def _are_last_assignments_valid(assignments, output_vars, ignore_exception=True):
        """Check whether the last assignments are SSAReturn of the output variables."""
        assert isinstance(assignments, collections.OrderedDict)
        if len(assignments) == 0:
            return False
        last_assignments = []
        for assign_outvar in reversed(assignments):
            last_assignments.append([assign_outvar, assignments[assign_outvar]])
            if len(last_assignments) == len(output_vars):
                break
        last_assignments = list(reversed(last_assignments))  # proper order
        for i, (assign_outvar, expr) in enumerate(last_assignments):
            if not(assign_outvar == output_vars[i]) or \
                    not(isinstance(expr, SSAReturn)) or \
                    not(expr.args[0] not in output_vars):
                if not ignore_exception:
                    last_assignments_vrepr = [(k.vrepr(), v.vrepr()) for (k, v) in last_assignments]
                    raise ValueError("last assignments are not of the form "
                                     "output_var <- SSAReturn(non_output_var)"
                                     f"\noutput vars = {output_vars}"
                                     f"\nlast assignments {last_assignments_vrepr}"
                                     f"\n{assignments}")
                return False
        return True

    @staticmethod
    def _make_last_assignments_valid(assignments, output_vars, input_vars):
        assert isinstance(assignments, collections.OrderedDict)
        # avoid using MemoizationTable since duplicate expr will be added

        # any output var appearing twice or more in the list of output_vars
        # is replaced by a new variable with a different name
        for i in range(len(output_vars)):
            while any(output_vars[i] == v for v in output_vars[:i]):
                old_v_i = output_vars[i]
                # new name just _ to avoid collision with MemoizationTable and _replace_multiuse_vars
                new_v_i = core.Variable(f"{old_v_i.name}_", old_v_i.width)
                if new_v_i in assignments or new_v_i in input_vars or new_v_i in output_vars:
                    raise ValueError(f"new variable new_v_{i}={new_v_i} already in SSA, "
                                     f"input_vars={input_vars}, output_vars={output_vars},"
                                     f"\nassignments={assignments}")

                # duplicate expr in assignments are allowed
                assignments[new_v_i] = operation.BvIdentity(old_v_i)

                for j in range(i, len(output_vars)):
                    if output_vars[j] == old_v_i:
                        output_vars[j] = new_v_i

        assert len(set(output_vars)) == len(output_vars)

        for i in range(len(output_vars)):
            # SSAReturn is used to ensure the output variables
            # come from the last operations in the MemoizationTable
            # (SSAReturn used instead of BvIdentity to ensure that
            # SSAReturn inputs are not outputs of previous SSAReturn)
            v = output_vars[i]
            new_v_i = core.Variable(f"{v.name}_out", v.width)
            new_expr = SSAReturn(v)
            if new_v_i in assignments or new_v_i in input_vars or new_v_i in output_vars or \
                    new_expr in set(assignments.values()):
                raise ValueError(f"new assignment new_v_{i}={new_v_i} <- {new_expr.vrepr()} "
                                 f"already in SSA (either the output variable or the expr), "
                                 f"input_vars={input_vars}, output_vars={output_vars},"
                                 f"\nassignments={assignments}")
            output_vars[i] = new_v_i
            assignments[new_v_i] = new_expr

        assert len(set(output_vars)) == len(output_vars)

    def __str__(self):
        if self.external_vars:
            aux_str = f" external_vars=[{', '.join([str(v) for v in self.external_vars])}],"
        else:
            aux_str = ""
        return "{}(input_vars={}, output_vars={},{} assignments={})".format(
            type(self).__name__,
            # str(self.input_vars) ignored to print it in a list-like way
            f"[{', '.join([str(v) for v in self.input_vars])}]",
            f"[{', '.join([str(v) for v in self.output_vars])}]",
            aux_str,
            f"[{', '.join([f'({v}, {e})' for v, e in self.assignments.items()])}]",
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that
        ``eval(self.vrepr())`` returns a new `SSA` object
        with the same content.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvXor, BvIdentity
            >>> from cascada.bitvector.ssa import BvFunction
            >>> z = Variable("z", 8)
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):  return y ^ z, x ^ 1
            >>> MyFoo.to_ssa(["x", "y"], "i").vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "SSA(input_vars=[Variable('x', width=8), Variable('y', width=8)],
                output_vars=[Variable('i0_out', width=8), Variable('i1_out', width=8)],
                assignments=[(Variable('i0', width=8), BvXor(Variable('y', width=8), Variable('z', width=8))),
                    (Variable('i1', width=8),
                        make_partial_operation(BvXor, (None, Constant(0b00000001, width=8)))(Variable('x', width=8))),
                    (Variable('i0_out', width=8), SSAReturn(Variable('i0', width=8))),
                    (Variable('i1_out', width=8), SSAReturn(Variable('i1', width=8)))])"

        .. Implementation details:
            Since the equality operator is not implemented,
            ``eval(self.vrepr()) == self`` does NOT hold.

            This method does not store all the information,
            only the info needed to be recreated the object.

        """
        return "{}(input_vars={}, output_vars={}, assignments={}{})".format(
            type(self).__name__,
            f"[{', '.join([v.vrepr() for v in self.input_vars])}]",
            f"[{', '.join([v.vrepr() for v in self.output_vars])}]",
            f"[{', '.join([f'({v.vrepr()}, {e.vrepr()})' for v, e in self.assignments.items()])}]",
            "" if not self._replace_multiuse_vars_bool else ", replace_multiuse_vars=True",
        )

    def get_C_code(self, C_function_name):
        """Return two strings (header and body) of a C function evalauting the SSA.

        This method returns two strings (the function header and the function body)
        of a function in the C programming language that computes the SSA.

        The C function is of ``void`` type and its list of arguments consists of
        the input variables, the external variables (if any) and the output variables.
        The output variables are defined as pointers to store the results.

        See also `crepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvXor, BvIdentity
            >>> from cascada.bitvector.ssa import BvFunction
            >>> z = Variable("z", 8)
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):  return y ^ z, x ^ 1
            >>> header, body = MyFoo.to_ssa(["x", "y"], "i").get_C_code("MyFoo")
            >>> header
            'void MyFoo(uint8_t x, uint8_t y, uint8_t z, uint8_t *i0_out, uint8_t *i1_out);'
            >>> print(body)  # doctest: +NORMALIZE_WHITESPACE
            #include <stdint.h>
            void MyFoo(uint8_t x, uint8_t y, uint8_t z, uint8_t *i0_out, uint8_t *i1_out){
                uint8_t i0 = y ^ z;
                uint8_t i1 = x ^ 0x1U;
                *i0_out = i0;
                *i1_out = i1;
            };

        """
        from cascada.bitvector.printing import BvCCodePrinter

        width2type = BvCCodePrinter._width2C_type

        # in C, * binds to the declarator, not the type specifier
        input_vars_c = ', '.join(["{} {}".format(width2type(v.width), v.name) for v in self.input_vars])
        output_vars_c = ', '.join(["{} *{}".format(width2type(v.width), v.name) for v in self.output_vars])
        if self.external_vars:
            external_vars_c = ', '.join(["{} {}".format(width2type(v.width), v.name) for v in self.external_vars])
            external_vars_c = external_vars_c + ", "
        else:
            external_vars_c = ""

        aux = f"void {C_function_name}({input_vars_c}, {external_vars_c}{output_vars_c})"
        header = f"{aux};"
        body = f"#include <stdint.h>\n{aux}{{"  # stdint for uint_*

        outvar2outvar_c = {v: core.Variable("*" + v.name, v.width, allowed_symbols="*") for v in self.output_vars}

        def primary_assignment2C_code(my_var, my_expr):
            assert isinstance(my_expr, (core.Constant, core.Variable, operation.PrimaryOperation))
            if my_var in self.output_vars:
                return f"*{my_var} = {my_expr.crepr()};"
            else:
                return f"{width2type(my_var.width)} {my_var} = {my_expr.crepr()};"

        for var, expr in self.assignments.items():
            expr = expr.xreplace(outvar2outvar_c)
            if isinstance(expr, operation.SecondaryOperation):
                expr = expr.doit(eval_sec_ops=True)
            body += f"\n\t{primary_assignment2C_code(var, expr)}"
        body += "\n};"

        return header, body

    def eval(self, input_vals, external_var2val=None, C_code=False, **kwargs):
        """Evaluate the SSA with `Constant` inputs and return the `Constant` outputs.

        If some external variable is not given, it is replaced by the zero bit-vector.

            >>> from cascada.bitvector.core import Variable, Constant
            >>> from cascada.bitvector.ssa import BvFunction
            >>> z = Variable("z", 8)
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):  return y ^ z, x
            >>> my_ssa = MyFoo.to_ssa(["x", "y"], "i")
            >>> my_ssa  # doctest: +NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[i0_out, x_out], external_vars=[z],
                assignments=[(i0, y ^ z), (i0_out, Id(i0)), (x_out, Id(x))])
            >>> my_ssa.external_vars
            (z,)
            >>> my_ssa.eval([Constant(0, 8), Constant(0, 8)])
            (0x00, 0x00)

        Args:
            input_vals: a list of `Constant` objects representing the inputs.
            external_var2val: a dictionary mapping the external variables
                to its `Constant` values.
            C_code: whether to evaluate the SSA by compiling and running
                its C code (see also `get_C_code`).

        """
        if external_var2val is None:
            external_var2val = {}

        if C_code:
            # avoid extra asserts in C_code
            header, body = self.get_C_code("eval_ssa")
            pymod, tmpdir = _compile_C_code(header, body, verbose=False)

            from cascada.bitvector.printing import BvCCodePrinter

            ffi = pymod.ffi

            width2type = BvCCodePrinter._width2C_type

            input_vals = [ffi.cast(width2type(x.width), int(x)) for x in input_vals]
            external_vals = [ffi.cast(width2type(x.width), int(external_var2val.get(x, 0)))
                             for x in self.external_vars]
            output_vals = [ffi.new(f"{width2type(x.width)} *") for x in self.output_vars]

            args = input_vals + external_vals + output_vals

            pymod.lib.eval_ssa(*args)

            offset = len(input_vals) + len(external_vals)
            for i in range(len(output_vals)):
                # ·[0] to access the content of the pointer
                output_vals[i] = core.Constant(args[offset + i][0], self.output_vars[i].width)
            del args  # free pointers

            tmpdir.cleanup()
            return tuple(output_vals)
        else:
            if kwargs.get("symbolic_inputs", False) is False:
                assert all(isinstance(x, core.Constant) for x in input_vals)
            assert len(input_vals) == len(self.input_vars)

            substitutions = {}

            for var, val in zip(self.input_vars, input_vals):
                substitutions[var] = val

            for var in self.external_vars:
                substitutions[var] = external_var2val.get(var, core.Constant(0, var.width))

            for var, expr in self.assignments.items():
                expr = expr.xreplace(substitutions)
                if kwargs.get("symbolic_inputs", False) is False:
                    assert isinstance(expr, core.Constant), "{} <- {}".format(var, expr)
                substitutions[var] = expr

            output_vals = []
            for var in self.output_vars:
                output_vals.append(substitutions[var])

            return tuple(output_vals)

    def to_bvfunction(self):
        """Return a `BvFunction` that evaluates  ``self``.

        The returned `BvFunction` stores ``self`` in the ``_ssa`` attribute.
        Moreover, the method `BvFunction.to_ssa` of the returned `BvFunction`
        raises an exception unless it is called with the same arguments that
        were used to create ``_ssa`` (in this case ``_ssa`` is returned).

            >>> from cascada.bitvector.ssa import BvFunction
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):  return x ^ y, x
            >>> MyFoo.to_ssa(["x", "y"], "a")  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
                assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])
            >>> MyFoo_v2 = MyFoo.to_ssa(["x", "y"], "a").to_bvfunction()
            >>> MyFoo_v2.to_ssa(["x", "y"], "a")  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
                assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])

        """
        _input_widths = [v.width for v in self.input_vars]
        _output_widths = [v.width for v in self.output_vars]

        class MyBvFunction(BvFunction):
            input_widths = _input_widths
            output_widths = _output_widths
            _ssa = self

            @classmethod
            def vrepr(cls):
                return f"{self.vrepr()}.to_bvfunction()"

            @classmethod
            def to_ssa(cls, input_names, id_prefix, decompose_sec_ops=False, **ssa_options):
                old_names = [str(v) for v in cls._ssa.input_vars]
                if list(input_names) != old_names or decompose_sec_ops is True or \
                        ssa_options.get("replace_multiuse_vars", False) != cls._ssa._replace_multiuse_vars_bool:
                    raise ValueError(f"the arguments of to_ssa(input_names={input_names}, "
                                     f"decompose_sec_ops={decompose_sec_ops}, {ssa_options})"
                                     f" are not the arguments used to generate {cls._ssa}")
                return cls._ssa

            @classmethod
            def eval(cls, *input_vals):
                d = {}

                for var, val in zip(cls._ssa.input_vars, input_vals):
                    d[var] = val

                for outvar, expr in cls._ssa.assignments.items():
                    # avoiding expr.xreplace(d)
                    # (only calls Operation.__new__ if there is some substitution)
                    out_val = type(expr)(*[d.get(arg, arg) for arg in expr.args])
                    d[outvar] = out_val

                output_vals = []
                for var in cls._ssa.output_vars:
                    output_vals.append(d.get(var, var))

                return tuple(output_vals)

        return MyBvFunction

    def copy(self):
        """Return a deep copy of ``self``.

            >>> from cascada.bitvector.ssa import BvFunction
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8, 8], [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y):  return x ^ y, x
            >>> ssa = MyFoo.to_ssa(["x", "y"], "a")
            >>> ssa  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
                assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])
            >>> ssa_copy = ssa.copy()
            >>> ssa_copy  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
                assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])
            >>> ssa_copy.input_vars = tuple()
            >>> ssa_copy.output_vars = tuple()
            >>> ssa_copy.assignments = collections.OrderedDict()
            >>> ssa_copy
            SSA(input_vars=[], output_vars=[], assignments=[])
            >>> ssa  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a0_out, x_out],
                assignments=[(a0, x ^ y), (a0_out, Id(a0)), (x_out, Id(x))])

        """
        import copy as python_copy
        my_ssa = python_copy.copy(self)  # fast shallow copy (avoid __init__)
        my_ssa.input_vars = my_ssa.input_vars[:]
        my_ssa.output_vars = my_ssa.output_vars[:]
        my_ssa.assignments = my_ssa.assignments.copy()
        return my_ssa

    def split(self, var_separators):
        """Split into multiple `SSA` objects given the list of variable separators.

        Given the `SSA` :math:`s`, this method returns a list of `SSA` objects
        :math:`s_1, s_2, ..., s_n`, such that their composition
        :math:`s_n \circ s_{n-1} \dots \circ s_1` is functionally equivalent to
        :math:`s`.

        The argument ``var_separators`` is a list containing lists of variables.
        The :math:`i`-th variable list denote the last variables of :math:`s_i`.
        In other words, the list of assignments of :math:`s_{i+1}` immediately
        starts after the last assignments of the last variable in
        ``var_separators[i]``.

        To split into :math:`n` `SSA` objects, ``var_separators`` must contain
        :math:`n-1` lists, as the variable list for :math:`s_n` is not given
        (its last variables are the output variables of :math:`s`).

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.ssa import BvFunction
            >>> z = Variable("z", 8)
            >>> var_separators = []
            >>> class MyFoo(BvFunction):
            ...     input_widths, output_widths = [8], [8]
            ...     @classmethod
            ...     def eval(cls, x0):
            ...         var_separators.clear()
            ...         x1, y1 = (x0 + x0), x0
            ...         non_used_var = x1 + y1
            ...         var_separators.append([x1, y1, non_used_var])
            ...         return [(x1 | y1) & z]
            >>> ssa = MyFoo.to_ssa(["x"], "a")
            >>> ssa  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x], output_vars=[a3_out], external_vars=[z],
                assignments=[(a0, x + x), (a2, a0 | x), (a3, a2 & z), (a3_out, Id(a3))])
            >>> var_separators
            [[a0, x, a1]]
            >>> for sub_ssa in ssa.split(var_separators): print(sub_ssa)  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x], output_vars=[a0_out, x_out],
                assignments=[(a0, x + x), (a0_out, Id(a0)), (x_out, Id(x))])
            SSA(input_vars=[a0, x], output_vars=[a3_out], external_vars=[z],
                assignments=[(a2, a0 | x), (a3, a2 & z), (a3_out, Id(a3))])

        """
        assert len(var_separators) >= 1

        _listify = lambda s: list(s) if isinstance(s, collections.abc.Sequence) else [s]

        input_vars_set = set(self.input_vars)
        external_vars_set = set(self.external_vars)

        # remove input and external vars from var_separators
        var_separators = [[x for x in _listify(vs_list)
                           if x not in input_vars_set|external_vars_set]
                          for vs_list in var_separators]

        assert tuple(var_separators[-1]) != tuple(self.output_vars)

        for i in range(len(var_separators)):
            if len(var_separators[i]) == 0:
                raise ValueError(f"no non-input/non-external vars in var_separators[{i}]")
            for j in range(len(var_separators[i])):
                var = var_separators[i][j]
                if j == len(var_separators[i]) - 1 and self.singleuse_var2multiuse_var.get(var, None) in external_vars_set:
                    is_last_copy = self.multiuse_var2singleuse_vars[self.singleuse_var2multiuse_var[var]][-1] == var
                    if not is_last_copy:
                        raise ValueError(f"split does not support copies of external variables ({var}) in var_separators")
                if var not in self.assignments:
                    assert var in self._old_outvar_assignments, \
                        f"var {var} is not in SSA\n - {self}\nand not in old SSA\n - {self._old_outvar_assignments}"
                    old_index = self._old_outvar_assignments.index(var)
                    for new_var_index in reversed(range(old_index)):
                        new_var = self._old_outvar_assignments[new_var_index]
                        if new_var in self.assignments:
                            # it doesn't matter if new_var in var_separators[i]
                            # (duplicates removed later)
                            warnings.warn(f"var_separators[{i}][{j}] = {var} replaced by {new_var}")
                            var_separators[i][j] = new_var
                            break
                    else:
                        raise ValueError(f"var_separators[{i}][{j}] = {var} not in assignments\n{self}")
            # removing duplicates
            var_separators[i] = set(var_separators[i])
            for prev_i in range(0, i):
                var_separators[i].difference_update(var_separators[prev_i])
            if len(var_separators[i]) == 0:
                raise ValueError(f"var_separators[{i}] only contains variables from "
                                 f"var_separators[j] with j < {i}")

        # var_terminators is var_separators w/ output vars (entries required to be sets)
        var_terminators = var_separators + [set(self.output_vars)]

        class SubSSA(object):
            replace_multiuse_vars = self._replace_multiuse_vars_bool

            def __init__(self_subssa):
                self_subssa.input_vars = []
                self_subssa.output_vars = []
                self_subssa.assignments = collections.OrderedDict()
                self_subssa.external_vars = []

            def __str__(self_subssa):
                msg = "SubSSA:"
                msg += f"\n - input_vars: {self_subssa.input_vars}"
                msg += f"\n - output_vars: {self_subssa.output_vars}"
                msg += f"\n - external_vars: {self_subssa.external_vars}"
                msg += f"\n - assignments: {self_subssa.assignments}"
                return msg

        sub_ssa_list = [SubSSA() for _ in range(len(var_terminators))]
        sub_ssa_list[0].input_vars = self.input_vars[:]
        sub_ssa_list[-1].output_vars = self.output_vars[:]

        # fill assignments and external_vars.
        # when var_terminators[i] is empty we move to the next sub_ssa
        i = 0
        for assign_out_var, assign_expr in self.assignments.items():
            sub_ssa_list[i].assignments[assign_out_var] = assign_expr
            var_terminators[i].discard(assign_out_var)
            for v in assign_expr.args:
                assert isinstance(v, core.Variable)
                if v in external_vars_set:
                    sub_ssa_list[i].external_vars.append(v)
            # # debugging
            # print(f" - added {(assign_out_var, assign_expr)} to sub_ssa_list[{i}] and added"
            #       f"{[v for v in assign_expr.args if v in external_vars_set]} to "
            #       f"sub_ssa_list[{i}].external_vars")
            if len(var_terminators[i]) == 0:
                i += 1
        assert i == len(sub_ssa_list)

        def get_input_vars(my_sub_ssa):
            vars_not_defined = []  # input vars
            vars_defined = set(my_sub_ssa.external_vars)
            for assign_out_var, assign_expr in my_sub_ssa.assignments.items():
                for arg in assign_expr.args:
                    if isinstance(arg, core.Variable) and arg not in vars_defined:
                        if arg not in vars_not_defined:  # avoid duplicates
                            # print(f"   - var {arg} added to {vars_not_defined} from {(assign_out_var, assign_expr)}")
                            vars_not_defined.append(arg)
                vars_defined.add(assign_out_var)
            assert len(vars_defined.intersection(set(vars_not_defined))) == 0
            # input vars also contain output vars not defined
            assert len(my_sub_ssa.output_vars) > 0
            vars_defined |= set(vars_not_defined)  # ows input_vars contain duplicates
            for output_var in my_sub_ssa.output_vars:
                if output_var not in vars_defined:
                    # output_var can still be used in some assignment expression
                    vars_not_defined.append(output_var)
            return vars_not_defined

        def check_output_vars(my_sub_ssa, my_i):
            # check all output vars are in input_vars or assignments
            aux_set = set(my_sub_ssa.input_vars) | set(my_sub_ssa.assignments)
            assert all(v in aux_set for v in my_sub_ssa.output_vars), \
                f"sub_ssa_list[{my_i}] | output_vars={my_sub_ssa.output_vars} not in " \
                f"input_vars={my_sub_ssa.input_vars} or assignments={my_sub_ssa.assignments}"

        # fill sub_ssa[i].input_vars from get_input_vars_from_sub_ssa(sub_ssa[i])
        # fill sub_ssa[i-1].output_vars from sub_ssa[i].input_vars
        for i in reversed(range(len(sub_ssa_list))):
            if i == 0:
                assert len(sub_ssa_list[i].input_vars) > 0
            else:
                # copy vars to sub_ssa_list to avoid sharing the vars between multiple objects
                sub_ssa_list[i].input_vars = get_input_vars(sub_ssa_list[i])[:]
                sub_ssa_list[i-1].output_vars = sub_ssa_list[i].input_vars[:]
            check_output_vars(sub_ssa_list[i], i)

        for i in range(len(sub_ssa_list)):
            # after calling SSA(), the suffix of output_vars might change
            # (some ``_`` or ``_out`` might be added)
            # but better not to use the new output_vars for the input vars
            # of the next sub_ssa (to avoid variables with suffix ``_out_out_out``)
            if i >= 1:
                # check that the base name (except the suffix) has not changed
                for v, prev_v in zip(sub_ssa_list[i].input_vars, sub_ssa_list[i-1].output_vars):
                    assert v.name == prev_v.name[:len(v.name)]

            check_output_vars(sub_ssa_list[i], i)

            sub_ssa_list[i] = SSA(input_vars=sub_ssa_list[i].input_vars,
                                  output_vars=sub_ssa_list[i].output_vars,
                                  assignments=sub_ssa_list[i].assignments,
                                  replace_multiuse_vars=SubSSA.replace_multiuse_vars)

            for ev in sub_ssa_list[i].external_vars:
                if ev not in external_vars_set:
                    raise ValueError(f"sub_ssa_list[{i}].external_vars {sub_ssa_list[i].external_vars} "
                                     f"not a subset of parent SSA external_vars {external_vars_set}")

        return sub_ssa_list

    def get_round_separators(self):
        """Return the round separators if the SSA was obtained from a `RoundBasedFunction`.

        If the `SSA` object was obtained from `RoundBasedFunction.to_ssa` of a
        `RoundBasedFunction` including `add_round_outputs` calls in its ``eval``,
        this method returns a list with the round outputs delimiting the rounds.
        Otherwise, ``None`` is returned.

        In the first case, this list contains ``num_rounds - 1`` entries,
        where the ``i``-th entry is the list of outputs of the ``i``-th round.
        In particular, the outputs of the last round are not included in this list.

        The list returned by this method is meant to be used as the argument
        of `split` to get the `SSA` object of each round.

            >>> from cascada.bitvector.ssa import RoundBasedFunction
            >>> class MyFoo(RoundBasedFunction):
            ...     input_widths, output_widths, num_rounds = [8, 8], [8, 8], 1
            ...     @classmethod
            ...     def eval(cls, x, y):
            ...         for _ in range(cls.num_rounds):
            ...             x, y = y, x + 1
            ...             cls.add_round_outputs(x, y)
            ...         return x, y
            ...     @classmethod
            ...     def set_num_rounds(cls, new_num_rounds):  cls.num_rounds = new_num_rounds
            >>> MyFoo.set_num_rounds(3)
            >>> ssa = MyFoo.to_ssa(["x", "y"], "a")
            >>> ssa  # doctest: +NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[a1_out, a2_out],
                assignments=[(a0, x + 0x01), (a1, y + 0x01), (a2, a0 + 0x01),
                    (a1_out, Id(a1)), (a2_out, Id(a2))])
            >>> round_separators = ssa.get_round_separators()
            >>> round_separators
            ((y, a0), (a0, a1))
            >>> for sub_ssa in ssa.split(round_separators): print(sub_ssa)  # doctest:+NORMALIZE_WHITESPACE
            SSA(input_vars=[x, y], output_vars=[y_out, a0_out],
                assignments=[(a0, x + 0x01), (y_out, Id(y)), (a0_out, Id(a0))])
            SSA(input_vars=[y, a0], output_vars=[a0_out, a1_out],
                assignments=[(a1, y + 0x01), (a0_out, Id(a0)), (a1_out, Id(a1))])
            SSA(input_vars=[a0, a1], output_vars=[a1_out, a2_out],
                assignments=[(a2, a0 + 0x01), (a1_out, Id(a1)), (a2_out, Id(a2))])

        """
        if getattr(self, "_rounds_outputs", None) is None:
            return None
        if len(self._rounds_outputs) == 0:
            return None
        return self._rounds_outputs[:-1]

    def dotprinting(self, repeat=True, vrepr_label=False, **kwargs):
        """Return the DOT description of the expression tree of the assignments.

        See also `printing.dotprinting`.

        Args:
            repeat: whether to use different nodes for common subexpressions
                (default True)
            vrepr_label: whether to use the verbose representation (`Term.vrepr`)
                to label the nodes (default False)
            kwargs: additional arguments passed to `printing.dotprinting`

        """
        from cascada.bitvector.printing import dotprinting
        from sympy.core.containers import Tuple
        expr = Tuple(*self.assignments.items())
        return dotprinting(expr, repeat=repeat, vrepr_label=vrepr_label, **kwargs)


@functools.lru_cache(maxsize=None)
def get_random_bvfunction(width, num_inputs, num_outputs, num_assignments, seed,
                          external_variable_prefix=None, operation_set_index=0, num_rounds=None,
                          extra_operations=None):
    """Return a random `BvFunction` with given shape.

    Args:
        width: the common bitsize of the input and output variables of the function
        num_inputs: the number of inputs of the function
        num_outputs: the number of outputs of the function
        num_assignments: an estimation of the number of operations within the function
        seed: the seed used when sampling
        external_variable_prefix: if not ``None``, external variables are used
            with the given prefix (at least one)
        operation_set_index: four set of operations to choose indexed by 0, 1, 2 and 3
        num_rounds: if not ``None``, returns a random `RoundBasedFunction` with
            the given number of rounds
        extra_operations: an optional `tuple` containing `Operation` subclasses
            to add to the list of operations to choose

    ::

        >>> from cascada.bitvector.ssa import get_random_bvfunction
        >>> my_foo = get_random_bvfunction(4, 2, 2, 2, seed=0)
        >>> my_foo.to_ssa(["x", "y"], "a")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[x, y], output_vars=[a0_out, a1_out],
            assignments=[(a0, y >> 0x2), (a1, ~y), (a0_out, Id(a0)), (a1_out, Id(a1))])
        >>> my_foo = get_random_bvfunction(4, 1, 1, 2, seed=20)
        >>> my_foo.to_ssa(["x"], "a")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[x], output_vars=[a7_out],
            assignments=[(a0, x[0]), (a1, x[1]), (a2, x[2]), (a3, x[3]),
            (a4, a0 :: a1), (a5, a4 :: a2), (a6, a5 :: a3), (a7, a6 + 0xe), (a7_out, Id(a7))])

    """
    assert num_inputs + num_assignments >= num_outputs

    import random
    import functools
    from cascada.bitvector.operation import (
        BvAnd, BvOr, BvXor, BvShl, BvLshr, RotateLeft, RotateRight, BvAdd,
        BvSub, BvNeg, BvNot, Concat, BvIdentity, make_partial_operation
    )
    from cascada.bitvector.secondaryop import BvIf, BvMaj

    PRNG = random.Random()
    PRNG.seed(seed)

    # SimpleReverse contains Concat and Extract
    # implemented as a class to later use issubclass()
    class SimpleReverse(object):
        def __new__(cls, bv):
            return functools.reduce(Concat, [bv[i] for i in range(bv.width)])

    if operation_set_index == 0:
        list_ops = (
            BvNot, BvXor, BvAnd, BvOr,
            BvAdd, BvSub, BvNeg,
            BvIdentity,
            BvIf, BvMaj,
            RotateLeft, RotateRight,
            BvShl, BvLshr,
            SimpleReverse
            # Concat, Extract, Ite,
            # BvComp, BvUlt, BvUle, BvUgt, BvUge,
            # BvMul, BvUdiv, BvUrem,
            # PopCount, Reverse, PopCountSum2, PopCountSum3, PopCountDiff, LeadingZeros,
        )
    elif operation_set_index == 1:
        list_ops = (
            BvNot, BvXor, BvAnd, BvOr,
            BvAdd, BvSub, BvNeg,
            BvIdentity,
            BvIf, BvMaj,
            RotateLeft, RotateRight,
            make_partial_operation(BvShl, (None, core.Constant(PRNG.randint(1, width - 1), width))),
            make_partial_operation(BvLshr, (None, core.Constant(PRNG.randint(1, width - 1), width))),
        )
    elif operation_set_index == 2:
        list_ops = (
            BvNot, BvXor, BvAnd, BvOr,
            BvAdd, BvSub,
            BvIdentity,
            RotateLeft, RotateRight,
            BvShl, BvLshr,
            SimpleReverse
        )
    elif operation_set_index == 3:
        list_ops = (
            BvNot, BvXor, BvAnd, BvOr,
            BvAdd, BvSub, BvNeg,
            BvIdentity,
            BvIf, BvMaj,
            RotateLeft, RotateRight,
            BvShl, BvLshr,
            SimpleReverse
        )
    else:
        raise ValueError("operation_set_index must be 0, 1, 2, or 3")

    extra_operations_vrepr = None
    if extra_operations is not None:
        list_ops += tuple(extra_operations)
        extra_operations_vrepr = f"({','.join(op.__name__ for op in extra_operations)},)"

    while True:  # outer loop to check RandomBvFunction does not return Constant
        list_of_lambda_assignments = []

        class RandomBvFunction(BvFunction):
            input_widths = [width] * num_inputs
            output_widths = [width] * num_outputs
            
            @classmethod
            def vrepr(cls):
                evp = external_variable_prefix.__repr__()
                return f"get_random_bvfunction({width}, {num_inputs}, {num_outputs}, " \
                       f"{num_assignments}, {seed}, {evp}, {operation_set_index}, {num_rounds}, " \
                       f"{extra_operations_vrepr})"

        if external_variable_prefix is not None:
            RandomBvFunction.round_keys = []

        def get_random_var_index():
            if len(list_of_lambda_assignments) >= num_inputs + 2:
                min_index = num_inputs
            else:
                min_index = 0
            return PRNG.randint(min_index, len(list_of_lambda_assignments) + num_inputs - 1)

        def get_random_var_indices(num_indices, my_unique_indices):
            indices = []
            while True:
                new_index = get_random_var_index()
                if not my_unique_indices or new_index not in indices:
                    indices.append(new_index)
                if len(indices) == num_indices:
                    break
            return indices

        while True:
            _op = list_ops[PRNG.randint(0, len(list_ops) - 1)]

            if _op in [RotateLeft, RotateRight]:  # only operations with scalar inputs
                class Op(object):  # need a class to store the randomness
                    op = _op
                    my_index = get_random_var_index()
                    offset = PRNG.randint(1, width - 1)  # != 0
                    # kwargs required for external vars (see below)
                    def __new__(cls, args, **kwargs): return cls.op(args[cls.my_index], cls.offset)

            elif _op == SimpleReverse:
                class Op(object):
                    op = _op
                    my_index = get_random_var_index()
                    def __new__(cls, args, **kwargs): return cls.op(args[cls.my_index])

            elif _op in [BvShl, BvLshr]:  # ensure 2nd operand ct
                class Op(object):
                    op = _op
                    my_index = get_random_var_index()
                    offset = core.Constant(PRNG.randint(1, width - 1), width)
                    def __new__(cls, args, **kwargs): return cls.op(args[cls.my_index], cls.offset)

            else:
                assert issubclass(_op, operation.Operation)
                assert _op.arity[1] == 0

                if _op.arity[0] >= 2 and PRNG.randint(0, 3) == 0:
                    # 1 in 4 to have a ct or external var (each 50%)
                    if external_variable_prefix is None or PRNG.randint(0, 1) == 0:
                        extra_arg = core.Constant(PRNG.randint(1, 2**width - 2), width)  # != 0, allones

                        class Op(object):
                            op = _op
                            indices = get_random_var_indices(_op.arity[0] - 1, False)
                            ct = extra_arg
                            def __new__(cls, args, **kwargs): return cls.op(*([args[i] for i in cls.indices] + [cls.ct]))
                    else:
                        evi = PRNG.randint(0, len(RandomBvFunction.round_keys))
                        if evi == len(RandomBvFunction.round_keys):
                            RandomBvFunction.round_keys.append(core.Variable(f"{external_variable_prefix}{evi}", width))

                        class Op(object):
                            op = _op
                            indices = get_random_var_indices(_op.arity[0] - 1, False)
                            ev_index = evi
                            # kwargs["round_keys"] instead of RandomBvFunction.round_keys
                            # to provide round_keys from cls in running time (in an argument of Op(
                            # (otherwise round_keys cannot be changed)
                            def __new__(cls, args, **kwargs): return cls.op(*([args[i] for i in cls.indices] +
                                                                              [kwargs["round_keys"][cls.ev_index]]))
                else:
                    unique_indices = _op in [BvAnd, BvOr, BvXor, BvSub]
                    if unique_indices and len(list_of_lambda_assignments) + num_inputs == 1:
                        # avoid using duplicated inputs for these operations
                        continue

                    class Op(object):
                        op = _op
                        indices = get_random_var_indices(_op.arity[0], unique_indices)
                        def __new__(cls, args, **kwargs): return cls.op(*[args[i] for i in cls.indices])

            if external_variable_prefix is not None and len(RandomBvFunction.round_keys) == 0:
                # ensure at least 1 external variable
                _op = BvXor
                evi = 0
                RandomBvFunction.round_keys.append(core.Variable(f"{external_variable_prefix}{evi}", width))

                class Op(object):
                    op = _op
                    indices = get_random_var_indices(_op.arity[0] - 1, False)
                    ev_index = evi
                    def __new__(cls, args, **kwargs): return cls.op(*([args[i] for i in cls.indices] +
                                                                      [kwargs["round_keys"][cls.ev_index]]))

            ## debugging
            # print(f"{len(list_of_lambda_assignments)+1}/{num_assignments} | op: {Op.op}")
            # if hasattr(Op, "my_index"):  print("\tmy_index:", Op.my_index)
            # if hasattr(Op, "indices"):  print("\tindices:", Op.indices)
            # if hasattr(Op, "ct"):  print("\tct:", Op.ct)
            # if hasattr(Op, "offset"):  print("\toffset:", Op.offset)
            # if hasattr(Op, "ev_index"):  print("\tev_index:", Op.ev_index)

            list_of_lambda_assignments.append(Op)
            if len(list_of_lambda_assignments) == num_assignments:
                break

        list_of_lambda_assignments = tuple(list_of_lambda_assignments)

        @classmethod
        def eval_method(cls, *args):
            assert isinstance(args, collections.abc.Sequence)
            all_vars = list(args)
            round_keys = getattr(cls, "round_keys", None)
            ## debugging
            # print(f"eval_method({args}):")
            for index_assign, lambda_assign in enumerate(list_of_lambda_assignments):
                result = lambda_assign(all_vars, round_keys=round_keys)

                Op = lambda_assign
                ffo_arg0, ffo_arg1, ffo_arg2 = "", "", ""
                if hasattr(Op, "my_index"):
                    ffo_arg0 = all_vars[Op.my_index]
                else:
                    assert hasattr(Op, "indices")
                    ffo_arg0 = all_vars[Op.indices[0]]
                    if len(Op.indices) >= 2:
                        ffo_arg1 = all_vars[Op.indices[1]]
                    if len(Op.indices) >= 3:
                        ffo_arg2 = all_vars[Op.indices[2]]
                ffo_extra_args = ""
                if hasattr(Op, "ct"):
                    ffo_extra_args = Op.ct
                elif hasattr(Op, "offset"):
                    ffo_extra_args = Op.offset
                elif hasattr(Op, "ev_index"):
                    ffo_extra_args = round_keys[Op.ev_index]
                format_string = "assignment {}/{}: {}({}, {}, {}, extra={}) = {}"
                ffo = [index_assign, len(list_of_lambda_assignments) - 1,
                       Op.op.__name__, ffo_arg0, ffo_arg1, ffo_arg2,
                       ffo_extra_args, result]
                cls.log_msg(format_string, ffo)

                ## debugging
                # print(f"\t{result.width}-width op: {Op.op.__name__}")
                # if hasattr(Op, "my_index"):  print("\t\targ:", all_vars[Op.my_index])
                # if hasattr(Op, "indices"):  print("\t\targs:", [all_vars[i] for i in Op.indices])
                # if hasattr(Op, "ct"):  print("\t\tct:", Op.ct)
                # if hasattr(Op, "offset"):  print("\t\toffset:", Op.offset)
                # if hasattr(Op, "ev_index"):  print("\t\text_arg:", round_keys[Op.ev_index])
                # print(f"\t\tresult:", result)
                all_vars.append(result)
            return all_vars[-num_outputs:]

        RandomBvFunction.eval = eval_method

        # check RandomBvFunction does not return Constant
        try:
            RandomBvFunction(*[core.Variable(f"x{i}", width) for i in range(num_inputs)],
                             symbolic_inputs=True, simplify=False)
        except ValueError as e:
            if not str(e).startswith("if symbolic_inputs, expected no Constant values"):
                raise e
            else:
                continue

        if num_rounds is None:
            return RandomBvFunction
        else:
            _num_rounds = num_rounds

            class RandomRBF(RandomBvFunction, RoundBasedFunction):
                num_rounds = _num_rounds

                @classmethod
                def eval(cls, *args):
                    for index_round in range(cls.num_rounds):
                        args = super().eval(*args)

                        if index_round < cls.num_rounds - 1:
                            if len(args) < len(cls.input_widths):
                                args = args * ((len(cls.input_widths) // len(args)) + 1)
                            args = args[:len(cls.input_widths)]

                        cls.add_round_outputs(*args)

                        format_string = "round {}/{}: outputs = (" + "{}, "*len(args) + ")"
                        ffo = [index_round, cls.num_rounds - 1] + list(args)
                        cls.log_msg(format_string, ffo)
                    return args

                @classmethod
                def set_num_rounds(cls, new_num_rounds):
                    cls.num_rounds = new_num_rounds

            # check RandomRBF does not return Constant
            try:
                RandomRBF(*[core.Variable(f"x{i}", width) for i in range(num_inputs)],
                          symbolic_inputs=True, simplify=False)
            except ValueError as e:
                if not str(e).startswith("if symbolic_inputs, expected no Constant values"):
                    raise e
                else:
                    continue
            else:
                return RandomRBF


def _compile_C_code(header, body, return_unloaded=False, verbose=False):
    """Compile a C function given its C code as two strings (function header and body).

    This method returns a list of two objects: ``pymod`` and ``tmpdir``.

    The first object ``pymod``  is a python CFFI wrapper of the compiled
    the C function. The C function can be called as ``pymod.lib.foo_name()``
    where ``foo_name`` is the name of the C function given in the C code.

    The second object ``tmpdir`` is the temporary directory (obtained from
    `tempfile.TemporaryDirectory` where the python module and the DLL are stored).
    This directory can be emptied by ``tmpdir.cleanup()``.

    If ``return_unloaded`` is True, then the path of the (unloaded) python
    wrapper, ``module_name`` and ``tmpdir`` are returned instead.

    Sources:
    https://cffi.readthedocs.io/en/latest/overview.html#api-mode-calling-the-c-standard-library
    """
    import importlib
    import tempfile
    import uuid

    import cffi

    module_name = "module_" + uuid.uuid4().hex

    if "__uint128" in header:
        raise ValueError("_compile_C_code does not support bit-vector widths "
                         "larger than 64 bits (cffi does not support __uint128)")

    ffibuilder = cffi.FFI()
    ffibuilder.cdef(header)
    ffibuilder.set_source(module_name, body)

    tmpdir = tempfile.TemporaryDirectory()
    lib_path = ffibuilder.compile(tmpdir=tmpdir.name, verbose=verbose)

    if return_unloaded:
        return lib_path, module_name, tmpdir

    # dynamic import
    # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    spec = importlib.util.spec_from_file_location(module_name, lib_path)
    pymod_parent = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    spec.loader.exec_module(pymod_parent)

    pymod = pymod_parent

    return pymod, tmpdir
