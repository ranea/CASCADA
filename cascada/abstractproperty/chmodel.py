"""Manage bit-vector models of characteristics w.r.t an abstract property.

.. autosummary::
   :nosignatures:

    ChModelSigType
    ChModel
    EncryptionChModel
    CipherChModel
"""
import enum
import collections
import functools
import string
import warnings

from cascada.bitvector import core
from cascada.bitvector import ssa as cascada_ssa
from cascada.bitvector import operation
from cascada.abstractproperty import property
from cascada.abstractproperty import opmodel as cascada_opmodel
from cascada.primitives import blockcipher


zip = functools.partial(zip, strict=True)


class ChModelSigType(enum.Enum):
    """List of signature types for a `ChModel`.

    Attributes:
        Unique: the signature includes all the properties in the
            characteristic model needed to uniquely identity
            the characteristic model; this might include
            the input or output `Property` values,
            the non-constant external `Property` objects of the `ChModel`,
            or the input or output `Property` values
            of the non-trivial assignments.
        InputOutput: the signature only includes the input and output
            `Property` values  of the `ChModel`

    """
    Unique = enum.auto()
    InputOutput = enum.auto()


class ChModel(object):
    """Represent bit-vector models of characteristics w.r.t some property over bit-vector functions.

    A (bit-vector) model of a characteristic over a bit-vector function
    :math:`f` for a particular `Property` is a set of bit-vector constraints
    that models the probability of the characteristic.

    For the definition of characteristic and its probability
    see `abstractproperty.characteristic.Characteristic`.

    A `ChModel` is mainly given by three bit-vector formulas or constraints:
    the validity constraint (see `validity_assertions`),
    the weight constraint (see `weight_assertions`),
    and probability-one constraint (see `pr_one_assertions`)

    A `ChModel` object is defined for a type of `Property`, a `BvFunction`
    :math:`f`, the names of the input property words and the `Property`
    of the external variables of the `SSA` of :math:`f`.

    A `ChModel` object also contains the symbolic properties :math:`\Delta_{x_i}`
    in the trail as `Variable` objects and the models of the bit-vector operations
    (`abstractproperty.opmodel.OpModel` objects) of the non-trivial assignments
    :math:`x_{i+1} \leftarrow f_i(x_i)` of the `SSA` representation of :math:`f`.

    The *trivial assignments* are those that propagate properties
    deterministically (with probability one). The models for these assignments
    are not included in `ChModel`. Instead, the symbolic output property
    of each trivial assignment is automatically computed from
    the symbolic input property.

    .. note::

        A `ChModel` object can also be seen as a *symbolic* characteristic,
        where all the properties in the trail are represented by `Variable` objects.
        A particular instance of a characteristic with constant values of
        properties is represented by a
        `abstractproperty.characteristic.Characteristic` object.

    A dictionary mapping `abstractproperty.opmodel.OpModel` classes to dictionaries
    (mapping class attributes names to their values) can be given in
    the argument ``op_model_class2options``.
    In that case, the class attributes of the `abstractproperty.opmodel.OpModel`
    objects in the trail will be set to the values given in ``op_model_class2options``.
    For example, ``op_model_class2options={XorModelBvAddCt: {'precision':0}}``
    will set ``precision`` to 0 for all the `XorModelBvAddCt` models in the trail.

    If the given bit-vector function :math:`f` is a `RoundBasedFunction`
    including `add_round_outputs` calls in its ``eval``, then the
    characteristic model over each round can be obtaining by combining
    `split` and `get_round_separators`.

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic models w.r.t some property,
    (see `differential.chmodel.ChModel`, `linear.chmodel.ChModel`
    or `algebraic.chmodel.ChModel` for some examples).

    Attributes:
        func: the `BvFunction` :math:`f`.
        prop_type: the type of `Property` of the characteristic.
        input_prop: a list of `Property` objects containing
            the (symbolic) input property.
        output_prop: a list of `Property` objects containing
            the (symbolic) output property.
        external_var2prop: a `collections.OrderedDict` mapping the external
            variables of `ssa` to their associated (symbolic or
            constant) `Property` objects.
        assign_outprop2op_model: a `collections.OrderedDict` mapping the
            output `Property` :math:`\Delta_{x_{i+1}}` of
            the non-trivial assignment :math:`x_{i+1} \leftarrow f_i(x_i)` to
            the `abstractproperty.opmodel.OpModel` of this assignment.
        ssa: the `SSA` of :math:`f`, where the
            input, output and intermediate values share the same names
            as the input, output and intermediate properties of
            the characteristic.
        var2prop: a `collections.OrderedDict` mapping each variable in `ssa`
            to its associated `Property` object.

    """
    _prop_label = None

    def _func2ssa(self, func, names, prefix):
        return func.to_ssa(names, id_prefix=prefix, decompose_sec_ops=False, replace_multiuse_vars=False)

    def _propagate(self, op, input_prop, outvar_op):
        return self.prop_type.propagate(op, input_prop)

    def _get_property_missing_external_var(self, prop_type, expr_arg, external_var2prop):
        # If the `SSA` of the `BvFunction` contains external `Variable` objects
        # and their `Property` objects are not provided in the initialization
        # through ``external_var2prop``, this method is called
        raise ValueError(f"found external variable {expr_arg} with not property associated "
                         f" in external_var2prop={external_var2prop} (prop_type == {prop_type})")

    def _check_input_vars_not_used(self, func_ssa):
        return None

    def _prop_init(self, func, prop_type, input_prop_names, prefix,
                   external_var2prop, op_model_class2options):
        # auxiliary method called by __init__ and _prop_new (with "prop" labels in arguments)

        if isinstance(func, blockcipher.Cipher):
            raise ValueError(f"func argument of {self.__class__.__name__} must be a subclass of"
                             f" BvFunction (not a subclass of Cipher)")

        assert issubclass(func, cascada_ssa.BvFunction)
        assert issubclass(prop_type, property.Property)

        self.func = func
        self.prop_type = prop_type

        if len(input_prop_names) != len(func.input_widths):
            raise ValueError("len(input_prop_names) = {} != {} = len(func.input_widths)".format(
                len(input_prop_names), len(func.input_widths)))
        if len(set(input_prop_names)) < len(input_prop_names):
            raise ValueError(f"duplicated input property names not supported")

        # needed for vrepr
        self._prefix = prefix
        self._input_prop_names = input_prop_names

        self.input_prop = []
        for name, width in zip(input_prop_names, func.input_widths):
            self.input_prop.append(prop_type(core.Variable(name, width)))
        self.input_prop = tuple(self.input_prop)

        op_model_class2new_class = {}
        if op_model_class2options is not None:
            for op_model_class, options in op_model_class2options.items():
                assert issubclass(op_model_class, cascada_opmodel.OpModel)
                assert isinstance(options, dict)

                class OpModel(op_model_class):
                    pass
                for opt_str, opt_value in options.items():
                    setattr(OpModel, opt_str, opt_value)
                OpModel.__name__ = op_model_class.__name__
                op_model_class2new_class[op_model_class] = OpModel

        self._op_model_class2options = op_model_class2options

        names = [d.val.name for d in self.input_prop]
        ssa = self._func2ssa(func, names, prefix)

        self.ssa = ssa

        self._check_input_vars_not_used(ssa)

        self._logger = func._logger
        if issubclass(func, cascada_ssa.RoundBasedFunction):
            assert func._rounds_outputs == ssa._rounds_outputs
            self._rounds_outputs = ssa._rounds_outputs
        else:
            self._rounds_outputs = None

        # SSA.__init__ does not ensure but only warns if last assignments are
        # not the SSAReturn of the output variables
        last_assignments = []
        for table_key in reversed(ssa.assignments):
            last_assignments.append([table_key, ssa.assignments[table_key]])
            if len(last_assignments) == len(ssa.output_vars):
                break
        last_assignments = list(reversed(last_assignments))
        for i, (var, expr) in enumerate(last_assignments):
            if not(var == ssa.output_vars[i]) or not(isinstance(expr, cascada_ssa.SSAReturn)) or \
                    not(expr.args[0] not in ssa.output_vars):
                raise ValueError("last assignments are not of the form output_var = SSAReturn(non_output_var)"
                                 f"\noutput vars = {ssa.output_vars}"
                                 f"\nlast assignments {last_assignments}")

        self.var2prop = {}  # Variable to Property
        for var, prop in zip(ssa.input_vars, self.input_prop):
            self.var2prop[var] = prop

        if external_var2prop is not None:
            if isinstance(external_var2prop, (list, tuple)):
                self.external_var2prop = collections.OrderedDict(external_var2prop)
            else:
                self.external_var2prop = external_var2prop.copy()
            external_var2prop = None  # avoid using argument external_var2prop
            assert all(k in ssa.external_vars and isinstance(v, prop_type)
                       for k, v in self.external_var2prop.items())
            self.var2prop.update(self.external_var2prop)
        else:
            self.external_var2prop = {}  # sorted later

        self.assign_outprop2op_model = collections.OrderedDict()
        for var, expr in ssa.assignments.items():
            expr_op = type(expr)
            expr_args = []

            for expr_arg in expr.args:
                # no scalar args in expr (partial operations detected in SSA)
                assert not isinstance(expr_arg, int)
                if expr_arg not in self.var2prop:
                    assert expr_arg in ssa.external_vars and expr_arg not in self.external_var2prop
                    expr_arg_diff = self._get_property_missing_external_var(prop_type, expr_arg, self.external_var2prop)
                    self.external_var2prop[expr_arg] = expr_arg_diff
                    self.var2prop[expr_arg] = expr_arg_diff

                expr_args.append(self.var2prop[expr_arg])

            if isinstance(expr, operation.PartialOperation) and expr.base_op == operation.Extract:
                prev_op_model = self.assign_outprop2op_model.get(self.var2prop[expr.args[0]], None)
                if prev_op_model is not None and isinstance(prev_op_model, cascada_opmodel.BranchNumberModel):
                    raise ValueError("Extract cannot be used with the output of BranchNumberModel-based "
                                     "operations (use PropExtract instead)")

            assert all(isinstance(arg, prop_type) for arg in expr_args), \
                f"invalid properties in {expr_args}"

            assign_propagation = self._propagate(expr_op, expr_args, var)

            ## debugging
            # print(f"assignments {var} ‹- {expr} leads to property propagation "
            #       f"{assign_propagation} = prop_type.propagate({expr_op.__name__}, {expr_args})")

            if isinstance(assign_propagation, cascada_opmodel.OpModel):
                for op_model_class, op_model_new_class in op_model_class2new_class.items():
                    if isinstance(assign_propagation, op_model_class):
                        assign_propagation = op_model_new_class(assign_propagation.input_prop)

                assign_outprop = prop_type(var)
                self.var2prop[var] = assign_outprop
                self.assign_outprop2op_model[assign_outprop] = assign_propagation
            else:
                # deterministic propagation, no model added to assign_outprop2op_model
                self.var2prop[var] = assign_propagation

        self.output_prop = []
        for var in ssa.output_vars:
            od = prop_type(var)  # as input prop
            assert od == self.var2prop[var]  # present in ssa
            self.output_prop.append(od)
        self.output_prop = tuple(self.output_prop)

        # self.external_var2prop sorted as ssa.external_vars
        self.external_var2prop = collections.OrderedDict(
            [(v, self.external_var2prop[v]) for v in ssa.external_vars]
        )

    def __init__(self, func, prop_type, input_prop_names, prefix="px",
                 external_var2prop=None, op_model_class2options=None):
        self._prop_init(
            func, prop_type=prop_type, input_prop_names=input_prop_names, prefix=prefix,
            external_var2prop=external_var2prop, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _prop_new(cls, func, prop_type, input_prop_names, prefix,
                  external_var2prop, op_model_class2options):
        # create a new object with "prop" labels in arguments
        # (__init__ argument names might be overridden)
        obj = cls.__new__(cls)  # does not call __init__
        obj._prop_init(
            func, prop_type=prop_type, input_prop_names=input_prop_names, prefix=prefix,
            external_var2prop=external_var2prop, op_model_class2options=op_model_class2options)
        return obj

    @classmethod
    def _get_Characteristic_cls(cls):
        raise NotImplementedError("subclasses must override this method")

    def __str__(self):
        # prop_type not needed since it appears in _prop and assign_outprop2op_model
        if self.external_var2prop:
            ev2d_str = f"[{', '.join([f'({v}, {d})' for v, d in self.external_var2prop.items()])}]"
        prop = self.__class__._prop_label
        format_str = f"{{}}(func={{}}, input_{prop}={{}}, output_{prop}={{}}{{}}, " \
                     f"assign_out{prop}2op_model={{}})"
        return format_str.format(
            type(self).__name__,
            self.func.get_name(),
            # str(self.input_vars) ignored to print it in a list-like way
            f"[{', '.join([str(v) for v in self.input_prop])}]",
            f"[{', '.join([str(v) for v in self.output_prop])}]",
            "" if not self.external_var2prop else f", external_var2{prop}=" + ev2d_str,
            f"[{', '.join([f'({v}, {e})' for v, e in self.assign_outprop2op_model.items()])}]",
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `ChModel` object with the same content.

        .. Implementation details:
            Since the equality operator is not implemented,
            ``eval(self.vrepr()) == self`` does NOT hold.

            This method does not store all the information,
            only the info needed to be recreated the object.

        """
        if hasattr(self.func, "vrepr"):
            func_vrepr = self.func.vrepr()
        else:
            func_vrepr = self.func.__name__
        if self.external_var2prop:
            fv2d_vrepr = f"[{', '.join([f'({v.vrepr()}, {d.vrepr()})' for v, d in self.external_var2prop.items()])}]"
        if self._op_model_class2options:
            omc2o_vrepr = self._op_model_class2options.__repr__()
        prop = self.__class__._prop_label  # {prop}
        format_string = f"{{}}(func={{}}, {prop}_type={{}}, input_{prop}_names={{}}, prefix={{}}{{}}{{}})"
        return format_string.format(
            type(self).__name__,
            func_vrepr,
            self.prop_type.__name__,
            f"[{', '.join([v.val.name.__repr__() for v in self.input_prop])}]",  # repr for quotes
            self._prefix.__repr__(),  # repr for quotes
            "" if not self.external_var2prop else f", external_var2{prop}={fv2d_vrepr}",
            "" if not self._op_model_class2options else f", op_model_class2options={omc2o_vrepr}",
        )

    def validity_assertions(self):
        """Return the validity constraint as a list of assertions.

        The validity constraint is a symbolic bit-vector expression
        (depending on the properties of the trail), and it is True
        if and only if the characteristic has non-zero probability.

        The list of assertions is a list of symbolic `Term` objects
        that form the validity constraint when combined with the
        `BvAnd` operation.

        The assertion list is composed of the `OpModel.validity_constraint`
        of the models of the non-trivial assignments.
        """
        assertions = []
        for outprop, op_model in self.assign_outprop2op_model.items():
            a = op_model.validity_constraint(outprop)
            if a == core.Constant(0, 1):
                warnings.warn(f"found False validity_constraint for outprop={outprop} op_model={op_model} in {self}")
                return core.Constant(0, 1),
            if a != core.Constant(1, 1):
                assertions.append(a)
        return tuple(assertions)

    def pr_one_assertions(self):
        """Return the probability-one constraint as a list of assertions.

        Return the constraint that evaluates to True if the characteristic
        has probability-one as a list of assertions.

        The probability-one constraint is a symbolic bit-vector expression
        (depending on the properties of the trail), and it is True
        if and only if the characteristic has probability 1.

        The list of assertions is a list of symbolic `Term` objects
        that form the constraint when combined with the
        `BvAnd` operation.

        The assertion list is composed of the ``OpModel.pr_one_constraint``
        of the models of the non-trivial assignments.
        """
        assertions = []
        for outprop, op_model in self.assign_outprop2op_model.items():
            a = op_model.pr_one_constraint(outprop)
            if a == core.Constant(0, 1):
                # # no need to throw warning
                # warnings.warn(f"found False pr_one_constraint for outprop={outprop} op_model={op_model} in {self}")
                return core.Constant(0, 1),
            if a != core.Constant(1, 1):
                assertions.append(a)
        return tuple(assertions)

    def weight_assertions(self, ch_weight_variable, assign_weight_variables, truncate=True):
        """Return the weight constraint as a list of assertions.

        The weight constraint is a symbolic bit-vector expression
        that depends on the properties of the trail, the characteristic
        weight variable and the weight variables of the
        models of the non-trivial assignments.
        This expression is True if and only if the weight
        (negative binary logarithm of the probability) of the characteristic
        is equals to the characteristic weight variable.

        The list of assertions is a list of symbolic `Term` objects
        that form the weight constraint when combined with the
        `BvAnd` operation.

        The assertion list is composed of the `OpModel.weight_constraint`
        of the models of the non-trivial assignments, plus
        a final assertion that sets the characteristic weight variable as
        the sum of the assignment weight variables.

        Similar as `abstractproperty.opmodel.OpModel`, the characteristic weight variable
        can be interpreted as a rational value with `num_frac_bits`
        fractional bits, and the weight constraint can be True even if
        the characteristic weight variable is equals to the characteristic weight
        up to some error bounded by `error`.

        If ``truncate`` is ``True``, the sum of the assignment weight variables
        is truncated (fractional bits are removed) and the result
        is compared to the characteristic weight variable.
        In other words, the characteristic weight variable represents
        the integer part of the weight of the characteristic probability
        if ``truncate=True`` (default case).
        """
        def zero_ext_right(var, num_zeros):
            """Expand with zeros to the right."""
            return var if num_zeros == 0 else operation.Concat(var, core.Constant(0, num_zeros))

        max_fb = self.num_frac_bits()
        max_width_wo_truncate = self.weight_width(truncate=False)
        assertions = []
        nonzero_awv_with_max_fb_and_width = []  # awv = assign_weight_variable
        found_nontrivial_opmodel_trivial_constraint = False

        for i, (outprop, op_model) in enumerate(self.assign_outprop2op_model.items()):
            var = assign_weight_variables[i]
            constraint = op_model.weight_constraint(outprop, var)
            if constraint == core.Constant(0, 1):
                warnings.warn(f"found False weight_constraint for outprop={outprop} op_model={op_model} in {self}")
                return core.Constant(0, 1),
            # avoid using zero weight vars W (with trivial constraint 0 == W) in the summation later
            trivial_constraint = operation.BvComp(var, core.Constant(0, var.width))
            if constraint == trivial_constraint or op_model.pr_one_constraint(outprop) == core.Constant(1, 1):
                if op_model.max_weight() > 0:
                    found_nontrivial_opmodel_trivial_constraint = True
                assertions.append(trivial_constraint)
            else:
                assert op_model.max_weight() > 0
                assertions.append(constraint)
                var = zero_ext_right(var, max_fb - op_model.num_frac_bits())
                assert max_width_wo_truncate - var.width >= 0
                var = operation.zero_extend(var, max_width_wo_truncate - var.width)
                nonzero_awv_with_max_fb_and_width.append(var)

        if len(nonzero_awv_with_max_fb_and_width) > 0:
            sum_assign_weight_variables = sum(nonzero_awv_with_max_fb_and_width)
        else:
            assert max_fb == 0
            sum_assign_weight_variables = core.Constant(0, 1)

        if truncate:
            if max_fb >= sum_assign_weight_variables.width:
                if sum_assign_weight_variables == 0:
                    sum_assign_weight_variables = core.Constant(0, ch_weight_variable.width)
                else:
                    from decimal import Decimal
                    aux_max_weight = self.max_weight(truncate=False) / Decimal(2**max_fb)
                    assert 0 < aux_max_weight < 1
                    raise ValueError(f"weight_assertions(..., truncate=True) cannot truncate characteristic weight"
                                     f" with 0 < {aux_max_weight}=(max_weight(truncate=False) / num_frac_bits()) < 1")
            else:
                sum_assign_weight_variables = sum_assign_weight_variables[:max_fb]

        if found_nontrivial_opmodel_trivial_constraint:
            # then ch_weight_variable.width > 1 (due to nontrivial opmodel) but sum_assign_weight_variables might be 1-bit
            offset = ch_weight_variable.width - sum_assign_weight_variables.width
            sum_assign_weight_variables = operation.zero_extend(sum_assign_weight_variables, offset)
        # otherwise, ch_weight_variable and sum_assign_weight_variables should have the same width

        assertions.append(operation.BvComp(ch_weight_variable, sum_assign_weight_variables))

        return tuple(assertions)

    def external_vars_validity_assertions(self):
        """Return the union of external variables of the constraints from `validity_assertions`.

        See `abstractproperty.opmodel.OpModel` for the definition of
        an external variable of a constraint from a model.
        """
        external_vars = []
        for outprop, op_model in self.assign_outprop2op_model.items():
            constraint = op_model.validity_constraint(outprop)
            if constraint == core.Constant(0, 1):
                return []  # False assertion found
            if not isinstance(constraint, core.Constant):
                external_vars.extend(op_model.external_vars_validity_constraint(outprop))
        return external_vars

    def external_vars_pr_one_assertions(self):
        """Return the union of external variables of the constraints from `pr_one_assertions`.

        See `abstractproperty.opmodel.OpModel` for the definition of
        an external variable of a constraint from a model.
        """
        external_vars = []
        for outprop, op_model in self.assign_outprop2op_model.items():
            constraint = op_model.pr_one_constraint(outprop)
            if constraint == core.Constant(0, 1):
                return []  # False assertion found
            if not isinstance(constraint, core.Constant):
                external_vars.extend(op_model.external_vars_pr_one_constraint(outprop))
        return external_vars

    def external_vars_weight_assertions(self, assign_weight_variables):
        """Return the union of external variables of the constraints from `weight_assertions`.

        See `abstractproperty.opmodel.OpModel` for the definition of
        an external variable of a constraint from a model.
        """
        external_vars = []
        for i, (outprop, op_model) in enumerate(self.assign_outprop2op_model.items()):
            awv_i = assign_weight_variables[i]
            constraint = op_model.weight_constraint(outprop, awv_i)
            if constraint == core.Constant(0, 1):
                return []  # False assertion found
            if not isinstance(constraint, core.Constant):
                external_vars.extend(op_model.external_vars_weight_constraint(outprop, awv_i))
        return external_vars

    def max_weight(self, truncate=True):
        """Return the maximum value the ch. weight variable can achieve in `weight_assertions`.

        If ``truncate`` is ``True``, fractional bits are not considered
        in the ch. weight variable (see `weight_assertions`).
        """
        ch_max_weight = 0
        max_frac_bits = self.num_frac_bits()
        for om in self.assign_outprop2op_model.values():
            ch_max_weight += om.max_weight() << (max_frac_bits - om.num_frac_bits())
        if truncate:
            ch_max_weight = ch_max_weight >> max_frac_bits
        return ch_max_weight

    def weight_width(self, truncate=True):
        """Return the width of the ch. weight variable used `weight_assertions`.

        If ``truncate`` is ``True``, fractional bits are not considered
        in the ch. weight variable (see `weight_assertions`).

        For the doctests see `max_weight`.
        """
        max_weight = self.max_weight(truncate=truncate)
        width = max(max_weight.bit_length(), 1)

        # ch_weight_variable width is at least the maximum of assignment weight variable widths
        max_frac_bits = self.num_frac_bits()
        for om in self.assign_outprop2op_model.values():
            if truncate:
                offset = - om.num_frac_bits()
            else:
                offset = max_frac_bits - om.num_frac_bits()
            width = max(width, om.weight_width() + offset)

        return width

    def num_frac_bits(self):
        """Return the number of fractional bits of the ch. weight variable used in
        `weight_assertions`.

        If the number of fractional bits is ``k``, then the bit-vector
        characteristic weight variable ``w`` of `weight_assertions`
        represents the number ``2^{-k} * bv2int(w)``.
        In particular, if ``k == 0``, then ``w`` represents an integer number.
        Otherwise, the ``k`` least significant bits of ``w`` denote the
        fractional part of the number represented by ``w``.

        For the doctests see `max_weight`.
        """
        return max((om.num_frac_bits() for om in self.assign_outprop2op_model.values()), default=0)

    def error(self):
        """Return the maximum difference between `weight_assertions` and the exact weight.

        The exact weight is exact value (without error) of the negative binary
        logarithm (weight) of the characteristic probability.

        This method returns an upper bound (in absolute value) of the maximum difference
        (over all properties in the trail) between the bit-vector characteristic weight
        from `weight_assertions` and the exact weight.
        """
        my_error = 0
        for op_model in self.assign_outprop2op_model.values():
            my_error += op_model.error()
        return my_error

    def signature(self, sig_type):
        """Return the signature of the characteristic model.

        The argument ``sig_type`` is a type of the signature,
        see `ChModelSigType`.

        The signature is an identifier of the characteristic model
        (used for comparing).
        Depending on the signature type, the identifier might
        uniquely represent the characteristic model or not.

        The signature does not include the `Property` objects
        of the characteristic but their (symbolic) values
        (`Variable` objects).
        """
        if sig_type == ChModelSigType.Unique:
            raise NotImplementedError("subclasses must implemented signature() for "
                                      "sig_type == ChModelSigType.Unique")
        elif sig_type == ChModelSigType.InputOutput:
            # already checked output_prop contains unique props
            sig_var = [d.val for d in self.input_prop]
            for d in self.output_prop:
                assert d.val not in sig_var  # output prop names are unique
                sig_var.append(d.val)
            assert len(set(sig_var)) == len(sig_var)
            return sig_var
        else:
            raise ValueError("invalid sig_type: {}".format(sig_type))

    def split(self, prop_separators):
        """Split into multiple `ChModel` objects given the list of property separators.

        Given the `ChModel` with underlying `SSA` :math:`s`, this method returns
        a list of ch. models with underlying `SSA` objects :math:`s_1, s_2, ..., s_n`,
        such that their composition :math:`s_n \circ s_{n-1} \dots \circ s_1`
        is functionally equivalent to :math:`s`.

        The argument ``prop_separators`` is a list containing lists of properties.
        The :math:`i`-th property list denote the last properties of the
        :math:`i`-th ch. model.  In other words, the :math:`(i+1)`-th ch. model
        immediately starts after the last property in ``prop_separators[i]``.

        To split into :math:`n` ch. models, ``prop_separators`` must contain
        :math:`n-1` lists, as the property list of the last ch. model is not given
        (its last properties are the output properties of the main model).

        Note the underlying functions of the new ch. models
        are `BvFunction` (and not `RoundBasedFunction`) even if
        `func` is a `RoundBasedFunction`.
        """
        assert len(prop_separators) >= 1

        _tuplify = lambda s: s if isinstance(s, collections.abc.Sequence) else tuple([s])

        # no need to check prop_separators (will be checked in ssa.split)
        prop_separators = [_tuplify(vs_list) for vs_list in prop_separators]

        # no need to check prop_separators (checked in ssa.split)

        # # no need to use different prefixes
        # def number2letters(my_number):
        #     s = ""
        #     while my_number >= len(string.ascii_lowercase):
        #         remainder = my_number % len(string.ascii_lowercase)
        #         s += string.ascii_lowercase[remainder]
        #         my_number = (my_number - remainder) / len(string.ascii_lowercase)
        #     s += string.ascii_lowercase[my_number]
        #     return s

        class SubChModel(object):
            func = self.func
            prop_type = self.prop_type
            op_model_class2options = self._op_model_class2options

            def __init__(self_chmodel, i):
                self_chmodel.input_prop_names = []
                # self_chmodel.prefix = self._prefix + number2letters(i)
                self_chmodel.prefix = self._prefix
                self_chmodel.external_var2prop = collections.OrderedDict()

            def __str__(self_chmodel):
                msg = "SubChModel:"
                msg += f"\n - input_prop_names: {self_chmodel.input_prop_names}"
                msg += f"\n - prefix: {self_chmodel.prefix}"
                msg += f"\n - external_var2prop: {self_chmodel.external_var2prop}"
                msg += f"\n - func: {self_chmodel.func}"
                return msg

        sub_chmodel_list = [SubChModel(i) for i in range(len(prop_separators) + 1)]

        var_separators = [[d.val for d in dt_list] for dt_list in prop_separators]
        sub_ssa_list = self.ssa.split(var_separators)
        assert len(sub_chmodel_list) == len(sub_ssa_list) == len(prop_separators) + 1

        for i in range(len(sub_chmodel_list)):
            sub_chmodel_list[i].func = sub_ssa_list[i].to_bvfunction()
            sub_chmodel_list[i].func.__name__ = self.func.get_name() + f"_{i}S"
            for v in sub_ssa_list[i].external_vars:
                if v in self.external_var2prop:
                    sub_chmodel_list[i].external_var2prop[v] = self.external_var2prop[v]

        aux_external_var2prop = collections.OrderedDict()
        aux_assign_op_models = []
        for i in range(len(sub_chmodel_list)):
            # if i == 0:
            #     print("# Main ChModel and SSA")
            #     print(self)
            #     print(self.ssa)
            # print(f"\n# Computing sub_chmodel_list[{i}/{len(sub_chmodel_list)-1}]")
            # print("ssa:", sub_ssa_list[i])

            # after calling ChModel(), the whole output_prop name is changed
            # (due to the new prefix), and we used the new output_prop name
            # for the input prop name of the next SubChModel to show
            # more clear the link between the SubChModel (no risk as in SSA.split)

            if i == 0:
                input_prop_names = self._input_prop_names
                assert list(input_prop_names) == [str(v) for v in sub_ssa_list[0].input_vars]
            else:
                # after calling ChModel() (which internally calls SSA()),
                # the output_prop might include ``_`` or ``_out``,
                # better to use input vars from sub_ssa_list
                input_prop_names = [str(v) for v in sub_ssa_list[i].input_vars]
                # input_prop_names = [d.val.name for d in sub_chmodel_list[i - 1].output_prop]
            sub_chmodel_list[i].input_prop_names = input_prop_names

            # print("sub_ch_model (SubChModel):", sub_chmodel_list[i])

            # ensuring sub_chmodel_list[i] is an instance of ChM and not of EncChM
            if issubclass(self.__class__, EncryptionChModel):
                for ch_model_cls in self.__class__.__bases__:
                    if not issubclass(ch_model_cls, EncryptionChModel):
                        break
                else:
                    raise ValueError(f"ChModel not in parents of {self.__class__} = "
                                     f"{self.__class__.__bases__}")
            else:
                ch_model_cls = self.__class__

            sub_chmodel_list[i] = ch_model_cls._prop_new(
                func=sub_chmodel_list[i].func,
                prop_type=sub_chmodel_list[i].prop_type,
                input_prop_names=sub_chmodel_list[i].input_prop_names,
                prefix=sub_chmodel_list[i].prefix,
                external_var2prop=sub_chmodel_list[i].external_var2prop,
                op_model_class2options=sub_chmodel_list[i].op_model_class2options)

            assert sub_chmodel_list[i].ssa == sub_ssa_list[i]

            # print("sub_ch_model (ChModel):", sub_chmodel_list[i])

            for v, d in sub_chmodel_list[i].external_var2prop.items():
                # an external var might be used in multiple split SSA
                assert aux_external_var2prop.get(v, d) == d
                aux_external_var2prop[v] = d

            for out_prop, op_model in sub_chmodel_list[i].assign_outprop2op_model.items():
                if not isinstance(op_model, cascada_opmodel.ModelIdentity):
                    assert out_prop not in aux_assign_op_models
                    aux_assign_op_models.append(op_model)

        assert self.external_var2prop == aux_external_var2prop

        # ensure the number and shape of the op models is the same (excluding ModelIdentity)
        old_assign_op_models = [om for om in self.assign_outprop2op_model.values()
                                if not isinstance(om, cascada_opmodel.ModelIdentity)]
        assert len(aux_assign_op_models) == len(old_assign_op_models)
        for old_om, new_om in zip(aux_assign_op_models, old_assign_op_models):
            assert type(old_om) == type(new_om), f"\n{aux_assign_op_models}\n{old_assign_op_models}"

        return sub_chmodel_list

    def get_formatted_logged_msgs(self):
        """Return the list of logged messages.

        If `func` includes `log_msg` calls in its ``eval``, this method
        return the list of messages logged with the format field objects
        applied. Otherwise, an empty list is returned.

        In the first case, the `Variable` objects of `ssa` appearing in the
        format field objects are replaced with their associated symbolic
        `Property` objects.
        """
        if self._logger is None:
            return []
        list_msgs = []
        replacements = {v: p.val for v, p in self.var2prop.items()}
        for format_string, format_field_objects in self._logger:
            for i in range(len(format_field_objects)):
                expr = format_field_objects[i]
                if isinstance(expr, core.Term) and not isinstance(expr, core.Constant):
                    format_field_objects[i] = self.prop_type(expr.xreplace(replacements))
            list_msgs.append(format_string.format(*format_field_objects))
        return list_msgs

    def get_round_separators(self):
        """Return the round separators if `func` is a `RoundBasedFunction`.

        If `func` includes `add_round_outputs` calls in its ``eval``,
        this method returns a list with the round `Property` outputs
        delimiting the rounds. Otherwise, ``None`` is returned.

        In the first case, this list contains ``num_rounds - 1`` entries,
        where the ``i``-th entry is the list of `Property` outputs of the
        ``i``-th round. In particular, the output properties of the last
        round are not included in this list.

        The list returned by this method is meant to be used as the argument
        of `ChModel.split` to get the `ChModel` object of each round.
        """
        if getattr(self, "_rounds_outputs", None) is None:
            return None
        if len(self._rounds_outputs) == 0:
            return None
        return tuple(tuple(self.prop_type(v) for v in lv)
                     for lv in self._rounds_outputs[:-1])

    # dotprinting last method
    def dotprinting(self, repeat=True, vrepr_label=False, **kwargs):
        """Return the DOT description of the expression tree of `assign_outprop2op_model`.

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
        from sympy.core.basic import Basic
        # sympy_property = type(self.prop_type.__name__, (Basic,), {})
        op_model_class2sympy_op_model = {}
        expr = []
        for out_prop, op_model in self.assign_outprop2op_model.items():
            if type(op_model) not in op_model_class2sympy_op_model:
                sympy_op_model = type(type(op_model).__name__, (Basic,), {})
                op_model_class2sympy_op_model[type(op_model)] = sympy_op_model
            else:
                sympy_op_model = op_model_class2sympy_op_model[type(op_model)]
            expr.append(Tuple(
                # sympy_property(out_prop.val),
                out_prop.val,
                # sympy_op_model(Tuple(*[sympy_property(d.val) for d in op_model.input_prop]))
                sympy_op_model(*[d.val for d in op_model.input_prop])
            ))
        expr = Tuple(*expr)
        return dotprinting(expr, repeat=repeat, vrepr_label=vrepr_label, **kwargs)


class EncryptionChModel(object):
    """Represent characteristic models of encryption functions w.r.t some property.

    Given a `Cipher`, an `EncryptionChModel` is a bit-vector model
    (see `ChModel`) of a characteristic for a particular `Property` over
    the `Cipher.encryption` (where the `Cipher.key_schedule`
    is ignored).

    See also `abstractproperty.characteristic.EncryptionCharacteristic`.

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic models over encryption functions,
    (see `differential.chmodel.EncryptionChModel`,
    `linear.chmodel.EncryptionChModel` or `algebraic.chmodel.EncryptionChModel`
    for some examples).

    Attributes:
        cipher: the block cipher as a `Cipher` object

    .. Implementation details:

        This class does not subclass `ChModel`, but subclasses
        of this class defined for a particular `Property`
        must subclass the corresponding ``ChModel``.

    """
    _prefix = None

    def __init__(self, cipher, prop_type, op_model_class2options=None, round_keys_prefix=None):
        assert issubclass(cipher, blockcipher.Cipher)

        prefix = self.__class__._prefix
        assert isinstance(prefix, str)

        num_inputs = len(cipher.encryption.input_widths)
        input_prop_names = [f"{prefix}p" + str(i) for i in range(num_inputs)]
        encryption_prefix = f"{prefix}x"

        if round_keys_prefix is not None:
            warnings.warn("the EncryptionChModel parameter round_keys_prefix is experimental")
            round_keys_prefix = prefix
        _round_keys = []
        for i, width in enumerate(cipher.key_schedule.output_widths):
            _round_keys.append(core.Variable(f"{round_keys_prefix}k" + str(i), width))

        class Encryption(cipher.encryption):
            round_keys = tuple(_round_keys)
        Encryption.__name__ = cipher.encryption.__name__

        # avoid prop_type=prop_type (super might not abstract)
        super().__init__(
            Encryption, prop_type, input_prop_names, prefix=encryption_prefix,
            op_model_class2options=op_model_class2options)

        all_round_keys = set(_round_keys)
        round_keys_found = set()
        for ext_var in self.ssa.external_vars:
            if ext_var not in all_round_keys:
                raise ValueError("found external variable {} not in round_keys {} in {}\n{}".format(
                    ext_var, _round_keys, Encryption.__name__, self.ssa
                ))
            round_keys_found.add(ext_var)
        if len(round_keys_found) < len(all_round_keys):
            raise ValueError(f"round keys {all_round_keys-round_keys_found} not used in {Encryption}")

        round_keys_prop_found = set(ext_var for ext_var in self.external_var2prop)
        if len(round_keys_prop_found) < len(all_round_keys):
            raise ValueError(f"round key properties of {all_round_keys - round_keys_prop_found} not found")

        self.cipher = cipher
        self._round_keys_prefix = round_keys_prefix

    @classmethod
    def _get_EncryptionCharacteristic_cls(cls):
        raise NotImplementedError("subclasses must override this method")

    def __str__(self):
        # similar to ChModel.__str__ but changing func by cipher
        if self.external_var2prop:
            ev2d_str = f"[{', '.join([f'({v}, {d})' for v, d in self.external_var2prop.items()])}]"
        prop = self.__class__._prop_label
        format_str = f"{{}}(cipher={{}}, input_{prop}={{}}, output_{prop}={{}}{{}}, " \
                     f"assign_out{prop}2op_model={{}})"
        return format_str.format(
            type(self).__name__,
            self.cipher.get_name(),
            f"[{', '.join([str(v) for v in self.input_prop])}]",
            f"[{', '.join([str(v) for v in self.output_prop])}]",
            "" if not self.external_var2prop else f", external_var2{prop}=" + ev2d_str,
            f"[{', '.join([f'({v}, {e})' for v, e in self.assign_outprop2op_model.items()])}]",
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `EncryptionChModel` object with the same content.

        .. Implementation details:
            Since the equality operator is not implemented,
            ``eval(self.vrepr()) == self`` does NOT hold.

            This method does not store all the information,
            only the info needed to be recreated the object.

        """
        if hasattr(self.cipher, "vrepr"):
            cipher_vrepr = self.cipher.vrepr()
        else:
            cipher_vrepr = self.cipher.__name__
        if self._op_model_class2options:
            omc2o_vrepr = self._op_model_class2options.__repr__()
        return "{}(cipher={}, {}_type={}{})".format(
            type(self).__name__,
            cipher_vrepr,
            self.__class__._prop_label,
            self.prop_type.__name__,
            "" if not self._op_model_class2options else f", op_model_class2options={omc2o_vrepr}",
            "" if not self._round_keys_prefix else f", round_keys_prefix={self._round_keys_prefix}",
        )

    def split(self, prop_separators):
        """Split into multiple `ChModel` objects given the list of property separators.

        Given an `EncryptionChModel`, this method calls `ChModel.split` to
        split the characteristic model.

        .. note::

            The new split characteristic models are instances of `ChModel`
            and not of `EncryptionChModel`.

        """
        return super().split(prop_separators)


class CipherChModel(object):
    """Represent characteristic models of ciphers w.r.t some property.

    Given a `Cipher`, a `CipherChModel` is a bit-vector model of a
    characteristic for a particular `Property` over the cipher.

    See also `abstractproperty.characteristic.CipherCharacteristic`.

    A `CipherChModel` consists of a pair of `ChModel` where one models
    the characteristic over the `Cipher.key_schedule`, and the other one
    models the characteristic over the `Cipher.encryption`.

    .. note::

        The model of the characteristic over the `Cipher.encryption`
        is an instance of `ChModel` and not an instance of `EncryptionChModel`.

    The round key properties in the encryption characteristic
    (the properties of the external variables of the encryption `SSA`)
    are set to the output properties of the key-schedule characteristic.

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic models over ciphers,
    (eee `differential.chmodel.CipherChModel`
    or `algebraic.chmodel.CipherChModel` for some examples).

    Attributes:
        cipher: the block cipher as a `Cipher` object
        prop_type: the type of `Property` of the characteristic
        ks_ch_model: the `ChModel` of the key schedule characteristic
        enc_ch_model: the `ChModel` of the encryption characteristic

    """
    _ChModel_cls = None
    _prefix = None

    def __init__(self, cipher, prop_type, op_model_class2options=None):
        assert issubclass(cipher, blockcipher.Cipher)

        prefix = self.__class__._prefix
        assert isinstance(prefix, str)

        myChModel = self.__class__._ChModel_cls

        func = cipher.key_schedule
        input_prop_names = tuple([f"{prefix}mk" + str(i) for i in range(len(func.input_widths))])
        ks_prefix = f"{prefix}k"
        ks_ch_model = myChModel(func, prop_type, input_prop_names, prefix=ks_prefix,
                                op_model_class2options=op_model_class2options)

        if ks_ch_model.ssa.external_vars or ks_ch_model.external_var2prop:
            raise ValueError("found external variables {} in {}\n{}".format(
                ks_ch_model.ssa.external_vars, func.__name__, ks_ch_model.ssa
            ))

        class Encryption(cipher.encryption):
            round_keys = ks_ch_model.ssa.output_vars
        Encryption.__name__ = cipher.encryption.__name__

        func = Encryption
        input_prop_names = [f"{prefix}p" + str(i) for i in range(len(func.input_widths))]
        enc_prefix = f"{prefix}x"
        rkvar2rkprop = {}
        for prop in ks_ch_model.output_prop:
            rkvar2rkprop[prop.val] = prop
        enc_ch_model = myChModel._prop_new(  # _prop_new to use external_var2prop
            func, prop_type, input_prop_names, prefix=enc_prefix,
            external_var2prop=rkvar2rkprop, op_model_class2options=op_model_class2options)

        all_round_keys = set(ks_ch_model.ssa.output_vars)
        round_keys_found = set()
        for ext_var in enc_ch_model.ssa.external_vars:
            if ext_var not in all_round_keys:
                raise ValueError("found external variable {} not in round_keys {} in {}\n{}".format(
                    ext_var, ks_ch_model.ssa.output_vars, Encryption.__name__, enc_ch_model.ssa
                ))
            round_keys_found.add(ext_var)
        if len(round_keys_found) < len(all_round_keys):
            raise ValueError(f"round keys {all_round_keys-round_keys_found} not used in {Encryption}")

        round_keys_prop_found = set(ext_var for ext_var in enc_ch_model.external_var2prop)
        if len(round_keys_prop_found) < len(all_round_keys):
            raise ValueError(f"round key properties of {all_round_keys - round_keys_prop_found} not found")

        self.prop_type = prop_type
        self.ks_ch_model = ks_ch_model
        self.enc_ch_model = enc_ch_model
        self.cipher = cipher

        # for vrepr
        self._op_model_class2options = op_model_class2options

    @classmethod
    def _get_CipherCharacteristic_cls(cls):
        raise NotImplementedError("subclasses must override this method")

    def __str__(self):
        return "{}(ks_ch_model={}, enc_ch_model={})".format(
            type(self).__name__,
            self.ks_ch_model.__str__(),
            self.enc_ch_model.__str__(),
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `CipherChModel` object with the same content.

        .. Implementation details:
            Since the equality operator is not implemented,
            ``eval(self.vrepr()) == self`` does NOT hold.

            This method does not store all the information,
            only the info needed to be recreated the object.

        """
        if hasattr(self.cipher, "vrepr"):
            cipher_vrepr = self.cipher.vrepr()
        else:
            cipher_vrepr = self.cipher.__name__
        if self._op_model_class2options:
            omc2o_vrepr = self._op_model_class2options.__repr__()
        assert self.ks_ch_model._prop_label == self.enc_ch_model._prop_label
        return "{}(cipher={}, {}_type={}{})".format(
            type(self).__name__,
            cipher_vrepr,
            self.ks_ch_model._prop_label,
            self.prop_type.__name__,
            "" if not self._op_model_class2options else f", op_model_class2options={omc2o_vrepr}",
        )

    def signature(self, ch_signature_type):
        """Return the signature of the characteristic model over the cipher.

        See also `ChModel.signature`.
        """
        return self.ks_ch_model.signature(ch_signature_type) + self.enc_ch_model.signature(ch_signature_type)

    def get_formatted_logged_msgs(self):
        """Return the messages logged in the evaluation of the function
        with the format field objects applied.

        See also `ChModel.get_formatted_logged_msgs`.
        """
        return self.ks_ch_model.get_formatted_logged_msgs() + \
               self.enc_ch_model.get_formatted_logged_msgs()
