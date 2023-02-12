"""Search for characteristics by modeling the search as a sequence of SMT problems.

.. autosummary::
   :nosignatures:

    INCREMENT_NUM_ROUNDS
    ChModelAssertType
    PrintingMode
    MissingVarWarning
    ChFinder
    CipherChFinder
    round_based_ch_search
    round_based_cipher_ch_search
"""
import collections
import decimal
import enum
import functools
import itertools
import datetime
import math
import warnings

from pysmt import environment
from pysmt import logics

from cascada.bitvector import core
from cascada.bitvector import context
from cascada.bitvector import operation
from cascada.bitvector import ssa as cascada_ssa
from cascada import abstractproperty
from cascada.primitives import blockcipher
from cascada.smt import pysmttypes


zip = functools.partial(zip, strict=True)
CurrentSignatureType = abstractproperty.chmodel.ChModelSigType.Unique
INCREMENT_NUM_ROUNDS = "increment_num_rounds"
"""Message to increase the current number of rounds by one (see `round_based_ch_search`)"""


def _get_time():
    now = datetime.datetime.now()
    return "{}-{}:{}".format(now.day, now.hour, now.minute)


def _get_smart_print(filename=None):
    def smart_print(*msg, **kwargs):
        if kwargs.get("prepend_time", False):
            msg = list(msg)
            msg[0] = f"{_get_time()} | {msg[0]}"
            del kwargs["prepend_time"]
        if isinstance(filename, str):
            with open(filename, "a") as fh:
                print(*msg, file=fh, flush=True, **kwargs)
        elif hasattr(filename, "write"):
            print(*msg, file=filename, **kwargs)
        else:
            print(*msg, flush=True, **kwargs)
    return smart_print


def _merge_weights(w0, w1):
    sum_pr = (decimal.Decimal(2) ** (-w0)) + (decimal.Decimal(2) ** (-w1))
    return min(- abstractproperty.opmodel.log2_decimal(sum_pr), w0, w1)  # avoid rounding errors


class ChModelAssertType(enum.Enum):
    """Represent the options available for the type of constraints
    of the characteristic model.

    See also `abstractproperty.chmodel.ChModel`.

    Attributes:
        ValidityAndWeight: consider `abstractproperty.chmodel.ChModel.validity_assertions`
            and `abstractproperty.chmodel.ChModel.weight_assertions`
        ProbabilityOne: only consider `abstractproperty.chmodel.ChModel.pr_one_assertions`
        Validity: only consider `abstractproperty.chmodel.ChModel.validity_assertions`

    """
    ValidityAndWeight = enum.auto()
    Validity = enum.auto()
    ProbabilityOne = enum.auto()


class PrintingMode(enum.Enum):
    """Represent the options available for the information to print.

    Attributes:
        Silent: nothing is printed
        WeightsAndSrepr: prints every time the target weight is increased
            or the final weight is modified, and prints the
            `abstractproperty.characteristic.Characteristic.srepr` method
            of all non-returned characteristics (together with the current time)
        WeightsAndVrepr: similar as `WeightsAndSrepr`, but the
            `abstractproperty.characteristic.Characteristic.vrepr` method is
            printed instead.
        Debug: similar as `WeightsAndSrepr`, but also prints all
            the constraints generated during the search

    """
    Silent = enum.auto()
    WeightsAndSrepr = enum.auto()
    WeightsAndVrepr = enum.auto()
    Debug = enum.auto()


class MissingVarWarning(UserWarning):  # _check_initial_constraints
    """The class of warnings when a variable from an additional constraint
    is missing in the SMT problem (see `ChFinder`)."""
    pass


class ChFinder(object):
    """Search for characteristics by modeling the search as a sequence of SMT problems.

    Given a characteristic model (`abstractproperty.chmodel.ChModel`
    o `abstractproperty.chmodel.EncryptionChModel`)
    defined for a particular `Property` (e.g., `XorDiff` or `LinearMask`),
    this class finds characteristics
    (`abstractproperty.characteristic.Characteristic`
    o `abstractproperty.characteristic.EncryptionCharacteristic`)
    satisfying the characteristic model by modelling the search as a sequence
    of SMT problems in the bit-vector theory.

    Depending on ``assert_type``, the SMT problems contain the validity,
    probability-one and/or weight assertions from the
    `abstractproperty.chmodel.ChModel`.
    They might also contain a constraint fixing the
    characteristic weight variable to a constant value
    (in the case of `find_next_ch_increasing_weight`) or
    additional constraints provided in ``initial_constraints``
    or derived from ``var2ct_prop`` or ``exclude_zero_input_prop``.

    .. note::

        The optional initialization argument ``var2ct_prop`` is a
        `collections.OrderedDict` mapping symbolic properties
        of the characteristic model to constant values.
        From each ``(sp, cp)`` pair in  ``var2ct_prop``,
        where ``sp`` is a symbolic `Property` and ``cp`` a constant `Property`,
        the constraint ``sp == cp`` is added to ``initial_constraints``.
        The dictionary ``var2ct_prop`` can also be filled with
        pairs of symbolic `Term` and `Constant` objects;
        in this case, they are first automatically converted to
        `Property` objects.

        If ``exclude_zero_input_prop`` is ``True``, and additional
        constraint is added preventing the input property to be zero.

        By defaut, an exception is raised if an additional constraint
        (one of the constraint from ``initial_constraints`` or one of the
        constraints derived from ``var2ct_prop`` or ``exclude_zero_input_prop``)
        contains a variable that does not appear in the SMT problem
        (i.e., `chmodel_asserts`).
        If the initialization argument ``raise_exception_missing_var`` is False,
        a warning with category `MissingVarWarning` is printed
        instead of raising an exception.

    .. note::

        If ``ch_model`` is an algebraic characteristic model (defined for
        the property `BitValue` or `WordValue`), the assertion type
        `ValidityAndWeight` in  ``assert_type`` is not supported
        (and `Validity` and `ProbabilityOne` are
        equivalent due to the definition of characteristic probability,
        see also `algebraic.chmodel.ChModel`).

        If the algebraic characteristic model does not contain
        external variables and a ciphertext value is provided in
        ``var2ct_prop`` or ``initial_constraints``, then the
        search for algebraic characteristics is equivalent to the
        search for preimages of the given ciphertext value.
        On the other hand, if the algebraic characteristic model
        contains external variables (e.g., round keys) and a
        plaintext-ciphertext pair is provided in
        ``var2ct_prop`` or ``initial_constraints``, then the
        search for algebraic characteristics is equivalent to the
        search for the external values (e.g., round keys)
        that make the underlying bit-vector function maps
        the given plaintext to the given ciphertext.

    The SMT problems are solved through pySMT_, which calls an
    off-the-shelf SMT solver supported by pySMT given by ``solver_name``
    (e.g., ``solver_name='btor'`` sets Boolector_ as the SMT solver).
    The pySMT documentation_ explains how to install an SMT solver.

    .. _pySMT: https://github.com/pysmt/pysmt
    .. _Boolector: https://boolector.github.io
    .. _documentation: https://github.com/pysmt/pysmt

    This class provides three methods to search for characteristics:
    `find_next_ch`, `find_next_ch_increasing_weight`
    and `find_next_ch_increasing_weight_fixed_in_out`.
    These methods are Python `generator` functions, returning an `iterator` that
    yields the `abstractproperty.characteristic.Characteristic` objects
    found in the search (see also this_).
    The characteristics returned are defined for
    the `Property` of the characteristic model.

    .. _this: https://docs.python.org/3/howto/functional.html?highlight=generator#generators

    .. note::

        The characteristics can be obtained in a for-loop over the iterator
        or retrieved one at a time with the `next` function, that is,

        .. code:: python

            [...]
            ch_finder = ChFinder(...)
            iterator = ch_finder.search_all()
            first_ch_found = next(iterator)
            for next_ch_found in iterator:
                [...]

        If `next` is used but not characteristic is found,
        an `StopIteration` exception is raised.

    ::

        >>> # example of SMT problem of XorDiff-EncryptionCharacteristic of Speck32
        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, XorDiff)
        >>> assert_type = ChModelAssertType.ValidityAndWeight
        >>> ch_finder = ChFinder(ch_model, assert_type, "btor", exclude_zero_input_prop=True, solver_seed=0)
        >>> ch_finder.formula_size()
        801
        >>> print(ch_finder.hrepr(full_repr=False))  # doctest: +NORMALIZE_WHITESPACE
        ; initial constraints
        assert ~((dp0 :: dp1) == 0x00000000)
        ; characteristic model assertions
        assert ((~(... << ...) ^ (dp1 << 0x0001)) & (~(... << ...) ^ (dx1 << 0x0001)) &
            ((... >>> ...) ^ dp1 ^ dx1 ^ ((dp0 >>> 7) << 0x0001))) == 0x0000
        assert ((~(... << ...) ^ ((... ^ ...) << 0x0001)) & (~(... << ...) ^ (dx6 << 0x0001)) &
            ((... >>> ...) ^ ... ^ ... ^ dx6 ^ ((dx1 >>> 7) << 0x0001))) == 0x0000
        assert dx6 == dx7_out
        assert ((((dp1 <<< 2) ^ dx1) <<< 2) ^ dx6) == dx9_out
        assert w0 == PopCount(~((~... ^ dp1) & (~... ^ dx1))[14:])
        assert w1 == PopCount(~((~... ^ ... ^ ...) & (~... ^ dx6))[14:])
        assert w2 == 0b0
        assert w3 == 0b0
        assert w == ((0b0 :: w0) + (0b0 :: w1))


    ::

        >>> # example of SMT problem of LinearMask-Characteristic of Speck32.key_schedule
        >>> from cascada.bitvector.core import Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import ChModel
        >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
        >>> v2c = {ch_model.input_mask[2]: core.Constant(0, 16)}  # setting input_mask[2] to 0
        >>> at = ChModelAssertType.ProbabilityOne
        >>> ch_finder = ChFinder(ch_model, at, "btor", var_prop2ct_prop=v2c, solver_seed=0)
        >>> ch_finder.formula_size()
        80
        >>> print(ch_finder.hrepr(full_repr=False))  # doctest: +NORMALIZE_WHITESPACE
        ; initial constraints
        assert mk2 == 0x0000
        ; characteristic model assertions
        assert (~((mx1 ^ (... >>> ...)) | (mx1 ^ mk2__0)) == 0xffff) &
            (((mx1 ^ (mk1 >>> 7) ^ mk2__0)[:1]) == 0b000000000000000)
        assert ((mk2__1 <<< 2) == mx1) & ((mk2__1 <<< 2) == mx3)
        assert (~((mx5 ^ (... >>> ...)) | (mx5 ^ mx3__0)) == 0xffff) &
            (((mx5 ^ (mk0 >>> 7) ^ mx3__0)[:1]) == 0b000000000000000)
        assert ((mx3__1 <<< 2) == mx5) & ((mx3__1 <<< 2) == mx8)
        assert (mk2 ^ mk2__0 ^ mk2__1) == mk2_out
        assert (mx3 ^ mx3__0 ^ mx3__1) == mx3_out
        assert mx8 == mx8_out

    Attributes:
        ch_model: the underlying characteristic model (a subclass of
            `abstractproperty.chmodel.ChModel` or
            `abstractproperty.chmodel.EncryptionChModel`)
        assert_type: the type of assertions (an element from `ChModelAssertType`)
        solver_name: the pySMT solver name_
        initial_constraints: the list of additional constraints (given as `Term`)
            to add to the SMT problems (by default an empty list)
        printing_mode: the information to print (an element of `PrintingMode`,
            by default `PrintingMode.Silent`)
        filename: the filename where the messages
            will be printed (by default the standard output is used)
        weight_prefix: the prefix to label the weight variables
        solver_seed: the seed for the SMT solver
        env: the associated `pysmt.environment.Environment`
            (by default a new one is created)
        chmodel_asserts: the list containing the main assertions (`Term` objects)
            common to the SMT problems

    .. _name: https://github.com/pysmt/pysmt#solvers-support
    """

    def __init__(self, ch_model, assert_type, solver_name,
                 initial_constraints=None, exclude_zero_input_prop=False,
                 var_prop2ct_prop=None, raise_exception_missing_var=True,
                 printing_mode=PrintingMode.Silent, filename=None,
                 weight_prefix="w", solver_seed=None, env=None):
        assert isinstance(ch_model, abstractproperty.chmodel.ChModel)
        assert isinstance(assert_type, ChModelAssertType)

        from cascada.algebraic.value import Value
        if issubclass(ch_model.prop_type, Value):
            if assert_type == ChModelAssertType.ValidityAndWeight:
                raise ValueError("searching for algebraic characteristics with"
                                 " assert_type == ValidityAndWeight is not supported")
            if isinstance(ch_model, abstractproperty.chmodel.EncryptionChModel) and \
                    initial_constraints is None and var_prop2ct_prop is None:
                warnings.warn("searching for algebraic EncryptionCharacteristic is usually done"
                              " to find the round keys given additional constraints fixing the input"
                              " and output values, but no additional constraints were provided")
            elif ch_model.external_var2prop and initial_constraints is None and var_prop2ct_prop is None:
                warnings.warn("searching for algebraic characteristics is usually done"
                              " with initial constraints fixing the input and/or output values,"
                              " but no additional constraints were provided")

        for v in ch_model.var2prop:
            if str(v).startswith(weight_prefix):
                raise ValueError(f"characteristic model cannot contain variable names starting"
                                 f" with the weight prefix {weight_prefix} (found {v})")

        assert isinstance(printing_mode, PrintingMode)
        assert filename is None or isinstance(filename, str) or hasattr(filename, "write")

        if env is None:
            environment.push_env()
            env = environment.get_env()
        else:
            assert env == environment.get_env()

        assert solver_name in env.factory.all_solvers(logics.QF_BV), \
            f"solver_name={solver_name} not in the list of available pySMT solvers " \
            f"supporting the logic QF_BV = {env.factory._all_solvers}"

        # initialize initial_constraints

        if initial_constraints is None:
            initial_constraints = []
        else:
            assert isinstance(initial_constraints, collections.abc.Sequence)
            initial_constraints = initial_constraints[:]

        if exclude_zero_input_prop:
            compact_input = functools.reduce(operation.Concat, [d.val for d in ch_model.input_prop])
            initial_constraints.append(
                operation.BvNot(operation.BvComp(compact_input, core.Constant(0, compact_input.width))))

        if var_prop2ct_prop is not None:
            new_var_prop2ct_prop = collections.OrderedDict()
            for vp, cp in var_prop2ct_prop.items():
                if not isinstance(vp, ch_model.prop_type):
                    assert not isinstance(vp, core.Constant)
                    vp = ch_model.prop_type(vp)
                if not isinstance(cp, ch_model.prop_type):
                    assert isinstance(cp, core.Constant)
                    cp = ch_model.prop_type(cp)
                new_var_prop2ct_prop[vp] = cp
            var_prop2ct_prop = new_var_prop2ct_prop
            for vp, cp in var_prop2ct_prop.items():
                initial_constraints.append(operation.BvComp(vp.val, cp.val))
        else:
            var_prop2ct_prop = {}

        #

        if assert_type == ChModelAssertType.ValidityAndWeight:
            ch_weight = core.Variable(weight_prefix, ch_model.weight_width())
            # awvs = assignment weight variables
            awvs = [core.Variable(f"{weight_prefix}{i}", om.weight_width())
                    for i, om in enumerate(ch_model.assign_outprop2op_model.values())]
            # error is the ch_model error without the weight truncation error
            error = ch_model.error()  # avoid recomputing
        else:
            ch_weight = None
            awvs = None
            error = 0
            
        #

        chmodel_asserts = ChFinder._get_chmodel_asserts(ch_model, assert_type, ch_weight=ch_weight, awvs=awvs)

        vars_in_constraints = set()
        for c in chmodel_asserts:
            if c == core.Constant(0, 1):
                warnings.warn("found assertion False in chmodel_asserts")
            vars_in_constraints |= c.atoms(core.Variable)

        ChFinder._check_initial_constraints(
            ch_model, initial_constraints, chmodel_asserts,
            exclude_zero_input_prop, var_prop2ct_prop,
            vars_in_constraints, raise_exception_missing_var
        )

        self.ch_model = ch_model
        self.assert_type = assert_type
        self.solver_name = solver_name
        self.initial_constraints = initial_constraints
        self.printing_mode = printing_mode
        self.filename = filename
        self.weight_prefix = weight_prefix
        self.solver_seed = solver_seed
        self._env = env  # .env is a property
        self.chmodel_asserts = chmodel_asserts

        # variables not added in docstring (private variables)
        self._exclude_zero_input_prop = exclude_zero_input_prop
        self._var_prop2ct_prop = var_prop2ct_prop
        self._ch_weight = ch_weight
        self._awvs = awvs
        self._error = error
        self._vars_in_constraints = vars_in_constraints
        self._raise_exception_missing_var = raise_exception_missing_var

    @property
    def env(self):
        assert self._env == environment.get_env()
        return self._env

    def __del__(self):
        if hasattr(self, "_env") and self._env in environment.ENVIRONMENTS_STACK:
            environment.ENVIRONMENTS_STACK.remove(self._env)

    @staticmethod
    def _get_chmodel_asserts(ch_model, assert_type, ch_weight=None, awvs=None):
        if assert_type == ChModelAssertType.Validity:
            return ch_model.validity_assertions()
        elif assert_type == ChModelAssertType.ValidityAndWeight:
            assert ch_weight is not None and awvs is not None
            assertions = ch_model.validity_assertions()
            assertions += ch_model.weight_assertions(ch_weight, awvs, truncate=True)
            return assertions
        elif assert_type == ChModelAssertType.ProbabilityOne:
            return ch_model.pr_one_assertions()
        else:
            raise ValueError("invalid assert_type")

    @staticmethod
    def _check_initial_constraints(
            ch_model, initial_constraints, chmodel_asserts,
            exclude_zero_input_prop, var_prop2ct_prop,
            vars_in_constraints, raise_exception_missing_var
    ):
        for var_prop, ct_prop in var_prop2ct_prop.items():
            if var_prop.val not in vars_in_constraints:
                msg = f"var {var_prop.val} from var_prop2ct_prop is not " \
                      f"present in chmodel_asserts = {chmodel_asserts}"
                if raise_exception_missing_var:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, category=MissingVarWarning)

        if exclude_zero_input_prop:
            for p in ch_model.input_prop:
                if p.val not in vars_in_constraints:
                    msg = f"exclude_zero_input_prop is True but some " \
                          f"input var ({p.val}) is not present in " \
                          f"chmodel_asserts = {chmodel_asserts}"
                    if raise_exception_missing_var:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg, category=MissingVarWarning)

        for i, c in enumerate(initial_constraints):
            if c == core.Constant(0, 1):
                raise ValueError("found constraint False in initial_constraints")
            for v in c.atoms(core.Variable):
                if v not in vars_in_constraints:
                    msg = f"var {v} from initial_constraints[{i}] = {c} is not " \
                          f"present in chmodel_asserts = {chmodel_asserts}"
                    if raise_exception_missing_var:
                        raise ValueError(msg)
                    else:
                        warnings.warn(msg, category=MissingVarWarning)

    def _pysmt_model2ch(self, solution_var2ct, target_weight=None, is_pysmt_model=True, is_sat=True):
        if is_pysmt_model:
            solution_var2ct = pysmttypes.pysmt_model2bv_model(solution_var2ct)
        else:
            solution_var2ct = solution_var2ct.copy()

        # to build a characteristic using get_properties_for_initialization,
        # all input/external properties, and output properties of non-Identity are needed
        # (note that these needed properties are not free properties (not affecting the ch)
        # (Unique signature cannot be used since it ignores op_model.max_weight() != 0)

        def get_needed_vars(my_ch_model):
            var_needed = [p.val for p in my_ch_model.input_prop if p.val not in my_ch_model.ssa._input_vars_not_used]
            for ext_var, prop in my_ch_model.external_var2prop.items():
                if not isinstance(prop.val, core.Constant):
                    var_needed.append(ext_var)
            for outprop, op_model in my_ch_model.assign_outprop2op_model.items():
                # if op_model.max_weight() != 0:
                if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                    var_needed.append(outprop.val)
            return var_needed

        missing_signature_vars = []
        for v in get_needed_vars(self.ch_model):
            if v not in solution_var2ct:
                missing_signature_vars.append(v)
                solution_var2ct[v] = core.Constant(0, v.width)

        # universally-invalid characteristics are invalid regardless of non-input non-output properties
        in_out_vars = [p.val for p in self.ch_model.input_prop + self.ch_model.output_prop]
        if missing_signature_vars and (is_sat or (
                self.printing_mode != PrintingMode.Silent or
                any(v in in_out_vars for v in missing_signature_vars))
        ):
            smart_print = _get_smart_print(self.filename)
            smart_print(f"Found {'satisfiable' if is_sat else 'unsatisfiable'} assignment "
                        f"of SMT problem for all values of {missing_signature_vars}; "
                        f"setting {self.ch_model.prop_type.__name__} of {missing_signature_vars} "
                        f"to 0 in yielded characteristic")

        if target_weight is not None and int(solution_var2ct[self._ch_weight]) != target_weight:
            raise ValueError(f"SMT ch. weight = {solution_var2ct[self._ch_weight]} "
                             f"!= {target_weight} = target_weight")

        if isinstance(self.ch_model, abstractproperty.chmodel.EncryptionChModel):
            Characteristic_cls = self.ch_model.__class__._get_EncryptionCharacteristic_cls()
        else:
            assert isinstance(self.ch_model, abstractproperty.chmodel.ChModel)
            Characteristic_cls = self.ch_model.__class__._get_Characteristic_cls()

        init_props = Characteristic_cls.get_properties_for_initialization(self.ch_model, solution_var2ct)
        input_prop, output_prop, external_props, assign_outprop_list = init_props

        # # debugging
        # print("\n_pysmt_model2ch")
        # print("ch model:", self.ch_model)
        # print("solution_var2ct:", solution_var2ct)
        # print("vars needed:", get_needed_vars(self.ch_model))
        # print("get_properties_for_initialization():", init_props, "\n")
        #

        # avoid *prop=*prop
        last_ch_found = Characteristic_cls(
            input_prop, output_prop, assign_outprop_list, self.ch_model, external_props,
            empirical_ch_weight=None, empirical_data_list=None, is_valid=is_sat)

        # checks

        assert isinstance(last_ch_found, Characteristic_cls), f"{last_ch_found}"

        assert not(self.assert_type == ChModelAssertType.ProbabilityOne
                   and last_ch_found.ch_weight != 0), f"{last_ch_found}"

        if self._exclude_zero_input_prop:
            compact_input = functools.reduce(
                operation.Concat, [d.val for d in last_ch_found.input_prop])
            if compact_input == core.Constant(0, compact_input.width):
                raise ValueError("exclude_zero_input_prop is True but characteristic input is "
                                 f"{last_ch_found.input_prop}\n{last_ch_found}")

        last_ch_found_v2c = {}
        for var_prop, ct_prop in last_ch_found.var_prop2ct_prop.items():
            if solution_var2ct.get(var_prop.val, ct_prop.val) != ct_prop.val:
                raise ValueError(f"SMT solution contains {var_prop.val} = {solution_var2ct[var_prop.val]}"
                                 f" but characteristic contains {var_prop.val} = {ct_prop.val}")
            last_ch_found_v2c[var_prop.val] = ct_prop.val

        if self._var_prop2ct_prop:
            for var_prop, ct_prop in self._var_prop2ct_prop.items():
                ch_ct_prop = type(ct_prop)(var_prop.val.xreplace(last_ch_found_v2c))
                if ct_prop != ch_ct_prop:
                    raise ValueError(f"({var_prop}, {ct_prop}) was added in var_prop2ct_prop"
                                     f" but {var_prop} has value {ch_ct_prop} in the characteristic"
                                     f"\nFull solution: {last_ch_found_v2c}\n{last_ch_found}")

        with context.Simplification(False):
            # solution_var2ct include weight variables
            last_ch_found_v2c = {**last_ch_found_v2c, **solution_var2ct}
            chmodel_asserts = [a.xreplace(last_ch_found_v2c) for a in self.chmodel_asserts]
            chmodel_asserts = functools.reduce(operation.BvAnd, chmodel_asserts)
            if (is_sat or isinstance(chmodel_asserts, core.Constant)) and chmodel_asserts != is_sat:
                # if is_sat=False, some variables might be missing
                raise ValueError(f"{is_sat} != chmodel_asserts = ({chmodel_asserts}) for the characteristic found"
                                 f"\nSMT solution = {solution_var2ct})"
                                 f"\nFull solution: {last_ch_found_v2c}\n{last_ch_found}")

        if target_weight is not None:
            last_ch_found._check_bv_weights(
                self._ch_weight.xreplace(solution_var2ct),
                [w.xreplace(solution_var2ct) for w in self._awvs])

            smt_weight = target_weight  # bv-weight (w/ truncate=True)
            ch_weight = last_ch_found.ch_weight  # decimal.Decimal recomputed
            abs_error = (smt_weight - ch_weight).copy_abs()

            # extra error due to truncate=True
            # e.g., if fb=2, max error = 0.11b (= 0.75 decimal) = 1 - 2**2
            # last factor to avoid Python decimal error
            fb = self.ch_model.num_frac_bits()
            extra_error = 1 - (decimal.Decimal(2) ** (-fb))
            assert not (fb == 0 and extra_error != 0)
            # extra_error = max(abs(ch_weight - math.ceil(ch_weight)), abs(ch_weight - math.floor(ch_weight)))

            max_abs_error = self._error + extra_error
            max_abs_error = decimal.Decimal(max_abs_error).quantize(
                decimal.Decimal("1." + "0" * (decimal.getcontext().prec // 2)), rounding=decimal.ROUND_UP)
            if abs_error > max_abs_error:
                aux_ws = [(v, c) for v, c in solution_var2ct.items() if str(v).startswith(self.weight_prefix)]
                raise ValueError(f"absolute error between integer weight {smt_weight} "
                                 f"(found by the SMT solver from {aux_ws}) and decimal weight {ch_weight} "
                                 f"(recomputed in Characteristic) is {abs_error}, "
                                 f"which is greater than maximum absolute error given by "
                                 f"ch_model-error={self._error} + extra_error={extra_error}"
                                 f"\n{last_ch_found}")

        return last_ch_found

    def find_next_ch(self, yield_assignment=False):
        """Return an iterator that yields the characteristics found in the SMT-based search.

        .. note::
            This method requires that `assert_type` is either
            `Validity` or `ProbabilityOne` (and not `ValidityAndWeight`).

        This method searches for characteristic using SMT solvers.
        The decision problem of whether there exists a characteristic
        (following the characteristic model ``ch_model``)
        is encoded as an SMT problem and given to the SMT solver,
        which checks its satisfiability.

        If the SMT solver finds the first problem satisfiable,
        an assignment of the variables that makes the problem satisfiable is
        obtained, and a `abstractproperty.characteristic.Characteristic`
        object is created and *yielded*.

        .. note::
            If ``yield_assignment`` is ``True``, the assignment
            (as a dictionary mapping `Variable` to `Constant` objects)
            is yielded instead of the characteristic.

        Afterwards, an additional constraint is added to the SMT problem
        to exclude the characteristic yielded and this procedure is repeated
        until all characteristics are found.

            >>> # example of search for LinearMask-Characteristic of Speck32.key_schedule
            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> v2c = {ch_model.input_mask[2]: core.Constant(0, 16)}  # setting input_mask[2] to 0
            >>> at = ChModelAssertType.ProbabilityOne
            >>> ch_finder = ChFinder(ch_model, at, "btor", var_prop2ct_prop=v2c, solver_seed=0)
            >>> for ch in ch_finder.find_next_ch(): print(ch.srepr()) # all 2-rounds probability-one trails
            Ch(w=0, id=0000 0000 0000, od=0000 0000 0000)
            Ch(w=0, id=0080 0080 0000, od=4001 4000 0001)
            Ch(w=0, id=0080 0000 0000, od=0000 4001 0001)
            Ch(w=0, id=0000 0080 0000, od=4001 0001 0000)
            >>> # example of search for BitValue-EncryptionCharacteristic of Speck32
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import EncryptionChModel
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = EncryptionChModel(Speck32, BitValue)
            >>> input_output_vals = ch_model.input_val + ch_model.output_val
            >>> # setting each input and output word to 0x0001
            >>> v2c = {v: core.Constant(1, 16) for v in input_output_vals}
            >>> at = ChModelAssertType.Validity
            >>> ch_finder = ChFinder(ch_model, at, "btor", var_prop2ct_prop=v2c, solver_seed=0)
            >>> found_ch = next(ch_finder.find_next_ch())
            >>> round_sep = found_ch.get_round_separators()
            >>> for round_ch in found_ch.split(round_sep): print(round_ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
                input_val=[0x0001, 0x0001], output_val=[0x0004, 0x0000], external_vals=[0x0205],
                assign_outval_list=[0x0201, 0x0004, 0x0000])
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
                input_val=[0x0004, 0x0000], output_val=[0x0001, 0x0001], external_vals=[0x0801],
                assign_outval_list=[0x0800, 0x0001, 0x0001])

        """
        if self.assert_type == ChModelAssertType.ValidityAndWeight:
            raise ValueError("find_next_ch requires assert_type != ChModelAssertType.ValidityAndWeight")

        smart_print = _get_smart_print(self.filename)
        symbolic_sig = self.ch_model.signature(CurrentSignatureType)

        parse_shifts_rotations = True if self.solver_name == "btor" else False
        bv2pysmt = functools.partial(
            pysmttypes.bv2pysmt, env=self.env, parse_shifts_rotations=parse_shifts_rotations)

        solver_kwargs = {}
        if self.solver_seed is not None:
            if self.solver_name == "btor":
                solver_kwargs = {"solver_options": {"seed": int(self.solver_seed) % 2**32}}  # btor seed uint32
            else:
                solver_kwargs = {"random_seed": self.solver_seed}
        solver = self.env.factory.Solver(name=self.solver_name, logic=logics.QF_BV, **solver_kwargs)

        for c in itertools.chain(self.initial_constraints, self.chmodel_asserts):
            solver.add_assertion(bv2pysmt(c, boolean=True))

        bv_model = None
        last_ch_found = None
        while True:
            if bv_model is not None:
                if len(symbolic_sig) == 0:
                    warnings.warn(f"empty signature of {self.ch_model}")
                    break
                if last_ch_found is not None:
                    last_ch_sig = last_ch_found.signature(CurrentSignatureType)
                else:
                    last_ch_sig = []
                    for v in symbolic_sig:
                        ct = self.ch_model.var2prop[v].val.xreplace(bv_model)
                        assert isinstance(ct, core.Constant)
                        last_ch_sig.append(ct)
                # disable simplification due to recursion error
                with context.Simplification(False):
                    exclude_last_ch = functools.reduce(
                        operation.BvOr,
                        [~operation.BvComp(ss, ls) for ss, ls in zip(symbolic_sig, last_ch_sig)]
                    )
                solver.add_assertion(bv2pysmt(exclude_last_ch, boolean=True))
                if self.printing_mode == PrintingMode.Debug:
                    smart_print(f"exclude_last_ch: {exclude_last_ch}", prepend_time=True)

            satisfiable = solver.solve()

            if satisfiable:
                self._last_model = solver.get_model()
                bv_model = pysmttypes.pysmt_model2bv_model(self._last_model)
                if yield_assignment:
                    yield bv_model
                else:
                    last_ch_found = self._pysmt_model2ch(bv_model, is_pysmt_model=False)
                    yield last_ch_found
            else:
                break

        solver.exit()

    def _compute_empirical_ch_weight(self, ch_found, empirical_weight_options):
        ch_found.compute_empirical_ch_weight(**empirical_weight_options)

    def _get_empirical_ch_weight(self, ch_found):
        return ch_found.empirical_ch_weight

    def _get_ch_weight(self, ch_found):
        return ch_found.ch_weight

    def _new_final_weight(self, ch_found, prev_final_weight):
        """Computes the new final weight given the previous characteristic yielded.

        Let B the error bound of the characteristic model.
        For the just found characteristic F, let F.W be the decimal weight
        and F.SW the SMT weight.

        If there exists a better characteristics G, then F.W ‹ G.W
        and G.SW ›= F.SW. But then, G.SW cannot be greater than
        F.W + B. Thus, the new final weight is at least F.W + B.
        """
        new_final_weight = int(math.ceil(ch_found.ch_weight + self._error))
        return min(prev_final_weight, new_final_weight)

    def find_next_ch_increasing_weight(
            self, initial_weight, final_weight=None, empirical_weight_options=None,
            stop_after_optimal=True, yield_weight=False
    ):
        """Return an iterator that yields the characteristics found in the SMT-based search
        with increasing weight order.

        .. note::
            This method requires that `assert_type` is `ValidityAndWeight`
            (and not `Validity` nor `ProbabilityOne`).

        This method searches for optimal characteristics (with optimal probability)
        using SMT solvers as follows.

        First, the probability space is decomposed into many intervals
        :math:`I_w = (2^{-w-1}, 2^{-w}]`,
        where ``w = initial_weight, initial_weight + 1, ..., final_weight``.
        For each interval, the decision problem of whether there exists a
        characteristic (following the characteristic model ``ch_model``)
        with probability :math:`p \in I_w` is encoded as an
        SMT problem. Note that a characteristic has probability :math:`p \in I_w`
        if and only if its integer weight (the integer part of the weight)
        is equal to :math:`w`.

        .. note::

            See `abstractproperty.characteristic.Characteristic` for
            the characteristic probability and weight considered here.

        The SMT problems are provided to the SMT solver,
        which checks their satisfiability in increasing weight order.
        When the SMT solver finds the first satisfiable problem,
        an assignment of the variables that makes the problem satisfiable is
        obtained, and a `abstractproperty.characteristic.Characteristic`
        object is created and *yielded*.

        .. note::

            If ``yield_weight`` is ``True``, a tuple is yielded instead,
            containing the target weight :math:`w` and the characteristic.

            If ``empirical_weight_options`` is provided, before yielding
            the characteristic, the empirical weight
            of the characteristic is computed by calling the method
            `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight`
            with the given options as arguments (see below for an explanation
            of ``empirical_weight_options``).
            Importantly, if the empirical weight computed is ``math.inf``,
            this characteristic is NOT yielded and the search continues.
            If ``empirical_weight_options`` is not provided, all characteristics
            found are yielded.

        If the error bound of the associated characteristic model
        (see `abstractproperty.chmodel.ChModel.error`) is zero,
        the first characteristic yielded is optimal in the sense that
        there are no characteristics with integer weight strictly smaller,
        and the search finishes (if ``stop_after_optimal is True``,
        otherwise it continues until all characteristics are found).

        .. note::

            Note the first characteristic yielded is optimal for the
            characteristic probability here considered, which might be
            an approximation of the actual characteristic probability
            (e.g., see `differential.characteristic.Characteristic`
            or `linear.characteristic.Characteristic`).

        Let :math:`\hat{w}` the weight of the first characteristic yielded.
        If the error bound of the associated characteristic model is
        :math:`e > 0`, then the search continues yielding all characteristics
        with weights between :math:`\hat{w}` and :math:`\hat{w} + e`.
        After all these characteristics have been yielded,
        the optimal characteristic (the characteristic among all yielded
        characteristics with the lowest
        `abstractproperty.characteristic.Characteristic.ch_weight`)
        is yielded again and the search finishes
        (if ``stop_after_optimal is True``,  otherwise it continues until all
        characteristics are found).

        .. note::

            After the iterator is exhausted, the last characteristic yielded
            is always the optimal, but the optimal characteristic is only yielded
            twice if it is different from the previous characteristic yielded.

            Moreoever, if ``yield_weight`` is ``True``, the second time
            the optimal characteristic is yielded a tuple is also yielded,
            but the first entry in the tuple contains ``None``
            (the target weight of the optimal characteristic is
            yielded the first time the optimal characteristic is yielded).

        ::

            >>> # example of search for XorDiff-EncryptionCharacteristic of Speck32
            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff, RXDiff
            >>> from cascada.differential.chmodel import ChModel, EncryptionChModel
            >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = EncryptionChModel(Speck32, XorDiff)
            >>> assert_type = ChModelAssertType.ValidityAndWeight
            >>> ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0)
            >>> # no need to exclude the input zero XOR difference if initial_weight != 0
            >>> ewo = {"seed": 0}  # no need to specify all args
            >>> next(ch_finder.find_next_ch_increasing_weight(1, empirical_weight_options=ewo))  # doctest: +NORMALIZE_WHITESPACE
            EncryptionCharacteristic(ch_weight=1, empirical_ch_weight=1.027020213933709037746664618,
                assignment_weights=[1, 0, 0, 0],
                input_diff=[0x0010, 0x2000], output_diff=[0x8000, 0x8002], external_diffs=[0x0000, 0x0000],
                assign_outdiff_list=[0x0000, 0x8000, 0x8000, 0x8002])
            >>> # example of search for RXDiff-Characteristic of Speck32.key_schedule
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])
            >>> dmk0, dmk1 = ch_model.input_diff[:2]
            >>> v2c = {dmk0: core.Constant(0, 16), dmk1: core.Constant(0, 16)}
            >>> ch_finder = ChFinder(ch_model, assert_type, "btor", var_prop2ct_prop=v2c, solver_seed=0)
            >>> for ch in ch_finder.find_next_ch_increasing_weight(1): print(ch.srepr())  # doctest: +ELLIPSIS
            Ch(w=2.830, id=0000 0000 0000, od=0000 0001 0007)
            ...
            Ch(w=3.830, id=0000 0000 8000, od=8000 8002 800b)
            ...
            Ch(w=4.830, id=0000 0000 0001, od=0001 0005 001b)
            ...
            Ch(w=4.830, id=0000 0000 8000, od=8000 8002 800e)
            Ch(w=2.830, id=0000 0000 0000, od=0000 0001 0007)

        Args:
            initial_weight: the initial weight to start the search
            final_weight: the last weight to consider in the search
                (by default ``math.inf``)
            empirical_weight_options: (optional) a dictionary containing
                the arguments of
                `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight`
                (used similar as ``**kwargs``, that is,
                ``compute_empirical_ch_weight(**empirical_weight_options)``
            yield_weight: if ``True``, the target weight of the SMT problem
                is also yielded (default ``False``)

        """
        assert isinstance(initial_weight, int) and initial_weight >= 0

        if initial_weight > 0 and self._exclude_zero_input_prop:
            warnings.warn("the constraint excluding the zero input property is being added"
                          " to the SMT problem but might be redundant since initial_weight > 0")

        if final_weight is None:
            final_weight = math.inf
        else:
            assert isinstance(final_weight, int)
        final_weight = min(2**self._ch_weight.width - 1, final_weight)

        assert initial_weight <= final_weight

        if empirical_weight_options is not None:
            if isinstance(empirical_weight_options, (list, tuple)):
                assert len(empirical_weight_options) == 2
                ewos = empirical_weight_options
            else:
                ewos = [empirical_weight_options]
            args = {"num_input_samples", "num_external_samples", "split_by_max_weight",
                    "split_by_rounds", "seed", "C_code", "num_parallel_processes"}
            assert all(all(k in args for k in ewo.keys()) for ewo in ewos if ewo is not None)

        if self.assert_type != ChModelAssertType.ValidityAndWeight:
            raise ValueError("find_next_ch_increasing_weight requires assert_type == "
                             "ChModelAssertType.ValidityAndWeight")

        smart_print = _get_smart_print(self.filename)
        symbolic_sig = self.ch_model.signature(CurrentSignatureType)

        parse_shifts_rotations = True if self.solver_name == "btor" else False
        bv2pysmt = functools.partial(
            pysmttypes.bv2pysmt, env=self.env, parse_shifts_rotations=parse_shifts_rotations)

        solver_kwargs = {}
        if self.solver_seed is not None:
            if self.solver_name == "btor":
                solver_kwargs = {"solver_options": {"seed": int(self.solver_seed) % 2**32}}  # btor seed uint32
            else:
                solver_kwargs = {"random_seed": self.solver_seed}
        solver = self.env.factory.Solver(name=self.solver_name, logic=logics.QF_BV, **solver_kwargs)
        for c in itertools.chain(self.initial_constraints, self.chmodel_asserts):
            solver.add_assertion(bv2pysmt(c, boolean=True))

        best_ch_found = None
        best_ch_weight = None
        last_ch_found = None
        target_weight = initial_weight

        while target_weight < final_weight or (target_weight == final_weight and last_ch_found is None):
            solver.push()
            target_weight_constraint = operation.BvComp(
                self._ch_weight, core.Constant(target_weight, self._ch_weight.width))
            solver.add_assertion(bv2pysmt(target_weight_constraint, boolean=True))
            if self.printing_mode != PrintingMode.Silent:
                smart_print(f"Solving for weight = {target_weight}", prepend_time=True)

            while True:  # find all ch with given target_weight
                if last_ch_found is not None:
                    if len(symbolic_sig) == 0:
                        warnings.warn(f"empty signature of {self.ch_model}")
                        break
                    last_ch_sig = last_ch_found.signature(CurrentSignatureType)
                    # disable simplification due to recursion error
                    with context.Simplification(False):
                        exclude_last_ch = functools.reduce(
                            operation.BvOr,
                            [~operation.BvComp(ss, ls) for ss, ls in zip(symbolic_sig, last_ch_sig)]
                        )
                    solver.add_assertion(bv2pysmt(exclude_last_ch, boolean=True))
                    if self.printing_mode == PrintingMode.Debug:
                        smart_print(f"exclude_last_ch: {exclude_last_ch}", prepend_time=True)

                satisfiable = solver.solve()

                if not satisfiable:
                    break
                else:
                    self._last_model = solver.get_model()
                    last_ch_found = self._pysmt_model2ch(self._last_model, target_weight)

                    valid_ch = True
                    if empirical_weight_options is not None:
                        self._compute_empirical_ch_weight(last_ch_found, empirical_weight_options)
                        if self._get_empirical_ch_weight(last_ch_found) == math.inf:
                            if self.printing_mode == PrintingMode.WeightsAndSrepr:
                                smart_print(f"Invalid characteristic found | {last_ch_found.srepr()}", prepend_time=True)
                            elif self.printing_mode in [PrintingMode.WeightsAndVrepr, PrintingMode.Debug]:
                                smart_print(f"Invalid characteristic found | {last_ch_found.vrepr()}", prepend_time=True)
                            valid_ch = False

                    if valid_ch:
                        if yield_weight:
                            yield (target_weight, last_ch_found)
                        else:
                            yield last_ch_found

                        if best_ch_weight is None or self._get_ch_weight(last_ch_found) < best_ch_weight:
                            best_ch_found = last_ch_found
                            best_ch_weight = self._get_ch_weight(best_ch_found)
                            new_final_weight = self._new_final_weight(best_ch_found, final_weight)
                            if stop_after_optimal and new_final_weight != final_weight:
                                final_weight = new_final_weight
                                del new_final_weight
                                if self.printing_mode != PrintingMode.Silent:
                                    smart_print(f"Final weight decreased to {final_weight}", prepend_time=True)
                                if final_weight <= target_weight:
                                    break  # found optimal

            # target_weight exhausted
            target_weight += 1
            solver.pop()

        if best_ch_found is not None:
            if best_ch_found != last_ch_found:
                # no need to return twice the same ch
                if yield_weight:
                    yield (None, best_ch_found)
                else:
                    yield best_ch_found
        else:
            raise ValueError(f"no characteristic found with weight <= {final_weight}")

        solver.exit()

    def find_next_ch_increasing_weight_fixed_in_out(
            self, input_prop, output_prop, initial_weight,
            final_weight=None, empirical_weight_options=None,
            use_empirical_weight=False,
    ):
        """Return an iterator that yields the characteristics found in the SMT-based search
        with increasing weight order and with fixed input and output properties.

        This method is similar as `find_next_ch_increasing_weight` with three differences:

         - This method only finds characteristics with input and output properties
           (`Characteristic.input_prop` and `Characteristic.output_prop`) equal
           to the input and output properties given by ``input_prop`` and ``output_prop``
           (lists containing `Constant` objects or constant `Property` objects).
         - When the SMT solver finds a satisfiable problem, a tuple of
           two elements is yielded: the first element is the cumulative weight
           and the second element is the characteristic found
           (as a `abstractproperty.characteristic.Characteristic` object).
           The cumulative weight is the weight (-log2) of the sum of the
           probabilities of all characteristics found in the search
           (including the characteristic just found).
         - After all these characteristics have been yielded,
           the optimal characteristic is NOT yielded again.

        By default, the cumulative weight is computed by taking the decimal weights
        (see `Characteristic.ch_weight`) of the found characteristic, transforming the
        decimal weights into probabitlies, then summing these probabilities
        and finally transforming the probability sum into a weight.
        However, if the argument ``empirical_weight_options`` is given
        (see `find_next_ch_increasing_weight`) and the argument ``use_empirical_weight``
        is ``True``, then the empirical weights are used instead of the decimal weights.

        .. note::

            For the `Difference`/`LinearMask` property types,
            the cumulative weight estimates the weight of the probability
            of the differential/hull with the given input and output
            differences/masks.

        ::

            >>> # example of search for XorDiff-EncryptionCharacteristic of Speck32 with fixed input/output difference
            >>> from cascada.bitvector.core import Constant
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import EncryptionChModel
            >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(4)
            >>> ch_model = EncryptionChModel(Speck32, XorDiff)
            >>> assert_type = ChModelAssertType.ValidityAndWeight
            >>> ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0)
            >>> best_ch = next(ch_finder.find_next_ch_increasing_weight(1))
            >>> best_ch  # doctest: +NORMALIZE_WHITESPACE
            EncryptionCharacteristic(ch_weight=5, assignment_weights=[2, 0, 1, 2, 0, 0],
                input_diff=[0x2800, 0x0010], output_diff=[0x8000, 0x840a], external_diffs=[0x0000, 0x0000, 0x0000, 0x0000],
                assign_outdiff_list=[0x0040, 0x8000, 0x8100, 0x8000, 0x8000, 0x840a])
            >>> iterator = ch_finder.find_next_ch_increasing_weight_fixed_in_out
            >>> # using the input and output differences of the best 5-round characteristic
            >>> for w, ch in iterator(best_ch.input_prop, best_ch.output_prop, 0): print(w, "|", ch)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            5 | EncryptionCharacteristic(ch_weight=5, assignment_weights=[2, 0, 1, 2, 0, 0],
                input_diff=[0x2800, 0x0010], output_diff=[0x8000, 0x840a], external_diffs=[0x0000, 0x0000, 0x0000, 0x0000],
                assign_outdiff_list=[0x0040, 0x8000, 0x8100, 0x8000, 0x8000, 0x840a])
            4.999999999916024096260838242 | EncryptionCharacteristic(ch_weight=39, assignment_weights=[6, 13, 11, 9, 0, 0],
                input_diff=[0x2800, 0x0010], output_diff=[0x8000, 0x840a], external_diffs=[0x0000, 0x0000, 0x0000, 0x0000],
                assign_outdiff_list=[0x03a0, 0x4f1f, 0x837f, 0x8000, 0x8000, 0x840a])
            ...
            4.999999999894210043141894841 | EncryptionCharacteristic(ch_weight=49, assignment_weights=[10, 15, 10, 14, 0, 0],
                input_diff=[0x2800, 0x0010], output_diff=[0x8000, 0x840a], external_diffs=[0x0000, 0x0000, 0x0000, 0x0000],
                assign_outdiff_list=[0x3fa0, 0xa00f, 0xff3f, 0x8000, 0x8000, 0x840a])
            >>> print(w)  # weight of differential
            4.999999999894210043141894841

        """
        if use_empirical_weight:
            assert empirical_weight_options is not None

        old_var_prop2ct_prop = self._var_prop2ct_prop.copy()
        old_initial_constraints = self.initial_constraints[:]

        # no need to use _get_ch_weight or _get_empirical_ch_weight

        for var_prop, ct_prop in itertools.chain(
                zip(self.ch_model.input_prop, input_prop),
                zip(self.ch_model.output_prop, output_prop),
        ):
            if not isinstance(ct_prop, self.ch_model.prop_type):
                assert isinstance(ct_prop, core.Constant)
                ct_prop = self.ch_model.prop_type(ct_prop)
            self._var_prop2ct_prop[var_prop] = ct_prop
            self.initial_constraints.append(operation.BvComp(var_prop.val, ct_prop.val))

        self._check_initial_constraints(
            self.ch_model, self.initial_constraints, self.chmodel_asserts,
            self._exclude_zero_input_prop, self._var_prop2ct_prop,
            self._vars_in_constraints, self._raise_exception_missing_var
        )

        cumulative_w = None

        for yielded_weight, ch_found in self.find_next_ch_increasing_weight(
            initial_weight, final_weight=final_weight,
            empirical_weight_options=empirical_weight_options,
            stop_after_optimal=False, yield_weight=True
        ):
            if yielded_weight is None:  # second time optimal ch is yielded
                assert cumulative_w is not None
                continue

            if use_empirical_weight:
                next_weight = ch_found.empirical_ch_weight
            else:
                next_weight = ch_found.ch_weight

            if cumulative_w is None:
                cumulative_w = next_weight
            else:
                cumulative_w = _merge_weights(next_weight, cumulative_w)

            yield cumulative_w, ch_found

        self._var_prop2ct_prop = old_var_prop2ct_prop
        self.initial_constraints = old_initial_constraints

    def formula_size(self, measure=None):
        """Return the size of the underlying SMT problem.

        See `pysmt.oracles.SizeOracle` for choosing ``measure``.
        """
        environment.push_env()
        env = environment.get_env()
        assert env != self._env
        bv2pysmt = functools.partial(pysmttypes.bv2pysmt, env=env)
        size = 0
        for c in itertools.chain(self.chmodel_asserts, self.initial_constraints):
            size += env.sizeo.get_size(bv2pysmt(c, boolean=True), measure)
        environment.pop_env()
        assert environment.get_env() == self._env
        return size

    def hrepr(self, full_repr=False):
        """Return a human-readable representation of the base SMT problem.

        The base SMT problem is the SMT problem containing the validity,
        probability-one and/or weight assertions (depending on ``assert_type``)
        and the additional constraints from `initial_constraints`,
        but excluding constraints created during the search such as
        the constraints fixing the characteristic weight variable to a
        constant value in `find_next_ch_increasing_weight` or
        the constraints fixing the input and output properties
        in `find_next_ch_increasing_weight_fixed_in_out`.

        If ``full_repr`` is False, the short string representation srepr is used.
        """
        representation = []
        if self.initial_constraints:
            representation.append("; initial constraints")
            for c in self.initial_constraints:
                representation.append(f"assert {c if full_repr else c.srepr()}")
        representation.append("; characteristic model assertions")
        for c in self.chmodel_asserts:
            representation.append(f"assert {c if full_repr else c.srepr()}")
        return "\n".join(representation)


class CipherChFinder(ChFinder):
    """Search for cipher characteristics by modeling the search as a sequence of SMT problems.

    Given a characteristic model of a `Cipher`
    (`abstractproperty.chmodel.CipherChModel`)
    defined for a particular `Property` (e.g., `XorDiff` or `BitValue`),
    this class finds characteristics
    (`abstractproperty.characteristic.CipherCharacteristic`)
    satisfying the characteristic model by modelling the search
    as a sequence of SMT problems in the bit-vector theory.

    To initialize a `CipherChFinder` object, first two auxiliary instances of
    `ChFinder` are created:

    - ``ks_finder``: a `ChFinder` with arguments
      ``ch_model.ks_ch_model``, ``ks_assert_type``
      ``ks_exclude_zero_input_prop`` and ``ks_weight_prefix``
    - ``enc_finder``: a `ChFinder` with arguments
      ``ch_model.enc_ch_model``, ``enc_assert_type``,
      ``enc_exclude_zero_input_prop`` amd ``enc_weight_prefix``

    Both ``ks_finder`` and ``enc_finder`` (together with the `CipherChFinder` object)
    share the arguments `solver_name`, `printing_mode`,
    `filename`,  `solver_seed` and `env`.

    Then, these two auxiliary `ChFinder` objects are merged into a `CipherChFinder`
    (which is also an instance of `ChFinder`) as follows:

    - ``solver_name``, ``printing_mode``, ``filename``,  ``solver_seed``
      ``env`` are the same as the ones from ``ks_finder`` and ``enc_finder``
    - ``ch_model`` is set to the characteristic model of the cipher
      (a subclass of `abstractproperty.chmodel.CipherChModel`)
    - ``assert_type`` is set as the *largest* assertion type, following
      the order `ValidityAndWeight` > `Validity` > `ProbabilityOne`
    - ``initial_constraints`` contains all initial constraints
      (including the ones derived from ``ks_exclude_zero_input_prop``,
      ``enc_exclude_zero_input_prop`` and ``var_prop2ct_prop``)
    - ``chmodel_asserts`` is the union of `chmodel_asserts` of
      ``ks_finder`` and ``enc_finder``

    See also `ChFinder`.

        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import CipherChModel
        >>> from cascada.smt.chsearch import CipherChFinder, ChModelAssertType
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = CipherChModel(Speck32, XorDiff)
        >>> assert_type = ChModelAssertType.ValidityAndWeight
        >>> ch_finder = CipherChFinder(ch_model, assert_type, assert_type, "btor",
        ...                            enc_exclude_zero_input_prop=True, solver_seed=0)
        >>> ch_finder.formula_size()
        1290
        >>> print(ch_finder.hrepr(full_repr=False))  # doctest: +NORMALIZE_WHITESPACE
        ; initial constraints
        assert ~((dp0 :: dp1) == 0x00000000)
        ; characteristic model assertions
        assert ((~(... << ...) ^ (dmk1 << 0x0001)) & (~(... << ...) ^ (dk1 << 0x0001)) &
            ((... >>> ...) ^ dmk1 ^ dk1 ^ ((dmk0 >>> 7) << 0x0001))) == 0x0000
        assert dmk1 == dmk1_out
        assert ((dmk1 <<< 2) ^ dk1) == dk3_out
        assert wk0 == PopCount(~((~... ^ dmk1) & (~... ^ dk1))[14:])
        assert wk1 == 0b0
        assert wk2 == 0b0
        assert wk == wk0
        assert ((~(... << ...) ^ (dp1 << 0x0001)) & (~(... << ...) ^ (dx1 << 0x0001)) &
            ((... >>> ...) ^ dp1 ^ dx1 ^ ((dp0 >>> 7) << 0x0001))) == 0x0000
        assert ((~(... << ...) ^ ((... ^ ...) << 0x0001)) & (~(... << ...) ^ (dx6 << 0x0001)) &
            ((... >>> ...) ^ ... ^ ... ^ dx6 ^ (((... ^ ...) >>> 7) << 0x0001))) == 0x0000
        assert (dx6 ^ dk3_out) == dx7_out
        assert ((((dp1 <<< 2) ^ dx1 ^ dmk1_out) <<< 2) ^ dx6 ^ dk3_out) == dx9_out
        assert we0 == PopCount(~((~... ^ dp1) & (~... ^ dx1))[14:])
        assert we1 == PopCount(~((~... ^ ... ^ ...) & (~... ^ dx6))[14:])
        assert we2 == 0b0
        assert we3 == 0b0
        assert we == ((0b0 :: we0) + (0b0 :: we1))

    """

    def __init__(self, ch_model, ks_assert_type, enc_assert_type, solver_name,
                 initial_constraints=None,
                 ks_exclude_zero_input_prop=False, enc_exclude_zero_input_prop=False,
                 var_prop2ct_prop=None, raise_exception_missing_var=True,
                 printing_mode=PrintingMode.Silent, filename=None,
                 ks_weight_prefix="wk", enc_weight_prefix="we", solver_seed=None, env=None):
        assert isinstance(ch_model, abstractproperty.chmodel.CipherChModel)

        ks_finder = ChFinder(
            ch_model.ks_ch_model, assert_type=ks_assert_type, solver_name=solver_name,
            initial_constraints=None, exclude_zero_input_prop=ks_exclude_zero_input_prop,
            var_prop2ct_prop=None, raise_exception_missing_var=False,
            printing_mode=PrintingMode.Silent, filename=None,
            weight_prefix=ks_weight_prefix, env=env,
        )

        # initial_constraints=[] to avoid algebraic warning
        enc_finder = ChFinder(
            ch_model.enc_ch_model, assert_type=enc_assert_type, solver_name=solver_name,
            initial_constraints=[], exclude_zero_input_prop=enc_exclude_zero_input_prop,
            var_prop2ct_prop=None, raise_exception_missing_var=False,
            printing_mode=PrintingMode.Silent, filename=None,
            weight_prefix=enc_weight_prefix, env=ks_finder.env,
        )

        assert ks_finder.env == enc_finder.env

        # initialize initial_constraints
        
        if initial_constraints is None:
            initial_constraints = []
        else:
            assert isinstance(initial_constraints, collections.abc.Sequence)
            initial_constraints = initial_constraints[:]

        initial_constraints.extend(ks_finder.initial_constraints)
        initial_constraints.extend(enc_finder.initial_constraints)

        # *_exclude_zero_input_prop processed in *_finder's init method

        prop_type = ch_model.ks_ch_model.prop_type
        assert ch_model.enc_ch_model.prop_type == prop_type

        if var_prop2ct_prop is not None:
            new_var_prop2ct_prop = collections.OrderedDict()
            for vp, cp in var_prop2ct_prop.items():
                if not isinstance(vp, prop_type):
                    assert not isinstance(vp, core.Constant)
                    vp = prop_type(vp)
                if not isinstance(cp, prop_type):
                    assert isinstance(cp, core.Constant)
                    cp = prop_type(cp)
                new_var_prop2ct_prop[vp] = cp
            var_prop2ct_prop = new_var_prop2ct_prop
            for vp, cp in var_prop2ct_prop.items():
                initial_constraints.append(operation.BvComp(vp.val, cp.val))
        else:
            var_prop2ct_prop = {}

        #

        kat, eat = ks_finder.assert_type, enc_finder.assert_type
        CMAT = ChModelAssertType
        if kat == CMAT.ValidityAndWeight and eat == CMAT.ValidityAndWeight:
            max_weight = ch_model.ks_ch_model.max_weight(truncate=True)
            max_weight += ch_model.enc_ch_model.max_weight(truncate=True)
            max_width = max(max_weight.bit_length(), ks_finder._ch_weight.width, enc_finder._ch_weight.width)
            ch_weight = operation.zero_extend(ks_finder._ch_weight, max_width - ks_finder._ch_weight.width)
            ch_weight += operation.zero_extend(enc_finder._ch_weight, max_width - enc_finder._ch_weight.width)
            error = ks_finder.ch_model.error() + enc_finder.ch_model.error()
            assert_type = CMAT.ValidityAndWeight  # for super() calls
        elif kat in [CMAT.Validity, CMAT.ProbabilityOne] and eat == CMAT.ValidityAndWeight:
            assert ks_finder._ch_weight is None and ks_finder._error == 0
            ch_weight = enc_finder._ch_weight
            error = enc_finder.ch_model.error()
            assert_type = CMAT.ValidityAndWeight
        elif kat == CMAT.ValidityAndWeight and eat in [CMAT.Validity, CMAT.ProbabilityOne]:
            assert enc_finder._ch_weight is None and enc_finder._error == 0
            ch_weight = ks_finder._ch_weight
            error = ks_finder.ch_model.error()
            assert_type = CMAT.ValidityAndWeight
        else:
            assert kat in [CMAT.Validity, CMAT.ProbabilityOne] and eat in [CMAT.Validity, CMAT.ProbabilityOne]
            assert ks_finder._ch_weight is None and ks_finder._error == 0
            assert enc_finder._ch_weight is None and enc_finder._error == 0
            ch_weight = None
            error = 0
            if CMAT.Validity not in [kat, eat]:
                assert_type = CMAT.ProbabilityOne
            else:
                assert_type = CMAT.Validity

        #

        chmodel_asserts = ks_finder.chmodel_asserts + enc_finder.chmodel_asserts
        vars_in_constraints = ks_finder._vars_in_constraints | enc_finder._vars_in_constraints

        ChFinder._check_initial_constraints(
            ks_finder.ch_model, initial_constraints, chmodel_asserts,
            ks_exclude_zero_input_prop, var_prop2ct_prop, vars_in_constraints,
            raise_exception_missing_var
        )
        ChFinder._check_initial_constraints(
            enc_finder.ch_model, initial_constraints, chmodel_asserts,
            enc_exclude_zero_input_prop, var_prop2ct_prop, vars_in_constraints,
            raise_exception_missing_var
        )

        self.ks_finder = ks_finder
        self.enc_finder = enc_finder

        self.ch_model = ch_model
        self.assert_type = assert_type
        self.solver_name = solver_name
        self.initial_constraints = initial_constraints
        self.printing_mode = printing_mode
        self.filename = filename
        self.weight_prefix = None
        self.solver_seed = solver_seed
        self._env = ks_finder.env
        self.chmodel_asserts = chmodel_asserts

        # variables not added in docstring (private variables)
        self._exclude_zero_input_prop = ks_exclude_zero_input_prop and enc_exclude_zero_input_prop
        self._var_prop2ct_prop = var_prop2ct_prop
        self._ch_weight = ch_weight
        self._error = error
        self._vars_in_constraints = vars_in_constraints

    def _pysmt_model2ch(self, solution_var2ct, target_weight=None, is_pysmt_model=True, is_sat=True):
        assert is_sat is True

        if is_pysmt_model:
            solution_var2ct = pysmttypes.pysmt_model2bv_model(solution_var2ct)
        else:
            solution_var2ct = solution_var2ct.copy()

        def _get_needed_vars(my_ch_model):
            var_needed = [p.val for p in my_ch_model.input_prop if p.val not in my_ch_model.ssa._input_vars_not_used]
            # # ks_ch_model has no external vars and enc_ch_model gets those from ks_ch_model.output
            # for ext_var, prop in my_ch_model.external_var2prop.items():
            #     if not isinstance(prop.val, core.Constant):
            #         var_needed.append(ext_var)
            for outprop, op_model in my_ch_model.assign_outprop2op_model.items():
                # if op_model.max_weight() != 0:
                if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                    var_needed.append(outprop.val)
            return var_needed

        def get_needed_vars(my_cipher_ch_model):
            return _get_needed_vars(my_cipher_ch_model.ks_ch_model) + _get_needed_vars(my_cipher_ch_model.enc_ch_model)

        missing_signature_vars = []
        for v in get_needed_vars(self.ch_model):
            if v not in solution_var2ct:
                missing_signature_vars.append(v)
                solution_var2ct[v] = core.Constant(0, v.width)
        if missing_signature_vars:
            smart_print = _get_smart_print(self.filename)
            smart_print(f"Found {'satisfiable' if is_sat else 'unsatisfiable'} assignment "
                        f"of SMT problem for all values of {missing_signature_vars}; "
                        f"setting {self.ch_model.prop_type.__name__} of {missing_signature_vars} "
                        f"to 0 in yielded characteristic")

        if target_weight is not None and \
                int(self._ch_weight.xreplace(solution_var2ct)) != target_weight:
            raise ValueError(f"SMT ch. weight = {solution_var2ct[self._ch_weight]} "
                             f"!= {target_weight} = target_weight")

        CipherCharacteristic_cls = self.ch_model.__class__._get_CipherCharacteristic_cls()

        init_props = CipherCharacteristic_cls.get_properties_for_initialization(self.ch_model, solution_var2ct)
        assert len(init_props) == 6
        ks_input_prop, ks_output_prop, ks_assign_outprop_list = init_props[:3]
        enc_input_prop, enc_output_prop, enc_assign_outprop_list = init_props[-3:]

        # # debugging
        # print("\n_pysmt_model2ch")
        # print("ch model:", self.ch_model)
        # print("ch model ks ssa :", self.ch_model.ks_ch_model.ssa)
        # print("ch model enc ssa :", self.ch_model.enc_ch_model.ssa)
        # print("solution_var2ct:", solution_var2ct)
        # print("vars needed:", get_needed_vars(self.ch_model))
        # print("missing_signature_vars:", missing_signature_vars), "\n")
        #

        # avoid *_props=*_props (super might not abstract)
        last_ch_found = CipherCharacteristic_cls(
            ks_input_prop,
            ks_output_prop,
            ks_assign_outprop_list,
            enc_input_prop,
            enc_output_prop,
            enc_assign_outprop_list,
            self.ch_model,
            # ks_free_props,
            # enc_free_props,
            ks_empirical_ch_weight=None,
            ks_empirical_data_list=None,
            enc_empirical_ch_weight=None,
            enc_empirical_data_list=None,
            ks_is_valid=is_sat,
            enc_is_valid=is_sat,
        )

        assert isinstance(last_ch_found, CipherCharacteristic_cls), f"{last_ch_found}"

        assert not(self.ks_finder.assert_type == ChModelAssertType.ProbabilityOne
                   and last_ch_found.ks_characteristic.ch_weight != 0), f"{last_ch_found}"
        assert not(self.enc_finder.assert_type == ChModelAssertType.ProbabilityOne
                   and last_ch_found.enc_characteristic.ch_weight != 0), f"{last_ch_found}"

        for i, aux_finder in enumerate([self.ks_finder, self.enc_finder]):
            aux_prefix = "ks" if i == 0 else "enc"
            aux_ch = last_ch_found.ks_characteristic if i == 0 else last_ch_found.enc_characteristic
            if aux_finder._exclude_zero_input_prop:
                compact_input = functools.reduce(operation.Concat, [d.val for d in aux_ch.input_prop])
                if compact_input == core.Constant(0, compact_input.width):
                    raise ValueError(f"exclude_{aux_prefix}_zero_input_prop is True but {aux_prefix}-"
                                     f"characteristic input is {aux_ch.input_prop}\n{last_ch_found}")

        last_ch_found_v2c = {}
        for var_prop, ct_prop in itertools.chain(
                last_ch_found.ks_characteristic.var_prop2ct_prop.items(),
                last_ch_found.enc_characteristic.var_prop2ct_prop.items()
        ):
            if solution_var2ct.get(var_prop.val, ct_prop.val) != ct_prop.val:
                raise ValueError(f"SMT solution contains {var_prop.val} = {solution_var2ct[var_prop.val]}"
                                 f" but characteristic contains {var_prop.val} = {ct_prop.val}")
            last_ch_found_v2c[var_prop.val] = ct_prop.val

        if self._var_prop2ct_prop:
            for var_prop, ct_prop in self._var_prop2ct_prop.items():
                ch_ct_prop = type(ct_prop)(var_prop.val.xreplace(last_ch_found_v2c))
                if ct_prop != ch_ct_prop:
                    raise ValueError(f"({var_prop}, {ct_prop}) was added in var_prop2ct_prop"
                                     f"but {var_prop} has value {ch_ct_prop} in the characteristic"
                                     f"\nFull solution: {last_ch_found_v2c}\n{last_ch_found}")

        with context.Simplification(False):
            # solution_var2ct include weight variables
            last_ch_found_v2c = {**last_ch_found_v2c, **solution_var2ct}
            chmodel_asserts = [a.xreplace(last_ch_found_v2c) for a in self.chmodel_asserts]
            chmodel_asserts = functools.reduce(operation.BvAnd, chmodel_asserts)
            if (is_sat or isinstance(chmodel_asserts, core.Constant)) and chmodel_asserts != is_sat:
                raise ValueError(f"{is_sat} != chmodel_asserts = ({chmodel_asserts}) for the characteristic found"
                                 f"\nSMT solution = {solution_var2ct})"
                                 f"\nFull solution: {last_ch_found_v2c}\n{last_ch_found}")

        if target_weight is not None:
            if self.ks_finder.assert_type == ChModelAssertType.ValidityAndWeight:
                last_ch_found.ks_characteristic._check_bv_weights(
                    self.ks_finder._ch_weight.xreplace(solution_var2ct),
                    [w.xreplace(solution_var2ct) for w in self.ks_finder._awvs])
            if self.enc_finder.assert_type == ChModelAssertType.ValidityAndWeight:
                last_ch_found.enc_characteristic._check_bv_weights(
                    self.enc_finder._ch_weight.xreplace(solution_var2ct),
                    [w.xreplace(solution_var2ct) for w in self.enc_finder._awvs])

            smt_weight = target_weight
            ch_weight = self._get_ch_weight(last_ch_found, add_aux_probability=False)
            abs_error = (smt_weight - ch_weight).copy_abs()

            # extra error due to truncate=True
            ks_extra_error, enc_extra_error = decimal.Decimal(0), decimal.Decimal(0)
            if self.ks_finder.assert_type == ChModelAssertType.ValidityAndWeight:
                ks_fb = self.ch_model.ks_ch_model.num_frac_bits()
                ks_extra_error = 1 - (decimal.Decimal(2) ** (-ks_fb))
                assert not (ks_fb == 0 and ks_extra_error != 0)
            if self.enc_finder.assert_type == ChModelAssertType.ValidityAndWeight:
                enc_fb = self.ch_model.enc_ch_model.num_frac_bits()
                enc_extra_error = 1 - (decimal.Decimal(2) ** (-enc_fb))
                assert not (enc_fb == 0 and enc_extra_error != 0)
            # # the following extra_error does not consider individual ks and enc error
            # extra_error = max(abs(ch_weight - math.ceil(ch_weight)), abs(ch_weight - math.floor(ch_weight)))

            max_abs_error = self._error + ks_extra_error + enc_extra_error
            max_abs_error = decimal.Decimal(max_abs_error).quantize(
                decimal.Decimal("1." + "0" * (decimal.getcontext().prec // 2)), rounding=decimal.ROUND_UP)
            if abs_error > max_abs_error:
                aux_ws = []
                kfwp, ecwp = self.ks_finder.weight_prefix, self.enc_finder.weight_prefix
                for v, c in solution_var2ct.items():
                    if (isinstance(kfwp, str) and str(v).startswith(kfwp)) or \
                            (isinstance(ecwp, str) and str(v).startswith(ecwp)):
                        aux_ws.append((v, c))
                raise ValueError(f"absolute error between integer weight {smt_weight} "
                                 f"(found by the SMT solver from {aux_ws}) and decimal weight {ch_weight} "
                                 f"(recomputed in CipherCharacteristic) is {abs_error}, "
                                 f"which is greater than maximum absolute error given by "
                                 f"ch_model-error={self._error} + extra_error={ks_extra_error+enc_extra_error}"
                                 f"\n{last_ch_found}")

        return last_ch_found

    def find_next_ch(self):
        """Return an iterator that yields the characteristics found in the SMT-based search.

        See also `ChFinder.find_next_ch`.

            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import CipherChModel
            >>> from cascada.smt.chsearch import CipherChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = CipherChModel(Speck32, BitValue)
            >>> input_output_vals = ch_model.enc_ch_model.input_val + ch_model.enc_ch_model.output_val
            >>> # setting each input and output plaintext word to 0x0001
            >>> v2c = {v: core.Constant(1, 16) for v in input_output_vals}
            >>> assert_type = ChModelAssertType.Validity
            >>> ch_finder = CipherChFinder(ch_model, assert_type, assert_type, "btor",
            ...                            var_prop2ct_prop=v2c, solver_seed=0)
            >>> found_enc_ch = next(ch_finder.find_next_ch()).enc_characteristic
            >>> round_sep = found_enc_ch.get_round_separators()
            >>> for round_enc_ch in found_enc_ch.split(round_sep): print(round_enc_ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
                input_val=[0x0001, 0x0001], output_val=[0x0004, 0x0000], external_vals=[0x0205],
                assign_outval_list=[0x0201, 0x0004, 0x0000])
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
                input_val=[0x0004, 0x0000], output_val=[0x0001, 0x0001], external_vals=[0x0801],
                assign_outval_list=[0x0800, 0x0001, 0x0001])

        """
        return super().find_next_ch()

    def fix_key_schedule_ch_weight(self, integer_weight):
        """Add the constraint that fixes the key-schedule weight to the given integer weight."""
        weight = core.Constant(integer_weight, self.ks_finder._ch_weight.width)
        self.initial_constraints.append(operation.BvComp(self.ks_finder._ch_weight, weight))

    def fix_encryption_ch_weight(self, integer_weight):
        """Add the constraint that fixes the encryption weight to the given integer weight."""
        weight = core.Constant(integer_weight, self.enc_finder._ch_weight.width)
        self.initial_constraints.append(operation.BvComp(self.enc_finder._ch_weight, weight))

    def _compute_empirical_ch_weight(self, ch_found, empirical_weight_options):
        if empirical_weight_options[0] is not None:
            ch_found.ks_characteristic.compute_empirical_ch_weight(**empirical_weight_options[0])
        if empirical_weight_options[1] is not None:
            ch_found.enc_characteristic.compute_empirical_ch_weight(**empirical_weight_options[1])

    def _get_empirical_ch_weight(self, ch_found):
        kew = ch_found.ks_characteristic.empirical_ch_weight
        eew = ch_found.enc_characteristic.empirical_ch_weight
        assert not(kew is None and eew is None)
        if kew is None:
            kew = 0
        if eew is None:
            eew = 0
        if math.inf in [kew, eew]:
            return math.inf
        else:
            return kew + eew

    def _get_ch_weight(self, ch_found, add_aux_probability=True):
        kat, eat = self.ks_finder.assert_type, self.enc_finder.assert_type
        CMAT = ChModelAssertType
        if kat == CMAT.ValidityAndWeight and eat == CMAT.ValidityAndWeight:
            return ch_found.ks_characteristic.ch_weight + ch_found.enc_characteristic.ch_weight
        elif kat in [CMAT.Validity, CMAT.ProbabilityOne] and eat == CMAT.ValidityAndWeight:
            # if ks-weight == 0, aux_probability = 1 - 1 = 0, and returned weight = enc-weight
            # if ks-weight > 0, aux_probability = > 0, and returned weight = enc-weight + epsilon
            if add_aux_probability:
                aux_probability = 1 - (decimal.Decimal(2)**(-ch_found.ks_characteristic.ch_weight))
                assert aux_probability >= 0, f"{aux_probability}"
            else:
                aux_probability = 0
            return ch_found.enc_characteristic.ch_weight + aux_probability
        elif kat == CMAT.ValidityAndWeight and eat in [CMAT.Validity, CMAT.ProbabilityOne]:
            if add_aux_probability:
                aux_probability = 1 - (decimal.Decimal(2)**(-ch_found.enc_characteristic.ch_weight))
                assert aux_probability >= 0, f"{aux_probability}"
            else:
                aux_probability = 0
            return ch_found.ks_characteristic.ch_weight + aux_probability
        else:
            raise ValueError(f"invalid assert types: {kat}, {eat}")

    def _new_final_weight(self, ch_found, prev_final_weight):
        new_ch_weight = self._get_ch_weight(ch_found, add_aux_probability=False)
        new_final_weight = int(math.ceil(new_ch_weight + self._error))
        return min(prev_final_weight, new_final_weight)

    def find_next_ch_increasing_weight(
            self, initial_weight, final_weight=None,
            ks_empirical_weight_options=None, enc_empirical_weight_options=None,
            stop_after_optimal=True, yield_weight=False,
    ):
        """Return an iterator that yields the characteristics found in the SMT-based search.

        This method searches for optimal characteristics (with optimal probability)
        using SMT solvers (see `ChFinder`).

        In particular, the search creates decision problems
        of whether there exists a characteristic with probability
        :math:`p \in I_w`, where a characteristic has probability
        :math:`p \in I_w` if and only if its integer weight
        (the integer part of the weight) is equal to :math:`w`.

        The main difference between `ChFinder.find_next_ch_increasing_weight`
        and this method is that this method defined the weight of a
        ``CipherCharacteristic`` depending on the type of the assertions.

        - If both ``ks_assert_type`` and ``enc_assert_type`` are
          `ValidityAndWeight`, then the weight of a
          ``CipherCharacteristic``
          is defined as the sum of the weight of the key-schedule
          characteristic and the weight of the encryption characteristic.
        - If only ``ks_assert_type`` is `ValidityAndWeight`,
          then the weight is defined as the weight of the key-schedule
          characteristic.
        - If only ``enc_assert_type`` is `ValidityAndWeight`,
          then the weight is defined as the weight of the encryption
          characteristic.

        .. note::

            For example, if only ``enc_assert_type`` is `ValidityAndWeight`,
            the search starts finding characteristics where
            the integer weight of the encryption characteristic is
            equal to the initial weight.

        The argument ``ks_empirical_weight_options``
        (resp. ``enc_empirical_weight_options``) specifies the options
        for the computation of the empirical weight over the key-schedule
        (resp. encryption) characteristic. If only one of them is given
        (and the other one is ``None``), then only one of the empirical
        weights is computed and used to determine whether to yield
        the characteristic found.

        See also `ChFinder.find_next_ch_increasing_weight`.

            >>> # example of search for RXDiff-CipherCharacteristic of Speck32
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.smt.chsearch import CipherChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = CipherChModel(Speck32, RXDiff)
            >>> assert_type = ChModelAssertType.ValidityAndWeight
            >>> ch_finder = CipherChFinder(ch_model, assert_type, assert_type, "btor", solver_seed=0)
            >>> iterator = ch_finder.find_next_ch_increasing_weight(1)
            >>> print(next(iterator).srepr())
            Ch(ks_ch=Ch(w=1.415, id=0000 0000, od=0000 0000), enc_ch=Ch(w=2.830, id=0000 0000, od=0000 0000))
            >>> print(next(iterator).srepr())
            Ch(ks_ch=Ch(w=1.415, id=0000 8000, od=8000 8002), enc_ch=Ch(w=2.830, id=0040 0000, od=8003 8003))

        """
        if ks_empirical_weight_options is None and enc_empirical_weight_options is None:
            empirical_weight_options = None
        else:
            empirical_weight_options = [ks_empirical_weight_options, enc_empirical_weight_options]
        return super().find_next_ch_increasing_weight\
            (initial_weight, final_weight=final_weight,
             empirical_weight_options=empirical_weight_options,
             stop_after_optimal=stop_after_optimal, yield_weight=yield_weight)

    def find_next_ch_increasing_weight_fixed_in_out(
            self, ks_input_prop, enc_input_prop, enc_output_prop, initial_weight, final_weight=None,
            ks_empirical_weight_options=None, enc_empirical_weight_options=None, use_empirical_weight=False,
    ):
        """Return an iterator that yields the characteristics found in the SMT-based search
        with increasing weight order and with fixed input and output properties.

        .. note::
          This method requires that both``ks_assert_type`` and ``enc_assert_type``
          are `ValidityAndWeight`.

        This method is similar to `ChFinder.find_next_ch_increasing_weight_fixed_in_out`,
        but internally `CipherChFinder.find_next_ch_increasing_weight` is used
        instead of `ChFinder.find_next_ch_increasing_weight`.

        In particular, this method finds cipher characteristics with the key-schedule
        input property, the encryption input property and the encryption output property
        equal to the properties given by ``ks_input_prop``, ``enc_input_prop`` and
        ``enc_output_prop`` (lists containing `Constant` objects or constant `Property` objects).

        Note that the cumulative weight is the weight (-log2) of the sum of the
        probabilities of all cipher characteristics found in the search
        (including the characteristic just found),
        where the probability of a cipher characteristic here considered is the
        product of the probabilities of the key-schedule and encryption
        characteristics.
        In other words, the cumulative weight is computed as in
        `ChFinder.find_next_ch_increasing_weight_fixed_in_out`, but
        the decimal weight of a cipher characteristic is taken
        as the sum of the decimal weights of the key-schedule
        and the encryption characteristics.

        .. note::

            For the `Difference` property types, the cumulative weight estimates
            the weight of the probability of the related-key differential
            with given masterkey difference, plaintext difference and
            ciphertext difference.

        ::

            >>> # example of search for RXDiff-CipherCharacteristic of Speck32 with fixed input/output difference
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.smt.chsearch import CipherChFinder, ChModelAssertType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = CipherChModel(Speck32, RXDiff)
            >>> assert_type = ChModelAssertType.ValidityAndWeight
            >>> ch_finder = CipherChFinder(ch_model, assert_type, assert_type, "btor", solver_seed=0)
            >>> best_ch = next(ch_finder.find_next_ch_increasing_weight(1))
            >>> best_ch.ks_characteristic.ch_weight + best_ch.enc_characteristic.ch_weight
            Decimal('4.244980417176049505485850841')
            >>> best_ch  # doctest: +NORMALIZE_WHITESPACE
            CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=1.414993472392016501828616947,
                assignment_weights=[1.414993472392016501828616947, 0, 0],
                input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000],
                assign_outdiff_list=[0x0000, 0x0000, 0x0000]),
            enc_characteristic=Characteristic(ch_weight=2.829986944784033003657233894,
                assignment_weights=[1.414993472392016501828616947, 1.414993472392016501828616947, 0, 0],
                input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000], external_diffs=[0x0000, 0x0000],
                assign_outdiff_list=[0x0000, 0x0000, 0x0000, 0x0000]))
            >>> iterator = ch_finder.find_next_ch_increasing_weight_fixed_in_out
            >>> # using the input and output differences of the best 2-round characteristic
            >>> ks_ip = best_ch.ks_characteristic.input_prop
            >>> enc_ip, enc_op = best_ch.enc_characteristic.input_prop, best_ch.enc_characteristic.output_prop
            >>> for w, ch in iterator(ks_ip, enc_ip, enc_op, 1): print(w, "|", ch)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            4.244980417176049505485850841 | CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=1.414993472392016501828616947,
                    assignment_weights=[1.414993472392016501828616947, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000],
                    assign_outdiff_list=[0x0000, 0x0000, 0x0000]),
                enc_characteristic=Characteristic(ch_weight=2.829986944784033003657233894,
                    assignment_weights=[1.414993472392016501828616947, 1.414993472392016501828616947, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000], external_diffs=[0x0000, 0x0000],
                    assign_outdiff_list=[0x0000, 0x0000, 0x0000, 0x0000]))
            3.244980417176049505485850840 | CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=1.414993472392016501828616947,
                    assignment_weights=[1.414993472392016501828616947, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0001],
                    assign_outdiff_list=[0x0001, 0x0000, 0x0001]),
                enc_characteristic=Characteristic(ch_weight=2.829986944784033003657233894,
                    assignment_weights=[1.414993472392016501828616947, 1.414993472392016501828616947, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000], external_diffs=[0x0000, 0x0001],
                    assign_outdiff_list=[0x0000, 0x0001, 0x0000, 0x0000]))
            ...
            3.192516141583136911276041810 | CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=17,
                    assignment_weights=[17, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x7fff],
                    assign_outdiff_list=[0x7fff, 0x0000, 0x7fff]),
                enc_characteristic=Characteristic(ch_weight=18.41499347239201650182861695,
                    assignment_weights=[1.414993472392016501828616947, 17, 0, 0],
                    input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000], external_diffs=[0x0000, 0x7fff],
                    assign_outdiff_list=[0x0000, 0x7fff, 0x0000, 0x0000]))
            >>> print(w)  # weight (sum of the key and encryption weights) of the related-key differential
            3.192516141583136911276041810

        See also `ChFinder.find_next_ch_increasing_weight_fixed_in_out`.
        """
        assert self.ks_finder.assert_type == ChModelAssertType.ValidityAndWeight
        assert self.enc_finder.assert_type == ChModelAssertType.ValidityAndWeight
        if use_empirical_weight:
            assert ks_empirical_weight_options is not None and enc_empirical_weight_options is not None

        old_var_prop2ct_prop = self._var_prop2ct_prop.copy()
        old_initial_constraints = self.initial_constraints[:]

        for var_prop, ct_prop in itertools.chain(
                zip(self.ch_model.ks_ch_model.input_prop, ks_input_prop),
                zip(self.ch_model.enc_ch_model.input_prop, enc_input_prop),
                zip(self.ch_model.enc_ch_model.output_prop, enc_output_prop),
        ):
            if not isinstance(ct_prop, self.ch_model.prop_type):
                assert isinstance(ct_prop, core.Constant)
                ct_prop = self.ch_model.prop_type(ct_prop)
            self._var_prop2ct_prop[var_prop] = ct_prop
            self.initial_constraints.append(operation.BvComp(var_prop.val, ct_prop.val))

        for aux_finder in [self.ks_finder, self.enc_finder]:
            ChFinder._check_initial_constraints(
                aux_finder.ch_model, self.initial_constraints, self.chmodel_asserts,
                aux_finder._exclude_zero_input_prop, self._var_prop2ct_prop,
                self._vars_in_constraints, aux_finder._raise_exception_missing_var
            )

        cumulative_w = None

        for yielded_weight, ch_found in self.find_next_ch_increasing_weight(
            initial_weight, final_weight=final_weight,
            ks_empirical_weight_options=ks_empirical_weight_options,
            enc_empirical_weight_options=enc_empirical_weight_options,
            stop_after_optimal=False, yield_weight=True
        ):
            if yielded_weight is None:  # second time optimal ch is yielded
                assert cumulative_w is not None
                continue

            if use_empirical_weight:
                next_weight = ch_found.ks_characteristic.empirical_ch_weight + \
                              ch_found.enc_characteristic.empirical_ch_weight
            else:
                next_weight = ch_found.ks_characteristic.ch_weight + \
                                 ch_found.enc_characteristic.ch_weight

            if cumulative_w is None:
                cumulative_w = next_weight
            else:
                assert cumulative_w is not None
                cumulative_w = _merge_weights(next_weight, cumulative_w)

            yield cumulative_w, ch_found

        self._var_prop2ct_prop = old_var_prop2ct_prop
        self.initial_constraints = old_initial_constraints


def round_based_ch_search(
        func, initial_num_rounds, final_num_rounds, prop_type, assert_type, solver_name,
        extra_chmodel_args=None,  # op_model_class2options,
        extra_chfinder_args=None,  # exclude, ic, v2c, pm, fn, env, weight_prefix, solver_seed
        extra_findnextchweight_args=None,  # initial_weight, final_weight, ewo
        **kwargs  # find_cipher_ch
        ):
    """Search for characteristics of round-based functions over multiple number of rounds.

    This function searches for characteristics of ``func`` (a `RoundBasedFunction`)
    by modelling the search as a sequence of SMT problems (using `ChFinder`),
    but the search is perfomed iteratively over the number of rounds of ``func``.
    That is, first characteristics covering ``initial_num_rounds`` rounds
    are searched, then ``initial_num_rounds + 1``, until ``final_num_rounds``.

    This function proceed as follows:

    1. Set the number of rounds of ``func`` to ``initial_num_rounds``.
    2. Create a `abstractproperty.chmodel.ChModel`
       (resp. `abstractproperty.chmodel.EncryptionChModel`) object using as
       arguments ``func``, ``prop_type`` and ``extra_chmodel_args``.
    3. Create a `ChFinder` object with arguments the characteristic model
       created in step 2, ``assert_type``, ``solver_name`` and
       ``extra_chfinder_args``.
    4. Loop over the generator `ChFinder.find_next_ch` or
       `ChFinder.find_next_ch_increasing_weight` (depending on
       ``assert_type``), and yield all characteristics from the
       generator (together with the current number of rounds).
    5. After the generator has been exhausted, the search is finished if
       the current number of rounds is ``final_num_rounds``.
       Otherwise, increase the number of rounds by one, set ``func``
       to this number of rounds, and go to step 2.

    In particular, this function is a Python `generator` function
    (see `ChFinder`), returning an `iterator` that yields
    tuples containing the current number of rounds and the last
    characteristic (an `abstractproperty.characteristic.Characteristic`
    if ``func`` is a `RoundBasedFunction` object, or
    a `abstractproperty.characteristic.EncryptionCharacteristic` if
    ``func`` is a `RoundBasedFunction`-encryption function
    of a `Cipher`) found in the search.

    The argument ``prop_type`` is a particular `Property` such as `XorDiff`
    or `LinearMask`. For ``assert_type`` and ``solver_name``, see `ChFinder`.
    The optional arguments ``extra_chmodel_args``, ``extra_chfinder_args``
    and ``extra_findnextchweight_args`` can be given as dictionaries
    (in the form of ``**kwargs``) containing additional arguments
    for ``ChModel/EncryptionChModel``, `ChFinder` and
    `ChFinder.find_next_ch_increasing_weight` calls respectively.

    It is possible to abort the current search for the current number of rounds
    and start the search with one more round by passing_ the
    value `INCREMENT_NUM_ROUNDS`
    to the generator iterator with `generator.send`.

    .. _passing: https://docs.python.org/3/howto/functional.html?highlight=generator#passing-values-into-a-generator

    .. note::

        In other words, step 4 in the previous description can be early
        aborted as follows

        .. code:: python

            [...]
            iterator = round_based_ch_search(...)
            n1, ch_found = next(iterator)
            iterator.send(INCREMENT_NUM_ROUNDS)
            n2, ch_found = next(iterator)  # n2 > n1
            [...]

        Although `generator.send` yields a new value of the iteration,
        this function does not yield anything meaningful in the ``send`` calls.

    The function `round_based_ch_search` is mostly meant to be used with
    `ChModelAssertType.ValidityAndWeight`, as the minimum weight obtain
    in one round is used as the initial weight for the next round.
    In other words, if all the characteristics covering :math:`r` number
    of rounds were found with SMT problems for integer weights
    (see `ChFinder.find_next_ch_increasing_weight`)
    larger than :math:`w`, then :math:`w` is set as the initial weight
    for the search for characteristics covering :math:`r+1` rounds.

    .. note::

        All SMT problems modelling a characteristic covering
        :math:`r+1` rounds for integer weights less or equal
        than :math:`r - 1` are unsatisfiable due to our definition
        of characteristic weight (see
        `abstractproperty.characteristic.Characteristic` and
        `differential.characteristic.Characteristic` or
        `linear.characteristic.Characteristic` for some examples)

    ::

        >>> # example of searching for XorDiff Characteristic over a BvFunction
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.smt.chsearch import ChModelAssertType, round_based_ch_search, INCREMENT_NUM_ROUNDS
        >>> from cascada.primitives import speck
        >>> Speck32_ks = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> assert_type = ChModelAssertType.ValidityAndWeight
        >>> iterator = round_based_ch_search(Speck32_ks, 2, 5, XorDiff, assert_type, "btor",
        ...     extra_chfinder_args={"exclude_zero_input_prop": True, "solver_seed":0},
        ...     extra_findnextchweight_args={"initial_weight": 0})
        >>> for (num_rounds, ch) in iterator:
        ...     print(num_rounds, ":", ch.srepr())
        2 : Ch(w=0, id=0040 0000 0000, od=0000 0000 8000)
        3 : Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000)
        4 : Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000 8002)
        5 : Ch(w=1, id=0040 0000 0000 0000, od=0000 0000 0000 8000 8002 8008)
        >>> # example of searching for LinearMask EncryptionCharacteristic over a Cipher
        >>> from cascada.linear.mask import LinearMask
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> assert_type = ChModelAssertType.ProbabilityOne
        >>> iterator = round_based_ch_search(Speck32, 2, 3, LinearMask, assert_type, "btor",
        ...     extra_chfinder_args={"solver_seed":0})
        >>> num_rounds, ch = next(iterator)
        >>> print(num_rounds, ":", ch.srepr())
        2 : Ch(w=0, id=0000 0000, od=0000 0000)
        >>> num_rounds, ch = next(iterator)
        >>> print(num_rounds, ":", ch.srepr())
        2 : Ch(w=0, id=0080 4021, od=0201 0200)
        >>> iterator.send(INCREMENT_NUM_ROUNDS)  # stop current num_rounds and increment by 1
        >>> num_rounds, ch = next(iterator)
        >>> print(num_rounds, ":", ch.srepr())
        3 : Ch(w=0, id=0000 0000, od=0000 0000)

    """
    if not (issubclass(func, cascada_ssa.RoundBasedFunction) or
            (issubclass(func, blockcipher.Cipher) and
             issubclass(func.encryption, cascada_ssa.RoundBasedFunction))):
        raise ValueError(f"{func} is not a RoundBasedFunction or a Cipher")

    assert isinstance(initial_num_rounds, int) and isinstance(final_num_rounds, int)
    assert 0 < initial_num_rounds <= final_num_rounds

    if extra_chmodel_args is None:
        extra_chmodel_args = {}
    else:
        extra_chmodel_args = extra_chmodel_args.copy()
    if extra_chfinder_args is None:
        extra_chfinder_args = {}
    else:
        extra_chfinder_args = extra_chfinder_args.copy()
    if extra_findnextchweight_args is None:
        extra_findnextchweight_args = {}
    else:
        extra_findnextchweight_args = extra_findnextchweight_args.copy()

    printing_mode = extra_chfinder_args.get("printing_mode", PrintingMode.Silent)
    filename = extra_chfinder_args.get("filename", None)
    smart_print = _get_smart_print(filename)

    find_cipher_ch = kwargs.pop("find_cipher_ch", False)

    if kwargs:
        raise ValueError(f"invalid arguments: {kwargs}")

    if find_cipher_ch:
        list_assert_type = assert_type
    else:
        list_assert_type = [assert_type]

    if ChModelAssertType.ValidityAndWeight in list_assert_type:
        if "yield_weight" in extra_findnextchweight_args:
            raise ValueError("extra_findnextchweight_args cannot contain yield_weight")
        extra_findnextchweight_args["yield_weight"] = True
        if "initial_weight" not in extra_findnextchweight_args:
            extra_findnextchweight_args["initial_weight"] = 0
            warnings.warn("setting extra_findnextchweight_args['initial_weight'] to 0")
        # warning already raised in find_next_ch_increasing_weight if exclude zero input + non-zero weight

    from cascada.differential.difference import XorDiff, RXDiff
    from cascada.linear.mask import LinearMask
    from cascada.algebraic.value import BitValue, WordValue

    if prop_type in [XorDiff, RXDiff]:
        from cascada.differential.chmodel import ChModel, EncryptionChModel, CipherChModel
    elif prop_type == LinearMask:
        assert find_cipher_ch is False
        from cascada.linear.chmodel import ChModel, EncryptionChModel
    elif prop_type in [BitValue, WordValue]:
        from cascada.algebraic.chmodel import ChModel, EncryptionChModel, CipherChModel
    else:
        raise ValueError(f"prop_type not in {[XorDiff, RXDiff, LinearMask, BitValue, WordValue]}")

    num_rounds = initial_num_rounds
    while True:
        func.set_num_rounds(num_rounds)

        if printing_mode != PrintingMode.Silent:
            if num_rounds != initial_num_rounds:
                smart_print("")
            smart_print(f"Current number of rounds: {num_rounds}", prepend_time=True)

        if issubclass(func, blockcipher.Cipher):
            if find_cipher_ch:
                ch_model = CipherChModel(func, prop_type, **extra_chmodel_args)
            else:
                ch_model = EncryptionChModel(func, prop_type, **extra_chmodel_args)
        else:
            prefix = EncryptionChModel._prefix
            input_prop_names = [f"{prefix}p{i}" for i in range(len(func.input_widths))]
            ch_model = ChModel(func, prop_type, input_prop_names, **extra_chmodel_args)

        if printing_mode == PrintingMode.Debug:
            smart_print(f"Characteristic model: {ch_model}")

        if find_cipher_ch:
            ch_finder = CipherChFinder(ch_model, assert_type[0], assert_type[1], solver_name, **extra_chfinder_args)
        else:
            ch_finder = ChFinder(ch_model, assert_type, solver_name, **extra_chfinder_args)

        if printing_mode == PrintingMode.Debug:
            smart_print("Size of the base SMT problem:", ch_finder.formula_size())
            smart_print(f"Base SMT problem:\n{ch_finder.hrepr()}")

        found_ch = False

        if ChModelAssertType.ValidityAndWeight in list_assert_type:
            min_target_weight = math.inf
            for target_weight, ch in ch_finder.find_next_ch_increasing_weight(**extra_findnextchweight_args):
                found_ch = True
                min_target_weight = min(min_target_weight, target_weight)
                sent_value = (yield (num_rounds, ch))
                if sent_value is not None:
                    if sent_value == INCREMENT_NUM_ROUNDS:
                        yield None
                        break
                    else:
                        warnings.warn(f"value {sent_value} is sent to the generator "
                                      f"but only sending INCREMENT_NUM_ROUNDS"
                                      f" affects the generator")
            extra_findnextchweight_args["initial_weight"] = min_target_weight
        else:
            for ch in ch_finder.find_next_ch():
                found_ch = True
                sent_value = (yield (num_rounds, ch))
                if sent_value is not None:
                    if sent_value == INCREMENT_NUM_ROUNDS:
                        yield None
                        break
                    else:
                        warnings.warn(f"value {sent_value} is sent to the generator "
                                      f"but only sending INCREMENT_NUM_ROUNDS"
                                      f" affects the generator")

        if not found_ch:
            if printing_mode == PrintingMode.Debug:
                smart_print("No characteristic found", prepend_time=True)
            break
        elif num_rounds == final_num_rounds:
            break
        else:
            num_rounds += 1
            assert num_rounds <= final_num_rounds


def round_based_cipher_ch_search(
        cipher, initial_num_rounds, final_num_rounds, prop_type, ks_assert_type, enc_assert_type, solver_name,
        extra_cipherchmodel_args=None, extra_cipherchfinder_args=None, extra_findnextchweight_args=None,
        ):
    """Search for characteristics of iterated ciphers over multiple number of rounds.

    This function is similar to `round_based_cipher_ch_search` but searching for
    `abstractproperty.characteristic.CipherCharacteristic` instead of
    `abstractproperty.characteristic.Characteristic`.

    In particular, this function creates an `abstractproperty.chmodel.CipherChModel`
    and an `CipherChFinder` objects instead of an `abstractproperty.chmodel.ChModel`
    and an `ChFinder` objects.

        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.smt.chsearch import ChModelAssertType, round_based_cipher_ch_search
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> at = ChModelAssertType.ValidityAndWeight
        >>> iterator = round_based_cipher_ch_search(Speck32, 2, 5, XorDiff, at, at, "btor",
        ...     extra_cipherchfinder_args={"ks_exclude_zero_input_prop": True, "solver_seed": 0},
        ...     extra_findnextchweight_args={"initial_weight": 0})
        >>> for (num_rounds, ch) in iterator:
        ...     print(num_rounds, ":", ch.srepr())  # doctest: +NORMALIZE_WHITESPACE
        2 : Ch(ks_ch=Ch(w=0, id=0040 0000, od=0000 8000),
              enc_ch=Ch(w=0, id=0000 0000, od=8000 8000))
        3 : Ch(ks_ch=Ch(w=0, id=0040 0000 0000, od=0000 0000 8000),
              enc_ch=Ch(w=0, id=0000 0000, od=8000 8000))
        4 : Ch(ks_ch=Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000),
              enc_ch=Ch(w=0, id=0000 0000, od=8000 8000))
        5 : Ch(ks_ch=Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000 8002),
              enc_ch=Ch(w=1, id=0000 0000, od=0102 0100))

    """
    return round_based_ch_search(
        cipher, initial_num_rounds, final_num_rounds, prop_type, [ks_assert_type, enc_assert_type], solver_name,
        extra_chmodel_args=extra_cipherchmodel_args,
        extra_chfinder_args=extra_cipherchfinder_args,
        extra_findnextchweight_args=extra_findnextchweight_args,
        find_cipher_ch=True
    )
