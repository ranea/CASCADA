"""Wrap bit-vector functions for characteristics with one non-trivial transition."""
import collections
import functools
import hashlib
import warnings

from cascada import abstractproperty
from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import context
from cascada.bitvector import ssa as cascada_ssa
from cascada.primitives import blockcipher
from cascada.smt import chsearch


zip = functools.partial(zip, strict=True)


class ChFuncAsOp(operation.SecondaryOperation):
    """Represent bit-vector operations wrapping functions of characteristic models.

    Given a characteristic model
    (i.e., `abstractproperty.chmodel.ChModel`),
    a `ChFuncAsOp` operation is a `SecondaryOperation` that wraps
    the underlying bit-vector function of the characteristic model.

    The operands of the wrapping operation are the
    input operands of the bit-vector function and its external variables.
    The ``eval`` method of wrapping operation calls the ``eval`` method
    of the bit-vector function and concatenates (with `PropConcat`)
    the outputs of the bit-vector function into a single bit-vector.
    This single bit-vector is the output of the wrapping operation.

    This class is not meant to be instantiated but to provide a base
    class to define secondary operators wrapping bit-vector functions
    generated through `make_chfunc_as_op`.

    Attributes:
        ch_model: the characteristic model of the bit-vector function.

    """
    arity = None
    is_symmetric = False
    is_simple = False

    ch_model = None  # needed for vrepr

    def vrepr(self):
        ch_model = self.__class__.ch_model
        return f"{make_chfunc_as_op.__name__}({ch_model.vrepr()})"

    @classmethod
    def condition(cls, *args):
        ext_widths = [v.width for v in cls.ch_model.external_var2prop]
        widths = cls.ch_model.func.input_widths + ext_widths
        return len(args) == len(widths) and all(x.width == w for x, w in zip(args, widths))

    @classmethod
    def output_width(cls, *args):
        return sum(cls.ch_model.func.output_widths)

    @classmethod
    def eval(cls, *args):
        # avoid __new__ to avoid round outputs
        # ssa instead of _func due to the external variables
        func_ssa = cls.ch_model.ssa
        ne = len(func_ssa.external_vars)
        if ne == 0:
            # ·[:-0] returns []
            input_args, ext_args = args, []
        else:
            input_args, ext_args = args[:-ne], args[-ne:]
        external_var2val = {v: ext_args[i] for i, v in enumerate(func_ssa.external_vars)}
        out_func = func_ssa.eval(input_args, external_var2val=external_var2val, symbolic_inputs=True)
        return functools.reduce(abstractproperty.property.PropConcat, reversed(out_func))


def make_chfunc_as_op(ch_model):
    """Return the `ChFuncAsOp` of the given characteristic model.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.smt.wrappedchmodel import make_chfunc_as_op
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> plaintext = [Variable("p0", 16), Variable("p1", 16)]
        >>> round_keys = [Variable("k0", 16), Variable("k1", 16)]
        >>> Speck32.encryption.round_keys = round_keys
        >>> Speck32.encryption(*plaintext, symbolic_inputs=True)  # doctest: +NORMALIZE_WHITESPACE
        ((((((p0 >>> 7) + p1) ^ k0) >>> 7) + ((p1 <<< 2) ^ ((p0 >>> 7) + p1) ^ k0)) ^ k1,
        (((p1 <<< 2) ^ ((p0 >>> 7) + p1) ^ k0) <<< 2) ^ (((((p0 >>> 7) + p1) ^ k0) >>> 7) + ((p1 <<< 2) ^ ((p0 >>> 7) + p1) ^ k0)) ^ k1)
        >>> SpeckEncAsOp = make_chfunc_as_op(EncryptionChModel(Speck32, XorDiff))
        >>> SpeckEncAsOp(*(plaintext + round_keys))
        SpeckEncryption_2RAsOp(p0, p1, k0, k1)
        >>> SpeckEncAsOp(*(plaintext + round_keys)).doit()  # doctest: +NORMALIZE_WHITESPACE
        Id((p1 <<< 2 ^ (p0 >>> 7 + p1) ^ k0) <<< 2 ^ (((p0 >>> 7 + p1) ^ k0) >>> 7 + (p1 <<< 2 ^ (p0 >>> 7 + p1) ^ k0)) ^ k1) ::
        Id((((p0 >>> 7 + p1) ^ k0) >>> 7 + (p1 <<< 2 ^ (p0 >>> 7 + p1) ^ k0)) ^ k1)

    """
    assert isinstance(ch_model, abstractproperty.chmodel.ChModel)

    input_widths = ch_model.func.input_widths
    external_widths = [v.width for v in ch_model.external_var2prop]
    _ch_model = ch_model

    class MyChFuncAsOp(ChFuncAsOp):
        arity = [len(input_widths) + len(external_widths), 0]
        ch_model = _ch_model

    MyChFuncAsOp.__name__ = f"{ch_model.func.get_name()}AsOp"

    return MyChFuncAsOp


class ModelChFuncAsOp(object):
    """Represent models of `ChFuncAsOp` operations.

    The model of a `ChFuncAsOp` operation is the set of bit-vector constraints
    that models the propagation probability over the `ChFuncAsOp` operation
    (see `abstractproperty.opmodel.OpModel`).

    .. note::
        The model of a `ChFuncAsOp` operation models the probability
        that the input property :math:`\\alpha`
        propagates to the output property :math:`\\beta`
        over the `ChFuncAsOp` operation, and ignores the properties
        of the intermediate values of the `ChFuncAsOp` operation.

        For the `Difference`/`LinearMask` property types,
        the model of a `ChFuncAsOp` models the probability of the
        differential/hull (instead of the probability of the characteristic).

    The constraints of the model of a `ChFuncAsOp` operation are obtained
    from the underlying characteristic model.
    Specifically, the methods
    `abstractproperty.opmodel.OpModel.validity_constraint`,
    `abstractproperty.opmodel.OpModel.pr_one_constraint`
    and `abstractproperty.opmodel.OpModel.weight_constraint`
    of `ModelChFuncAsOp` used the methods
    `abstractproperty.chmodel.ChModel.validity_assertions`,
    `abstractproperty.chmodel.ChModel.pr_one_assertions` and
    `abstractproperty.chmodel.ChModel.weight_assertions`
    of the characteristic model respectively.
    Note that the external variables of ``validity_constraint``,
    ``pr_one_constraint`` and ``weight_constraint`` include
    the intermediate properties in the characteristic model
    (the external variables of ``weight_constraint`` also include
    the intermediate weight variables).

    The weight returned by the `abstractproperty.opmodel.OpModel.decimal_weight`
    method of `ModelChFuncAsOp` is computed by internally calling
    `ChFinder.find_next_ch` using the underlying characteristic model.
    In other words, the returned weight is a very weak approximation (with
    big `abstractproperty.opmodel.OpModel.error`) of the actual weight,
    since the returned weight is the weight of a random characteristic
    satisfying the underlying characteristic model.

    This class is not meant to be instantiated but to provide a base
    class for models generated through `make_chfuncasop_model`.

    .. Implementation details:

        This class does not subclass `abstractproperty.opmodel.OpModel`,
        but subclasses of this class defined for a particular `Property`
        must subclass the corresponding ``OpModel``.

    """
    @property
    def ch_model(self):
        """The underlying characteristic model of `ChFuncAsOp`."""
        return self.op.ch_model

    def _get_full_input_vars_xreplace_dict(self):
        self_full_input_vars = [p.val for p in self.input_prop]
        ch_full_input_vars = [p.val for p in self.ch_model.input_prop] + list(self.ch_model.external_var2prop)
        assert len(self_full_input_vars) == len(ch_full_input_vars)
        my_dict = collections.OrderedDict()
        for ch_var, self_var in zip(ch_full_input_vars, self_full_input_vars):
            my_dict[ch_var] = self_var
        return my_dict

    @property
    def _validity_constraint(self):
        if not hasattr(self, "_stored_validity_constraint") or \
                self._stored_validity_constraint[1] != self.input_prop:
            with context.Simplification(False), context.Cache(False):
                d = self._get_full_input_vars_xreplace_dict()
                self._stored_validity_constraint = (
                    functools.reduce(operation.BvAnd, (c.xreplace(d) for c in self.ch_model.validity_assertions())),
                    self.input_prop
                )
        return self._stored_validity_constraint[0]

    @property
    def _pr_one_constraint(self):
        if not hasattr(self, "_stored_pr_one_constraint") or \
                self._stored_pr_one_constraint[1] != self.input_prop:
            with context.Simplification(False), context.Cache(False):
                d = self._get_full_input_vars_xreplace_dict()
                self._stored_pr_one_constraint = (
                    functools.reduce(operation.BvAnd, (c.xreplace(d) for c in self.ch_model.pr_one_assertions())),
                    self.input_prop
                )
        return self._stored_pr_one_constraint[0]

    def _get_out_constraint(self, output_prop):
        assert output_prop.val.width == sum(p.val.width for p in self.ch_model.output_prop)
        cm_out_val = reversed([p.val for p in self.ch_model.output_prop])
        cm_out_val = functools.reduce(operation.Concat, cm_out_val)
        return operation.BvComp(output_prop.val, cm_out_val)

    def validity_constraint(self, output_prop):
        return self._get_out_constraint(output_prop) & self._validity_constraint

    def pr_one_constraint(self, output_prop):
        return self._get_out_constraint(output_prop) & self._pr_one_constraint

    def external_vars_validity_constraint(self, output_prop):
        external_vars = self._validity_constraint.atoms(core.Variable)
        # remove self_full_input_prop
        external_vars.difference_update(self._get_full_input_vars_xreplace_dict().values())
        for p in self.ch_model.output_prop:
            external_vars.add(p.val)
        assert all(v not in external_vars for v in self._get_full_input_vars_xreplace_dict())
        return sorted(external_vars, key=str)

    def external_vars_pr_one_constraint(self, output_prop):
        external_vars = self._pr_one_constraint.atoms(core.Variable)
        external_vars.difference_update(self._get_full_input_vars_xreplace_dict().values())
        for p in self.ch_model.output_prop:
            external_vars.add(p.val)
        assert all(v not in external_vars for v in self._get_full_input_vars_xreplace_dict())
        return sorted(external_vars, key=str)

    def _get_weight_variable_assign_weight_variables(self):
        cm = self.ch_model
        h = hashlib.blake2b(digest_size=16)
        data = cm.vrepr() + self.vrepr()
        h.update(data.encode())
        prefix = "_w_tmp" + h.hexdigest()
        weight_variable = core.Variable(f"{prefix}", cm.weight_width())
        assign_weight_variables = []
        for i, om in enumerate(cm.assign_outprop2op_model.values()):
            assign_weight_variables.append(core.Variable(f"{prefix}_{i}", om.weight_width()))
        return weight_variable, assign_weight_variables

    @property
    def _weight_constraint(self):
        if not hasattr(self, "_stored_weight_constraint") or \
                self._stored_weight_constraint[1] != self.input_prop:
            wv, awvs = self._get_weight_variable_assign_weight_variables()
            with context.Simplification(False), context.Cache(False):
                d = self._get_full_input_vars_xreplace_dict()
                wa = self.ch_model.weight_assertions(wv, awvs, truncate=True)
                self._stored_weight_constraint = (
                    functools.reduce(operation.BvAnd, (c.xreplace(d) for c in wa)),
                    self.input_prop,
                    wv,
                )
        return self._stored_weight_constraint

    def weight_constraint(self, output_prop, weight_variable):
        return self._get_out_constraint(output_prop) & \
               operation.BvComp(weight_variable, self._weight_constraint[-1]) & \
               self._weight_constraint[0]

    def external_vars_weight_constraint(self, output_prop, weight_variable):
        external_vars = self._weight_constraint[0].atoms(core.Variable)  # includes wa and awvs
        external_vars.difference_update(self._get_full_input_vars_xreplace_dict().values())
        for p in self.ch_model.output_prop:
            external_vars.add(p.val)
        assert all(v not in external_vars for v in self._get_full_input_vars_xreplace_dict())
        return sorted(external_vars, key=str)

    def max_weight(self):
        return self.ch_model.max_weight(truncate=True)

    def weight_width(self):
        return self.ch_model.weight_width(truncate=True)

    def decimal_weight(self, output_prop, **extra_chfinder_args):
        if not hasattr(self, "_inout_prop2decimal_weight"):
            self._inout_prop2decimal_weight = {}
        if (self.input_prop, output_prop) in self._inout_prop2decimal_weight:
            return self._inout_prop2decimal_weight[(self.input_prop, output_prop)]

        self._assert_preconditions_decimal_weight(output_prop)

        from pysmt import environment, logics

        if extra_chfinder_args is None:
            extra_chfinder_args = {}
        else:
            assert "assert_type" not in extra_chfinder_args
            assert "initial_constraints" not in extra_chfinder_args
        if "solver_name" in extra_chfinder_args:
            solver_name = extra_chfinder_args["solver_name"]
            del extra_chfinder_args["solver_name"]
        else:
            list_solver_name = list(environment.get_env().factory.all_solvers(logics.QF_BV))
            if "btor" in list_solver_name:
                solver_name = "btor"
            else:
                solver_name = list_solver_name[0]

        # ch_finder needs _get_full_input_vars_xreplace_dict because
        # ch_finder uses self.ch_model.validity_assertions and not self._validity_constraint
        # (however some vars from _get_full_input_vars_xreplace_dict might not appear
        # in validity_assertions like round keys in some cases)

        var_prop2ct_prop = collections.OrderedDict()
        ext_vars = set()
        for constraint in self.ch_model.validity_assertions():
            ext_vars |= constraint.atoms(core.Variable)
        for ch_var, self_var in self._get_full_input_vars_xreplace_dict().items():
            if ch_var in ext_vars:
                if isinstance(self_var, core.Variable):
                    assert self_var in ext_vars
                var_prop2ct_prop[self.ch_model.prop_type(ch_var)] = self.ch_model.prop_type(self_var)

        # # debugging
        # print("\nch_model:", self.ch_model)
        # print("ch_model ssa:", self.ch_model.ssa)
        # print("input_prop:", self.input_prop)
        # print("output_prop:", output_prop)
        # print("initial_constraints (_get_out_constraint):", self._get_out_constraint(output_prop))
        # print("var_prop2ct_prop:", var_prop2ct_prop)
        # print("-----")
        # print("validity_constraint:", self.validity_constraint(output_prop))
        # print("external_vars_validity_constraint:", self.external_vars_validity_constraint(output_prop))
        # print("-----")
        # print("ch_model validity_assertions:", self.ch_model.validity_assertions())
        # print("ch_model external_vars_validity_assertions:", self.ch_model.external_vars_validity_assertions(), "\n")
        # #

        ch_finder = chsearch.ChFinder(
            self.ch_model,
            assert_type=chsearch.ChModelAssertType.Validity,
            solver_name=solver_name,
            initial_constraints=[self._get_out_constraint(output_prop)],
            var_prop2ct_prop=var_prop2ct_prop,
            **extra_chfinder_args
        )
        for ch_found in ch_finder.find_next_ch():
            # # debugging
            # print("ch_found:", ch_found)
            #
            dw = ch_found.ch_weight
            break
        else:
            raise abstractproperty.opmodel.InvalidOpModelError(
                f"{self}.decimal_weight({output_prop}) cannot be obtained "
                f"since {output_prop} <- {self} is not valid")

        self._assert_postconditions_decimal_weight(output_prop, dw)

        self._inout_prop2decimal_weight[(self.input_prop, output_prop)] = dw
        return dw

    def num_frac_bits(self):
        return 0

    def error(self):
        return self.max_weight() + self.ch_model.error()


def make_chfuncasop_model(my_chfunc_as_op):
    """Return the `ModelChFuncAsOp` of the given `ChFuncAsOp`.

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.smt.wrappedchmodel import make_chfunc_as_op, make_chfuncasop_model
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> SpeckEncAsOp = make_chfunc_as_op(EncryptionChModel(Speck32, XorDiff))
        >>> ModelSpeckEncAsOp = make_chfuncasop_model(SpeckEncAsOp)
        >>> input_diff = [XorDiff(Constant(0, 16))]*4
        >>> f = ModelSpeckEncAsOp(input_diff)
        >>> print(f.vrepr())  # doctest: +NORMALIZE_WHITESPACE
        XorModelSpeckEncryption_2RAsOp([XorDiff(Constant(0x0000, width=16)), XorDiff(Constant(0x0000, width=16)),
            XorDiff(Constant(0x0000, width=16)), XorDiff(Constant(0x0000, width=16))])
        >>> f.validity_constraint(XorDiff(Constant(0, 32)))  # doctest: +NORMALIZE_WHITESPACE
        (0x00000000 == (dx9_out :: dx7_out)) &
        ((~(dx1 << 0x0001) & dx1) == 0x0000) &
        (((~((dx1 >>> 7) << 0x0001) ^ (dx1 << 0x0001)) & (~((dx1 >>> 7) << 0x0001) ^ (dx6 << 0x0001)) &
            ((dx1 >>> 7) ^ dx1 ^ dx6 ^ ((dx1 >>> 7) << 0x0001))) == 0x0000) &
        (dx6 == dx7_out) & (((dx1 <<< 2) ^ dx6) == dx9_out)
        >>> f.external_vars_validity_constraint(XorDiff(Constant(0, 32)))
        [dx1, dx6, dx7_out, dx9_out]
        >>> w = Variable("w", f.weight_width())
        >>> f.weight_constraint(XorDiff(Constant(0, 32)), w)  # doctest: +NORMALIZE_WHITESPACE
        (0x00000000 == (dx9_out :: dx7_out)) &
        (w == _w_tmp74c4f05762c354cc149091e363577700) &
        (_w_tmp74c4f05762c354cc149091e363577700_0 == PopCount(dx1[14:])) &
        (_w_tmp74c4f05762c354cc149091e363577700_1 == PopCount(~((~(dx1 >>> 7) ^ dx1) & (~(dx1 >>> 7) ^ dx6))[14:])) &
        (_w_tmp74c4f05762c354cc149091e363577700_2 == 0b0) &
        (_w_tmp74c4f05762c354cc149091e363577700_3 == 0b0) &
        (_w_tmp74c4f05762c354cc149091e363577700 == ((0b0 :: _w_tmp74c4f05762c354cc149091e363577700_0) +
            (0b0 :: _w_tmp74c4f05762c354cc149091e363577700_1)))
        >>> f.external_vars_weight_constraint(XorDiff(Constant(0, 32)), w)  # doctest: +NORMALIZE_WHITESPACE
        [_w_tmp74c4f05762c354cc149091e363577700, _w_tmp74c4f05762c354cc149091e363577700_0,
        _w_tmp74c4f05762c354cc149091e363577700_1, _w_tmp74c4f05762c354cc149091e363577700_2,
        _w_tmp74c4f05762c354cc149091e363577700_3, dx1, dx6, dx7_out, dx9_out]
        >>> f.max_weight(), f.weight_width(), f.error(), f.num_frac_bits()
        (30, 5, 30, 0)

    """
    assert issubclass(my_chfunc_as_op, ChFuncAsOp)

    from cascada.differential.difference import XorDiff, RXDiff
    from cascada.linear.mask import LinearMask
    from cascada.algebraic.value import WordValue, BitValue

    ch_model = my_chfunc_as_op.ch_model

    if ch_model.prop_type in [XorDiff, RXDiff]:
        from cascada.differential.opmodel import OpModel
    elif ch_model.prop_type == LinearMask:
        from cascada.linear.opmodel import OpModel
    elif ch_model.prop_type in [WordValue, BitValue]:
        from cascada.algebraic.opmodel import OpModel
    else:
        raise ValueError(f"property {ch_model.prop_type} not supported")

    class MyModelChFuncAsOp(ModelChFuncAsOp, OpModel):
        op = my_chfunc_as_op

    # if ch_model.prop_type in [XorDiff, RXDiff]:
    #     setattr(ModelFunc, "diff_type", ch_model.prop_type)

    if ch_model.prop_type == XorDiff:
        MyModelChFuncAsOp.diff_type = XorDiff
        prefix = "Xor"
    elif ch_model.prop_type == RXDiff:
        MyModelChFuncAsOp.diff_type = RXDiff
        prefix = "RX"
    elif ch_model.prop_type == LinearMask:
        assert MyModelChFuncAsOp.prop_type == LinearMask
        prefix = "Linear"
    elif ch_model.prop_type == WordValue:
        MyModelChFuncAsOp.val_type = WordValue
        prefix = "Word"
    elif ch_model.prop_type == BitValue:
        MyModelChFuncAsOp.val_type = BitValue
        prefix = "Bit"
    else:
        raise ValueError(f"property {ch_model.prop_type} not supported")

    MyModelChFuncAsOp.__name__ = f"{prefix}Model{my_chfunc_as_op.__name__}"

    return MyModelChFuncAsOp


def get_wrapped_chfunc(ch_model):
    """Wrap and return the underlying bit-vector function of a characteristic model.

    Given a characteristic model (i.e., `abstractproperty.chmodel.ChModel`)
    of a bit-vector function :math:`f`, returns a functionally-equivalent
    (same input-output behaviour) bit-vector function :math:`g` that contains
    the `ChFuncAsOp` of :math:`f` and the output extraction (using `PropExtract`)
    of the single-output of `ChFuncAsOp`.

    Note that the returned wrapped function :math:`g` is a
    `BvFunction` (and not a `RoundBasedFunction`)
    even if `func` is a `RoundBasedFunction`.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.ssa import RoundBasedFunction
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import ChModel
        >>> from cascada.smt.wrappedchmodel import get_wrapped_chfunc
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
        >>> WrappedSpeck32KS = get_wrapped_chfunc(ch_model)
        >>> ch_model.func.get_name(), WrappedSpeck32KS.get_name()
        ('SpeckKeySchedule_2R', 'WrappedSpeckKeySchedule_2R')
        >>> ch_model.func.to_ssa(["mk0", "mk1", "mk2"], "k")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[mk0, mk1, mk2], output_vars=[mk2_out, k3_out, k8_out],
            assignments=[(k0, mk1 >>> 7), (k1, k0 + mk2), (k2, mk2 <<< 2), (k3, k2 ^ k1),
            (k4, mk0 >>> 7), (k5, k4 + k3), (k6, k5 ^ 0x0001), (k7, k3 <<< 2), (k8, k7 ^ k6),
            (mk2_out, Id(mk2)), (k3_out, Id(k3)), (k8_out, Id(k8))])
        >>> WrappedSpeck32KS.to_ssa(["mk0", "mk1", "mk2"], "k") # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[mk0, mk1, mk2], output_vars=[k1_out, k2_out, k3_out],
            assignments=[(k0, SpeckKeySchedule_2RAsOp(mk0, mk1, mk2)),
            (k1, PropExtract_{·, 15, 0}(k0)), (k2, PropExtract_{·, 31, 16}(k0)), (k3, PropExtract_{·, 47, 32}(k0)),
            (k1_out, Id(k1)), (k2_out, Id(k2)), (k3_out, Id(k3))])
        >>> masterkey = [Constant(0, 16)]*3
        >>> ch_model.func(*masterkey)
        (0x0000, 0x0000, 0x0001)
        >>> WrappedSpeck32KS(*masterkey)
        (0x0000, 0x0000, 0x0001)
        >>> issubclass(ch_model.func, RoundBasedFunction), issubclass(WrappedSpeck32KS, RoundBasedFunction)
        (True, False)

    """
    assert isinstance(ch_model, abstractproperty.chmodel.ChModel)

    if hasattr(ch_model.func, "_chfunc_as_op"):
        raise ValueError("wrapping functions already wrapped is not supported")

    # - creating the Op

    chfunc_as_op = make_chfunc_as_op(ch_model)

    # - creating the OpModel and linking it to the Op

    from cascada.differential.difference import XorDiff, RXDiff
    from cascada.linear.mask import LinearMask
    from cascada.algebraic.value import WordValue, BitValue

    if ch_model.prop_type == XorDiff:
        prop_model_name = "xor_model"
    elif ch_model.prop_type == RXDiff:
        prop_model_name = "rx_model"
    elif ch_model.prop_type == LinearMask:
        prop_model_name = "linear_model"
    elif ch_model.prop_type == WordValue:
        prop_model_name = "word_model"
    elif ch_model.prop_type == BitValue:
        prop_model_name = "bit_model"
    else:
        raise ValueError(f"property {ch_model.prop_type} not supported")

    chfuncasop_model = make_chfuncasop_model(chfunc_as_op)

    def property_opmodel(cls, input_prop):
        return chfuncasop_model(input_prop)

    setattr(chfunc_as_op, prop_model_name, classmethod(property_opmodel))

    # - creating the wrapping function

    func = ch_model.func

    # subclassing BvFunction (and not func) to avoid subclassing RoundBasedFunction
    class WrappingFunc(cascada_ssa.BvFunction):
        input_widths = func.input_widths
        output_widths = func.output_widths

        _ignore_replace_multiuse_vars = True
        _chfunc_as_op = chfunc_as_op

        @classmethod
        def eval(cls, *args):
            if issubclass(cls, blockcipher.Encryption):
                # ch_model.ssa.external_vars might be different that round_keys
                assert len(cls.round_keys) == len(cls._chfunc_as_op.ch_model.ssa.external_vars)
                ext_vars = tuple(cls.round_keys)
            else:
                ext_vars = tuple(cls._chfunc_as_op.ch_model.ssa.external_vars)
            out_wop = cls._chfunc_as_op(*(args+ext_vars))
            out_func = []
            offset = 0
            for w in cls.output_widths:
                partial_propextract = abstractproperty.property.make_partial_propextract(w + offset - 1, offset)
                out_func.append(partial_propextract(out_wop))
                offset += w
            assert offset == sum(cls.output_widths)
            return tuple(out_func)

    WrappingFunc.__name__ = f"Wrapped{func.get_name()}"

    return WrappingFunc


def get_wrapped_chcipher(ch_model):
    """Wrap and return the underlying cipher of a characteristic model.

    Given a characteristic model (i.e., `abstractproperty.chmodel.CipherChModel`)
    of a `Cipher`, returns a functionally-equivalent cipher where the
    key-schedule and encryption functions are wrapped by `get_wrapped_chfunc`.

    If the given characteristic model is an `abstractproperty.chmodel.EncryptionChModel`
    of a `Cipher.encryption`, then a functionally-equivalent cipher is returned
    where only the encryption function is wrapped by `get_wrapped_chfunc`.

    Note that the returned wrapped cipher is not iterated (see `Cipher`)
    even if the given cipher is iterated.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.smt.wrappedchmodel import get_wrapped_chcipher
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, XorDiff)
        >>> WrappedSpeck32 = get_wrapped_chcipher(ch_model)
        >>> WrappedSpeck32Enc = WrappedSpeck32.encryption
        >>> ch_model.func.get_name(), WrappedSpeck32Enc.get_name()
        ('SpeckEncryption_2R', 'WrappedSpeckEncryption_2R')
        >>> ch_model.func.to_ssa(["p0", "p1"], "x")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[p0, p1], output_vars=[x7_out, x9_out], external_vars=[dk0, dk1],
            assignments=[(x0, p0 >>> 7), (x1, x0 + p1), (x2, x1 ^ dk0), (x3, p1 <<< 2), (x4, x3 ^ x2),
                (x5, x2 >>> 7), (x6, x5 + x4), (x7, x6 ^ dk1), (x8, x4 <<< 2), (x9, x8 ^ x7),
                (x7_out, Id(x7)), (x9_out, Id(x9))])
        >>> WrappedSpeck32Enc.round_keys = ch_model.func.round_keys[:]
        >>> WrappedSpeck32Enc.to_ssa(["p0", "p1"], "x")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[p0, p1], output_vars=[x1_out, x2_out], external_vars=[dk0, dk1],
            assignments=[(x0, SpeckEncryption_2RAsOp(p0, p1, dk0, dk1)),
                (x1, PropExtract_{·, 15, 0}(x0)), (x2, PropExtract_{·, 31, 16}(x0)),
                (x1_out, Id(x1)), (x2_out, Id(x2))])
        >>> plaintext = Constant(1, 16), Constant(2, 16)
        >>> list(ch_model.func(*plaintext, symbolic_inputs=True))  # doctest: +NORMALIZE_WHITESPACE
        [(((0x0202 ^ dk0) >>> 7) + (0x020a ^ dk0)) ^ dk1,
        ((0x020a ^ dk0) <<< 2) ^ (((0x0202 ^ dk0) >>> 7) + (0x020a ^ dk0)) ^ dk1]
        >>> [x.doit() for x in WrappedSpeck32Enc(*plaintext, symbolic_inputs=True)]  # doctest: +NORMALIZE_WHITESPACE
        [Id(((0x0202 ^ dk0) >>> 7 + (0x020a ^ dk0)) ^ dk1),
        Id((0x020a ^ dk0) <<< 2 ^ ((0x0202 ^ dk0) >>> 7 + (0x020a ^ dk0)) ^ dk1)]
        >>> Speck32.num_rounds, WrappedSpeck32.num_rounds
        (2, None)

    """
    if isinstance(ch_model, abstractproperty.chmodel.EncryptionChModel):
        wrapped_ks = ch_model.cipher.key_schedule
        aux_wrapped_enc = get_wrapped_chfunc(ch_model)
    else:
        assert isinstance(ch_model, abstractproperty.chmodel.CipherChModel)
        wrapped_ks = get_wrapped_chfunc(ch_model.ks_ch_model)
        aux_wrapped_enc = get_wrapped_chfunc(ch_model.enc_ch_model)

    class WrappedEnc(blockcipher.Encryption, aux_wrapped_enc):
        pass

    WrappedEnc.__name__ = aux_wrapped_enc.get_name()

    class WrappingCipher(blockcipher.Cipher):
        key_schedule = wrapped_ks
        encryption = WrappedEnc
        num_rounds = None

    WrappingCipher.__name__ = f"Wrapped{ch_model.cipher.get_name()}"

    return WrappingCipher


def get_wrapped_chmodel(ch_model):
    """Return a wrapped model of the given characteristic model.

    Given a characteristic model of a bit-vector function :math:`f`
    (i.e., `abstractproperty.chmodel.ChModel`),
    a wrapped model is a characteristic model of the wrapped function
    of :math:`f` through `get_wrapped_chfunc`.
    If the given characteristic model is an
    `abstractproperty.chmodel.EncryptionChModel`, the wrapped encryption
    is obtained through `get_wrapped_chcipher`.

    The wrapped model only has one non-`ModelIdentity` transition:
    the `ModelChFuncAsOp` of the `ChFuncAsOp` of :math:`f`.

    To avoid name collisions between the given characteristic model and the
    returned wrapped model (except for the external variables),
    an underscore is added to the variable prefix of the returned wrapped model.

        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.smt.wrappedchmodel import get_wrapped_chmodel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, XorDiff)
        >>> wrapped_ch_model = get_wrapped_chmodel(ch_model)
        >>> wrapped_ch_model  # doctest: +NORMALIZE_WHITESPACE
         EncryptionChModel(cipher=WrappedSpeckCipher_2R,
            input_diff=[XorDiff(_dp0), XorDiff(_dp1)],
            output_diff=[XorDiff(_dx1_out), XorDiff(_dx2_out)],
            external_var2diff=[(_dk0, XorDiff(0x0000)), (_dk1, XorDiff(0x0000))],
            assign_outdiff2op_model=[(XorDiff(_dx0), XorModelSpeckEncryption_2RAsOp([XorDiff(_dp0), XorDiff(_dp1), XorDiff(0x0000), XorDiff(0x0000)])),
                (XorDiff(_dx1_out), XorModelId(XorDiff(PropExtract_{·, 15, 0}(_dx0)))),
                (XorDiff(_dx2_out), XorModelId(XorDiff(PropExtract_{·, 31, 16}(_dx0))))])
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import ChModel
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
        >>> wrapped_ch_model = get_wrapped_chmodel(ch_model)
        >>> wrapped_ch_model  # doctest: +NORMALIZE_WHITESPACE
        ChModel(func=WrappedSpeckKeySchedule_2R,
            input_mask=[LinearMask(_mk0), LinearMask(_mk1), LinearMask(_mk2)],
            output_mask=[LinearMask(_mx1_out), LinearMask(_mx2_out), LinearMask(_mx3_out)],
            assign_outmask2op_model=[(LinearMask(_mx0), LinearModelSpeckKeySchedule_2RAsOp([LinearMask(_mk0), LinearMask(_mk1), LinearMask(_mk2)])),
                (LinearMask(_mx1_out), LinearModelId(LinearMask(PropExtract_{·, 15, 0}(_mx0)))),
                (LinearMask(_mx2_out), LinearModelId(LinearMask(PropExtract_{·, 31, 16}(_mx0)))),
                (LinearMask(_mx3_out), LinearModelId(LinearMask(PropExtract_{·, 47, 32}(_mx0))))])

    """
    assert isinstance(ch_model, abstractproperty.chmodel.ChModel)

    non_id_transitions = 0
    for op_model in ch_model.assign_outprop2op_model.values():
        if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
            non_id_transitions += 1
    if non_id_transitions <= 1:
        warnings.warn(f"wrapping a characteristic model with "
                      f"{non_id_transitions} non-ModelIdentity transition")

    if isinstance(ch_model, abstractproperty.chmodel.EncryptionChModel):
        wrapped_cipher = get_wrapped_chcipher(ch_model)

        class WrappedEncryptionChModel(type(ch_model)):
            _prefix = "_" + ch_model.__class__._prefix

        WrappedEncryptionChModel.__name__ = type(ch_model).__name__

        wrapped_ch_model = WrappedEncryptionChModel(
            wrapped_cipher, ch_model.prop_type,
            op_model_class2options=ch_model._op_model_class2options)
        wrapped_op = wrapped_cipher.encryption._chfunc_as_op
        ext_var_names_set = set()  # round keys are part of wrapped_op and not of ch_model
    else:
        # assert isinstance(ch_model, abstractproperty.chmodel.ChModel)
        wrapped_func = get_wrapped_chfunc(ch_model)
        prefix = "_" + ch_model._prefix
        input_prop_names = ["_" + v.val.name for v in ch_model.input_prop]
        while any(ipn.startswith(prefix) for ipn in input_prop_names):
            prefix += prefix[-1]
        wrapped_ch_model = type(ch_model)._prop_new(
            wrapped_func, ch_model.prop_type,
            input_prop_names=input_prop_names,
            prefix=prefix,
            external_var2prop=ch_model.external_var2prop,
            op_model_class2options=ch_model._op_model_class2options
        )
        wrapped_op = wrapped_func._chfunc_as_op
        ext_var_names_set = {str(v) for v in wrapped_ch_model.external_var2prop}

    # ensure no name collision
    intersection = {str(v) for v in wrapped_ch_model.var2prop}.intersection(
        {str(v) for v in wrapped_op.ch_model.var2prop})
    assert intersection == ext_var_names_set

    assert list(wrapped_ch_model.assign_outprop2op_model.values())[0].op == wrapped_op
    for op_model in wrapped_ch_model.assign_outprop2op_model.values():
        if op_model.op != wrapped_op and not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
            raise ValueError(f"invalid OpModel {op_model} in wrapped model {wrapped_ch_model}")

    wrapped_ch_model._unwrapped_ch_model = ch_model  # used in InvalidPropFinder._check

    for v1, v2 in zip(ch_model.external_var2prop.values(), wrapped_ch_model.external_var2prop.values()):
        if isinstance(v1.val, core.Constant) or isinstance(v2.val, core.Constant):
            assert v1 == v2, f"\n{ch_model}\n{wrapped_ch_model}"

    return wrapped_ch_model


def get_wrapped_cipher_chmodel(cipher_ch_model):
    """Return a wrapped model of the given cipher characteristic model.

    Given a characteristic model of a `Cipher`
    (i.e., `abstractproperty.chmodel.CipherChModel`),
    a wrapped model is a characteristic model of the wrapped cipher
    through `get_wrapped_chcipher`.

    The wrapped model only has two non-`ModelIdentity` transitions:
    the `ModelChFuncAsOp` of the `ChFuncAsOp` of the key-schedule function
    and the `ModelChFuncAsOp` of the `ChFuncAsOp` of the encryption function.

    To avoid name collisions between the given cipher characteristic model
    and the returned wrapped model, an underscore is added to the variable
    prefix of the returned wrapped model.

        >>> from cascada.differential.difference import RXDiff
        >>> from cascada.differential.chmodel import CipherChModel
        >>> from cascada.smt.wrappedchmodel import get_wrapped_cipher_chmodel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = CipherChModel(Speck32, RXDiff)
        >>> wrapped_ch_model = get_wrapped_cipher_chmodel(ch_model)
        >>> wrapped_ch_model  # doctest: +NORMALIZE_WHITESPACE
        CipherChModel(ks_ch_model=ChModel(func=WrappedSpeckKeySchedule_1R,
            input_diff=[RXDiff(_dmk0), RXDiff(_dmk1)],
            output_diff=[RXDiff(_dk1_out), RXDiff(_dk2_out)],
            assign_outdiff2op_model=[(RXDiff(_dk0), RXModelSpeckKeySchedule_1RAsOp([RXDiff(_dmk0), RXDiff(_dmk1)])),
                (RXDiff(_dk1_out), RXModelId(RXDiff(PropExtract_{·, 15, 0}(_dk0)))),
                (RXDiff(_dk2_out), RXModelId(RXDiff(PropExtract_{·, 31, 16}(_dk0))))]),
            enc_ch_model=ChModel(func=WrappedSpeckEncryption_2R,
                input_diff=[RXDiff(_dp0), RXDiff(_dp1)],
                output_diff=[RXDiff(_dx1_out), RXDiff(_dx2_out)],
                external_var2diff=[(_dk1_out, RXDiff(_dk1_out)), (_dk2_out, RXDiff(_dk2_out))],
                assign_outdiff2op_model=[(RXDiff(_dx0), RXModelSpeckEncryption_2RAsOp([RXDiff(_dp0), RXDiff(_dp1), RXDiff(_dk1_out), RXDiff(_dk2_out)])),
                    (RXDiff(_dx1_out), RXModelId(RXDiff(PropExtract_{·, 15, 0}(_dx0)))),
                    (RXDiff(_dx2_out), RXModelId(RXDiff(PropExtract_{·, 31, 16}(_dx0))))]))

    See also `get_wrapped_chmodel`.
    """
    assert isinstance(cipher_ch_model, abstractproperty.chmodel.CipherChModel)

    non_id_transitions = [0, 0]
    for i, cm in enumerate([cipher_ch_model.ks_ch_model, cipher_ch_model.enc_ch_model]):
        for op_model in cm.assign_outprop2op_model.values():
            if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                non_id_transitions[i] += 1
    if non_id_transitions[0] <= 1 and non_id_transitions[1] <= 1:
        warnings.warn(f"wrapping a cipher characteristic model with "
                      f"{non_id_transitions} non-ModelIdentity transitions")

    wrapped_cipher = get_wrapped_chcipher(cipher_ch_model)

    class WrappedCipherChModel(type(cipher_ch_model)):
        _prefix = "_" + cipher_ch_model.__class__._prefix

    WrappedCipherChModel.__name__ = type(cipher_ch_model).__name__

    wrapped_cipher_ch_model = WrappedCipherChModel(
        wrapped_cipher, cipher_ch_model.prop_type,
        op_model_class2options=cipher_ch_model._op_model_class2options)

    for cm in [wrapped_cipher_ch_model.ks_ch_model, wrapped_cipher_ch_model.enc_ch_model]:
        wrapped_op = cm.func._chfunc_as_op

        intersection = {str(v) for v in cm.var2prop}.intersection({str(v) for v in wrapped_op.ch_model.var2prop})
        assert len(intersection) == 0, f"{intersection}"

        assert list(cm.assign_outprop2op_model.values())[0].op == wrapped_op
        for op_model in cm.assign_outprop2op_model.values():
            if op_model.op != wrapped_op and not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                raise ValueError(f"invalid OpModel {op_model} in wrapped model {wrapped_cipher_ch_model}")

    wrapped_cipher_ch_model._unwrapped_cipher_ch_model = cipher_ch_model  # used in InvalidCipherPropFinder._check
    wrapped_cipher_ch_model.ks_ch_model._unwrapped_ch_model = cipher_ch_model.ks_ch_model
    wrapped_cipher_ch_model.enc_ch_model._unwrapped_ch_model = cipher_ch_model.enc_ch_model

    assert len(wrapped_cipher_ch_model.ks_ch_model.external_var2prop.values()) == 0
    assert len(cipher_ch_model.ks_ch_model.external_var2prop.values()) == 0
    for v1, v2 in zip(
            wrapped_cipher_ch_model.enc_ch_model.external_var2prop.values(),
            cipher_ch_model.enc_ch_model.external_var2prop.values()
    ):
        if isinstance(v1.val, core.Constant) or isinstance(v2.val, core.Constant):
            assert v1 == v2, f"\n{cipher_ch_model}\n{wrapped_cipher_ch_model}"

    return wrapped_cipher_ch_model
