"""Manage bit-vector models of linear characteristics."""
import functools
import warnings

from cascada.bitvector import operation, core
from cascada.linear.mask import LinearMask
from cascada.linear import opmodel as cascada_opmodel

from cascada import abstractproperty


ChModelSigType = abstractproperty.chmodel.ChModelSigType


class ChModel(abstractproperty.chmodel.ChModel):
    """Represent bit-vector models of linear characteristics over bit-vector functions.

    Internally, this class is a subclass of `abstractproperty.chmodel.ChModel`,
    where the `Property` is a `LinearMask` type and the probability of the
    characteristic here considered is the product of the
    absolute linear correlation of the linear approximations
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` composing
    the characteristic (see `linear.characteristic.Characteristic`).

    To handle the propagation of masks through branching subroutines
    (i.e., :math:`x \mapsto (x, x)`), the `SSA` of the `BvFunction` is computed
    with the initialization argument ``replace_multiuse_vars`` set to ``True``
    and the new free masks are wrapped in `LinearModelFreeBranch` objects
    (see also `SSA` and `LinearModelFreeBranch`).

    If the `SSA` of the `BvFunction` contains external `Variable` objects
    and their `LinearMask` objects are not provided in the initialization
    through ``external_var2mask``, these free external masks
    are wrapped in `LinearMask` objects.

    Linear characteristic models of functions with unused input vars
    is currently not supported.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import ChModel
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE
        ChModel(func=SpeckKeySchedule_2R,
                input_mask=[LinearMask(mk0), LinearMask(mk1), LinearMask(mk2)],
                output_mask=[LinearMask(mk2_out), LinearMask(mx3_out), LinearMask(mx8_out)],
                assign_outmask2op_model=[(LinearMask(mk2__0), LinearModelFreeBranch(LinearMask(mk2))),
                    (LinearMask(mk2__1), LinearModelFreeBranch(LinearMask(mk2))),
                    (LinearMask(mx1), LinearModelBvAdd([LinearMask(mk1 >>> 7), LinearMask(mk2__0)])),
                    (LinearMask(mx3), LinearModelBvXor([LinearMask(mk2__1 <<< 2), LinearMask(mx1)])),
                    (LinearMask(mx3__0), LinearModelFreeBranch(LinearMask(mx3))),
                    (LinearMask(mx3__1), LinearModelFreeBranch(LinearMask(mx3))),
                    (LinearMask(mx5), LinearModelBvAdd([LinearMask(mk0 >>> 7), LinearMask(mx3__0)])),
                    (LinearMask(mx8), LinearModelBvXor([LinearMask(mx3__1 <<< 2), LinearMask(mx5)])),
                    (LinearMask(mk2_out), LinearModelId(LinearMask(mk2 ^ mk2__0 ^ mk2__1))),
                    (LinearMask(mx3_out), LinearModelId(LinearMask(mx3 ^ mx3__0 ^ mx3__1))),
                    (LinearMask(mx8_out), LinearModelId(LinearMask(mx8)))])
        >>> ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[mk0, mk1, mk2],
            output_vars=[mk2_out, mx3_out, mx8_out],
            assignments=[(mx0, mk1 >>> 7), (mk2__0, Id(mk2)), (mk2__1, Id(mk2)), (mk2__2, Id(mk2)),
                    (mx1, mx0 + mk2__0), (mx2, mk2__1 <<< 2), (mx3, mx2 ^ mx1),
                (mx4, mk0 >>> 7), (mx3__0, Id(mx3)), (mx3__1, Id(mx3)), (mx3__2, Id(mx3)),
                    (mx5, mx4 + mx3__0), (mx6, mx5 ^ 0x0001), (mx7, mx3__1 <<< 2), (mx8, mx7 ^ mx6),
                (mk2_out, Id(mk2__2)), (mx3_out, Id(mx3__2)), (mx8_out, Id(mx8))])

    Attributes:
        mask_type: the mask type (currently only `LinearMask` is supported);
            alias of `abstractproperty.chmodel.ChModel.prop_type`.
        input_mask: a list of `LinearMask` objects containing
            the (symbolic) input mask (alias of
            `abstractproperty.chmodel.ChModel.input_prop`).
        output_mask: a list of `LinearMask` objects containing
            the (symbolic) output mask (alias of
            `abstractproperty.chmodel.ChModel.output_prop`).
        external_var2mask: a `collections.OrderedDict` mapping the external
            variables of `abstractproperty.chmodel.ChModel.ssa` to their
            associated (symbolic or constant) `LinearMask` objects (alias of
            `abstractproperty.chmodel.ChModel.external_var2prop`).
        assign_outmask2op_model: a `collections.OrderedDict` mapping the
            output `LinearMask` :math:`\Delta_{x_{i+1}}` of
            the non-trivial assignment  :math:`x_{i+1} \leftarrow f_i(x_i)` to
            the `linear.opmodel.OpModel` of this assignment (alias of
            `abstractproperty.chmodel.ChModel.assign_outprop2op_model`).
        var2mask: a `collections.OrderedDict` mapping each variable in
            `abstractproperty.chmodel.ChModel.ssa`
            to its associated `LinearMask` object (alias of
            `abstractproperty.chmodel.ChModel.var2prop`).

    """
    _prop_label = "mask"  # for str and vrepr

    @property
    def mask_type(self):
        return self.prop_type

    @property
    def input_mask(self):
        return self.input_prop

    @property
    def output_mask(self):
        return self.output_prop

    @property
    def external_var2mask(self):
        return self.external_var2prop

    @property
    def assign_outmask2op_model(self):
        return self.assign_outprop2op_model

    @property
    def var2mask(self):
        return self.var2prop

    @property
    def _input_mask_names(self):
        return self._input_prop_names

    def _func2ssa(self, func, names, prefix):
        return func.to_ssa(names, id_prefix=prefix, decompose_sec_ops=False, replace_multiuse_vars=True)

    def _propagate(self, op, input_mask, outvar_op):
        outvar_is_input_prop_extract = False
        if op == operation.BvIdentity:
            assert len(input_mask) == 1
            # check if this op belongs to a branching operation
            # branching notation: a single left variable branching into multiples right variables
            right_var = outvar_op
            if right_var in self.ssa.singleuse_var2multiuse_var:
                left_var = self.ssa.singleuse_var2multiuse_var[right_var]

                for ssa_var, ssa_expr in self.ssa.assignments.items():
                    if isinstance(ssa_expr, operation.PartialOperation) and \
                            ssa_expr.base_op == abstractproperty.property.PropExtract and \
                            ssa_expr.args[0] == right_var:
                        outvar_is_input_prop_extract = True
                        break

                if not outvar_is_input_prop_extract:
                    # check if right_var is last branch of a branching operation
                    if right_var == self.ssa.multiuse_var2singleuse_vars[left_var][-1]:
                        left_mask = self.var2mask[left_var].val
                        other_right_variables = self.ssa.multiuse_var2singleuse_vars[left_var][:-1]
                        other_right_masks = [self.var2mask[v].val for v in other_right_variables]
                        all_other_masks = [left_mask] + other_right_masks
                        return LinearMask(functools.reduce(operation.BvXor, all_other_masks))
                    else:
                        return cascada_opmodel.LinearModelFreeBranch(input_mask)

        if not outvar_is_input_prop_extract:
            assert outvar_op not in self.ssa.singleuse_var2multiuse_var

        output_mask = self.mask_type.propagate(op, input_mask)

        # check that no external mask is used in a non-trivial OpModel
        #   Example: function f(x0) = x0 ^ (k0 & 1).
        #   Without this check, its trail will contain the non-trivial OpModel (mk1, BvAndCt_1(mk0))
        #   which will either append a non-zero weight or fix the masks mk0 and mk1,
        #   (and might lead to an invalid trail if multiple non-trivial OpModel involving mk1, mk0).
        #   However, this OpModel is not needed since we can move (k0 & 1) to the key schedule.
        if not isinstance(output_mask, LinearMask) and \
                any(im in self.external_var2mask for im in input_mask):
            raise ValueError(f"linear ChModel does not support non-trivial assignments "
                             f"involving external variables, as {input_mask} -› {output_mask}, "
                             f"external_var2mask = {self.external_var2mask}")

        return output_mask

    def _get_property_missing_external_var(self, mask_type, ext_var, external_var2mask):
        return mask_type(ext_var)

    def _check_input_vars_not_used(self, func_ssa):
        # if f(x) = M(x) is a linear function then
        #   (alpha, beta) = (a, b) is a linear approx. with Cr. +-1 if a = M^t · b
        # let f(p0, p1) = (p0)  // p1 input var not used
        #   M = [I, 0], (a0, a1) = [I 0]^T (b0) = (b0, 0)
        # that is, an extra assertion p1 == 0 needs to be added for every p1 input var not used
        # currently, adding extra assertion without adding assignments it is not supported
        if func_ssa._input_vars_not_used:
            raise ValueError("linear characteristic models of functions with unused input vars is not supported")

    def __init__(self, func, mask_type, input_mask_names, prefix="mx",
                 external_var2mask=None, op_model_class2options=None):
        super().__init__(
            func, prop_type=mask_type, input_prop_names=input_mask_names, prefix=prefix,
            external_var2prop=external_var2mask, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_Characteristic_cls(cls):
        from cascada.linear.characteristic import Characteristic as Characteristic_cls
        return Characteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.ChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> ch_model.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), mask_type=LinearMask,
                     input_mask_names=['mk0', 'mk1', 'mk2'], prefix='mx')"

        """
        return super().vrepr()

    def validity_assertions(self):
        """Return the validity constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.validity_assertions`.

            >>> from functools import reduce
            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> assertions = ch_model.validity_assertions()
            >>> for a in assertions: print(a)  # doctest: +NORMALIZE_WHITESPACE
            ((~((mx1 ^ (mk1 >>> 7)) | (mx1 ^ mk2__0)) | _tmp5b87dfbff36ecd792bff9b4a837080cc) == 0xffff) &
                ((_tmp5b87dfbff36ecd792bff9b4a837080cc ^ (_tmp5b87dfbff36ecd792bff9b4a837080cc >> 0x0001) ^
                ((mx1 ^ (mk1 >>> 7) ^ mk2__0) >> 0x0001)) == 0x0000)
            ((mk2__1 <<< 2) == mx1) & ((mk2__1 <<< 2) == mx3)
            ((~((mx5 ^ (mk0 >>> 7)) | (mx5 ^ mx3__0)) | _tmp496354c626556f9b918d51527d988998) == 0xffff) &
                ((_tmp496354c626556f9b918d51527d988998 ^ (_tmp496354c626556f9b918d51527d988998 >> 0x0001) ^
                ((mx5 ^ (mk0 >>> 7) ^ mx3__0) >> 0x0001)) == 0x0000)
            ((mx3__1 <<< 2) == mx5) & ((mx3__1 <<< 2) == mx8)
            (mk2 ^ mk2__0 ^ mk2__1) == mk2_out
            (mx3 ^ mx3__0 ^ mx3__1) == mx3_out
            mx8 == mx8_out
            >>> validity_constraint = reduce(BvAnd, assertions)
            >>> validity_constraint  # doctest: +ELLIPSIS
            ((~((mx1 ^ (mk1 >>> 7)) | (mx1 ^ mk2__0)) | ...

        """
        return super().validity_assertions()

    def pr_one_assertions(self):
        """Return the probability-one constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.pr_one_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> for a in ch_model.pr_one_assertions(): print(a)  # doctest: +NORMALIZE_WHITESPACE
            (~((mx1 ^ (mk1 >>> 7)) | (mx1 ^ mk2__0)) == 0xffff) & (((mx1 ^ (mk1 >>> 7) ^ mk2__0)[:1]) == 0b000000000000000)
            ((mk2__1 <<< 2) == mx1) & ((mk2__1 <<< 2) == mx3)
            (~((mx5 ^ (mk0 >>> 7)) | (mx5 ^ mx3__0)) == 0xffff) & (((mx5 ^ (mk0 >>> 7) ^ mx3__0)[:1]) == 0b000000000000000)
            ((mx3__1 <<< 2) == mx5) & ((mx3__1 <<< 2) == mx8)
            (mk2 ^ mk2__0 ^ mk2__1) == mk2_out
            (mx3 ^ mx3__0 ^ mx3__1) == mx3_out
            mx8 == mx8_out

        """
        return super().pr_one_assertions()

    def weight_assertions(self, ch_weight_variable, assign_weight_variables, truncate=True):
        """Return the weight constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.weight_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> cm = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> cwv = Variable("w", cm.weight_width())
            >>> omvs = [Variable(f"w{i}", om.weight_width()) for i, om in enumerate(cm.assign_outmask2op_model.values())]
            >>> for a in cm.weight_assertions(cwv, omvs): print(a)  # doctest: +NORMALIZE_WHITESPACE
            w0 == 0b0
            w1 == 0b0
            (PopCount(_tmp5b87dfbff36ecd792bff9b4a837080cc[14:]) == w2) & ((_tmp5b87dfbff36ecd792bff9b4a837080cc ^
                (_tmp5b87dfbff36ecd792bff9b4a837080cc >> 0x0001) ^ ((mx1 ^ (mk1 >>> 7) ^ mk2__0) >> 0x0001)) == 0x0000)
            w3 == 0b0
            w4 == 0b0
            w5 == 0b0
            (PopCount(_tmp496354c626556f9b918d51527d988998[14:]) == w6) & ((_tmp496354c626556f9b918d51527d988998 ^
                (_tmp496354c626556f9b918d51527d988998 >> 0x0001) ^ ((mx5 ^ (mk0 >>> 7) ^ mx3__0) >> 0x0001)) == 0x0000)
            w7 == 0b0
            w8 == 0b0
            w9 == 0b0
            w10 == 0b0
            w == ((0b0 :: w2) + (0b0 :: w6))

        """
        return super().weight_assertions(ch_weight_variable, assign_weight_variables, truncate=truncate)

    def max_weight(self, truncate=True):
        """Return the maximum value the ch. weight variable can achieve in `weight_assertions`.

        See also `abstractproperty.chmodel.ChModel.max_weight`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> cm = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> cm.max_weight(), cm.weight_width(), cm.num_frac_bits()
            (30, 5, 0)

        """
        return super().max_weight(truncate=truncate)

    def error(self):
        """Return the maximum difference between `weight_assertions` and the exact weight.

        See also `abstractproperty.chmodel.ChModel.error`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"]).error()
            0

        """
        return super().error()

    def signature(self, sig_type):
        """Return the signature of the characteristic model.

        See also `abstractproperty.chmodel.ChModel.signature`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> ch_model.signature(ChModelSigType.Unique)
            [mk2, mk1, mk2__0, mx3, mk0, mx3__0, mk2_out, mx3_out, mx8_out]
            >>> ch_model.signature(ChModelSigType.InputOutput)
            [mk0, mk1, mk2, mk2_out, mx3_out, mx8_out]

        """
        if sig_type == ChModelSigType.Unique:
            # only the output, external and input masks of non-trivial assignments
            # are needed to fully propagate the characteristic
            sig_var = []
            for ext_var, prop in self.external_var2prop.items():
                if not isinstance(prop.val, core.Constant):
                    sig_var.append(ext_var)

            def sig_var_add_vars_from_expr(expr):
                for v in expr.atoms(core.Variable):
                    if v not in sig_var:
                        sig_var.append(v)

            for op_model in self.assign_outprop2op_model.values():
                if isinstance(op_model, cascada_opmodel.LinearModelFreeBranch):
                    sig_var_add_vars_from_expr(op_model.input_prop[0].val)
                elif op_model.max_weight() != 0:
                    for input_prop in op_model.input_prop:
                        sig_var_add_vars_from_expr(input_prop.val)

            for p in self.output_prop:
                if p.val not in sig_var:
                    sig_var.append(p.val)
            return sig_var
        else:
            return super().signature(sig_type)

    def split(self, mask_separators):
        """Split into multiple `ChModel` objects given the list of mask separators.

        See also `abstractproperty.chmodel.ChModel.split`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> tuple(ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((mx0, mk1 >>> 7), (mk2__0, Id(mk2)), (mk2__1, Id(mk2)), (mk2__2, Id(mk2)),
                (mx1, mx0 + mk2__0), (mx2, mk2__1 <<< 2), (mx3, mx2 ^ mx1),
            (mx4, mk0 >>> 7), (mx3__0, Id(mx3)), (mx3__1, Id(mx3)), (mx3__2, Id(mx3)),
                (mx5, mx4 + mx3__0), (mx6, mx5 ^ 0x0001), (mx7, mx3__1 <<< 2), (mx8, mx7 ^ mx6),
            (mk2_out, Id(mk2__2)), (mx3_out, Id(mx3__2)), (mx8_out, Id(mx8)))
            >>> mask_separators = [ (LinearMask(Variable("mx2", width=16)), LinearMask(Variable("mx3", width=16))), ]
            >>> for cm in ch_model.split(mask_separators): print(cm)  # doctest: +NORMALIZE_WHITESPACE
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_mask=[LinearMask(mk0), LinearMask(mk1), LinearMask(mk2)],
                    output_mask=[LinearMask(mk0_out), LinearMask(mx3_out), LinearMask(mk2__2_out)],
                    assign_outmask2op_model=[(LinearMask(mk2__0), LinearModelFreeBranch(LinearMask(mk2))),
                        (LinearMask(mk2__1), LinearModelFreeBranch(LinearMask(mk2))),
                        (LinearMask(mx1), LinearModelBvAdd([LinearMask(mk1 >>> 7), LinearMask(mk2__0)])),
                        (LinearMask(mx3), LinearModelBvXor([LinearMask(mk2__1 <<< 2), LinearMask(mx1)])),
                        (LinearMask(mk0_out), LinearModelId(LinearMask(mk0))),
                        (LinearMask(mx3_out), LinearModelId(LinearMask(mx3))),
                        (LinearMask(mk2__2_out), LinearModelId(LinearMask(mk2 ^ mk2__0 ^ mk2__1)))])
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_mask=[LinearMask(mk0), LinearMask(mx3), LinearMask(mk2__2)],
                    output_mask=[LinearMask(mk2_out), LinearMask(mx3_out), LinearMask(mx8_out)],
                    assign_outmask2op_model=[(LinearMask(mx3__0), LinearModelFreeBranch(LinearMask(mx3))),
                        (LinearMask(mx3__1), LinearModelFreeBranch(LinearMask(mx3))),
                        (LinearMask(mx5), LinearModelBvAdd([LinearMask(mk0 >>> 7), LinearMask(mx3__0)])),
                        (LinearMask(mx8), LinearModelBvXor([LinearMask(mx3__1 <<< 2), LinearMask(mx5)])),
                        (LinearMask(mk2_out), LinearModelId(LinearMask(mk2__2))),
                        (LinearMask(mx3_out), LinearModelId(LinearMask(mx3 ^ mx3__0 ^ mx3__1))),
                        (LinearMask(mx8_out), LinearModelId(LinearMask(mx8)))])

        """
        return super().split(mask_separators)

    def get_round_separators(self):
        """Return the round separators if `func` is a `RoundBasedFunction`.

        See also `abstractproperty.chmodel.ChModel.get_round_separators`.

            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, LinearMask, ["mk0", "mk1", "mk2"])
            >>> sub_cms = ch_model.split(ch_model.get_round_separators())
            >>> for cm in sub_cms: print(cm)  # doctest: +NORMALIZE_WHITESPACE
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_mask=[LinearMask(mk0), LinearMask(mk1), LinearMask(mk2)],
                    output_mask=[LinearMask(mk0_out), LinearMask(mx3_out), LinearMask(mk2__2_out)],
                    assign_outmask2op_model=[(LinearMask(mk2__0), LinearModelFreeBranch(LinearMask(mk2))),
                        (LinearMask(mk2__1), LinearModelFreeBranch(LinearMask(mk2))),
                        (LinearMask(mx1), LinearModelBvAdd([LinearMask(mk1 >>> 7), LinearMask(mk2__0)])),
                        (LinearMask(mx3), LinearModelBvXor([LinearMask(mk2__1 <<< 2), LinearMask(mx1)])),
                        (LinearMask(mk0_out), LinearModelId(LinearMask(mk0))),
                        (LinearMask(mx3_out), LinearModelId(LinearMask(mx3))),
                        (LinearMask(mk2__2_out), LinearModelId(LinearMask(mk2 ^ mk2__0 ^ mk2__1)))])
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_mask=[LinearMask(mk0), LinearMask(mx3), LinearMask(mk2__2)],
                    output_mask=[LinearMask(mk2_out), LinearMask(mx3_out), LinearMask(mx8_out)],
                    assign_outmask2op_model=[(LinearMask(mx3__0), LinearModelFreeBranch(LinearMask(mx3))),
                        (LinearMask(mx3__1), LinearModelFreeBranch(LinearMask(mx3))),
                        (LinearMask(mx5), LinearModelBvAdd([LinearMask(mk0 >>> 7), LinearMask(mx3__0)])),
                        (LinearMask(mx8), LinearModelBvXor([LinearMask(mx3__1 <<< 2), LinearMask(mx5)])),
                        (LinearMask(mk2_out), LinearModelId(LinearMask(mk2__2))),
                        (LinearMask(mx3_out), LinearModelId(LinearMask(mx3 ^ mx3__0 ^ mx3__1))),
                        (LinearMask(mx8_out), LinearModelId(LinearMask(mx8)))])

        """
        return super().get_round_separators()


class EncryptionChModel(abstractproperty.chmodel.EncryptionChModel, ChModel):
    """Represent linear characteristic models of encryption functions.

    Given a `Cipher`, an `EncryptionChModel` is a bit-vector model
    (see `ChModel`) of a linear characteristic over
    the `Cipher.encryption`
    (where the `Cipher.key_schedule` is ignored and round key masks
    are set to `LinearMask` objects containing `Variable` objects).

    See also `linear.characteristic.EncryptionCharacteristic`.

    In an `EncryptionChModel`, the plaintext masks start with the prefix ``"mp"``,
    the intermediate and output masks start with the prefix ``"mx"``,
    and the round key masks (the masks of the external variables
    of the encryption `SSA`) start with the prefix ``"mk"``.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import EncryptionChModel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, LinearMask)
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE
        EncryptionChModel(cipher=SpeckCipher_2R,
            input_mask=[LinearMask(mp0), LinearMask(mp1)],
            output_mask=[LinearMask(mx7_out), LinearMask(mx9_out)],
            external_var2mask=[(mk0, LinearMask(mk0)), (mk1, LinearMask(mk1))],
            assign_outmask2op_model=[(LinearMask(mp1__0), LinearModelFreeBranch(LinearMask(mp1))),
                (LinearMask(mx1), LinearModelBvAdd([LinearMask(mp0 >>> 7), LinearMask(mp1__0)])),
                (LinearMask(mx2), LinearModelBvXor([LinearMask(mx1), LinearMask(mk0)])),
                (LinearMask(mx2__0), LinearModelFreeBranch(LinearMask(mx2))),
                (LinearMask(mx4), LinearModelBvXor([LinearMask((mp1 ^ mp1__0) <<< 2), LinearMask(mx2__0)])),
                (LinearMask(mx4__0), LinearModelFreeBranch(LinearMask(mx4))),
                (LinearMask(mx6), LinearModelBvAdd([LinearMask((mx2 ^ mx2__0) >>> 7), LinearMask(mx4__0)])),
                (LinearMask(mx7), LinearModelBvXor([LinearMask(mx6), LinearMask(mk1)])),
                (LinearMask(mx7__0), LinearModelFreeBranch(LinearMask(mx7))),
                (LinearMask(mx9), LinearModelBvXor([LinearMask((mx4 ^ mx4__0) <<< 2), LinearMask(mx7__0)])),
                (LinearMask(mx7_out), LinearModelId(LinearMask(mx7 ^ mx7__0))),
                (LinearMask(mx9_out), LinearModelId(LinearMask(mx9)))])
        >>> ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[mp0, mp1], output_vars=[mx7_out, mx9_out], external_vars=[mk0, mk1],
        assignments=[(mx0, mp0 >>> 7), (mp1__0, Id(mp1)), (mp1__1, Id(mp1)),
            (mx1, mx0 + mp1__0), (mx2, mx1 ^ mk0), (mx3, mp1__1 <<< 2), (mx2__0, Id(mx2)), (mx2__1, Id(mx2)),
                (mx4, mx3 ^ mx2__0), (mx5, mx2__1 >>> 7), (mx4__0, Id(mx4)), (mx4__1, Id(mx4)),
            (mx6, mx5 + mx4__0), (mx7, mx6 ^ mk1), (mx8, mx4__1 <<< 2), (mx7__0, Id(mx7)), (mx7__1, Id(mx7)),
                (mx9, mx8 ^ mx7__0), (mx7_out, Id(mx7__1)), (mx9_out, Id(mx9))])

    """
    _prefix = "m"

    def __init__(self, cipher, mask_type, op_model_class2options=None):
        super().__init__(
            cipher, prop_type=mask_type, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_EncryptionCharacteristic_cls(cls):
        from cascada.linear.characteristic import EncryptionCharacteristic as EncryptionCharacteristic_cls
        return EncryptionCharacteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.EncryptionChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import EncryptionChModel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> EncryptionChModel(Speck32, LinearMask).vrepr()
            'EncryptionChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), mask_type=LinearMask)'

        """
        return super().vrepr()
