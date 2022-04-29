"""Manage bit-vector models of differential characteristics."""
from cascada.bitvector import core
from cascada.differential import difference

from cascada import abstractproperty


ChModelSigType = abstractproperty.chmodel.ChModelSigType


class ChModel(abstractproperty.chmodel.ChModel):
    """Represent bit-vector models of differential characteristics over bit-vector functions.

    Internally, this class is a subclass of `abstractproperty.chmodel.ChModel`,
    where the `Property` is a `Difference` type and the probability of the
    characteristic here considered is the product of the
    probabilities of the differentials
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` composing
    the characteristic (see `differential.characteristic.Characteristic`).

    If the `SSA` of the `BvFunction` contains external `Variable` objects
    and their `Difference` objects are not provided in the initialization
    through ``external_var2diff``, these free external differences
    are set to zero when the argument ``diff_type`` is `XorDiff`.

    .. note::

        Free external differences are not supported by `ChModel`
        when ``diff_type`` is `RXDiff`. For this type of difference,
        all external differences must be provided in ``external_var2diff``.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff, RXDiff
        >>> from cascada.differential.chmodel import ChModel
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
        >>> xor_ch_model  # doctest: +NORMALIZE_WHITESPACE
        ChModel(func=SpeckKeySchedule_2R,
            input_diff=[XorDiff(dmk0), XorDiff(dmk1), XorDiff(dmk2)],
            output_diff=[XorDiff(dmk2_out), XorDiff(dx3_out), XorDiff(dx8_out)],
            assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dmk1 >>> 7), XorDiff(dmk2)])),
                (XorDiff(dx5), XorModelBvAdd([XorDiff(dmk0 >>> 7), XorDiff((dmk2 <<< 2) ^ dx1)])),
                (XorDiff(dmk2_out), XorModelId(XorDiff(dmk2))),
                (XorDiff(dx3_out), XorModelId(XorDiff((dmk2 <<< 2) ^ dx1))),
                (XorDiff(dx8_out), XorModelId(XorDiff((((dmk2 <<< 2) ^ dx1) <<< 2) ^ dx5)))])
        >>> xor_ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[dmk0, dmk1, dmk2],
            output_vars=[dmk2_out, dx3_out, dx8_out],
            assignments=[(dx0, dmk1 >>> 7), (dx1, dx0 + dmk2), (dx2, dmk2 <<< 2), (dx3, dx2 ^ dx1),
                (dx4, dmk0 >>> 7), (dx5, dx4 + dx3), (dx6, dx5 ^ 0x0001), (dx7, dx3 <<< 2), (dx8, dx7 ^ dx6),
                (dmk2_out, Id(dmk2)), (dx3_out, Id(dx3)), (dx8_out, Id(dx8))])
        >>> rx_ch_model = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])
        >>> rx_ch_model  # doctest: +NORMALIZE_WHITESPACE
        ChModel(func=SpeckKeySchedule_2R,
                input_diff=[RXDiff(dmk0), RXDiff(dmk1), RXDiff(dmk2)],
                output_diff=[RXDiff(dmk2_out), RXDiff(dx3_out), RXDiff(dx8_out)],
                assign_outdiff2op_model=[(RXDiff(dx1), RXModelBvAdd([RXDiff(dmk1 >>> 7), RXDiff(dmk2)])),
                    (RXDiff(dx5), RXModelBvAdd([RXDiff(dmk0 >>> 7), RXDiff((dmk2 <<< 2) ^ dx1)])),
                    (RXDiff(dmk2_out), RXModelId(RXDiff(dmk2))),
                    (RXDiff(dx3_out), RXModelId(RXDiff((dmk2 <<< 2) ^ dx1))),
                    (RXDiff(dx8_out), RXModelId(RXDiff((((dmk2 <<< 2) ^ dx1) <<< 2) ^ dx5 ^ 0x0003)))])

    Attributes:
        diff_type: the type of `Difference` of the characteristic (alias of
            `abstractproperty.chmodel.ChModel.prop_type`).
        input_diff: a list of `Difference` objects containing
            the (symbolic) input difference (alias of
            `abstractproperty.chmodel.ChModel.input_prop`).
        output_diff: a list of `Difference` objects containing
            the (symbolic) output difference (alias of
            `abstractproperty.chmodel.ChModel.output_prop`).
        external_var2diff: a `collections.OrderedDict` mapping the external
            variables of `abstractproperty.chmodel.ChModel.ssa` to their
            associated (symbolic or constant) `Difference` objects (alias of
            `abstractproperty.chmodel.ChModel.external_var2prop`).
        assign_outdiff2op_model: a `collections.OrderedDict` mapping the
            output `Difference` :math:`\Delta_{x_{i+1}}` of
            the non-trivial assignment  :math:`x_{i+1} \leftarrow f_i(x_i)` to
            the `differential.opmodel.OpModel` of this assignment (alias of
            `abstractproperty.chmodel.ChModel.assign_outprop2op_model`).
        var2diff: a `collections.OrderedDict` mapping each variable in
            `abstractproperty.chmodel.ChModel.ssa`
            to its associated `Difference` object (alias of
            `abstractproperty.chmodel.ChModel.var2prop`).

    """
    _prop_label = "diff"  # for str and vrepr

    @property
    def diff_type(self):
        return self.prop_type

    @property
    def input_diff(self):
        return self.input_prop

    @property
    def output_diff(self):
        return self.output_prop

    @property
    def external_var2diff(self):
        return self.external_var2prop

    @property
    def assign_outdiff2op_model(self):
        return self.assign_outprop2op_model

    @property
    def var2diff(self):
        return self.var2prop

    @property
    def _input_diff_names(self):
        return self._input_prop_names

    def _get_property_missing_external_var(self, diff_type, ext_var, external_var2diff):
        if diff_type == difference.XorDiff:
            return diff_type(core.Constant(0, ext_var.width))
        elif diff_type == difference.RXDiff:
            raise ValueError(f"found external variable {ext_var} with not RX-difference associated "
                             f"in external_var2prop={external_var2diff} (required for RXDiff)")

    def __init__(self, func, diff_type, input_diff_names, prefix="dx",
                 external_var2diff=None, op_model_class2options=None):
        super().__init__(
            func, prop_type=diff_type, input_prop_names=input_diff_names, prefix=prefix,
            external_var2prop=external_var2diff, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_Characteristic_cls(cls):
        from cascada.differential.characteristic import Characteristic as Characteristic_cls
        return Characteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.ChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> ch_model.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), diff_type=XorDiff,
                     input_diff_names=['dmk0', 'dmk1', 'dmk2'], prefix='dx')"

        """
        return super().vrepr()

    def validity_assertions(self):
        """Return the validity constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.validity_assertions`.

            >>> from functools import reduce
            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> assertions = xor_ch_model.validity_assertions()
            >>> for a in assertions: print(a)  # doctest: +NORMALIZE_WHITESPACE
            ((~((dmk1 >>> 7) << 0x0001) ^ (dmk2 << 0x0001)) & (~((dmk1 >>> 7) << 0x0001) ^ (dx1 << 0x0001)) &
                ((dmk1 >>> 7) ^ dmk2 ^ dx1 ^ ((dmk1 >>> 7) << 0x0001))) == 0x0000
            ((~((dmk0 >>> 7) << 0x0001) ^ (((dmk2 <<< 2) ^ dx1) << 0x0001)) & (~((dmk0 >>> 7) << 0x0001) ^
                (dx5 << 0x0001)) & ((dmk0 >>> 7) ^ (dmk2 <<< 2) ^ dx1 ^ dx5 ^ ((dmk0 >>> 7) << 0x0001))) == 0x0000
            dmk2 == dmk2_out
            ((dmk2 <<< 2) ^ dx1) == dx3_out
            ((((dmk2 <<< 2) ^ dx1) <<< 2) ^ dx5) == dx8_out
            >>> validity_constraint = reduce(BvAnd, assertions)
            >>> validity_constraint  # doctest: +ELLIPSIS
            (((~((dmk1 >>> 7) << 0x0001) ^ (dmk2 << 0x0001)) & ...

        """
        return super().validity_assertions()

    def pr_one_assertions(self):
        """Return the probability-one constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.pr_one_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> for a in xor_ch_model.pr_one_assertions(): print(a)  # doctest: +NORMALIZE_WHITESPACE
            (((~((dmk1 >>> 7) << 0x0001) ^ (dmk2 << 0x0001)) & (~((dmk1 >>> 7) << 0x0001) ^ (dx1 << 0x0001)) &
                ((dmk1 >>> 7) ^ dmk2 ^ dx1 ^ ((dmk1 >>> 7) << 0x0001))) == 0x0000) & ((~((~(dmk1 >>> 7) ^ dmk2) &
                (~(dmk1 >>> 7) ^ dx1))[14:]) == 0b000000000000000)
            (((~((dmk0 >>> 7) << 0x0001) ^ (((dmk2 <<< 2) ^ dx1) << 0x0001)) & (~((dmk0 >>> 7) << 0x0001) ^
                (dx5 << 0x0001)) & ((dmk0 >>> 7) ^ (dmk2 <<< 2) ^ dx1 ^ dx5 ^ ((dmk0 >>> 7) << 0x0001))) == 0x0000) &
                ((~((~(dmk0 >>> 7) ^ (dmk2 <<< 2) ^ dx1) & (~(dmk0 >>> 7) ^ dx5))[14:]) == 0b000000000000000)
            dmk2 == dmk2_out
            ((dmk2 <<< 2) ^ dx1) == dx3_out
            ((((dmk2 <<< 2) ^ dx1) <<< 2) ^ dx5) == dx8_out

        """
        return super().pr_one_assertions()

    def weight_assertions(self, ch_weight_variable, assign_weight_variables, truncate=True):
        """Return the weight constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.weight_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff, RXDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> cm = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> cwv = Variable("w", cm.weight_width())
            >>> omvs = [Variable(f"w{i}", om.weight_width()) for i, om in enumerate(cm.assign_outdiff2op_model.values())]
            >>> for a in cm.weight_assertions(cwv, omvs): print(a)  # doctest: +NORMALIZE_WHITESPACE
            w0 == PopCount(~((~(dmk1 >>> 7) ^ dmk2) & (~(dmk1 >>> 7) ^ dx1))[14:])
            w1 == PopCount(~((~(dmk0 >>> 7) ^ (dmk2 <<< 2) ^ dx1) & (~(dmk0 >>> 7) ^ dx5))[14:])
            w2 == 0b0
            w3 == 0b0
            w4 == 0b0
            w == ((0b0 :: w0) + (0b0 :: w1))
            >>> cm = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])  # RXDiff for the fb
            >>> cwv = Variable("w", cm.weight_width())
            >>> omvs = [Variable(f"w{i}", om.weight_width()) for i, om in enumerate(cm.assign_outdiff2op_model.values())]
            >>> for a in cm.weight_assertions(cwv, omvs): print(a)  # doctest: +NORMALIZE_WHITESPACE
            w0 == (((0x0 :: PopCount(((((dmk1 >>> 7)[:1]) ^ (dx1[:1])) | ((dmk2[:1]) ^ (dx1[:1])))[13:])) << 0x03) +
                (Ite(~((dmk1[8]) ^ (dmk2[1]) ^ (dx1[1])), 0x0b, 0x18)))
            w1 == (((0x0 :: PopCount(((((dmk0 >>> 7)[:1]) ^ (dx5[:1])) | ((((dmk2 <<< 2) ^ dx1)[:1]) ^
                (dx5[:1])))[13:])) << 0x03) + (Ite(~((dmk0[8]) ^ (((dmk2 <<< 2) ^ dx1)[1]) ^ (dx5[1])), 0x0b, 0x18)))
            w2 == 0b0
            w3 == 0b0
            w4 == 0b0
            w == (((0b0 :: w0) + (0b0 :: w1))[:3])

        """
        return super().weight_assertions(ch_weight_variable, assign_weight_variables, truncate=truncate)

    def max_weight(self, truncate=True):
        """Return the maximum value the ch. weight variable can achieve in `weight_assertions`.

        See also `abstractproperty.chmodel.ChModel.max_weight`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff, RXDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_cm = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> xor_cm.max_weight(), xor_cm.weight_width(), xor_cm.num_frac_bits()
            (30, 5, 0)
            >>> rx_cm = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])
            >>> rx_cm.num_frac_bits()
            3
            >>> rx_cm.max_weight(), rx_cm.weight_width()
            (34, 6)
            >>> rx_cm.max_weight(truncate=False), rx_cm.weight_width(truncate=False)
            (272, 9)

        """
        return super().max_weight(truncate=truncate)

    def error(self):
        """Return the maximum difference between `weight_assertions` and the exact weight.

        See also `abstractproperty.chmodel.ChModel.error`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff, RXDiff
            >>> from cascada.differential.opmodel import RXModelBvAdd
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"]).error()
            0
            >>> ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"]).error()
            Decimal('2.102441455630541244332995062')
            >>> opt = {RXModelBvAdd: {"precision": 0}}
            >>> ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"], op_model_class2options=opt).error()
            Decimal('2.852441455630541244332995062')

        """
        return super().error()

    def signature(self, sig_type):
        """Return the signature of the characteristic model.

        See also `abstractproperty.chmodel.ChModel.signature`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> ch_model.signature(ChModelSigType.Unique)
            [dmk0, dmk1, dmk2, dx1, dx5]
            >>> ch_model.signature(ChModelSigType.InputOutput)
            [dmk0, dmk1, dmk2, dmk2_out, dx3_out, dx8_out]

        """
        if sig_type == ChModelSigType.Unique:
            # only the input, external and output differences of non-trivial assignments
            # are needed to fully propagate the characteristic
            sig_var = [p.val for p in self.input_prop if p.val not in self.ssa._input_vars_not_used]
            for ext_var, prop in self.external_var2prop.items():
                if not isinstance(prop.val, core.Constant):
                    sig_var.append(ext_var)
            for outprop, op_model in self.assign_outprop2op_model.items():
                if op_model.max_weight() != 0:
                    sig_var.append(outprop.val)
            return sig_var
        else:
            return super().signature(sig_type)


    def split(self, diff_separators):
        """Split into multiple `ChModel` objects given the list of difference separators.

        See also `abstractproperty.chmodel.ChModel.split`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> tuple(ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((dx0, dmk1 >>> 7), (dx1, dx0 + dmk2), (dx2, dmk2 <<< 2), (dx3, dx2 ^ dx1),
            (dx4, dmk0 >>> 7), (dx5, dx4 + dx3), (dx6, dx5 ^ 0x0001), (dx7, dx3 <<< 2), (dx8, dx7 ^ dx6),
            (dmk2_out, Id(dmk2)), (dx3_out, Id(dx3)), (dx8_out, Id(dx8)))
            >>> diff_separators = [ (XorDiff(Variable("dx2", width=16)), XorDiff(Variable("dx3", width=16))), ]
            >>> for cm in ch_model.split(diff_separators): print(cm)  # doctest: +NORMALIZE_WHITESPACE
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_diff=[XorDiff(dmk0), XorDiff(dmk1), XorDiff(dmk2)],
                    output_diff=[XorDiff(dmk0_out), XorDiff(dx3_out), XorDiff(dmk2_out)],
                    assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dmk1 >>> 7), XorDiff(dmk2)])),
                        (XorDiff(dmk0_out), XorModelId(XorDiff(dmk0))),
                        (XorDiff(dx3_out), XorModelId(XorDiff((dmk2 <<< 2) ^ dx1))),
                        (XorDiff(dmk2_out), XorModelId(XorDiff(dmk2)))])
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_diff=[XorDiff(dmk0), XorDiff(dx3), XorDiff(dmk2)],
                    output_diff=[XorDiff(dmk2_out), XorDiff(dx3_out), XorDiff(dx8_out)],
                    assign_outdiff2op_model=[(XorDiff(dx5), XorModelBvAdd([XorDiff(dmk0 >>> 7), XorDiff(dx3)])),
                        (XorDiff(dmk2_out), XorModelId(XorDiff(dmk2))),
                        (XorDiff(dx3_out), XorModelId(XorDiff(dx3))),
                        (XorDiff(dx8_out), XorModelId(XorDiff((dx3 <<< 2) ^ dx5)))])

        """
        # avoid diff_separators=diff_separators (parent might be abstract)
        return super().split(diff_separators)

    def get_round_separators(self):
        """Return the round separators if `func` is a `RoundBasedFunction`.

        See also `abstractproperty.chmodel.ChModel.get_round_separators`.

            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> sub_cms = ch_model.split(ch_model.get_round_separators())
            >>> for cm in sub_cms: print(cm)  # doctest: +NORMALIZE_WHITESPACE
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_diff=[XorDiff(dmk0), XorDiff(dmk1), XorDiff(dmk2)],
                    output_diff=[XorDiff(dmk0_out), XorDiff(dx3_out), XorDiff(dmk2_out)],
                    assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dmk1 >>> 7), XorDiff(dmk2)])),
                        (XorDiff(dmk0_out), XorModelId(XorDiff(dmk0))),
                        (XorDiff(dx3_out), XorModelId(XorDiff((dmk2 <<< 2) ^ dx1))),
                        (XorDiff(dmk2_out), XorModelId(XorDiff(dmk2)))])
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_diff=[XorDiff(dmk0), XorDiff(dx3), XorDiff(dmk2)],
                    output_diff=[XorDiff(dmk2_out), XorDiff(dx3_out), XorDiff(dx8_out)],
                    assign_outdiff2op_model=[(XorDiff(dx5), XorModelBvAdd([XorDiff(dmk0 >>> 7), XorDiff(dx3)])),
                        (XorDiff(dmk2_out), XorModelId(XorDiff(dmk2))),
                        (XorDiff(dx3_out), XorModelId(XorDiff(dx3))),
                        (XorDiff(dx8_out), XorModelId(XorDiff((dx3 <<< 2) ^ dx5)))])

        """
        return super().get_round_separators()


class EncryptionChModel(abstractproperty.chmodel.EncryptionChModel, ChModel):
    """Represent differential characteristic models of encryption functions.

    Given a `Cipher`, an `EncryptionChModel` is a bit-vector model
    (see `ChModel`) of a differential characteristic over
    the `Cipher.encryption` in the single-key setting
    (where the `Cipher.key_schedule` is ignored and round key differences are
    set to zero).

    .. note::

        `EncryptionChModel` only supports `XorDiff` for ``diff_type``
        (and not `RXDiff`).

    See also `differential.characteristic.EncryptionCharacteristic`.

    In an `EncryptionChModel`, the plaintext differences start with the prefix ``"dp"``,
    the intermediate and output differences start with the prefix ``"dx"``,
    and the round key differences (the differences of the external variables
    of the encryption `SSA`) start with the prefix ``"dk"``.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, XorDiff)
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE
        EncryptionChModel(cipher=SpeckCipher_2R, input_diff=[XorDiff(dp0), XorDiff(dp1)],
            output_diff=[XorDiff(dx7_out), XorDiff(dx9_out)],
            external_var2diff=[(dk0, XorDiff(0x0000)), (dk1, XorDiff(0x0000))],
            assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dp0 >>> 7), XorDiff(dp1)])),
                (XorDiff(dx6), XorModelBvAdd([XorDiff(dx1 >>> 7), XorDiff((dp1 <<< 2) ^ dx1)])),
                (XorDiff(dx7_out), XorModelId(XorDiff(dx6))),
                (XorDiff(dx9_out), XorModelId(XorDiff((((dp1 <<< 2) ^ dx1) <<< 2) ^ dx6)))])
        >>> ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[dp0, dp1], output_vars=[dx7_out, dx9_out], external_vars=[dk0, dk1],
            assignments=[(dx0, dp0 >>> 7), (dx1, dx0 + dp1), (dx2, dx1 ^ dk0), (dx3, dp1 <<< 2),
                (dx4, dx3 ^ dx2), (dx5, dx2 >>> 7), (dx6, dx5 + dx4), (dx7, dx6 ^ dk1),
                (dx8, dx4 <<< 2), (dx9, dx8 ^ dx7), (dx7_out, Id(dx7)), (dx9_out, Id(dx9))])

    """
    _prefix = "d"

    def __init__(self, cipher, diff_type, op_model_class2options=None):
        super().__init__(
            cipher, prop_type=diff_type, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_EncryptionCharacteristic_cls(cls):
        from cascada.differential.characteristic import EncryptionCharacteristic as EncryptionCharacteristic_cls
        return EncryptionCharacteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.EncryptionChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import EncryptionChModel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> EncryptionChModel(Speck32, XorDiff).vrepr()
            'EncryptionChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), diff_type=XorDiff)'

        """
        return super().vrepr()


class CipherChModel(abstractproperty.chmodel.CipherChModel):
    """Represent related-key differential characteristic models of ciphers.

    Given a `Cipher`, a `CipherChModel` is a bit-vector model
    of a related-key differential characteristic
    (see `differential.characteristic.CipherCharacteristic`) over the cipher.

    A `CipherChModel` consists of a pair of `ChModel` where one models
    the characteristic over the `Cipher.key_schedule`, and the other one
    models the characteristic over the `Cipher.encryption`.

    In a `CipherChModel`, the master key differences start with the prefix ``"dmk"``,
    the intermediate and output differences of the key schedule start with the prefix ``"dk"``,
    the plaintext differences start with the prefix ``"dp"`` and intermediate and
    output differences of the encryption with the prefix ``"dx"``.

    The round key differences in the encryption characteristic
    (the differences of the external variables of the encryption `SSA`)
    are set to the output differences of the key-schedule characteristic.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff, RXDiff
        >>> from cascada.differential.chmodel import CipherChModel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> xor_ch_model = CipherChModel(Speck32, XorDiff)
        >>> xor_ch_model  # doctest: +NORMALIZE_WHITESPACE
        CipherChModel(ks_ch_model=ChModel(func=SpeckKeySchedule_1R,
            input_diff=[XorDiff(dmk0), XorDiff(dmk1)],
            output_diff=[XorDiff(dmk1_out), XorDiff(dk3_out)],
            assign_outdiff2op_model=[(XorDiff(dk1), XorModelBvAdd([XorDiff(dmk0 >>> 7), XorDiff(dmk1)])),
                (XorDiff(dmk1_out), XorModelId(XorDiff(dmk1))),
                (XorDiff(dk3_out), XorModelId(XorDiff((dmk1 <<< 2) ^ dk1)))]),
        enc_ch_model=ChModel(func=SpeckEncryption_2R,
            input_diff=[XorDiff(dp0), XorDiff(dp1)],
            output_diff=[XorDiff(dx7_out), XorDiff(dx9_out)],
            external_var2diff=[(dmk1_out, XorDiff(dmk1_out)), (dk3_out, XorDiff(dk3_out))],
            assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dp0 >>> 7), XorDiff(dp1)])),
                (XorDiff(dx6), XorModelBvAdd([XorDiff((dx1 ^ dmk1_out) >>> 7), XorDiff((dp1 <<< 2) ^ dx1 ^ dmk1_out)])),
                (XorDiff(dx7_out), XorModelId(XorDiff(dx6 ^ dk3_out))),
                (XorDiff(dx9_out), XorModelId(XorDiff((((dp1 <<< 2) ^ dx1 ^ dmk1_out) <<< 2) ^ dx6 ^ dk3_out)))]))
        >>> xor_ch_model.ks_ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[dmk0, dmk1], output_vars=[dmk1_out, dk3_out],
            assignments=[(dk0, dmk0 >>> 7), (dk1, dk0 + dmk1), (dk2, dmk1 <<< 2),
                (dk3, dk2 ^ dk1), (dmk1_out, Id(dmk1)), (dk3_out, Id(dk3))])
        >>> xor_ch_model.enc_ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[dp0, dp1], output_vars=[dx7_out, dx9_out], external_vars=[dmk1_out, dk3_out],
            assignments=[(dx0, dp0 >>> 7), (dx1, dx0 + dp1), (dx2, dx1 ^ dmk1_out), (dx3, dp1 <<< 2),
                (dx4, dx3 ^ dx2), (dx5, dx2 >>> 7), (dx6, dx5 + dx4), (dx7, dx6 ^ dk3_out), (dx8, dx4 <<< 2),
                (dx9, dx8 ^ dx7), (dx7_out, Id(dx7)), (dx9_out, Id(dx9))])
        >>> rx_ch_model = CipherChModel(Speck32, RXDiff)
        >>> rx_ch_model  # doctest: +NORMALIZE_WHITESPACE
        CipherChModel(ks_ch_model=ChModel(func=SpeckKeySchedule_1R,
            input_diff=[RXDiff(dmk0), RXDiff(dmk1)],
            output_diff=[RXDiff(dmk1_out), RXDiff(dk3_out)],
            assign_outdiff2op_model=[(RXDiff(dk1), RXModelBvAdd([RXDiff(dmk0 >>> 7), RXDiff(dmk1)])),
                (RXDiff(dmk1_out), RXModelId(RXDiff(dmk1))),
                (RXDiff(dk3_out), RXModelId(RXDiff((dmk1 <<< 2) ^ dk1)))]),
            enc_ch_model=ChModel(func=SpeckEncryption_2R,
                input_diff=[RXDiff(dp0), RXDiff(dp1)],
                output_diff=[RXDiff(dx7_out), RXDiff(dx9_out)],
                external_var2diff=[(dmk1_out, RXDiff(dmk1_out)), (dk3_out, RXDiff(dk3_out))],
                assign_outdiff2op_model=[(RXDiff(dx1), RXModelBvAdd([RXDiff(dp0 >>> 7), RXDiff(dp1)])),
                    (RXDiff(dx6), RXModelBvAdd([RXDiff((dx1 ^ dmk1_out) >>> 7), RXDiff((dp1 <<< 2) ^ dx1 ^ dmk1_out)])),
                    (RXDiff(dx7_out), RXModelId(RXDiff(dx6 ^ dk3_out))),
                    (RXDiff(dx9_out), RXModelId(RXDiff((((dp1 <<< 2) ^ dx1 ^ dmk1_out) <<< 2) ^ dx6 ^ dk3_out)))]))

    """
    _ChModel_cls = ChModel
    _prefix = "d"

    def __init__(self, cipher, diff_type, op_model_class2options=None):
        super().__init__(
            cipher, prop_type=diff_type, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_CipherCharacteristic_cls(cls):
        from cascada.differential.characteristic import CipherCharacteristic as CipherCharacteristic_cls
        return CipherCharacteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.CipherChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> CipherChModel(Speck32, XorDiff).vrepr()
            'CipherChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), diff_type=XorDiff)'

        """
        return super().vrepr()

    def signature(self, ch_signature_type):
        """Return the signature of the characteristic model over the cipher.

        See also `abstractproperty.chmodel.CipherChModel.signature`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import CipherChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> rkch_model = CipherChModel(Speck32, XorDiff)
            >>> rkch_model.signature(ChModelSigType.Unique)
            [dmk0, dmk1, dk1, dp0, dp1, dmk1_out, dk3_out, dx1, dx6]
            >>> rkch_model.signature(ChModelSigType.InputOutput)  # doctest:+NORMALIZE_WHITESPACE
            [dmk0, dmk1, dmk1_out, dk3_out, dp0, dp1, dx7_out, dx9_out]

        """
        return super().signature(ch_signature_type)
