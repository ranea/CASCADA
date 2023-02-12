"""Manage models of algebraic characteristics.

.. autosummary::
   :nosignatures:

    ChModel
    EncryptionChModel
    CipherChModel
"""
from cascada.bitvector import core
from cascada.algebraic import value

from cascada import abstractproperty


ChModelSigType = abstractproperty.chmodel.ChModelSigType


class ChModel(abstractproperty.chmodel.ChModel):
    """Represent bit-vector models of algebraic characteristics over bit-vector functions.

    Internally, this class is a subclass of `abstractproperty.chmodel.ChModel`,
    where the `Property` is a `Value` type and the probability of the
    characteristic is the product of the propagation probabilities
    of the `Value` pairs :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})`
    composing the characteristic (see `algebraic.characteristic.Characteristic`).

    If the `SSA` of the `BvFunction` contains external `Variable` objects
    and their `Value` objects are not provided in the initialization
    through ``external_var2val``, these free external variables
    are wrapped in `Value` objects.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.algebraic.value import BitValue, WordValue
        >>> from cascada.algebraic.chmodel import ChModel
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        ChModel(func=SpeckKeySchedule_2R,
                input_val=[BitValue(vmk0), BitValue(vmk1), BitValue(vmk2)],
                output_val=[BitValue(vmk2_out), BitValue(vx3_out), BitValue(vx8_out)],
                assign_outval2op_model=[(BitValue(vx1), BitModelBvAdd([BitValue((vmk1[6:]) :: ...
        >>> ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[vmk0, vmk1, vmk2],
            output_vars=[vmk2_out, vx3_out, vx8_out],
            assignments=[(vx0, vmk1 >>> 7), (vx1, vx0 + vmk2), (vx2, vmk2 <<< 2), (vx3, vx2 ^ vx1),
                         (vx4, vmk0 >>> 7), (vx5, vx4 + vx3), (vx6, vx5 ^ 0x0001), (vx7, vx3 <<< 2), (vx8, vx7 ^ vx6),
                         (vmk2_out, Id(vmk2)), (vx3_out, Id(vx3)), (vx8_out, Id(vx8))])
        >>> ch_model = ChModel(Speck32_KS, WordValue, ["vmk0", "vmk1", "vmk2"])
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE
        ChModel(func=SpeckKeySchedule_2R,
                input_val=[WordValue(vmk0), WordValue(vmk1), WordValue(vmk2)],
                output_val=[WordValue(vmk2_out), WordValue(vx3_out), WordValue(vx8_out)],
                assign_outval2op_model=[(WordValue(vmk2_out), WordModelId(WordValue(vmk2))),
                                        (WordValue(vx3_out), WordModelId(WordValue(vmk2 <<< 2 ^ (vmk1 >>> 7 + vmk2)))),
                                        (WordValue(vx8_out), WordModelId(WordValue((vmk2 <<< 2 ^ (vmk1 >>> 7 + vmk2)) <<< 2 ^
                                            (vmk0 >>> 7 + (vmk2 <<< 2 ^ (vmk1 >>> 7 + vmk2))) ^ 0x0001)))])

    Attributes:
        val_type: alias of
            `abstractproperty.chmodel.ChModel.prop_type`.
        input_val: alias of
            `abstractproperty.chmodel.ChModel.input_prop`.
        output_val: alias of
            `abstractproperty.chmodel.ChModel.output_prop`.
        external_var2val: alias of
            `abstractproperty.chmodel.ChModel.external_var2prop`.
        assign_outval2op_model: alias of
            `abstractproperty.chmodel.ChModel.assign_outprop2op_model`.
        var2val: alias of
            `abstractproperty.chmodel.ChModel.var2prop`.

    """
    _prop_label = "val"  # for str and vrepr

    @property
    def val_type(self):
        return self.prop_type

    @property
    def input_val(self):
        return self.input_prop

    @property
    def output_val(self):
        return self.output_prop

    @property
    def external_var2val(self):
        return self.external_var2prop

    @property
    def assign_outval2op_model(self):
        return self.assign_outprop2op_model

    @property
    def var2val(self):
        return self.var2prop

    @property
    def _input_val_names(self):
        return self._input_prop_names

    def _get_property_missing_external_var(self, val_type, ext_var, external_var2val):
        return val_type(ext_var)

    def __init__(self, func, val_type, input_val_names, prefix="vx",
                 external_var2val=None, op_model_class2options=None):
        super().__init__(
            func, prop_type=val_type, input_prop_names=input_val_names, prefix=prefix,
            external_var2prop=external_var2val, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_Characteristic_cls(cls):
        from cascada.algebraic.characteristic import Characteristic as Characteristic_cls
        return Characteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.ChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> ch_model.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), val_type=BitValue,
                     input_val_names=['vmk0', 'vmk1', 'vmk2'], prefix='vx')"

        """
        return super().vrepr()

    def validity_assertions(self):
        """Return the validity constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.validity_assertions`.

            >>> from functools import reduce
            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> assertions = ch_model.validity_assertions()
            >>> for a in assertions: print(a)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            (0b0 == ((vmk1[7]) ^ (vmk2[0]) ^ (vx1[0]))) & ...
            (0b0 == ((vmk0[7]) ^ (vmk2[14]) ^ (vx1[0]) ^ (vx5[0]))) & ...
            ((vmk2[0]) == (vmk2_out[0])) & ((vmk2[1]) == (vmk2_out[1])) & ...
            (((vmk2[14]) ^ (vx1[0])) == (vx3_out[0])) & (((vmk2[15]) ^ ...
            (((vmk2[12]) ^ (vx1[14]) ^ ~(vx5[0])) == (vx8_out[0])) & ...

        """
        return super().validity_assertions()

    def pr_one_assertions(self):
        """Return the probability-one constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.pr_one_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> for a in ch_model.pr_one_assertions(): print(a)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            (0b0 == ((vmk1[7]) ^ (vmk2[0]) ^ (vx1[0]))) & (0b0 == ((vmk1[8]) ^ (vmk2[1]) ^ (vx1[1]) ^ ...
            (0b0 == ((vmk0[7]) ^ (vmk2[14]) ^ (vx1[0]) ^ (vx5[0]))) & (0b0 == ((vmk0[8]) ^ (vmk2[15]) ^ ...
            ((vmk2[0]) == (vmk2_out[0])) & ((vmk2[1]) == (vmk2_out[1])) & ((vmk2[2]) == (vmk2_out[2])) & ...
            (((vmk2[14]) ^ (vx1[0])) == (vx3_out[0])) & (((vmk2[15]) ^ (vx1[1])) == (vx3_out[1])) & ...
            (((vmk2[12]) ^ (vx1[14]) ^ ~(vx5[0])) == (vx8_out[0])) & (((vmk2[13]) ^ (vx1[15]) ^ ...

        """
        return super().pr_one_assertions()

    def weight_assertions(self, ch_weight_variable, assign_weight_variables, truncate=True):
        """Return the weight constraint as a list of assertions.

        See also `abstractproperty.chmodel.ChModel.weight_assertions`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue, WordValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> cm = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> cwv = Variable("w", cm.weight_width())
            >>> omvs = [Variable(f"w{i}", om.weight_width()) for i, om in enumerate(cm.assign_outval2op_model.values())]
            >>> for a in cm.weight_assertions(cwv, omvs): print(a)  # doctest: +NORMALIZE_WHITESPACE
            w0 == 0b0
            w1 == 0b0
            w2 == 0b0
            w3 == 0b0
            w4 == 0b0
            w == 0b0
            >>> cm = ChModel(Speck32_KS, WordValue, ["vmk0", "vmk1", "vmk2"])
            >>> cwv = Variable("w", cm.weight_width())
            >>> omvs = [Variable(f"w{i}", om.weight_width()) for i, om in enumerate(cm.assign_outval2op_model.values())]
            >>> for a in cm.weight_assertions(cwv, omvs): print(a)  # doctest: +NORMALIZE_WHITESPACE
            w0 == 0b0
            w1 == 0b0
            w2 == 0b0
            w == 0b0

        """
        return super().weight_assertions(ch_weight_variable, assign_weight_variables, truncate=truncate)

    def max_weight(self, truncate=True):
        """Return the maximum value the ch. weight variable can achieve in `weight_assertions`.

        See also `abstractproperty.chmodel.ChModel.max_weight`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue, WordValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> cm = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> cm.max_weight(), cm.weight_width(), cm.num_frac_bits()
            (0, 1, 0)
            >>> cm = ChModel(Speck32_KS, WordValue, ["vmk0", "vmk1", "vmk2"])
            >>> cm.num_frac_bits()
            0
            >>> cm.max_weight(), cm.weight_width()
            (0, 1)
            >>> cm.max_weight(truncate=False), cm.weight_width(truncate=False)
            (0, 1)

        """
        return super().max_weight(truncate=truncate)

    def error(self):
        """Return the maximum difference between `weight_assertions` and the exact weight.

        See also `abstractproperty.chmodel.ChModel.error`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue, WordValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"]).error()
            0
            >>> ChModel(Speck32_KS, WordValue, ["vmk0", "vmk1", "vmk2"]).error()
            0

        """
        return super().error()

    def signature(self, sig_type):
        """Return the signature of the characteristic model.

        See also `abstractproperty.chmodel.ChModel.signature`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> ch_model.signature(ChModelSigType.Unique)
            [vmk0, vmk1, vmk2]
            >>> ch_model.signature(ChModelSigType.InputOutput)
            [vmk0, vmk1, vmk2, vmk2_out, vx3_out, vx8_out]

        """
        if sig_type == ChModelSigType.Unique:
            # only the input and external values are needed to fully propagate the characteristic
            sig_var = [p.val for p in self.input_prop if p.val not in self.ssa._input_vars_not_used]
            for ext_var, prop in self.external_var2prop.items():
                if not isinstance(prop.val, core.Constant):
                    sig_var.append(ext_var)
            return sig_var
        else:
            return super().signature(sig_type)

    def split(self, val_separators):
        """Split into multiple `ChModel` objects given the list of value separators.

        See also `abstractproperty.chmodel.ChModel.split`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> tuple(ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((vx0, vmk1 >>> 7), (vx1, vx0 + vmk2), (vx2, vmk2 <<< 2), (vx3, vx2 ^ vx1),
            (vx4, vmk0 >>> 7), (vx5, vx4 + vx3), (vx6, vx5 ^ 0x0001), (vx7, vx3 <<< 2), (vx8, vx7 ^ vx6),
            (vmk2_out, Id(vmk2)), (vx3_out, Id(vx3)), (vx8_out, Id(vx8)))
            >>> val_separators = [ (BitValue(Variable("vx2", width=16)), BitValue(Variable("vx3", width=16))), ]
            >>> for cm in ch_model.split(val_separators): print(cm)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_val=[BitValue(vmk0), BitValue(vmk1), BitValue(vmk2)],
                    output_val=[BitValue(vmk0_out), BitValue(vx3_out), BitValue(vmk2_out)],
                    assign_outval2op_model=[(BitValue(vx1), BitModelBvAdd([BitValue((vmk1[6:]) :: ...
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_val=[BitValue(vmk0), BitValue(vx3), BitValue(vmk2)],
                    output_val=[BitValue(vmk2_out), BitValue(vx3_out), BitValue(vx8_out)],
                    assign_outval2op_model=[(BitValue(vx5), BitModelBvAdd([BitValue((vmk0[6:]) :: ...

        """
        return super().split(val_separators)

    def get_round_separators(self):
        """Return the round separators if `func` is a `RoundBasedFunction`.

        See also `abstractproperty.chmodel.ChModel.get_round_separators`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.bitvector.operation import BvAnd
            >>> from cascada.algebraic.value import BitValue, WordValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> sub_cms = ch_model.split(ch_model.get_round_separators())
            >>> for cm in sub_cms: print(cm)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            ChModel(func=SpeckKeySchedule_2R_0S,
                    input_val=[BitValue(vmk0), BitValue(vmk1), BitValue(vmk2)],
                    output_val=[BitValue(vmk0_out), BitValue(vx3_out), BitValue(vmk2_out)],
                    assign_outval2op_model=[(BitValue(vx1), BitModelBvAdd([BitValue((vmk1[6:]) :: ...
            ChModel(func=SpeckKeySchedule_2R_1S,
                    input_val=[BitValue(vmk0), BitValue(vx3), BitValue(vmk2)],
                    output_val=[BitValue(vmk2_out), BitValue(vx3_out), BitValue(vx8_out)],
                    assign_outval2op_model=[(BitValue(vx5), BitModelBvAdd([BitValue((vmk0[6:]) :: ...

        """
        return super().get_round_separators()


class EncryptionChModel(abstractproperty.chmodel.EncryptionChModel, ChModel):
    """Represent algebraic characteristic models of encryption functions.

    Given a `Cipher`, an `EncryptionChModel` is a bit-vector model
    (see `ChModel`) of an algebraic characteristic over
    the `Cipher.encryption` in the single-key setting
    (where the `Cipher.key_schedule` is ignored and round key values
    are set to `Value` objects containing `Variable` objects).

    See also `algebraic.characteristic.EncryptionCharacteristic`.

    In an `EncryptionChModel`, the plaintext values start with the prefix ``"vp"``,
    the intermediate and output values start with the prefix ``"vx"``,
    and the round key variables start with the prefix ``"vmk"``.

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.algebraic.value import BitValue, WordValue
        >>> from cascada.algebraic.chmodel import EncryptionChModel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, BitValue)
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        EncryptionChModel(cipher=SpeckCipher_2R,
            input_val=[BitValue(vp0), BitValue(vp1)],
            output_val=[BitValue(vx7_out), BitValue(vx9_out)],
            external_var2val=[(vk0, BitValue(vk0)), (vk1, BitValue(vk1))],
            assign_outval2op_model=[(BitValue(vx1), BitModelBvAdd([BitValue((vp0[6:]) :: ...
        >>> ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[vp0, vp1], output_vars=[vx7_out, vx9_out], external_vars=[vk0, vk1],
            assignments=[(vx0, vp0 >>> 7), (vx1, vx0 + vp1), (vx2, vx1 ^ vk0), (vx3, vp1 <<< 2), (vx4, vx3 ^ vx2),
                (vx5, vx2 >>> 7), (vx6, vx5 + vx4), (vx7, vx6 ^ vk1), (vx8, vx4 <<< 2), (vx9, vx8 ^ vx7),
                (vx7_out, Id(vx7)), (vx9_out, Id(vx9))])

    """
    _prefix = "v"

    def __init__(self, cipher, val_type, op_model_class2options=None, round_keys_prefix=None):
        super().__init__(
            cipher, prop_type=val_type, op_model_class2options=op_model_class2options, round_keys_prefix=round_keys_prefix
        )

    @classmethod
    def _get_EncryptionCharacteristic_cls(cls):
        from cascada.algebraic.characteristic import EncryptionCharacteristic as EncryptionCharacteristic_cls
        return EncryptionCharacteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.EncryptionChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.algebraic.value import BitValue, WordValue
            >>> from cascada.algebraic.chmodel import EncryptionChModel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> EncryptionChModel(Speck32, BitValue).vrepr()
            'EncryptionChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), val_type=BitValue)'

        """
        return super().vrepr()


class CipherChModel(abstractproperty.chmodel.CipherChModel):
    """Represent algebraic characteristic models of ciphers.

    Given a `Cipher`, a `CipherChModel` is a bit-vector model
    of an algebraic characteristic
    (see `algebraic.characteristic.CipherCharacteristic`) over the cipher.

    A `CipherChModel` consists of a pair of `ChModel` where one models
    the characteristic over the `Cipher.key_schedule`, and the other one
    models the characteristic over the `Cipher.encryption`.

    In a `CipherChModel`, the master key values start with the prefix ``"vmk"``,
    the intermediate and output values of the key schedule start with the prefix ``"vk"``,
    the plaintext values start with the prefix ``"vp"`` and intermediate and
    output values of the encryption with the prefix ``"vx"``.

    The round key values in the encryption characteristic
    (the values of the external variables of the encryption `SSA`)
    are set to the output values of the key-schedule characteristic.

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.algebraic.value import BitValue, WordValue
        >>> from cascada.algebraic.chmodel import CipherChModel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = CipherChModel(Speck32, BitValue)
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        CipherChModel(ks_ch_model=ChModel(func=SpeckKeySchedule_1R,
            input_val=[BitValue(vmk0), BitValue(vmk1)],
            output_val=[BitValue(vmk1_out), BitValue(vk3_out)],
            assign_outval2op_model=[(BitValue(vk1), BitModelBvAdd([BitValue((vmk0[6:]) :: ...
        enc_ch_model=ChModel(func=SpeckEncryption_2R,
            input_val=[BitValue(vp0), BitValue(vp1)],
            output_val=[BitValue(vx7_out), BitValue(vx9_out)],
            external_var2val=[(vmk1_out, BitValue(vmk1_out)), (vk3_out, BitValue(vk3_out))],
            assign_outval2op_model=[(BitValue(vx1), BitModelBvAdd([BitValue((vp0[6:]) :: ...
        >>> ch_model.ks_ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[vmk0, vmk1], output_vars=[vmk1_out, vk3_out],
            assignments=[(vk0, vmk0 >>> 7), (vk1, vk0 + vmk1), (vk2, vmk1 <<< 2), (vk3, vk2 ^ vk1),
                (vmk1_out, Id(vmk1)), (vk3_out, Id(vk3))])
        >>> ch_model.enc_ch_model.ssa  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[vp0, vp1], output_vars=[vx7_out, vx9_out], external_vars=[vmk1_out, vk3_out],
            assignments=[(vx0, vp0 >>> 7), (vx1, vx0 + vp1), (vx2, vx1 ^ vmk1_out), (vx3, vp1 <<< 2), (vx4, vx3 ^ vx2),
                (vx5, vx2 >>> 7), (vx6, vx5 + vx4), (vx7, vx6 ^ vk3_out), (vx8, vx4 <<< 2), (vx9, vx8 ^ vx7),
                (vx7_out, Id(vx7)), (vx9_out, Id(vx9))])
        >>> ch_model = CipherChModel(Speck32, WordValue)
        >>> ch_model  # doctest: +NORMALIZE_WHITESPACE
        CipherChModel(ks_ch_model=ChModel(func=SpeckKeySchedule_1R,
            input_val=[WordValue(vmk0), WordValue(vmk1)],
            output_val=[WordValue(vmk1_out), WordValue(vk3_out)],
            assign_outval2op_model=[(WordValue(vmk1_out), WordModelId(WordValue(vmk1))),
                (WordValue(vk3_out), WordModelId(WordValue(vmk1 <<< 2 ^ (vmk0 >>> 7 + vmk1))))]),
        enc_ch_model=ChModel(func=SpeckEncryption_2R,
            input_val=[WordValue(vp0), WordValue(vp1)],
            output_val=[WordValue(vx7_out), WordValue(vx9_out)],
            external_var2val=[(vmk1_out, WordValue(vmk1_out)), (vk3_out, WordValue(vk3_out))],
            assign_outval2op_model=[(WordValue(vx7_out), WordModelId(WordValue((((vp0 >>> 7 + vp1) ^ vmk1_out) >>>
                    7 + (vp1 <<< 2 ^ (vp0 >>> 7 + vp1) ^ vmk1_out)) ^ vk3_out))),
                (WordValue(vx9_out), WordModelId(WordValue((vp1 <<< 2 ^ (vp0 >>> 7 + vp1) ^ vmk1_out) <<< 2 ^
                    (((vp0 >>> 7 + vp1) ^ vmk1_out) >>> 7 + (vp1 <<< 2 ^ (vp0 >>> 7 + vp1) ^ vmk1_out)) ^ vk3_out)))]))

    """
    _ChModel_cls = ChModel
    _prefix = "v"

    def __init__(self, cipher, val_type, op_model_class2options=None):
        super().__init__(
            cipher, prop_type=val_type, op_model_class2options=op_model_class2options
        )

    @classmethod
    def _get_CipherCharacteristic_cls(cls):
        from cascada.algebraic.characteristic import CipherCharacteristic as CipherCharacteristic_cls
        return CipherCharacteristic_cls

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.chmodel.CipherChModel.vrepr`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import CipherChModel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> CipherChModel(Speck32, BitValue).vrepr()
            'CipherChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), val_type=BitValue)'

        """
        return super().vrepr()

    def signature(self, ch_signature_type):
        """Return the signature of the characteristic model over the cipher.

        See also `abstractproperty.chmodel.CipherChModel.signature`.

            >>> from cascada.bitvector.core import Variable
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import CipherChModel, ChModelSigType
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = CipherChModel(Speck32, BitValue)
            >>> ch_model.signature(ChModelSigType.Unique)
            [vmk0, vmk1, vp0, vp1, vmk1_out, vk3_out]
            >>> ch_model.signature(ChModelSigType.InputOutput)  # doctest:+NORMALIZE_WHITESPACE
            [vmk0, vmk1, vmk1_out, vk3_out, vp0, vp1, vx7_out, vx9_out]

        """
        return super().signature(ch_signature_type)
