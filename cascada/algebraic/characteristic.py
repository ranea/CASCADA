"""Manipulate non-symbolic algebraic characteristics.

.. autosummary::
   :nosignatures:

    Characteristic
    EncryptionCharacteristic
    CipherCharacteristic
"""
import functools

from cascada.bitvector import core
from cascada.algebraic import chmodel as cascada_chmodel

from cascada import abstractproperty


zip = functools.partial(zip, strict=True)
EmpiricalWeightData = abstractproperty.characteristic.EmpiricalWeightData


class Characteristic(abstractproperty.characteristic.Characteristic):
    """Represent algebraic characteristics over bit-vector functions.

    Internally, this class is a subclass of
    `abstractproperty.characteristic.Characteristic`,
    where the `Property` is a `Value` type.

    As mention in `abstractproperty.characteristic.Characteristic`,
    the characteristic probability is defined as the product of the propagation
    probability of the `Value` pairs
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` over :math:`f_i`.
    In other words, the characteristic probability is 0 or 1 depending on
    whether :math:`f(\Delta_{x_{i+1}}) = \Delta_{x_{i}}` for all the bit-vector
    `Operation` objects :math:`f_i` composing the `BvFunction` :math:`f`.

    .. note::
        Algebraic characteristics only support the default value
        ``True`` for the optional argument ``is_valid``.

    ::

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateLeft, RotateRight
        >>> from cascada.algebraic.value import BitValue, WordValue
        >>> from cascada.algebraic.chmodel import ChModel
        >>> from cascada.algebraic.characteristic import Characteristic
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> ch_model = ChModel(Speck32_KS, BitValue, ["dmk0", "dmk1", "dmk2"])
        >>> # assign_outval [vx1, vx5, vmk2_out, vx3_out, vx8_out]
        >>> zv = core.Constant(0, width=16)
        >>> mk0 = RotateLeft(Constant(1, width=16), 7)  # mk0 >>> 7 == 0b0···01,
        >>> x5 = RotateRight(mk0, 7)  # vx5 = mk0 >>> 7
        >>> x8 = x5 ^  0x0001 # vx8 = x5 ^ 0x0001
        >>> ch = Characteristic([mk0, zv, zv], [zv, zv, x8], [zv, x5, zv, zv, x8], ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0, 0],
            input_val=[0x0080, 0x0000, 0x0000], output_val=[0x0000, 0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0001, 0x0000, 0x0000, 0x0000])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outval2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0001), BitModelBvAdd([BitValue(0x0001), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000)))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000)))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000))))]
        >>> ch_model = ChModel(Speck32_KS, WordValue, ["vmk0", "vmk1", "vmk2"])
        >>> # assign_outval [vmk2_out, vx3_out, vx8_out]
        >>> ch = Characteristic([mk0, zv, zv], [zv, zv, x8], [zv, zv, x8], ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
            input_val=[0x0080, 0x0000, 0x0000], output_val=[0x0000, 0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0000, 0x0000])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outval2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (WordValue(0x0000), WordModelId(WordValue(0x0000)))),
        (Decimal('0'), (WordValue(0x0000), WordModelId(WordValue(0x0000)))),
        (Decimal('0'), (WordValue(0x0000), WordModelId(WordValue(0x0000))))]

    Attributes:
        input_val: alias of
            `abstractproperty.characteristic.Characteristic.input_prop`.
        output_val: alias of
            `abstractproperty.characteristic.Characteristic.output_prop`.
        external_vals: alias of
            `abstractproperty.characteristic.Characteristic.external_props`.
        tuple_assign_outval2op_model:  alias of
            `abstractproperty.characteristic.Characteristic.tuple_assign_outprop2op_model`.
        free_vals: alias of
            `abstractproperty.characteristic.Characteristic.free_props`.
        var_val2ct_val: alias of
            `abstractproperty.characteristic.Characteristic.var_prop2ct_prop`.

    """
    _prop_label = "val"  # for str and vrepr

    @property
    def input_val(self):
        return self.input_prop

    @property
    def output_val(self):
        return self.output_prop

    @property
    def external_vals(self):
        return self.external_props

    @property
    def tuple_assign_outval2op_model(self):
        return self.tuple_assign_outprop2op_model

    @property
    def free_vals(self):
        return self.free_props

    @property
    def var_val2ct_val(self):
        return self.var_prop2ct_prop

    def __init__(self, input_val, output_val, assign_outval_list,
                 ch_model, external_vals=None, free_vals=None,
                 empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        if is_valid is not True:
            raise ValueError("invalid algebraic characteristics are not supported")
        if ch_model.ssa.external_vars:
            assert external_vals is not None
        super().__init__(
            input_prop=input_val, output_prop=output_val, assign_outprop_list=assign_outval_list,
            ch_model=ch_model, external_props=external_vals, free_props=free_vals,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid
        )
        # quick check
        input_vals = [iv.val for iv in self.input_val]
        if self.external_vals is not None:
            external_var2val = {k: v.val for k, v in zip(ch_model.ssa.external_vars, self.external_vals)}
        else:
            external_var2val = None
        output_vals = self.ch_model.ssa.eval(input_vals=input_vals, external_var2val=external_var2val)
        expected_output_vals = tuple(ov.val for ov in self.output_val)
        if tuple(output_vals) != expected_output_vals:
            raise ValueError(f"SSA(input_vals={input_vals}, external_var2val={external_vals}) = "
                             f"{output_vals} != {expected_output_vals} = output_vals")

    def vrepr(self, ignore_external_vals=False):
        """Return an executable string representation.

        See also `abstractproperty.characteristic.Characteristic.vrepr`.

            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateLeft, RotateRight
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> zv = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 7)
            >>> x5 = RotateRight(mk0, 7)
            >>> x8 = x5 ^  0x0001
            >>> ch = Characteristic([mk0, zv, zv], [zv, zv, x8], [zv, x5, zv, zv, x8], ch_model)
            >>> ch.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "Characteristic(input_val=[Constant(0x0080, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
                output_val=[Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
                assign_outval_list=[Constant(0x0000, width=16), Constant(0x0001, width=16),
                    Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
                ch_model=ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), val_type=BitValue,
                    input_val_names=['vmk0', 'vmk1', 'vmk2'], prefix='vx'))"
            >>> ch.srepr()
            'Ch(w=0, id=0080 0000 0000, od=0000 0000 0000)'

        """
        return super().vrepr(ignore_external_vals)

    def split(self, val_separators):
        """Split into multiple `Characteristic` objects given the list of value separators.

        See also `abstractproperty.characteristic.Characteristic.split`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.operation import RotateLeft, RotateRight
            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.algebraic.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> tuple(ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((vx0, vmk1 >>> 7), (vx1, vx0 + vmk2), (vx2, vmk2 <<< 2), (vx3, vx2 ^ vx1),
            (vx4, vmk0 >>> 7), (vx5, vx4 + vx3), (vx6, vx5 ^ 0x0001), (vx7, vx3 <<< 2), (vx8, vx7 ^ vx6),
            (vmk2_out, Id(vmk2)), (vx3_out, Id(vx3)), (vx8_out, Id(vx8)))
            >>> val_separators = [ (BitValue(Variable("vx2", width=16)), BitValue(Variable("vx3", width=16))), ]
            >>> zv = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 7)
            >>> x5 = RotateRight(mk0, 7)
            >>> x8 = x5 ^  0x0001
            >>> ch = Characteristic([mk0, zv, zv], [zv, zv, x8], [zv, x5, zv, zv, x8], ch_model)
            >>> for ch in ch.split(val_separators): print(ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
                           input_val=[0x0080, 0x0000, 0x0000], output_val=[0x0080, 0x0000, 0x0000],
                           assign_outval_list=[0x0000, 0x0080, 0x0000, 0x0000])
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
                           input_val=[0x0080, 0x0000, 0x0000], output_val=[0x0000, 0x0000, 0x0000],
                           assign_outval_list=[0x0001, 0x0000, 0x0000, 0x0000])

        """
        return super().split(val_separators)

    def compute_empirical_ch_weight(self, num_input_samples=None, num_external_samples=None,
                                    split_by_max_weight=None, split_by_rounds=False,
                                    seed=None, C_code=False, num_parallel_processes=None):
        """Compute and store the empirical weight.

        This method simply stores `ch_weight` in `empirical_ch_weight`
        and a trivial `EmpiricalWeightData` object in `empirical_data_list`.

        See also
        `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight`.
        """
        ew_list = [self.ch_weight]
        data = EmpiricalWeightData(
            aux_weights=ew_list, num_input_samples=0,
            num_external_samples=0, seed=seed, C_code=C_code)
        self.empirical_ch_weight = ew_list[0]
        self.empirical_data_list = [data]

    @classmethod
    def _sample_outprop_opmodel(cls, prop_type, ct_op_model, outprop_width, get_random_bv):
        ct_outprop = prop_type(ct_op_model.op(*[iv.val for iv in ct_op_model.input_val]))
        assert bool(ct_op_model.validity_constraint(ct_outprop))
        return ct_outprop

    @classmethod
    def random(cls, ch_model, seed, external_vals=None):
        """Return a random `Characteristic` with given `algebraic.chmodel.ChModel`.

        See also `abstractproperty.characteristic.Characteristic.random`.

            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import ChModel
            >>> from cascada.algebraic.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> ch_model = ChModel(Speck32_KS, BitValue, ["vmk0", "vmk1", "vmk2"])
            >>> Characteristic.random(ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0, 0],
                input_val=[0xc53e, 0xd755, 0x14ba], output_val=[0x14ba, 0x9280, 0x5a09],
                assign_outval_list=[0xc068, 0x100a, 0x14ba, 0x9280, 0x5a09])

        """
        return super().random(ch_model, seed, external_vals)


class EncryptionCharacteristic(abstractproperty.characteristic.EncryptionCharacteristic, Characteristic):
    """Represent algebraic characteristics over encryption functions.

    Given a `Cipher`, an `EncryptionCharacteristic` is an
    algebraic characteristic  (see `Characteristic`) over
    the `Cipher.encryption` in the single-key setting
    (where the `Cipher.key_schedule` is ignored and round key values
    are given as constant `Value`).

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.algebraic.value import BitValue
        >>> from cascada.algebraic.chmodel import EncryptionChModel
        >>> from cascada.algebraic.characteristic import EncryptionCharacteristic
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, BitValue)
        >>> zv = core.Constant(0, width=16)
        >>> vk1 = core.Constant(1, 16)
        >>> ch = EncryptionCharacteristic([zv, zv], [vk1, vk1], [zv, zv, vk1, vk1], ch_model, [zv, vk1])
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        EncryptionCharacteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
            input_val=[0x0000, 0x0000], output_val=[0x0001, 0x0001], external_vals=[0x0000, 0x0001],
            assign_outval_list=[0x0000, 0x0000, 0x0001, 0x0001])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outval2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0001), BitModelId(BitValue(0x0001)))),
         (Decimal('0'), (BitValue(0x0001), BitModelId(BitValue(0x0001))))]

    """

    def __init__(self, input_val, output_val, assign_outval_list, ch_model, external_vals=None,
                 free_vals=None, empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        assert isinstance(ch_model, cascada_chmodel.EncryptionChModel)
        if ch_model.ssa.external_vars:
            assert external_vals is not None
        super().__init__(
            input_val, output_val, assign_outval_list, ch_model, external_props=external_vals, free_props=free_vals,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid)

    @classmethod
    def random(cls, ch_model, seed):
        """Return a random `EncryptionCharacteristic` with given `algebraic.chmodel.EncryptionChModel`.

        See also `abstractproperty.characteristic.EncryptionCharacteristic.random`.

            >>> from cascada.algebraic.value import BitValue
            >>> from cascada.algebraic.chmodel import EncryptionChModel
            >>> from cascada.algebraic.characteristic import EncryptionCharacteristic
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = EncryptionChModel(Speck32, BitValue)
            >>> EncryptionCharacteristic.random(ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            EncryptionCharacteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
                input_val=[0xc53e, 0xd755], output_val=[0x6322, 0x17ea], external_vals=[0x14ba, 0x8490],
                assign_outval_list=[0x54df, 0xe7b2, 0x6322, 0x17ea])

        """
        return super().random(ch_model, seed)


class CipherCharacteristic(abstractproperty.characteristic.CipherCharacteristic):
    """Represent algebraic characteristics over ciphers.

    A `CipherCharacteristic` is given by one algebraic `Characteristic` over
    the `Cipher.key_schedule` and another algebraic `Characteristic`
    over the `Cipher.encryption`, where the round key values in the encryption
    characteristic (the values of the external variables of the encryption `SSA`)
    are set to the output values of the key-schedule characteristic.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateRight, RotateLeft
        >>> from cascada.algebraic.value import BitValue, WordValue
        >>> from cascada.algebraic.chmodel import CipherChModel
        >>> from cascada.algebraic.characteristic import CipherCharacteristic
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = CipherChModel(Speck32, BitValue)
        >>> zv = core.Constant(0, width=16)
        >>> ch = CipherCharacteristic(
        ...     [zv, zv], [zv, zv], [zv, zv, zv],
        ...     [zv, zv], [zv, zv], [zv, zv, zv, zv], ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
            input_val=[0x0000, 0x0000], output_val=[0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0000, 0x0000]),
        enc_characteristic=Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
            input_val=[0x0000, 0x0000], output_val=[0x0000, 0x0000], external_vals=[0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0000, 0x0000, 0x0000]))
        >>> ch.vrepr()  # doctest: +NORMALIZE_WHITESPACE
        'CipherCharacteristic(ks_input_val=[Constant(0x0000, width=16), Constant(0x0000, width=16)],
            ks_output_val=[Constant(0x0000, width=16), Constant(0x0000, width=16)],
            ks_assign_outval_list=[Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
            enc_input_val=[Constant(0x0000, width=16), Constant(0x0000, width=16)],
            enc_output_val=[Constant(0x0000, width=16), Constant(0x0000, width=16)],
            enc_assign_outval_list=[Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
            cipher_ch_model=CipherChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), val_type=BitValue))'
        >>> ks_ch, enc_ch = ch.ks_characteristic, ch.enc_characteristic
        >>> list(zip(ks_ch.assignment_weights, ks_ch.tuple_assign_outval2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000)))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000))))]
        >>> list(zip(enc_ch.assignment_weights, enc_ch.tuple_assign_outval2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0000), BitModelBvAdd([BitValue(0x0000), BitValue(0x0000)]))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000)))),
         (Decimal('0'), (BitValue(0x0000), BitModelId(BitValue(0x0000))))]
        >>> cipher_ch_model = CipherChModel(Speck32, WordValue)
        >>> ch = CipherCharacteristic(
        ...     [zv, zv], [zv, zv], [zv, zv, zv],
        ...     [zv, zv], [zv, zv], [zv, zv, zv, zv], ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=0, assignment_weights=[0, 0, 0],
            input_val=[0x0000, 0x0000], output_val=[0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0000, 0x0000]),
        enc_characteristic=Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
            input_val=[0x0000, 0x0000], output_val=[0x0000, 0x0000], external_vals=[0x0000, 0x0000],
            assign_outval_list=[0x0000, 0x0000, 0x0000, 0x0000]))
        >>> ch.srepr()  # doctest: +NORMALIZE_WHITESPACE
        'Ch(ks_ch=Ch(w=0, id=0000 0000, od=0000 0000), enc_ch=Ch(w=0, id=0000 0000, od=0000 0000))'

    """
    _Characteristic_cls = Characteristic

    def __init__(
            self, ks_input_val, ks_output_val, ks_assign_outval_list,
            enc_input_val, enc_output_val, enc_assign_outval_list,
            cipher_ch_model, ks_free_vals=None, enc_free_vals=None,
            ks_empirical_ch_weight=None, ks_empirical_data_list=None,
            enc_empirical_ch_weight=None, enc_empirical_data_list=None,
            ks_is_valid=True, enc_is_valid=True
    ):
        # avoid *_props=*_props (super might not abstract)
        super().__init__(
            ks_input_val, ks_output_val, ks_assign_outval_list,
            enc_input_val, enc_output_val, enc_assign_outval_list,
            cipher_ch_model, ks_free_vals, enc_free_vals,
            ks_empirical_ch_weight=ks_empirical_ch_weight, ks_empirical_data_list=ks_empirical_data_list,
            enc_empirical_ch_weight=enc_empirical_ch_weight, enc_empirical_data_list=enc_empirical_data_list,
            ks_is_valid=ks_is_valid, enc_is_valid=enc_is_valid
        )

    @classmethod
    def random(cls, cipher_ch_model, seed):
        """Return a random `CipherCharacteristic` with given `algebraic.chmodel.CipherChModel`.

        See also `abstractproperty.characteristic.CipherCharacteristic.random`.

            >>> from cascada.algebraic.value import WordValue
            >>> from cascada.algebraic.chmodel import CipherChModel
            >>> from cascada.algebraic.characteristic import CipherCharacteristic
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = CipherChModel(Speck32, WordValue)
            >>> CipherCharacteristic.random(ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=0,
                assignment_weights=[0, 0],
                input_val=[0xc53e, 0xd755], output_val=[0xd755, 0x0988],
                assign_outval_list=[0xd755, 0x0988]),
            enc_characteristic=Characteristic(ch_weight=0,
                assignment_weights=[0, 0],
                input_val=[0xe53f, 0x11fb], output_val=[0x2b81, 0x2e71], external_vals=[0xd755, 0x0988],
                assign_outval_list=[0x2b81, 0x2e71]))

        """
        return super().random(cipher_ch_model, seed)
