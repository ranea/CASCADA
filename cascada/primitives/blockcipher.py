"""Represent encryption functions and block ciphers."""
import collections
import functools
import warnings

from cascada.bitvector import core
from cascada.bitvector import ssa as cascada_ssa


class Encryption(object):
    """Represent encryption functions of block ciphers.

    An encryption function is a bit-vector function (see `BvFunction`) that takes
    the plaintext as input and returns the ciphertext for some fixed master key.

    An encryption function together with a key-schedule function forms a `Cipher`.

    The tuple of round keys (the outputs of the associated key-schedule function)
    are accessible through the class attribute `round_keys`.

    The encryption function is not responsible for the creation of the round keys,
    and the ``eval`` method of `Encryption` can assume that `round_keys` is a tuple
    of bit-vectors with the bit-sizes given by ``Cipher.key_schedule.output_widths``.

    .. note::

        The ``eval`` method of `Encryption` is meant to be called from the ``eval``
        method of `Cipher`, and in the latter ``eval`` the round keys for the
        given master key are automatically created and temporarily stored in
        `round_keys` (until the end of ``eval``).

    To use the encryption function as a regular `BvFunction`, `round_keys`
    must be filled with a tuple of `Constant` or `Variable` objects.
    This can be easily done by `Cipher.set_round_keys`
    If `round_keys` contains `Variable` objects, the round keys are represented
    in the `SSA` of the encryption function as external variables.

    To define an encryption function for a new `Cipher`, you must
    create a new class with two parent/base classes: `Encryption` and
    either `BvFunction` or `RoundBasedFunction`.

    .. note::

        `Encryption` must always be the first parent/base class so that
        the `Encryption` methods can override the `BvFunction` or
        `RoundBasedFunction` methods.

    Attributes:
        round_keys: the round keys as a tuple of `Constant` or `Variable` objects

    .. Implementation details:

        Encryption doesn't implement the round keys as an eval argument since
        it does not have access to `Cipher.key_schedule.output_widths`.

    """
    round_keys = None

    def __new__(cls, *args, **options):
        if cls.round_keys is None:
            raise ValueError(f"{cls.get_name()} cannot be evaluated without setting the round_keys")
        return super().__new__(cls, *args, **options)

    @classmethod
    def to_ssa(cls, input_names, id_prefix, decompose_sec_ops=False, **ssa_options):
        if cls.round_keys is None:
            raise ValueError("round_keys must be set before calling to_ssa()")

        my_ssa = super().to_ssa(input_names, id_prefix, decompose_sec_ops=decompose_sec_ops, **ssa_options)

        all_round_keys = set(cls.round_keys)
        round_keys_found = set()
        for ext_var in my_ssa.external_vars:
            if ext_var not in all_round_keys:
                raise ValueError("found external variable {} not in round_keys {} in {}\n{}".format(
                    ext_var, cls.round_keys, cls.__name__, my_ssa
                ))
            round_keys_found.add(ext_var)
        if len(round_keys_found) < len(all_round_keys):
            raise ValueError(f"round keys {all_round_keys - round_keys_found} not used in {cls.__name__}")

        return my_ssa


class Cipher(object):
    """Represent block ciphers.

    A block cipher is a pair of bit-vector functions (see `BvFunction`):
    the key schedule that computes the round keys from a master key, and
    the `Encryption` function.

    Similar to `BvFunction`, `Cipher` is evaluated with the operator ``()``
    that is, ``Cipher(plaintext, masterkey)`` returns the ciphertext.
    The arguments ``plaintext`` and ``masterkey`` are both lists of
    `Constant` objects or integers (integers are automatically converted
    to `Constant` objects using the bit-sizes given by
    ``Cipher.key_schedule.input_widths`` and ``Cipher.encryption.input_widths``.

    An iterated block cipher is a `Cipher` where both the key-schedule and the
    encryption functions are `RoundBasedFunction` objects.

    This class is not meant to be instantiated but to provide a base class for
    block ciphers. To define a block cipher, subclass `Cipher` and set the class
    attributes `key_schedule` and `encryption`.
    For an iterated block cipher, the method `set_num_rounds` must be implemented.

    .. note::
        The method `set_num_rounds` must set the number of rounds of `encryption`
        to the given number of rounds, but the number of rounds of
        `key_schedule` might differ.

        For an iterated block cipher ``C`` where the `key_schedule`
        and the `encryption` share the number of rounds (i.e.,
        ``C.key_schedule.num_rounds == C.encryption.num_rounds``),
        the usual implement of `set_num_rounds` is

        .. code:: python

            @classmethod
            def set_num_rounds(cls, encryption_new_num_rounds):
                cls.key_schedule.set_num_rounds(encryption_new_num_rounds)
                cls.encryption.set_num_rounds(encryption_new_num_rounds)

        However, an iterated block cipher can also contain a `key_schedule`
        with a different number of rounds than the `encryption`.
        An example of `set_num_rounds` for an iterated cipher where
        the `key_schedule` has one less round that the  `encryption` is

        .. code:: python

            @classmethod
            def set_num_rounds(cls, encryption_new_num_rounds):
                cls.key_schedule.set_num_rounds(encryption_new_num_rounds - 1)
                cls.encryption.set_num_rounds(encryption_new_num_rounds)


    ::

        >>> from cascada.primitives.blockcipher import Cipher
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> issubclass(Speck32, Cipher)
        True
        >>> Speck32(plaintext=[0, 0], masterkey=[0, 0, 0, 0])  # Automatic Constant Conversion
        (0x2bb9, 0xc642)
        >>> Speck32.get_name()
        'SpeckCipher_22R'
        >>> Speck32.vrepr()
        'SpeckCipher.set_num_rounds_and_return(22)'
        >>> Speck32.set_round_keys(masterkey=[0, 0, 0, 0])
        >>> Speck32.encryption.round_keys  # doctest: +NORMALIZE_WHITESPACE
        (0x0000, 0x0000, 0x0001, 0x0007, 0x0018, 0x027c, 0x0189, 0x0fab, 0x7904, 0x8f0d, 0x911f,
        0xa5da, 0x49d1, 0xba62, 0xeda2, 0xd3da, 0x6c70, 0x0da9, 0x86c6, 0xa604, 0xef7d, 0x093e)
        >>> Speck32.set_num_rounds(2)
        >>> Speck32(plaintext=[0, 0], masterkey=[0, 0])  # masterkey is now a 2-word list
        (0x0000, 0x0000)
        >>> Speck32.get_name()
        'SpeckCipher_2R'
        >>> Speck32.vrepr()
        'SpeckCipher.set_num_rounds_and_return(2)'
        >>> Speck32.set_round_keys(symbolic_prefix="k")
        >>> for v in Speck32.encryption.round_keys: print(v.vrepr())
        Variable('k0', width=16)
        Variable('k1', width=16)

    Attributes:
        key_schedule: the key-schedule function (a subclass of `BvFunction`)
        encryption: the encryption function (a subclass of `Encryption` and `BvFunction`)

    """
    key_schedule = None
    encryption = None
    # _min_num_rounds

    def __new__(cls, plaintext, masterkey, **options):
        assert isinstance(plaintext, collections.abc.Sequence)
        assert isinstance(masterkey, collections.abc.Sequence)

        # best place to check that the Cipher subclass has been well-defined
        if not issubclass(cls.key_schedule, cascada_ssa.BvFunction):
            raise ValueError(f"{cls.get_name()}.key_schedule is not a subclass of BvFunction")
        if not issubclass(cls.encryption, Encryption) or not issubclass(cls.encryption, cascada_ssa.BvFunction):
            raise ValueError(f"{cls.get_name()}.encryption is not a subclass of Encryption or BvFunction")

        for parent_class in cls.encryption.__mro__:
            if parent_class == Encryption:
                break
            if parent_class in [cascada_ssa.BvFunction, cascada_ssa.RoundBasedFunction]:
                raise ValueError(f"{cls.get_name()}.encryption must subclass Encryption before BvFunction")

        if cls.num_rounds is not None:
            # ks_schedule does not need to be a RoundBasedFunction
            if not issubclass(cls.encryption, cascada_ssa.RoundBasedFunction):
                raise ValueError(f"{cls.get_name()}.encryption is not a subclass of "
                                 f"RoundBasedFunction but num_rounds is {cls.num_rounds}")
            if not isinstance(cls.num_rounds, int) or cls.num_rounds <= 0:
                raise ValueError(f"{cls.get_name()}.num_rounds must be a non-zero positive integer")
            if hasattr(cls, "_min_num_rounds") and cls.num_rounds < cls._min_num_rounds:
                raise ValueError(f"{cls.get_name()}._min_num_rounds = {cls._min_num_rounds}"
                                 f" > {cls.num_rounds} = {cls.get_name()}.num_rounds")

        if cls.encryption.round_keys is None:
            prev_round_keys = None
        else:
            if len(cls.encryption.round_keys) == len(cls.key_schedule.output_widths):
                prev_round_keys = tuple(cls.encryption.round_keys[:])
            else:
                warnings.warn(f"removing old round_keys ("
                              f"len({cls.encryption.get_name()}.round_keys) = "
                              f"{len(cls.encryption.round_keys)} but "
                              f"len({cls.key_schedule.get_name()}.output_widths) = "
                              f"{len(cls.key_schedule.output_widths)})")
                # if set_round_keys and then set_num_rounds, round_keys might be invalid
                prev_round_keys = None
        round_keys = cls.key_schedule(*masterkey, **options)

        if prev_round_keys is not None:
            assert len(prev_round_keys) == len(round_keys)

        cls.encryption.round_keys = tuple(round_keys)

        result = cls.encryption(*plaintext, **options)

        cls.encryption.round_keys = prev_round_keys

        return result

    @classmethod
    @property
    def num_rounds(cls):
        """The number of rounds of `encryption` if iterated, otherwise ``None``"""
        if issubclass(cls.encryption, cascada_ssa.RoundBasedFunction):
            return cls.encryption.num_rounds
        else:
            return None

    @classmethod
    def get_name(cls):
        """Return the class name and the current number of rounds (if iterated)."""
        if cls.num_rounds is not None:
            aux_str = f"_{cls.num_rounds}R"
        else:
            aux_str = ""
        return f"{cls.__name__}{aux_str}"

    @classmethod
    def vrepr(cls):
        """Return an executable string representation.

        This method returns a string so that ``eval(cls.vrepr())``
        returns a new `Cipher` object with the same content.
        """
        if cls.num_rounds is not None:
            return f"{cls.__name__}.set_num_rounds_and_return({cls.num_rounds})"
        else:
            return f"{cls.__name__}"

    @classmethod
    def set_num_rounds(cls, encryption_new_num_rounds):
        """Call `RoundBasedFunction.set_num_rounds` of `key_schedule` and `encryption` (if iterated)."""
        if cls.num_rounds is None:
            raise ValueError(f"{cls.get_name()} is not iterated")
        else:
            raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def set_num_rounds_and_return(cls, new_num_rounds):
        """Call `set_num_rounds` and return ``cls`` (if iterated)."""
        if cls.num_rounds is None:
            raise ValueError(f"{cls.get_name()} is not iterated")
        cls.set_num_rounds(new_num_rounds)
        return cls

    @classmethod
    def set_round_keys(cls, masterkey=None, symbolic_prefix=None):
        """Set ``cls.encryption.round_keys``.

        If ``masterkey`` is given,
        set the round keys as the output of ``cls.key_schedule(*masterkey)``.

        If ``symbolic_prefix`` is given,
        set the round keys as a tuple of `Variable` objects with
        prefix name ``symbolic_prefix`` and with bit-sizes given by
        ``cls.key_schedule.output_widths``.

        Args:
            masterkey: (optional) a list of `Constant` objects or integers
            symbolic_prefix: (optional) a string

        """
        if masterkey is not None:
            cls.encryption.round_keys = cls.key_schedule(*masterkey)
        else:
            assert symbolic_prefix is not None
            cls.encryption.round_keys = []
            for i, width in enumerate(cls.key_schedule.output_widths):
                var = core.Variable(symbolic_prefix + str(i), width)
                cls.encryption.round_keys.append(var)
        cls.encryption.round_keys = tuple(cls.encryption.round_keys)


@functools.lru_cache(maxsize=None)
def get_random_cipher(width, key_num_inputs, key_num_assignments,
                      enc_num_inputs, enc_num_outputs, enc_num_assignments, seed,
                      external_variable_prefix, operation_set_index=0, num_rounds=None,
                      extra_operations=None):
    """Return a random `Cipher` with given shape.

    Args:
        width: the common bitsize of the input and output variables of the function
        key_num_inputs: the number of inputs of the key schedule
        key_num_assignments: an estimation of the number of operations within the key schedule
        enc_num_inputs: the number of inputs of the encryption
        enc_num_assignments: an estimation of the number of operations within the encryption
        enc_num_outputs: the number of outputs of the encryption
        seed: the seed used when sampling
        operation_set_index: four set of operations to choose indexed by 0, 1, 2 and 3
        num_rounds: if not ``None``, sample the key-schedule and the encryption
            functions as random `RoundBasedFunction` objects with the given number of rounds
        extra_operations: an optional `tuple` containing `Operation` subclasses
            to add to the list of operations to choose

    ::

        >>> from cascada.bitvector.core import Variable
        >>> from cascada.primitives.blockcipher import get_random_cipher
        >>> my_cipher = get_random_cipher(4, 1, 1, 2, 2, 4, seed=1, external_variable_prefix="rk")
        >>> my_cipher.key_schedule.to_ssa(["mk0"], "k")
        SSA(input_vars=[mk0], output_vars=[k0_out], assignments=[(k0, mk0 & 0x5), (k0_out, Id(k0))])
        >>> my_cipher.encryption.round_keys = [Variable(f"rk{i}", 4) for i in range(1)]
        >>> my_cipher.encryption.to_ssa(["p0", "p1"], "x")  # doctest: +NORMALIZE_WHITESPACE
        SSA(input_vars=[p0, p1], output_vars=[x2_out, x3_out], external_vars=[rk0],
            assignments=[(x0, p0 ^ rk0), (x1, x0 - p1), (x2, p0 + p1), (x3, -x1), (x2_out, Id(x2)), (x3_out, Id(x3))])

    See also `get_random_bvfunction`.
    """
    while True:
        enc_seed = seed
        while True:
            random_enc = cascada_ssa.get_random_bvfunction(
                width, num_inputs=enc_num_inputs, num_outputs=enc_num_outputs,
                num_assignments=enc_num_assignments, seed=enc_seed,
                external_variable_prefix=external_variable_prefix, operation_set_index=operation_set_index,
                num_rounds=num_rounds, extra_operations=extra_operations
            )

            # repeat until random_enc doesn't contain a redundant assignment with an external var
            assert external_variable_prefix[0] not in ["x", "p"]
            enc_ssa = random_enc.to_ssa(["p" + str(i) for i in range(enc_num_inputs)], "x")
            if tuple(random_enc.round_keys) == tuple(enc_ssa.external_vars):
                break
            else:
                enc_seed += 1

        key_num_outputs = len(random_enc.round_keys)
        key_num_assignments = max(key_num_outputs, key_num_assignments)

        random_ks = cascada_ssa.get_random_bvfunction(
            width, num_inputs=key_num_inputs, num_outputs=key_num_outputs, num_assignments=key_num_assignments,
            seed=seed, external_variable_prefix=None, operation_set_index=operation_set_index, num_rounds=num_rounds,
            extra_operations=extra_operations
        )

        ## debugging
        # print("key_ssa:", random_ks.to_ssa(["mk" + str(i) for i in range(key_num_inputs)], "k"))
        # print("enc_ssa:", random_enc.to_ssa(["p" + str(i) for i in range(enc_num_inputs)], "x"))

        class RandomEncryption(Encryption, random_enc):
            num_rounds = getattr(random_enc, "num_rounds", None)
            round_keys = None

        assert RandomEncryption.round_keys != random_enc.round_keys

        class RandomKeySchedule(random_ks):
            num_rounds = getattr(random_ks, "num_rounds", None)

        extra_operations_vrepr = None
        if extra_operations is not None:
            extra_operations_vrepr = f"({','.join(op.__name__ for op in extra_operations)},)"

        class RandomCipher(Cipher):
            key_schedule = RandomKeySchedule
            encryption = RandomEncryption
            num_rounds = getattr(random_enc, "num_rounds", None)

            @classmethod
            def vrepr(cls):
                evp = external_variable_prefix.__repr__()
                return f"get_random_cipher({width}, {key_num_inputs}, {key_num_assignments}, " \
                       f"{enc_num_inputs}, {enc_num_outputs}, {enc_num_assignments}, " \
                       f"{seed}, {evp}, {operation_set_index}, {num_rounds}, {extra_operations_vrepr})"

            @classmethod
            def set_num_rounds(cls, new_num_rounds):
                if cls.num_rounds is None:
                    raise ValueError(f"{cls.get_name()} is not iterated")
                else:
                    assert new_num_rounds >= 0
                    cls.num_rounds = new_num_rounds
                    cls.encryption.set_num_rounds(new_num_rounds)
                    cls.key_schedule.set_num_rounds(new_num_rounds)

        # check RandomCipher does not return Constant
        try:
            plaintext = [core.Variable(f"p{i}", width) for i in range(enc_num_inputs)]
            masterkey = [core.Variable(f"mk{i}", width) for i in range(key_num_inputs)]
            RandomCipher(plaintext, masterkey, symbolic_inputs=True, simplify=False)
        except ValueError as e:
            if not str(e).startswith("if symbolic_inputs, expected no Constant values"):
                raise e
            else:
                seed += 1024  # avoid collisions with enc_seed
                continue
        break

    return RandomCipher
