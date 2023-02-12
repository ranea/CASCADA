Adding a primitive
==================

To implement a new primitive, the easiest way is to take a similar primitive
already implemented and adapt it to the new primitive
(see also `BvFunction` and `blockcipher`).
If the new primitive contains a bit-vector operation
with no associated property model (see `abstractproperty.opmodel.OpModel`),
you need to implement the property model (as in `simon` and `simon_rf`)
or use a `WeakModel`, `BranchNumberModel` or `WDTModel` (as in `aes`
or `skinny64`).

We will now describe step-by-step how to add a new primitive
from scratch using Speck32/64 as example.


Description of Speck
--------------------

Speck is a family of lightweight block ciphers. A member of the family is denoted by Speck :math:`2n/m·n`,
where the block size is :math:`2n` bits for :math:`n\in \{16,24,32,48,64\}`,
and the key size is :math:`m·n` bits for :math:`m \in \{2,3,4\}`, depending on the desired security level.

The round function of Speck :math:`f(x, y, k) = (x', y')`
receives two words :math:`x` and :math:`y`, and a round key :math:`k`,
all of size :math:`n`, and outputs two words of bitsize :math:`n`, :math:`x'` and :math:`y'`, such that

- :math:`x' = (x \ggg \alpha)\boxplus y)\oplus k`
- :math:`y' = x' \oplus (y \lll \beta))`

The Speck key-schedule algorithm uses the same round function to generate the round keys.
Let :math:`K = (l_{m-2},...,l_0,k_0)` be a master key for Speck :math:`2n/m·n`, where :math:`l_i, k_0`
are :math:`n`-bit words. The sequence of round keys :math:`k_i` (with :math:`i` starting from 0) is generated as

- :math:`l_{i+m-1} =  (l_i \ggg \alpha)\boxplus k_i)\oplus i`
- :math:`k_{i+1} = l_{i+m-1} \oplus (k_i\lll \beta)`

The rotation offset :math:`(\alpha, \beta)` is :math:`(7,2)` for Speck32/64, and :math:`(8,3)` for the larger versions.



Implementing the round function
-------------------------------

First, we will implement the round function :math:`f(x, y, k) = (x', y')` using the `bitvector` module.
To implement :math:`f`, we search in `bitvector.operation` and `bitvector.secondaryop` the classes
corresponding to the bit-vector operations used in :math:`f`.
In particular, :math:`f` uses the modular addition (corresponding to the class `BvAdd`),
the left and right rotations (`RotateLeft` and `RotateRight`),  and the XOR (`BvXor`).
Thus, we can implement :math:`f` with the following Python function

.. code:: python

    from cascada.bitvector.operation import BvXor, BvAdd, RotateRight, RotateLeft
    def f(x, y, k):
        x = BvXor(BvAdd(RotateRight(x, 7), y), k)
        y = BvXor(RotateLeft(y, 2), x)
        return x, y

In the unlikely case that :math:`f` would contain an operation not included in the `bitvector` module
or in the case that we would want to group some operations together to treat them as an atomic operation,
new operations can be defined as subclasses of `SecondaryOperation`
(see for example `simon_rf`).

We can test :math:`f` by evaluating it with symbolic and constant bit-vectors
(corresponding to the classes `Variable` and `Constant`).

.. code:: python

    >>> from cascada.bitvector.core import Constant, Variable
    >>> f(Variable("x", 16), Variable("y", 16), Variable("z", 16))
    (((x >>> 7) + y) ^ z, (y <<< 2) ^ ((x >>> 7) + y) ^ z)
    >>> f(Constant(0, 16), Constant(0, 16), Constant(0, 16))
    (0x0000, 0x0000)

Since :math:`f` is only meant to be used with bit-vector inputs (`Term` objects), we can use
the Python operators ``+, ^`` which are overloaded for `Term` objects with
the classes `BvAdd` and `BvXor` respectively. In other words, we can also implement
:math:`f` with the following Python function

.. code:: python

    def f(x, y, k):
        x = RotateRight(x, 7) + y ^ k
        y = RotateLeft(y, 2) ^ x
        return x, y


Implementing the key-schedule and the encryption functions
----------------------------------------------------------

We will now implement the key-schedule and the encryption functions of Speck32/64
using two Python functions and bit-vector operations.
To do so, we will implement the key-schedule as a function taking the
key words and returning the list of round keys, and the
encryption as a function taking the plaintext words and the round keys
and returning the ciphertext words.

.. code:: python

    from cascada.bitvector.core import Constant
    from cascada.bitvector.operation import RotateRight, RotateLeft

    num_rounds = 22

    def f(x, y, k):
        x = RotateRight(x, 7) + y ^ k
        y = RotateLeft(y, 2) ^ x
        return x, y

    def key_schedule(l2, l1, l0, k0):
        l = [None for i in range(num_rounds + 3)]
        round_keys = [None for _ in range(num_rounds)]
        round_keys[0] = k0
        l[0:3] = [l0, l1, l2]
        for i in range(num_rounds - 1):
            l[i+3], round_keys[i+1] = f(l[i], round_keys[i], Constant(i, 16))
        return round_keys

    def encryption(x, y, round_keys):
        for i in range(num_rounds):
            x, y = f(x, y, round_keys[i])
        return x, y

There are many ways to implement these key-schedule and encryption functions;
our only restriction is that the output of these two functions are list of
bit-vectors (`Term` objects) assuming that the inputs are `Term` objects.
We can test these two Python functions with the following Speck32/64 test vector

 - plaintext = (0x6574, 0x694c)
 - key = (0x1918, 0x1110, 0x0908, 0x0100)
 - ciphertext = (0xa868, 0x42f2)

.. code:: python

    >>> from cascada.bitvector.core import Constant
    >>> l2, l1 = Constant(0x1918, 16), Constant(0x1110, 16)
    >>> l0, k0 = Constant(0x0908, 16), Constant(0x0100, 16)
    >>> round_keys = key_schedule(l2, l1, l0, k0)
    >>> x, y = Constant(0x6574, 16), Constant(0x694c, 16)
    >>> ciphertext = encryption(x, y, round_keys)
    >>> ciphertext
    (0xa868, 0x42f2)


Wrapping the key-schedule function
----------------------------------

To implement Speck32/64 as a primitive supported by CASCADA
(i.e., as a `Cipher` subclass), we need to wrap the key-schedule
into a `RoundBasedFunction` subclass.

To do so, we need to set some attributes (``input_widths``,
``output_widths``, ``num_rounds``) and implement the methods
``eval`` and ``set_num_rounds`` (see `BvFunction` and `RoundBasedFunction`)
as follows

.. code:: python

    from cascada.bitvector.core import Constant
    from cascada.bitvector.ssa import RoundBasedFunction
    from cascada.bitvector.operation import RotateRight, RotateLeft

    def f(x, y, k):
        x = RotateRight(x, 7) + y ^ k
        y = RotateLeft(y, 2) ^ x
        return x, y

    class SpeckKeySchedule(RoundBasedFunction):
        input_widths = [16, 16, 16, 16]
        output_widths = [16 for _ in range(22)]
        num_rounds = 22

        @classmethod
        def eval(cls, l2, l1, l0, k0):
            l = [None for i in range(cls.num_rounds + 3)]
            round_keys = [None for _ in range(cls.num_rounds)]
            round_keys[0] = k0
            l[0:3] = [l0, l1, l2]
            for i in range(cls.num_rounds - 1):
                l[i+3], round_keys[i+1] = f(l[i], round_keys[i], Constant(i, 16))
            return round_keys

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
            cls.output_widths = [16 for _ in range(new_num_rounds)]

To implement the ``eval`` method, we took the previous Python function ``key_schedule`` working
on bit-vectors, but using the class attribute ``cls.num_rounds`` so that
we can change the current number of rounds with the method ``set_num_rounds``.

Similar as before, we can test ``SpeckKeySchedule`` by evaluating it with symbolic and constant bit-vectors.
To avoid long outputs, we will reduce the number of rounds for the test:

    >>> from cascada.bitvector.core import Constant, Variable
    >>> SpeckKeySchedule.set_num_rounds(3)
    >>> masterkey = Variable("l2", 16), Variable("l1", 16), Variable("l0", 16), Variable("k0", 16)
    >>> SpeckKeySchedule(*masterkey, symbolic_inputs=True)
    (k0, (k0 <<< 2) ^ ((l0 >>> 7) + k0), (((k0 <<< 2) ^ ((l0 >>> 7) + k0)) <<< 2) ^ ((l1 >>> 7) + ((k0 <<< 2) ^ ((l0 >>> 7) + k0))) ^ 0x0001)
    >>> masterkey = Constant(0, 16), Constant(0, 16), Constant(0, 16), Constant(0, 16)
    >>> SpeckKeySchedule(*masterkey)
    (0x0000, 0x0000, 0x0001)

By default evaluating `RoundBasedFunction` with symbolic inputs is not allowed,
but this can be enabled by passing the optional argument ``symbolic_inputs=True``.

Note that we can also use the method ``add_round_outputs`` within the ``eval``
to delimit the end of the rounds (as in `speck`), but this is optional and
we will not use it in this tutorial.


Wrapping the encryption function
----------------------------------

To implement Speck32/64 as a `Cipher` subclass, we also need to wrap the encryption
function into a `Encryption`/`RoundBasedFunction` subclass.

To do so, we need to set some attributes (``input_widths``,
``output_widths``, ``num_rounds``) and implement the methods
``eval`` and ``set_num_rounds``:

.. code:: python

    from cascada.bitvector.ssa import RoundBasedFunction
    from cascada.bitvector.operation import RotateRight, RotateLeft
    from cascada.primitives.blockcipher import Encryption

    def f(x, y, k):
        x = RotateRight(x, 7) + y ^ k
        y = RotateLeft(y, 2) ^ x
        return x, y

    class SpeckEncryption(Encryption, RoundBasedFunction):
        input_widths = [16, 16]
        output_widths = [16, 16]
        num_rounds = 22
        round_keys = None

        @classmethod
        def eval(cls, x, y):
            for i in range(cls.num_rounds):
                x, y = f(x, y, cls.round_keys[i])
            return x, y

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds

Similar as before, we took the previous Python function ``encryption`` working
on bit-vectors, but using the class attribute ``cls.num_rounds`` so that
we can change the current number of rounds with the method ``set_num_rounds``.

For the ``eval`` method of the encryption in particular, the round keys are obtained
from the class attribute ``cls.round_keys`` (later after we link
``SpeckKeySchedule`` and ``SpeckEncryption``, the class attribute ``cls.round_keys``
will be automatically set to the output of ``SpeckKeySchedule``).

We can test ``SpeckEncryption`` by evaluating it with symbolic and constant bit-vectors,
but we need to set the round keys manually for testing it individually.

    >>> from cascada.bitvector.core import Constant, Variable
    >>> num_rounds = 2
    >>> round_keys = [Variable("k" + str(i), 16) for i in range(num_rounds)]
    >>> SpeckEncryption.set_num_rounds(num_rounds)
    >>> SpeckEncryption.round_keys = round_keys
    >>> plaintext = [Variable("x", 16), Variable("y", 16)]
    >>> SpeckEncryption(*plaintext, symbolic_inputs=True)
    ((((((x >>> 7) + y) ^ k0) >>> 7) + ((y <<< 2) ^ ((x >>> 7) + y) ^ k0)) ^ k1,
    (((y <<< 2) ^ ((x >>> 7) + y) ^ k0) <<< 2) ^ (((((x >>> 7) + y) ^ k0) >>> 7) + ((y <<< 2) ^ ((x >>> 7) + y) ^ k0)) ^ k1)

Alternatively, we can also check it by computing its `SSA`
(a list of simple instructions representing the bit-vector function)

    >>> num_rounds = 2
    >>> round_keys = [Variable("k" + str(i), 16) for i in range(num_rounds)]
    >>> SpeckEncryption.set_num_rounds(num_rounds)
    >>> SpeckEncryption.round_keys = round_keys
    >>> SpeckEncryption.to_ssa(["x", "y"], "z")
    SSA(input_vars=[x, y], output_vars=[z7_out, z9_out], external_vars=[k0, k1],
        assignments=[
            (z0, x >>> 7), (z1, z0 + y), (z2, z1 ^ k0), (z3, y <<< 2), (z4, z3 ^ z2),
            (z5, z2 >>> 7), (z6, z5 + z4), (z7, z6 ^ k1), (z8, z4 <<< 2), (z9, z8 ^ z7),
            (z7_out, Id(z7)), (z9_out, Id(z9))])


Wrapping everything as a Cipher subclass
----------------------------------------

Finally,  we can implement Speck32/64 as a `Cipher` subclass
using ``SpeckKeySchedule``, ``SpeckEncryption`` and setting
some attributes:


.. code:: python

    from cascada.bitvector.core import Constant
    from cascada.bitvector.ssa import RoundBasedFunction
    from cascada.bitvector.operation import RotateRight, RotateLeft
    from cascada.primitives.blockcipher import Encryption, Cipher

    def f(x, y, k):
        x = RotateRight(x, 7) + y ^ k
        y = RotateLeft(y, 2) ^ x
        return x, y

    class SpeckKeySchedule(RoundBasedFunction):
        input_widths = [16, 16, 16, 16]
        output_widths = [16 for _ in range(22)]
        num_rounds = 22

        @classmethod
        def eval(cls, l2, l1, l0, k0):
            l = [None for i in range(cls.num_rounds + 3)]
            round_keys = [None for _ in range(cls.num_rounds)]
            round_keys[0] = k0
            l[0:3] = [l0, l1, l2]
            for i in range(cls.num_rounds - 1):
                l[i+3], round_keys[i+1] = f(l[i], round_keys[i], Constant(i, 16))
            return round_keys

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds
            cls.output_widths = [16 for _ in range(new_num_rounds)]

    class SpeckEncryption(Encryption, RoundBasedFunction):
        input_widths = [16, 16]
        output_widths = [16, 16]
        num_rounds = 22
        round_keys = None

        @classmethod
        def eval(cls, x, y):
            for i in range(cls.num_rounds):
                x, y = f(x, y, cls.round_keys[i])
            return x, y

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.num_rounds = new_num_rounds

    class SpeckCipher(Cipher):
        key_schedule = SpeckKeySchedule
        encryption = SpeckEncryption

        @classmethod
        def set_num_rounds(cls, new_num_rounds):
            cls.key_schedule.set_num_rounds(new_num_rounds)
            cls.encryption.set_num_rounds(new_num_rounds)


The subclass ``SpeckCipher`` is now a primitive that can be used in the automated methods
implemented by CASCADA (see :doc:`usage`).

Since the `differential`, `linear` and `algebraic` models are implemented
for `BvAdd`, `BvXor`, `RotateLeft` and `RotateRight` (the operations of Speck),
``SpeckCipher`` can be used directly in the automated methods of CASCADA
(e.g., `round_based_ch_search`).

However, if we implement a primitive containing an operation whose property model
is not implemented, then we need to implement it in order to search for characteristics
over this primitive. For example, if the primitive contains S-boxes (`LutOperation`)
or binary matrix-vector operations (`MatrixOperation`), no default property models
are assigned for these operations, but the generic models `WeakModel`,
`BranchNumberModel` and `WDTModel` can be used (as in `aes` or `skinny64`).

Note that the previous implementation of Speck32/64 is slightly different to the
implementations of Speck in `speck`. In particular,
`speck` includes `add_round_outputs` calls to delimit the
rounds and considers that a Speck cipher instance with :math:`r` rounds contains
:math:`r` encryption rounds but :math:`(r-1)` key-schedule rounds
(since the key-schedule does not perform any operation for the 1-round cipher).