Usage
=====

To search for characteristics (e.g., differential or linear characteristics)
or zero-probability properties (e.g., impossible differentials or zero-correlation approximations)
over a primitive, the primitive needs to be implemented following the CASCADA interface.
There are several primitives already implemented,
such as AES, CHAM, Chaskey, FEAL, HIGHT, LEA, MULTI2, NOEKEON, π-cipher, RECTANGLE, SHACAL-1, SHACAL-2,
Simeck, Simon, Speck, SKINNY, TEA, or XTEA (see `primitives_implemented`).

You can find a step-by-step tutorial about implement a new primitive
in :doc:`adding_primitive`.


Searching for characteristics
-----------------------------

Searching for characteristics can be easily done with `round_based_ch_search`
and `round_based_cipher_ch_search`. For example, searching for
optimal (i.e, lowest-weight) XOR differential characteristics
covering 2, 3, and 4 rounds of `speck_32_64` can be done as follows

.. code:: python

    >>> from cascada.differential.difference import XorDiff
    >>> from cascada.smt.chsearch import ChModelAssertType, round_based_ch_search
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> assert_type = ChModelAssertType.ValidityAndWeight
    >>> iterator = round_based_ch_search(Speck32, 2, 3, XorDiff, assert_type, "btor",
    ...     extra_chfinder_args={"exclude_zero_input_prop": True},
    ...     extra_findnextchweight_args={"initial_weight": 0})
    >>> for (num_rounds, ch) in iterator:
    ...     print(num_rounds, ":", ch.srepr())
    2 : Ch(w=0, id=0040 0000 0000, od=0000 0000 8000)
    3 : Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000)
    4 : Ch(w=0, id=0040 0000 0000 0000, od=0000 0000 0000 8000 8002)

The function `round_based_ch_search` can be used to search for
characteristics for any bit-vector property (e.g., `XorDiff`, `RXDiff`
or `LinearMask`), and `round_based_cipher_ch_search` similarly but
for pairs of characteristics (e.g., related-key differential characteristics).
See `round_based_ch_search` and `round_based_cipher_ch_search`
for a detailed explanation of these functions and their parameters.

The submodule `chsearch` also provides `ChFinder` and `CipherChFinder`
to search for characteristics with more options and flexibility.


Searching for zero-probability properties
-----------------------------------------

Searching for zero-probability properties can be easily done with `round_based_invalidprop_search`
and `round_based_invalidcipherprop_search`. For example, searching for zero-correlation hulls
covering 4 and 5 rounds of `speck_32_64` can be done as follows

.. code:: python

    >>> from cascada.linear.mask import LinearMask
    >>> from cascada.smt.invalidpropsearch import round_based_invalidprop_search, INCREMENT_NUM_ROUNDS
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> iterator = round_based_invalidprop_search(Speck32, 4, 5, LinearMask, "btor")
    >>> tuple_rounds, tuple_chs = next(iterator)
    >>> print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))  #  4 rounds
    (0, 2, 1, 1) : Ch(w=0, id=0080 4021, od=0201 0200), Ch(w=Infinity, id=0201 0200, od=0000 0001), Ch(w=0, id=0000 0001, od=0004 0004)
    >>> iterator.send(INCREMENT_NUM_ROUNDS)  # stop current num_rounds (4) and increment by 1
    >>> tuple_rounds, tuple_chs = next(iterator)
    >>> print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))  #  5 rounds
    (0, 2, 1, 2) : Ch(w=0, id=0080 4021, od=0201 0200), Ch(w=Infinity, id=0201 0200, od=0080 4021), Ch(w=0, id=0080 4021, od=0201 0200)

The function `round_based_invalidprop_search` searches for zero-probability
property pairs using a meet-in-the-middle approach, and so does `round_based_invalidcipherprop_search`
but for zero-probability properties of ciphers (e.g., related-key differentials).

To search for zero-probability properties with other automated methods,
the submodule `invalidpropsearch` provides `InvalidPropFinder`
with the methods `InvalidPropFinder.find_next_invalidprop_activebitmode`
and `InvalidPropFinder.find_next_invalidprop_quantified_logic`, and similarly
for `InvalidCipherPropFinder`.

For example, searching for related-key impossible rotational-XOR differentials
using quantified SMT problems over 3-round `speck_32_64` can be done as follows

.. code:: python

    >>> from cascada.differential.difference import RXDiff
    >>> from cascada.differential.chmodel import CipherChModel
    >>> from cascada.smt.invalidpropsearch import InvalidCipherPropFinder, get_wrapped_cipher_chmodel
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> Speck32.set_num_rounds(2)
    >>> ch_model = get_wrapped_cipher_chmodel(CipherChModel(Speck32, RXDiff))
    >>> invalid_prop_finder = InvalidCipherPropFinder(ch_model, "z3")
    >>> uni_inv_ch = next(invalid_prop_finder.find_next_invalidprop_quantified_logic())
    >>> print(uni_inv_ch.srepr())
    Ch(ks_ch=Ch(w=Infinity, id=c0a2 c0a2, od=0000 0000),  enc_ch=Ch(w=Infinity, id=c0a2 0000, od=c0a2 c0a2))

The :doc:`cascada` provides more information for all the functions and classes of CASCADA.