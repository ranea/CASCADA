Advanced Usage
==============

Before searching for characteristics or zero-probability properties
over a large number of rounds of a cryptographic primitive,
CASCADA provides a set of tools to run a fine-grained search over
a small number of rounds and check the results.
Additionally, if you are implementing a new primitive or a new property model,
CASCADA also includes several tests to check cipher implementations
and property models.

In this tutorial we will see how to check the bit-vector implementation of
a primitive, the property models of a bit-vector operations, a characteristic model,
and a characteristic obtained in the search.


Checking the bit-vector implementation of a primitive
-----------------------------------------------------

If you are using one of the primitives included in CASCADA (see `primitives_implemented`)
you can skip this section. Otherwise, you can check that the primitive is correct
by evaluating it with some test vectors. For example, you can evaluate
the cipher Speck32/64 with a test vector as follows:

.. code:: python

    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> plaintext = (0x6574, 0x694c)
    >>> key = (0x1918, 0x1110, 0x0908, 0x0100)
    >>> Speck32(plaintext, key) == (0xa868, 0x42f2)
    True

For a round-based primitive including `add_round_outputs` calls, you can also
check the automatic round splitting of the `SSA` form:

.. code:: python

    >>> SpeckCipher.set_num_rounds(2)
    >>> SpeckCipher.set_round_keys(symbolic_prefix="k")
    >>> ssa_2rounds = SpeckCipher.encryption.to_ssa(["p0", "p1"], "x")
    >>> ssa_2rounds
    SSA(input_vars=[p0, p1], output_vars=[x7_out, x9_out], external_vars=[k0, k1],
        assignments=[(x0, p0 >>> 7), (x1, x0 + p1), (x2, x1 ^ k0), (x3, p1 <<< 2), (x4, x3 ^ x2),
            (x5, x2 >>> 7), (x6, x5 + x4), (x7, x6 ^ k1), (x8, x4 <<< 2), (x9, x8 ^ x7), (x7_out, Id(x7)), (x9_out, Id(x9))])
    >>> ssa_round1, ssa_round2 = ssa_2rounds.split(ssa_2rounds.get_round_separators())
    >>> ssa_round1
    SSA(input_vars=[p0, p1], output_vars=[x2_out, x4_out], external_vars=[k0],
        assignments=[(x0, p0 >>> 7), (x1, x0 + p1), (x2, x1 ^ k0), (x3, p1 <<< 2), (x4, x3 ^ x2), (x2_out, Id(x2)), (x4_out, Id(x4))])
    >>> ssa_round2
    SSA(input_vars=[x2, x4], output_vars=[x7_out, x9_out], external_vars=[k1],
        assignments=[(x5, x2 >>> 7), (x6, x5 + x4), (x7, x6 ^ k1), (x8, x4 <<< 2), (x9, x8 ^ x7), (x7_out, Id(x7)), (x9_out, Id(x9))])

Here we can see the SSA of each round corresponds exactly to a round of Speck,
that the external variables are the round keys, and that the composition of
``ssa_round1`` and ``ssa_round2`` results in ``ssa_2rounds``.
Checking the SSA form of the cipher is beneficial as this is the main internal
representation used by CASCADA. Moreover, the SSA form might differ from the original
implementation (while still being functionally equivalent), as redundant assignments
are removed and additional transformations might be performed (more on that later).

Additionally, you can use the tests in ``smt/tests/test_primitives_eval.py``
(by adding your primitive to the list ``LIST_PRIMITIVES``)
or in ``smt/tests/test_primitives_smt.py``, or read more about
implementing primitives in :doc:`adding_primitive`, :py:mod:`cascada.bitvector.ssa`
and `blockcipher`.



Checking the property model of a bit-vector operation
-----------------------------------------------------

If you are using one the primitives included in CASCADA or only property
models included in CASCADA, you can skip this section.

If you implement a new property model from the generic models
`WeakModel`, `BranchNumberModel` or `WDTModel`, then you can
check the weights and the model error. For example, for a `WDTModel`:

.. code:: python

    >>> from decimal import Decimal
    >>> from math import inf
    >>> from cascada.bitvector.core import Variable, Constant
    >>> from cascada.bitvector.secondaryop import LutOperation
    >>> from cascada.differential.difference import XorDiff
    >>> from cascada.differential.opmodel import log2_decimal, get_wdt_model
    >>> class Sbox3b(LutOperation): lut = [Constant(i, 3) for i in (0, 1, 2, 3, 4, 6, 7, 5)]
    >>> ddt = [(8, 0, 0, 0, 0, 0, 0, 0), (0, 4, 4, 0, 0, 0, 0, 0), (0, 0, 4, 4, 0, 0, 0, 0), (0, 4, 0, 4, 0, 0, 0, 0),
    ...     (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2), (0, 0, 0, 0, 2, 2, 2, 2)]
    >>> wdt_dec = tuple([tuple(inf if x == 0 else -log2_decimal(x/Decimal(2**3)) for x in row) for row in ddt])
    >>> wdt_dec
    ((Decimal('0'), inf, inf, inf, inf, inf, inf, inf),
    (inf, Decimal('1'), Decimal('1'), inf, inf, inf, inf, inf),
    (inf, inf, Decimal('1'), Decimal('1'), inf, inf, inf, inf),
    (inf, Decimal('1'), inf, Decimal('1'), inf, inf, inf, inf),
    (inf, inf, inf, inf, Decimal('2'), Decimal('2'), Decimal('2'), Decimal('2')),
    (inf, inf, inf, inf, Decimal('2'), Decimal('2'), Decimal('2'), Decimal('2')),
    (inf, inf, inf, inf, Decimal('2'), Decimal('2'), Decimal('2'), Decimal('2')),
    (inf, inf, inf, inf, Decimal('2'), Decimal('2'), Decimal('2'), Decimal('2')))
    >>> XorWDTModelSbox3b = get_wdt_model(Sbox3b, XorDiff, wdt_dec)
    >>> XorWDTModelSbox3b(XorDiff(Variable("a", 3))).error()
    0
    >>> wdt_bv = [[None for _ in range(2**3)] for _ in range(2**3)]
    >>> for i in range(2**3):
    ...    for j in range(2**3):
    ...        f = XorWDTModelSbox3b(XorDiff(Constant(i, 3)))
    ...        if f.validity_constraint(XorDiff(Constant(j, 3))):
    ...            wdt_bv[i][j] = f.bv_weight(XorDiff(Constant(j, 3)))
    >>> wdt_bv
    [[0b00, None, None, None, None, None, None, None],
    [None, 0b01, 0b01, None, None, None, None, None],
    [None, None, 0b01, 0b01, None, None, None, None],
    [None, 0b01, None, 0b01, None, None, None, None],
    [None, None, None, None, 0b10, 0b10, 0b10, 0b10],
    [None, None, None, None, 0b10, 0b10, 0b10, 0b10],
    [None, None, None, None, 0b10, 0b10, 0b10, 0b10],
    [None, None, None, None, 0b10, 0b10, 0b10, 0b10]]

We can see that the error is 0 as the bit-vector weights from ``wdt_bv``
match the decimal weights from ``wdt_dec``. However, if the decimal weights
provided to ``get_wdt_model`` are not integers, you should check the error
and the bit-vector weights to choose the best trade-off for the precision
(more fractional bits give to less error but more complex models).

If you implement a new property model (not derived from
`WeakModel`, `BranchNumberModel` or `WDTModel`), then the CASCADA modules
``differential.tests.test_opmodel`` and ``linear.tests.test_opmodel``
provide the class ``TestOpModelGeneric`` to test the constraints of a differential
or linear model by sampling all inputs of small wordsize.
An example of how to use this class is the method ``TestOpModelsSmallWidth.test_*_models_slow``
from these modules, or the method ``TestOpModelsSmallWidth.test_Xor_models_slow`` from
``smt/tests/test_simon_rf`` (which might be easier to understand).
Additionally, you can check the error and the symbolic expressions of the constraints
of the model similar to the docstrings of ``smt/tests/test_simon_rf``, `differential.opmodel` and
`linear.opmodel`. You can read more about property models in
`abstractproperty.opmodel`.


Checking a characteristic model
-------------------------------

Before checking the characteristics found in the search, it is better to check first
the representation of a characteristic model (i.e., the symbolic characteristic associated with
any characteristics found) and its error.

For a round-based primitive including `add_round_outputs` calls, you can print
for example the `XorDiff` `differential.chmodel.EncryptionChModel` of each round and its underlying
SSA form as follows:

.. code:: python

    >>> from cascada.differential.difference import XorDiff
    >>> from cascada.differential.chmodel import EncryptionChModel
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> Speck32.set_num_rounds(2)
    >>> ch_model_2rounds = EncryptionChModel(Speck32, XorDiff)
    >>> ch_model_2rounds.error()
    0
    >>> ch_model_round1, ch_model_round2 = ch_model_2rounds.split(ch_model_2rounds.get_round_separators())
    >>> ch_model_round1.ssa
    SSA(input_vars=[dp0, dp1], output_vars=[dx2_out, dx4_out], external_vars=[dk0],
        assignments=[(dx0, dp0 >>> 7), (dx1, dx0 + dp1), (dx2, dx1 ^ dk0), (dx3, dp1 <<< 2), (dx4, dx3 ^ dx2), (dx2_out, Id(dx2)), (dx4_out, Id(dx4))])
    >>> ch_model_round1
    ChModel(func=SpeckEncryption_2R_0S, input_diff=[XorDiff(dp0), XorDiff(dp1)],
        output_diff=[XorDiff(dx2_out), XorDiff(dx4_out)],
        external_var2diff=[(dk0, XorDiff(0x0000))],
        assign_outdiff2op_model=[(XorDiff(dx1), XorModelBvAdd([XorDiff(dp0 >>> 7), XorDiff(dp1)])),
                                 (XorDiff(dx2_out), XorModelId(XorDiff(dx1))),
                                 (XorDiff(dx4_out), XorModelId(XorDiff((dp1 <<< 2) ^ dx1)))])
    >>> ch_model_round2.ssa
    SSA(input_vars=[dx2, dx4], output_vars=[dx7_out, dx9_out], external_vars=[dk1],
        assignments=[(dx5, dx2 >>> 7), (dx6, dx5 + dx4), (dx7, dx6 ^ dk1), (dx8, dx4 <<< 2), (dx9, dx8 ^ dx7), (dx7_out, Id(dx7)), (dx9_out, Id(dx9))])
    >>> ch_model_round2
    ChModel(func=SpeckEncryption_2R_1S, input_diff=[XorDiff(dx2), XorDiff(dx4)],
        output_diff=[XorDiff(dx7_out), XorDiff(dx9_out)],
        external_var2diff=[(dk1, XorDiff(0x0000))],
        assign_outdiff2op_model=[(XorDiff(dx6), XorModelBvAdd([XorDiff(dx2 >>> 7), XorDiff(dx4)])),
                                 (XorDiff(dx7_out), XorModelId(XorDiff(dx6))),
                                 (XorDiff(dx9_out), XorModelId(XorDiff((dx4 <<< 2) ^ dx6)))])

Excluding the trivial transitions with ``XorModelId``, we can check that there is only
one non-trivial transition per round that corresponds to the propagation of the modular addition:
``(XorDiff(dx1), XorModelBvAdd([XorDiff(dp0 >>> 7), XorDiff(dp1)]))`` in the first round,
and ``(XorDiff(dx6), XorModelBvAdd([XorDiff(dx2 >>> 7), XorDiff(dx4)]))`` in the second round.
You can also check the difference associated to each intermediate variable from
the attribute ``var2diff`` of the characteristic model.

Checking the underlying SSA is particularly important for linear characteristics,
as this SSA differs from the SSA of the original primitive; the underlying SSA
is computed with the initialization argument ``replace_multiuse_vars=True``.
(see `linear.chmodel.ChModel`).

We can also see here that the characteristic model of the two rounds
``ch_model_2rounds`` has error 0, as the model `XorModelBvAdd` has also error 0.
This error is important as it will affect the optimality of the
characteristics found in the search and the speed when computing the empirical weights.

You can read more about characteristic models in `abstractproperty.chmodel`.


Checking the characteristics found in the search
------------------------------------------------

If the hypothesis of stochastic equivalence does not hold for the target primitive,
(i.e., if the intermediate propagations of properties within the characteristic are not independent),
the actual weight of the characteristics obtained in the search might be different
than the ``ch_weight`` (the weight of the characteristic computed as the sum of
the ``assignment_weights``, see `abstractproperty.characteristic.Characteristic`).

To check the discrepancy between the ``ch_weight`` and the actual weight,
you can compute the ``empirical_ch_weight``, which is another approximation
of the actual weight of the characteristic
(see `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight` for
more details about the empirical weight).

In the following example, we use the method `ChFinder.find_next_ch_increasing_weight`
with the parameter ``empirical_weight_options`` to search for 2-round characteristics
of Speck32/64.

.. code:: python

    >>> from cascada.differential.difference import XorDiff
    >>> from cascada.differential.chmodel import EncryptionChModel
    >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType, PrintingMode
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> Speck32.set_num_rounds(2)
    >>> ch_model = EncryptionChModel(Speck32, XorDiff)
    >>> assert_type = ChModelAssertType.ValidityAndWeight
    >>> ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0,
    ...                      exclude_zero_input_prop=True, printing_mode=PrintingMode.Debug)
    >>> ewo = {"seed": 0, "C_code": True}
    >>> ch_found = next(ch_finder.find_next_ch_increasing_weight(0, empirical_weight_options=ewo))
    [...] | Solving for weight = 0
    [...] | Solving for weight = 1
    >>> ch_found
    EncryptionCharacteristic(ch_weight=1, empirical_ch_weight=1.037103994662739465661470304, assignment_weights=[1, 0, 0, 0],
        input_diff=[0x0010, 0x2000], output_diff=[0x8000, 0x8002], external_diffs=[0x0000, 0x0000], assign_outdiff_list=[0x0000, 0x8000, 0x8000, 0x8002])

We are using the method `ChFinder.find_next_ch_increasing_weight` instead of the function `round_based_ch_search`
since we are searching for characteristics with a fix number of rounds (and the iterative speedup of `round_based_ch_search`
is not needed) and to have more control over the characteristic model and the parameters of the search.

From the previous example we can see that that the difference between ``ch_weight`` and ``empirical_ch_weight``
is small, which is a good sign. You can see more information about the empirical weight
(such as the number of input samples used to compute it; see also  `EmpiricalWeightData`)
from the attribute ``empirical_data_list``:

.. code:: python

    >>> ch_found.empirical_data_list
    [EmpiricalWeightData(weight_avg_aux_prs=1.037103994662739465661470304, num_aux_weights=32, num_inf_aux_weights=0, num_input_samples=64, seed=0, C_code=True)]
    >>> ch_found.empirical_data_list[0].aux_weights
    [Decimal('1.045803689613124791193876401'), Decimal('1'), Decimal('1'), ..., Decimal('0.6424479953819163068340286924')]

The time to compute the empirical weight depends on the number of input samples.
If the number of samples is not given, this number is taken as a function of
the ``ch_weight`` and the error of the characteristic model.
Thus, if you are implementing a ``OpModel``, reducing its error will reduce the characteristic model error,
which in turn will speed up the computation of the empirical weight.

The empirical weight can be computed manually by calling the method ``compute_empirical_ch_weight``
after a characteristic is yielded (and this allows providing the
number of input of samples dynamically).  On the other hand, if the empirical weight
is computed within the method ``find_next_ch_increasing_weight``, then this allows
discarding characteristics with invalid empirical weights.
This is the main use of the empirical weight: to provide a filter to discard invalid characteristics
(rather than providing a better approximation of the actual weight).

In the following example we try to search for a `RXDiff` characteristic over the key-schedule of `lea`
(for 1 round) with the condition that the  input and output RX-differences are
``(0xc0781f3e, 0x00f03e70, 0x81e04bf8, 0xa303d940) -> (0x0870365c, 0x4381b2cc, 0x38186e64, 0x4381b2cc, 0x0fcf1c0c, 0x4381b2cc)``.

.. code:: python

    >>> from cascada.bitvector.core import Constant
    >>> from cascada.differential.difference import RXDiff
    >>> from cascada.differential.chmodel import ChModel
    >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType, PrintingMode
    >>> from cascada.primitives import lea
    >>> lea.LEACipher.key_schedule.set_num_rounds(1)
    >>> ch_model = ChModel(lea.LEACipher.key_schedule, RXDiff, ["dmk0", "dmk1", "dmk2", "dkm3"])
    >>> assert_type = ChModelAssertType.ValidityAndWeight
    >>> v2c = {v: Constant(c, 32) for v, c in zip(ch_model.input_diff, [0xc0781f3e, 0x00f03e70, 0x81e04bf8, 0xa303d940])}
    >>> for v, c in zip(ch_model.output_diff, [0x0870365c, 0x4381b2cc, 0x38186e64, 0x4381b2cc, 0x0fcf1c0c, 0x4381b2cc]): v2c[v] = Constant(c, 32)
    >>> ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0, printing_mode=PrintingMode.WeightsAndVrepr, var_prop2ct_prop=v2c)
    >>> ewo = {"seed": 0, "C_code": True, "split_by_max_weight": 20}
    >>> ch_found = next(ch_finder.find_next_ch_increasing_weight(0, empirical_weight_options=ewo))
    [...] | Solving for weight = 0
    [...]
    [...] | Solving for weight = 43
    [...] | Invalid characteristic found | Characteristic(input_diff=[Constant(0xc0781f3e, width=32), Constant(0x00f03e70, width=32), Constant(0x81e04bf8, width=32), Constant(0xa303d940, width=32)], output_diff=[Constant(0x0870365c, width=32), Constant(0x4381b2cc, width=32), Constant(0x38186e64, width=32), Constant(0x4381b2cc, width=32), Constant(0x0fcf1c0c, width=32), Constant(0x4381b2cc, width=32)], assign_outdiff_list=[Constant(0x04381b2e, width=32), Constant(0x88703659, width=32), Constant(0x90e061b9, width=32), Constant(0x8181f9e3, width=32), Constant(0x4381b2cc, width=32), Constant(0x4381b2cc, width=32), Constant(0x0870365c, width=32), Constant(0x4381b2cc, width=32), Constant(0x38186e64, width=32), Constant(0x4381b2cc, width=32), Constant(0x0fcf1c0c, width=32), Constant(0x4381b2cc, width=32)], ch_model=ChModel(func=LEAKeySchedule.set_num_rounds_and_return(1), diff_type=RXDiff, input_diff_names=['dmk0', 'dmk1', 'dmk2', 'dkm3'], prefix='dx'), empirical_ch_weight=inf, empirical_data_list=[EmpiricalWeightData(num_input_samples=724338, seed=0, C_code=True, aux_weights=[Decimal('16.14437544306481397560334771')]), EmpiricalWeightData(num_input_samples=1024735554, seed=0, C_code=True, aux_weights=[inf]), EmpiricalWeightData(num_input_samples=0, seed=0, C_code=True, aux_weights=[inf])])
    [...] | Solving for weight = 44
    [...]
    [...] | Solving for weight = 254
    ValueError: no characteristic found with weight <= 255

A single characteristic with weight 43 was found during the search, but it was discarded since ``empirical_ch_weight=inf``.
By using the parameter ``printing_mode=PrintingMode.WeightsAndVrepr``, we can find the attribute ``empirical_data_list``
in the invalid characteristic and obtain more information:

.. code:: python

    empirical_data_list=[
        EmpiricalWeightData(num_input_samples=724338, seed=0, C_code=True, aux_weights=[Decimal('16.14437544306481397560334771')]),
        EmpiricalWeightData(num_input_samples=1024735554, seed=0, C_code=True, aux_weights=[inf]),
        EmpiricalWeightData(num_input_samples=0, seed=0, C_code=True, aux_weights=[inf])])

As we can see, the characteristic was split into three sub-characteristics (following ``"split_by_max_weight": 20``),
and the second sub-characteristic turned out to be infinity (and thus the computation of the empirical weight stopped,
and the third sub-characteristic was ignored).

Apart from checking the hypothesis of stochastic equivalence and the empirical weight,
we can also debug a characteristic by printing additional information such as
the value of all its intermediate properties.
In the following example, we search for a 2-round characteristic
of Speck32/64, then we split it into the sub-characteristic of each round,
and for each sub-characteristic we are printing the list of propagations
(``tuple_assign_outdiff2op_model``) and the value of each difference
(``var_diff2ct_diff``).

.. code:: python

    >>> from cascada.differential.difference import XorDiff
    >>> from cascada.differential.chmodel import EncryptionChModel
    >>> from cascada.smt.chsearch import ChFinder, ChModelAssertType, PrintingMode
    >>> from cascada.primitives import speck
    >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> Speck32.set_num_rounds(2)
    >>> ch_model = EncryptionChModel(Speck32, XorDiff)
    >>> assert_type = ChModelAssertType.ValidityAndWeight
    >>> ch_finder = ChFinder(ch_model, assert_type, "btor", solver_seed=0, exclude_zero_input_prop=True)
    >>> ch_found = next(ch_finder.find_next_ch_increasing_weight(0))
    >>> for i, sub_ch in enumerate(ch_found.split(ch_found.get_round_separators())):
    ...     print(f"Round {i}: {sub_ch.srepr()}")
    ...     print(list(zip(sub_ch.assignment_weights, sub_ch.tuple_assign_outdiff2op_model)))
    ...     print(sub_ch.var_diff2ct_diff)
    ...     print()
    Round 0: Ch(w=1, id=0010 2000, od=0000 8000)
    [(Decimal('1'), (XorDiff(0x0000), XorModelBvAdd([XorDiff(0x2000), XorDiff(0x2000)]))),
     (Decimal('0'), (XorDiff(0x0000), XorModelId(XorDiff(0x0000)))),
     (Decimal('0'), (XorDiff(0x8000), XorModelId(XorDiff(0x8000))))]
    OrderedDict([(XorDiff(dp0), XorDiff(0x0010)), (XorDiff(dp1), XorDiff(0x2000)), (XorDiff(dk0), XorDiff(0x0000)), (XorDiff(dx0), XorDiff(0x2000)), (XorDiff(dx1), XorDiff(0x0000)), (XorDiff(dx2), XorDiff(0x0000)), (XorDiff(dx3), XorDiff(0x8000)), (XorDiff(dx4), XorDiff(0x8000)), (XorDiff(dx2_out), XorDiff(0x0000)), (XorDiff(dx4_out), XorDiff(0x8000))])

    Round 1: Ch(w=0, id=0000 8000, od=8000 8002)
    [(Decimal('0'), (XorDiff(0x8000), XorModelBvAdd([XorDiff(0x0000), XorDiff(0x8000)]))),
     (Decimal('0'), (XorDiff(0x8000), XorModelId(XorDiff(0x8000)))),
     (Decimal('0'), (XorDiff(0x8002), XorModelId(XorDiff(0x8002))))]
    OrderedDict([(XorDiff(dx2), XorDiff(0x0000)), (XorDiff(dx4), XorDiff(0x8000)), (XorDiff(dk1), XorDiff(0x0000)), (XorDiff(dx5), XorDiff(0x0000)), (XorDiff(dx6), XorDiff(0x8000)), (XorDiff(dx7), XorDiff(0x8000)), (XorDiff(dx8), XorDiff(0x0002)), (XorDiff(dx9), XorDiff(0x8002)), (XorDiff(dx7_out), XorDiff(0x8000)), (XorDiff(dx9_out), XorDiff(0x8002))])

You can read more about characteristics and the automated search methods in
`abstractproperty.characteristic` and `chsearch`.