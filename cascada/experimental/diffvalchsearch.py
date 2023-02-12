"""Search simultaneously for a characteristic and a pair of right inputs satisfying the characteristic.

In this script we construct a model that describe difference transitions
(`XorDiff` or `RXDiff`) and value transitions (`WordValue`) simultaneously.
This is inspired from Section 4.5 of https://eprint.iacr.org/2022/402
(which is inspired from https://eprint.iacr.org/2020/632).

To do this, we construct independently the following objects:

 - the model for the difference transitions
   (a `ChFinder` object given a `XorDiff` or `RXDiff` `EncryptionChModel`)
 - the model for the value transitions
   (two `ChFinder` objects given each one a `WordValue` `EncryptionChModel`)
 - (optional) the model for the round key value transitions
   (a `ChFinder` object given a `WordValue` `ChModel`)

After we construct the three models, we merge the constraints of all
the models, and then by adding constraints we connect:
(1) the difference transitions and the value transitions,
and (2) the value transitions and the round key value transitions.

Thus, the characteristics found for the final model are guaranteed to be
valid, and we recover from the SAT solution all the characteristics
involved.

Note that this script is experimental, and it has not been fully tested.
Moreover, this script does not support `CipherChModel` for the difference
transitions, although it shouldn't be difficult to adapt this script
to related-key difference characteristics-
"""
import functools
import warnings

from cascada.bitvector.operation import BvComp
from cascada.differential.difference import XorDiff, RXDiff
from cascada.differential import chmodel as diff_chmodel
from cascada.algebraic.value import WordValue
from cascada.algebraic import chmodel as alg_chmodel
from cascada.algebraic import characteristic as alg_characteristic
from cascada.smt.chsearch import ChFinder, ChModelAssertType, PrintingMode
from cascada.smt.pysmttypes import pysmt_model2bv_model

from cascada.primitives import speck


zip = functools.partial(zip, strict=True)

Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
cipher = Speck32
num_rounds = 4
diff_type = XorDiff


if __name__ == '__main__':
    cipher.set_num_rounds(num_rounds)

    diff_ch_model = diff_chmodel.EncryptionChModel(cipher, diff_type)
    input_mk_names = [f"mk{i}" for i in range(len(cipher.key_schedule.input_widths))]
    ks_ch_model = alg_chmodel.ChModel(cipher.key_schedule, WordValue, input_val_names=input_mk_names, prefix="rk")

    print("Symbolic difference transitions:", diff_ch_model)
    print("\nSymbolic round-key value transitions:", ks_ch_model)

    class EncryptionChModelA(alg_chmodel.EncryptionChModel):
        _prefix = "A"

    class EncryptionChModelB(alg_chmodel.EncryptionChModel):
        _prefix = "B"

    a_ch_model = EncryptionChModelA(cipher, WordValue, round_keys_prefix="C")
    b_ch_model = EncryptionChModelB(cipher, WordValue, round_keys_prefix="C")

    print("\nSymbolic value transitions (of first encryption):", a_ch_model)
    print("\nSymbolic value transition (of second encryption)", b_ch_model)

    assert str(diff_ch_model.ssa) == str(a_ch_model.ssa).replace("Ap", "dp").replace("Ax", "dx").replace("Ck", "dk")
    assert str(diff_ch_model.ssa) == str(b_ch_model.ssa).replace("Bp", "dp").replace("Bx", "dx").replace("Ck", "dk")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a_ch_finder = ChFinder(a_ch_model, ChModelAssertType.Validity, "btor", solver_seed=0)
        b_ch_finder = ChFinder(b_ch_model, ChModelAssertType.Validity, "btor", solver_seed=0)
        ks_ch_finder = ChFinder(ks_ch_model, ChModelAssertType.Validity, "btor", solver_seed=0)

    print("\nConstraints of round-key value transition:\n", ks_ch_finder.hrepr())
    print("\nConstraints of first-encryption value transition:\n", a_ch_finder.hrepr())
    print("\nConstraints of second-encryption value transition:\n", b_ch_finder.hrepr())

    initial_constraints = []

    # NOTE: disable the next block to ignore key-schedule
    initial_constraints += list(ks_ch_finder.chmodel_asserts)
    assert len(a_ch_model.external_var2val) == len(ks_ch_model.output_prop)
    for ext_a_var, ks_out_prop in zip(a_ch_model.external_var2val, ks_ch_model.output_prop):
        initial_constraints.append(BvComp(a_ch_model.external_var2val[ext_a_var].val, ks_out_prop.val))
    print("\nConnection constraints of round-keys:", initial_constraints)
    #

    initial_constraints += list(a_ch_finder.chmodel_asserts + b_ch_finder.chmodel_asserts)
    num_ic = len(initial_constraints)

    for xor_var, a_var, b_var in zip(diff_ch_model.var2prop, a_ch_model.var2prop, b_ch_model.var2prop):
        assert str(xor_var) == str(a_var).replace("A", "d").replace("C", "d")
        assert str(a_var) == str(b_var).replace("B", "A")
        if "C" in str(a_var):
            continue
        initial_constraints.append(
            BvComp(diff_ch_model.var2prop[xor_var].val,
                   diff_type.from_pair(a_ch_model.var2prop[a_var].val, b_ch_model.var2prop[b_var].val).val)
        )

    print("\nConnection constraints of differences and values:", initial_constraints[-num_ic:], "\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xor_ch_finder = ChFinder(diff_ch_model, ChModelAssertType.ValidityAndWeight, "btor",
                                 initial_constraints=initial_constraints, raise_exception_missing_var=False,
                                 exclude_zero_input_prop=True,
                                 solver_seed=0, printing_mode=PrintingMode.WeightsAndSrepr)

    # print("\nAll constraints (excluding weight constraints):")
    # print(xor_ch_finder.hrepr())

    xor_ch_found = next(xor_ch_finder.find_next_ch_increasing_weight(0))

    print("\nDifferential characteristic found:", xor_ch_found)

    solution = pysmt_model2bv_model(xor_ch_finder._last_model)

    # NOTE: disable the next block to ignore KS
    ks_ch_found = alg_characteristic.Characteristic(
        input_val=[symb_val.val.xreplace(solution) for symb_val in ks_ch_model.input_val],
        output_val=[symb_val.val.xreplace(solution) for symb_val in ks_ch_model.output_val],
        assign_outval_list=[symb_val.val.xreplace(solution) for symb_val in ks_ch_model.assign_outval2op_model],
        ch_model=ks_ch_model)
    print("\nValues of round keys found:", ks_ch_found)
    #

    a_ch_found = alg_characteristic.EncryptionCharacteristic(
        input_val=[symb_val.val.xreplace(solution) for symb_val in a_ch_model.input_val],
        output_val=[symb_val.val.xreplace(solution) for symb_val in a_ch_model.output_val],
        assign_outval_list=[symb_val.val.xreplace(solution) for symb_val in a_ch_model.assign_outval2op_model],
        ch_model=a_ch_model,
        external_vals=[symb_val.val.xreplace(solution) for symb_val in a_ch_model.external_var2val.values()])

    print("\nValues of first encryption found:", a_ch_found)

    b_ch_found = alg_characteristic.EncryptionCharacteristic(
        input_val=[symb_val.val.xreplace(solution) for symb_val in b_ch_model.input_val],
        output_val=[symb_val.val.xreplace(solution) for symb_val in b_ch_model.output_val],
        assign_outval_list=[symb_val.val.xreplace(solution) for symb_val in b_ch_model.assign_outval2op_model],
        ch_model=b_ch_model,
        external_vals=[symb_val.val.xreplace(solution) for symb_val in b_ch_model.external_var2val.values()])

    print("\nValues of second encryption found:", b_ch_found)
