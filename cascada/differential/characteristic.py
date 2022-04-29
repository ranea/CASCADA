"""Manipulate non-symbolic differential characteristics."""
import decimal
import collections
import functools
import itertools
import math
import multiprocessing
import random

from cascada.bitvector import core
from cascada.bitvector import context
from cascada.bitvector import ssa as cascada_ssa
from cascada.differential import difference
from cascada.differential import chmodel as cascada_chmodel
from cascada.differential import opmodel as cascada_opmodel

from cascada import abstractproperty


zip = functools.partial(zip, strict=True)
EmpiricalWeightData = abstractproperty.characteristic.EmpiricalWeightData


class Characteristic(abstractproperty.characteristic.Characteristic):
    """Represent differential characteristics over bit-vector functions.

    Internally, this class is a subclass of
    `abstractproperty.characteristic.Characteristic`,
    where the `Property` is a `Difference` type.

    As mentioned in `abstractproperty.characteristic.Characteristic`,
    the characteristic probability is defined as the product of the propagation
    probability (differential probability) of the `Difference` pairs (differentials)
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` over :math:`f_i`.
    If :math:`f` has external variables, the characteristic probability
    approximates the differential probability of the input-output difference pair
    of the characteristic averaged over the set of all values of the external variables.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateLeft
        >>> from cascada.differential.difference import XorDiff, RXDiff
        >>> from cascada.differential.chmodel import ChModel
        >>> from cascada.differential.characteristic import Characteristic
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(2)
        >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
        >>> zd = core.Constant(0, width=16)
        >>> mk0 = RotateLeft(Constant(1, width=16), 5)  # mk0 >>> 7 == 0b010···0
        >>> rr2 = RotateLeft(Constant(1, width=16), 14)  # rr2 == 0b010···0 == d5 == d11 (see ch_model)
        >>> xor_ch = Characteristic([mk0, zd, zd], [zd, zd, rr2], [zd, rr2, zd, zd, rr2], xor_ch_model)
        >>> xor_ch  # doctest: +NORMALIZE_WHITESPACE
        Characteristic(ch_weight=1, assignment_weights=[0, 1, 0, 0, 0],
            input_diff=[0x0020, 0x0000, 0x0000], output_diff=[0x0000, 0x0000, 0x4000],
            assign_outdiff_list=[0x0000, 0x4000, 0x0000, 0x0000, 0x4000])
        >>> list(zip(xor_ch.assignment_weights, xor_ch.tuple_assign_outdiff2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (XorDiff(0x0000), XorModelBvAdd([XorDiff(0x0000), XorDiff(0x0000)]))),
        (Decimal('1'), (XorDiff(0x4000), XorModelBvAdd([XorDiff(0x4000), XorDiff(0x0000)]))),
        (Decimal('0'), (XorDiff(0x0000), XorModelId(XorDiff(0x0000)))),
        (Decimal('0'), (XorDiff(0x0000), XorModelId(XorDiff(0x0000)))),
        (Decimal('0'), (XorDiff(0x4000), XorModelId(XorDiff(0x4000))))]
        >>> rx_ch_model = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])
        >>> zd = RXDiff(core.Constant(0, width=16))
        >>> td = RXDiff(core.Constant(3, width=16))
        >>> rx_ch = Characteristic([zd, zd, zd], [zd, zd, td], [zd, zd, zd, zd, td], rx_ch_model)
        >>> rx_ch  # doctest: +NORMALIZE_WHITESPACE
        Characteristic(ch_weight=2.829986944784033003657233894,
            assignment_weights=[1.414993472392016501828616947, 1.414993472392016501828616947, 0, 0, 0],
            input_diff=[0x0000, 0x0000, 0x0000], output_diff=[0x0000, 0x0000, 0x0003],
            assign_outdiff_list=[0x0000, 0x0000, 0x0000, 0x0000, 0x0003])
        >>> list(zip(rx_ch.assignment_weights, rx_ch.tuple_assign_outdiff2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('1.414993472392016501828616947'), (RXDiff(0x0000), RXModelBvAdd([RXDiff(0x0000), RXDiff(0x0000)]))),
        (Decimal('1.414993472392016501828616947'), (RXDiff(0x0000), RXModelBvAdd([RXDiff(0x0000), RXDiff(0x0000)]))),
        (Decimal('0'), (RXDiff(0x0000), RXModelId(RXDiff(0x0000)))),
        (Decimal('0'), (RXDiff(0x0000), RXModelId(RXDiff(0x0000)))),
        (Decimal('0'), (RXDiff(0x0003), RXModelId(RXDiff(0x0003))))]

    Attributes:
        input_diff: a list of `Difference` objects containing
            the (constant) input difference (alias of
            `abstractproperty.characteristic.Characteristic.input_prop`).
        output_diff: a list of `Difference` objects containing
            the (constant) output difference (alias of
            `abstractproperty.characteristic.Characteristic.output_prop`).
        external_diffs: a list containing the (constant) `Difference` of
            the external variables of the function (alias of
            `abstractproperty.characteristic.Characteristic.external_props`).
        tuple_assign_outdiff2op_model:  a tuple where each element is a pair
            containing: (1) the output (constant) `Difference` :math:`\Delta_{x_{i+1}}`
            of the non-trivial assignment  :math:`x_{i+1} \leftarrow f_i(x_i)`
            and (2) the `differential.opmodel.OpModel` of this assignment with
            a (constant) input `Difference` :math:`\Delta_{x_{i}}` (alias of
            `abstractproperty.characteristic.Characteristic.tuple_assign_outprop2op_model`).
        free_diffs: a list of (symbolic) `Difference` objects of
            the `Characteristic.ch_model`, whose values do not affect the
            characteristic, and were replaced by constant differences in `input_diff`,
            `output_diff`, `external_diffs` or `tuple_assign_outdiff2op_model`
            (alias of `abstractproperty.characteristic.Characteristic.free_props`).
        var_diff2ct_diff: a `collections.OrderedDict` mapping each
            symbolic `Difference` in the trail to its constant difference (alias of
            `abstractproperty.characteristic.Characteristic.var_prop2ct_prop`).

    """
    _prop_label = "diff"  # for str and vrepr

    @property
    def input_diff(self):
        return self.input_prop

    @property
    def output_diff(self):
        return self.output_prop

    @property
    def external_diffs(self):
        return self.external_props

    @property
    def tuple_assign_outdiff2op_model(self):
        return self.tuple_assign_outprop2op_model

    @property
    def free_diffs(self):
        return self.free_props

    @property
    def var_diff2ct_diff(self):
        return self.var_prop2ct_prop

    def __init__(self, input_diff, output_diff, assign_outdiff_list,
                 ch_model, external_diffs=None, free_diffs=None,
                 empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        super().__init__(
            input_prop=input_diff, output_prop=output_diff, assign_outprop_list=assign_outdiff_list,
            ch_model=ch_model, external_props=external_diffs, free_props=free_diffs,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid
        )

    def vrepr(self, ignore_external_diffs=False):
        """Return an executable string representation.

        See also `abstractproperty.characteristic.Characteristic.vrepr`.

            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateLeft
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> zd = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 5)
            >>> aux = RotateLeft(Constant(1, width=16), 14)
            >>> xor_ch = Characteristic([mk0, zd, zd], [zd, zd, aux], [zd, aux, zd, zd, aux], xor_ch_model)
            >>> xor_ch.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            "Characteristic(input_diff=[Constant(0x0020, width=16), Constant(0x0000, width=16), Constant(0x0000, width=16)],
                output_diff=[Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x4000, width=16)],
                assign_outdiff_list=[Constant(0x0000, width=16), Constant(0x4000, width=16),
                    Constant(0x0000, width=16), Constant(0x0000, width=16), Constant(0x4000, width=16)],
                ch_model=ChModel(func=SpeckKeySchedule.set_num_rounds_and_return(2), diff_type=XorDiff,
                    input_diff_names=['dmk0', 'dmk1', 'dmk2'], prefix='dx'))"
            >>> xor_ch.srepr()
            'Ch(w=1, id=0020 0000 0000, od=0000 0000 4000)'

        """
        # ignore ignore_external_diffs=ignore_external_diffs (parent class might be abstract)
        return super().vrepr(ignore_external_diffs)

    def split(self, diff_separators):
        """Split into multiple `Characteristic` objects given the list of difference separators.

        See also `abstractproperty.characteristic.Characteristic.split`.

            >>> from cascada.bitvector.core import Constant, Variable
            >>> from cascada.bitvector.operation import RotateLeft
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.differential.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> tuple(xor_ch_model.ssa.assignments.items())  # doctest: +NORMALIZE_WHITESPACE
            ((dx0, dmk1 >>> 7), (dx1, dx0 + dmk2), (dx2, dmk2 <<< 2), (dx3, dx2 ^ dx1),
            (dx4, dmk0 >>> 7), (dx5, dx4 + dx3), (dx6, dx5 ^ 0x0001), (dx7, dx3 <<< 2), (dx8, dx7 ^ dx6),
            (dmk2_out, Id(dmk2)), (dx3_out, Id(dx3)), (dx8_out, Id(dx8)))
            >>> diff_separators = [ (XorDiff(Variable("dx2", width=16)), XorDiff(Variable("dx3", width=16))), ]
            >>> zd = core.Constant(0, width=16)
            >>> mk0 = RotateLeft(Constant(1, width=16), 5)  # mk0 >>> 7 == 0b010···0
            >>> rr2 = RotateLeft(Constant(1, width=16), 14)  # rr2 == 0b010···0 == d5 == d11 (see ch_model)
            >>> xor_ch = Characteristic([mk0, zd, zd], [zd, zd, rr2], [zd, rr2, zd, zd, rr2], xor_ch_model)
            >>> for ch in xor_ch.split(diff_separators): print(ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0],
                input_diff=[0x0020, 0x0000, 0x0000], output_diff=[0x0020, 0x0000, 0x0000],
                assign_outdiff_list=[0x0000, 0x0020, 0x0000, 0x0000])
            Characteristic(ch_weight=1, assignment_weights=[1, 0, 0, 0],
                input_diff=[0x0020, 0x0000, 0x0000], output_diff=[0x0000, 0x0000, 0x4000],
                assign_outdiff_list=[0x4000, 0x0000, 0x0000, 0x4000])

        """
        return super().split(diff_separators)

    @staticmethod
    def _num_right_inputs2weight(num_right_inputs, num_input_samples):
        if num_right_inputs == 0:
            return math.inf
        else:
            pr = decimal.Decimal(num_right_inputs) / num_input_samples
            weight = - cascada_opmodel.log2_decimal(pr)
            assert weight >= 0
            return weight

    def _get_empirical_ch_weights_C(self, num_input_samples, num_external_samples, seed, num_parallel_processes,
                                    verbose=False, get_sampled_inputs=False):
        """Return a list of empirical weights (one for each ``num_external_samples``)
        by compiling and executing C code.

            .. Implementation details:
                Calling the C function get_num_right_inputs() returns the EW for one
                external sample, which is randomly sampled at the beginning
                get_num_right_inputs(). Thus, calling num_external_samples-times
                get_num_right_inputs() produces the list of empirical weights.
                This avoids returning a list in the C code.

                The only argument of get_num_right_inputs is the seed.
                The differences are hardcoded, and the input samples
                are randomly sampled using the hardcoded differences.

                PRNG used: http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/VERSIONS/C-LANG/mt19937-64.c

        """
        import uuid
        from cascada.bitvector.printing import BvCCodePrinter

        assert not (verbose and get_sampled_inputs)

        assert 1 <= num_input_samples <= 2 ** sum(d.val.width for d in self.input_diff)
        if self.external_diffs:
            assert 1 <= num_external_samples <= 2 ** sum(d.val.width for d in self.external_diffs)
        else:
            assert num_external_samples == 0

        if num_input_samples.bit_length() > 64:
            raise ValueError("max num_input_samples supported is 2**64 - 1")

        width2C_type = BvCCodePrinter._width2C_type
        get_and_mask_C_code = BvCCodePrinter._get_and_mask_C_code

        ctype_num_right_inputs = width2C_type(num_input_samples.bit_length())

        # 1 - Build the C code

        header = f"{ctype_num_right_inputs} get_num_right_inputs(uint64_t seed);"

        if hasattr(self.ch_model, "_unwrapped_ch_model"):
            # avoid gcc error due to huge line statements
            ssa = self.ch_model._unwrapped_ch_model.ssa
        else:
            ssa = self.ch_model.ssa
        eval_ssa_code_function_name = "eval_ssa"
        _, eval_ssa_code_body = ssa.get_C_code(eval_ssa_code_function_name)  # ignore header

        from pathlib import Path
        mt19937_filename = Path(__file__).parent.resolve() / "mt19937.c"

        def rand(my_width):
            if my_width == 1:
                rand_str = "genrand64_int1"
            elif my_width <= 8:
                rand_str = "genrand64_int8"
            elif my_width <= 16:
                rand_str = "genrand64_int16"
            elif my_width <= 32:
                rand_str = "genrand64_int32"
            elif my_width <= 64:
                rand_str = "genrand64_int64"
            else:
                raise ValueError("random bit-vectors with more than 64 bits are not supported")
            return f"{rand_str}(){get_and_mask_C_code(var.width)}"

        # stdint already in eval_ssa and in mt19937
        body = f'#include "{mt19937_filename}"\n{eval_ssa_code_body}'
        body += f"\n{ctype_num_right_inputs} get_num_right_inputs(uint64_t seed){{"

        to_sample_all_iv = num_input_samples == 2 ** sum(d.val.width for d in self.input_diff)

        if self.external_diffs or not to_sample_all_iv:
            body += "\n\tinit_genrand64(seed);"

        body += f"\n\t{ctype_num_right_inputs} num_right_inputs = 0U;"

        # to_sample_all_ev cannot be used since get_num_right_inputs only returns 1 EW
        if ssa.external_vars:
            ev_other_names = [f"{v.crepr()}_{uuid.uuid4().hex}" for v in ssa.external_vars]
            for i in range(len(ssa.external_vars)):
                var = ssa.external_vars[i]
                body += f"\n\t{width2C_type(var.width)} {var.crepr()} = {rand(var.width)};"
                other_var = core.Variable(ev_other_names[i], var.width)
                other_val = self.external_diffs[i].get_pair_element(var)
                body += f"\n\t{width2C_type(other_var.width)} {other_var.crepr()} = {other_val.crepr()};"
                if verbose:
                    body += f'\n\tprintf("\\nexternal_vars[%u] = (%x, %x)", {i}U, {var.crepr()}, {other_var.crepr()});'
            ev_args = ', '.join([v.crepr() for v in ssa.external_vars]) + ", "
            ev_other_args = ', '.join(ev_other_names) + ", "
        else:
            ev_args = ""
            ev_other_args = ""

        # start for

        if not to_sample_all_iv:
            body += f"\n\tfor ({ctype_num_right_inputs} i = 0U; i < {num_input_samples}U; ++i) {{"

        iv_other_names = [f"{v.crepr()}_{uuid.uuid4().hex}" for v in ssa.input_vars]
        for i in range(len(ssa.input_vars)):
            var = ssa.input_vars[i]
            if to_sample_all_iv:
                v = var.crepr()
                body += f"\n\t\tfor ({width2C_type(var.width+1)} {v} = 0U; {v} < {2**(var.width)}U; ++{v}) {{"
            else:
                body += f"\n\t\t{width2C_type(var.width)} {var.crepr()} = {rand(var.width)};"
            other_var = core.Variable(iv_other_names[i], var.width)
            other_val = self.input_diff[i].get_pair_element(var)
            body += f"\n\t\t{width2C_type(other_var.width)} {other_var.crepr()} = {other_val.crepr()};"
            if verbose:
                body += f'\n\t\tprintf("\\ninput sample i=%u | input_vars[%u] = (%x, %x)", ' \
                        f'i, {i}U, {var.crepr()}, {other_var.crepr()});'
        if get_sampled_inputs:
            for i in range(len(ssa.input_vars)):
                aux_prefix = "[" if i == 0 else ""
                aux_suffix = "]," if i == len(ssa.input_vars) - 1 else ""
                body += f'\n\t\tprintf("{aux_prefix}0x%x,{aux_suffix}", {ssa.input_vars[i].crepr()});'
        iv_args = ', '.join([v.crepr() for v in ssa.input_vars])
        iv_other_args = ', '.join(iv_other_names)

        ov_other_names = [f"{v.crepr()}_{uuid.uuid4().hex}" for v in ssa.output_vars]
        for i in range(len(ssa.output_vars)):
            var = ssa.output_vars[i]
            other_var = core.Variable(ov_other_names[i], var.width)
            # var passed by reference later (no need to declare them as pointers)
            body += f"\n\t\t{width2C_type(var.width)} {var.crepr()}, {other_var.crepr()};"
        ov_args = ', '.join(["&" + v.crepr() for v in ssa.output_vars])
        ov_other_args = ', '.join(["&" + name for name in ov_other_names])

        body += f"\n\t\t{eval_ssa_code_function_name}({iv_args}, {ev_args}{ov_args});"
        body += f"\n\t\t{eval_ssa_code_function_name}({iv_other_args}, {ev_other_args}{ov_other_args});"

        if_conditions = []
        for i in range(len(ssa.output_vars)):
            var = ssa.output_vars[i]
            other_var = core.Variable(ov_other_names[i], var.width)
            current_d = self.ch_model.diff_type.from_pair(var, other_var).val
            expected_d = self.output_diff[i].val
            # casting current_d is necessary for the comparison
            if_conditions.append(f"( ({width2C_type(current_d.width)})({current_d.crepr()}) == {expected_d.crepr()} )")
            if verbose:
                body += f'\n\t\tprintf("\\n                 | output_vars[%u] = (%x, %x)", ' \
                        f'{i}U, {var.crepr()}, {other_var.crepr()});'
                body += f'\n\t\tprintf("\\n                 | current_expected[%u] = (%x, %x)", ' \
                        f'{i}U, {current_d.crepr()}, {expected_d.crepr()});'
        body += f"\n\t\tif ( {' && '.join(if_conditions)} ){{"
        body += "\n\t\t\tnum_right_inputs += 1U;"
        body += "\n\t\t}"

        if to_sample_all_iv:
            body += "\n\t\t" + "}"*len(ssa.input_vars)
        else:
            body += "\n\t}"

        # end for

        if verbose:
            body += f'\n\tprintf("\\nnum_right_inputs = %u\\n", num_right_inputs);'

        body += "\n\treturn num_right_inputs;\n}"

        # 2 - Run the C code

        if num_parallel_processes is None or num_external_samples <= 1:
            pymod, tmpdir = cascada_ssa._compile_C_code(header, body, verbose=verbose)

            PRNG = random.Random()
            PRNG.seed(seed)

            aux_empirical_weights = []
            for _ in range(max(1, num_external_samples)):  # num_external_samples can be 0
                num_right_inputs = pymod.lib.get_num_right_inputs(PRNG.randrange(0, 2**64))
                aux_empirical_weights.append(
                    self._num_right_inputs2weight(num_right_inputs, num_input_samples)
                )
        else:
            lib_path, module_name, tmpdir = cascada_ssa._compile_C_code(header, body, return_unloaded=True, verbose=verbose)

            num_parallel_processes = min(num_parallel_processes, num_external_samples)

            PRNG = random.Random()
            PRNG.seed(seed)
            external_seeds = [PRNG.randrange(0, 2**64) for _ in range(num_external_samples)]

            chunk = num_external_samples // num_parallel_processes
            if num_external_samples % num_parallel_processes == 0:
                extra_chunk = 0
            else:
                extra_chunk = num_external_samples % num_parallel_processes
            assert chunk*(num_parallel_processes-1) + chunk + extra_chunk == num_external_samples

            aux_empirical_weights = []
            with multiprocessing.Pool(num_parallel_processes) as pool:
                async_results = [None for _ in range(num_parallel_processes)]
                for ar_index in range(len(async_results)):
                    async_results[ar_index] = pool.apply_async(
                        _run_C_code_get_num_right_inputs,
                        (
                            external_seeds[ar_index*chunk],
                            module_name,
                            lib_path,
                            chunk+extra_chunk if ar_index == len(async_results) - 1 else chunk
                        )
                    )
                pool.close()
                pool.join()  # blocking call
                for ar_index in range(len(async_results)):
                    # type(process_list[process_index]) == AsyncResult
                    # and AsyncResult.get() blocks until result obtained
                    list_num_right_inputs = async_results[ar_index].get()
                    aux_empirical_weights.extend(
                        [self._num_right_inputs2weight(nri, num_input_samples) for nri in list_num_right_inputs]
                    )

            assert len(aux_empirical_weights) == num_external_samples

        tmpdir.cleanup()

        return aux_empirical_weights

    def _get_empirical_ch_weights_Python(self, num_input_samples, num_external_samples, seed, list_input_samples=None):
        """Return a list of empirical weights (one for each ``num_external_samples``)."""
        assert 1 <= num_input_samples <= 2 ** sum(d.val.width for d in self.input_diff)
        if self.external_diffs:
            assert 1 <= num_external_samples <= 2 ** sum(d.val.width for d in self.external_diffs)
        else:
            assert num_external_samples == 0

        PRNG = random.Random()
        PRNG.seed(seed)

        def get_random_bv(width):
            return core.Constant(PRNG.randrange(2 ** width), width)

        aux_empirical_weights = []

        with context.Simplification(False), context.Cache(False):
            input_widths = [d.val.width for d in self.input_diff]
            if list_input_samples is not None:
                assert len(list_input_samples) == num_input_samples
                def get_next_input_pair():
                    for pt in list_input_samples:
                        assert len(self.input_diff) == len(pt)
                        other_pt = []
                        for aux_i, diff in enumerate(self.input_diff):
                            pt[aux_i] = core.Constant(pt[aux_i], diff.val.width)
                            other_pt.append(diff.get_pair_element(pt[aux_i]))
                        yield pt, other_pt
            elif num_input_samples == 2 ** sum(input_widths):
                # input_samples == whole input space
                def get_next_input_pair():
                    for input_sample in itertools.product(*[range(2**w) for w in input_widths]):
                        pt = [core.Constant(p_i, w) for p_i, w in zip(input_sample, input_widths)]
                        other_pt = [diff.get_pair_element(pt[i]) for i, diff in enumerate(self.input_diff)]
                        yield pt, other_pt
            else:
                def get_next_input_pair():
                    for _ in range(num_input_samples):
                        pt = []
                        other_pt = []
                        for diff in self.input_diff:
                            random_bv = get_random_bv(diff.val.width)
                            pt.append(random_bv)
                            other_pt.append(diff.get_pair_element(random_bv))
                        yield pt, other_pt

            found_external_vars = len(self.external_diffs) > 0
            if found_external_vars:
                external_widths = [d.val.width for d in self.external_diffs]
                if num_external_samples == 2 ** sum(external_widths):
                    # external_samples == whole external space
                    external_space = itertools.product(*[range(2**w) for w in external_widths])
                else:
                    external_space = range(num_external_samples)  # num_external_samples > 0

                def get_next_ssa_pair_external_fixed():
                    for external_sample in external_space:
                        ssa = self.ch_model.ssa.copy()
                        v2c = collections.OrderedDict()
                        if isinstance(external_sample, int):
                            for v in ssa.external_vars:
                                v2c[v] = get_random_bv(v.width)
                        else:
                            for i_v, v in enumerate(ssa.external_vars):
                                v2c[v] = core.Constant(external_sample[i_v], external_widths[i_v])
                        for outvar in ssa.assignments:
                            ssa.assignments[outvar] = ssa.assignments[outvar].xreplace(v2c)
                        ssa.external_vars = []
                        if all(d.val == 0 for d in self.external_diffs) and \
                                self.ch_model.diff_type == difference.XorDiff:
                            other_ssa = ssa
                        else:
                            # RXDiff 0-difference does not mean ssa == other_ssa
                            other_ssa = self.ch_model.ssa.copy()
                            other_v2c = collections.OrderedDict()
                            for i, (v, c) in enumerate(v2c.items()):
                                other_v2c[v] = self.external_diffs[i].get_pair_element(v2c[v])
                            for outvar in other_ssa.assignments:
                                other_ssa.assignments[outvar] = other_ssa.assignments[outvar].xreplace(other_v2c)
                            other_ssa.external_vars = []
                        # print("# new external samples")
                        # print("v2c:", v2c)
                        # print("other_v2c:", other_v2c)  # might throw error
                        # print("external_diffs:", self.external_diffs)
                        # print("base ssa:", self.ch_model.ssa)
                        # print("ssa:", ssa)
                        # print("other_ssa:", other_ssa)  # might throw error
                        yield ssa, other_ssa

                gen_next_ssa_pair_external_fixed = get_next_ssa_pair_external_fixed()

            for _ in range(max(1, num_external_samples)):  #  num_external_samples can be 0
                num_right_inputs = 0

                if found_external_vars:
                    ssa, other_ssa = next(gen_next_ssa_pair_external_fixed)
                    # print("> ssa:", ssa)
                    # print("> other_ssa:", other_ssa)

                for pt, other_pt in get_next_input_pair():
                    if not found_external_vars:  # func() faster than SSA.eval()
                        try:
                            ct, other_ct = self.ch_model.func(*pt), self.ch_model.func(*other_pt)
                        except ValueError as e:
                            if str(e).startswith("expected a tuple of Constant values returned"):
                                # func might return redundant symbolic outputs (e.g., k ^ k = 0)
                                ct, other_ct = self.ch_model.ssa.eval(pt), self.ch_model.ssa.eval(other_pt)
                            else:
                                raise e
                    else:
                        ct, other_ct = ssa.eval(pt), other_ssa.eval(other_pt)

                    for x in itertools.chain(ct, other_ct):
                        if not isinstance(x, core.Constant):
                            raise ValueError(f"found symbolic output in ct={ct} or other_ct={other_ct}")

                    # print("pt, other_pt:", pt, "\t", other_pt)
                    # print("ct, other_ct:", ct, "\t", other_ct)
                    # print("expect diff:", self.output_diff)
                    # print("diff found :", [self.ch_model.diff_type.from_pair(ct[i], other_ct[i]) for i in range(len(ct))])
                    # print()

                    for i, diff in enumerate(self.output_diff):
                        if self.ch_model.diff_type.from_pair(ct[i], other_ct[i]) != diff:
                            break
                    else:
                        num_right_inputs += 1

                aux_empirical_weights.append(
                    self._num_right_inputs2weight(num_right_inputs, num_input_samples)
                )

        return aux_empirical_weights

    def _get_empirical_data_complexity(self, num_input_samples, num_external_samples):
        if num_input_samples is None:
            # data complexity = small_ct * p^(-1), p == ch. probability
            # w = -log2(p) = log2(p^(-1)) ==> 2**w = p^(-1)
            p = 2 ** (self.ch_weight + self.ch_model.error())
            # small_ct == number of input bits
            small_ct = sum(d.val.width for d in self.ch_model.input_diff)
            num_input_samples = int(math.ceil(small_ct * p))
        num_input_samples = max(
            1, min(num_input_samples, 2**sum(d.val.width for d in self.ch_model.input_diff)))

        if not self.external_diffs:
            num_external_samples = 0
        else:
            if num_external_samples is None:
                # an external sample for each external bit
                num_external_samples = sum([d.width for d in self.ch_model.external_var2diff])
            num_external_samples = max(
                1, min(num_external_samples, 2 ** sum(d.width for d in self.ch_model.external_var2diff)))
        return num_input_samples, num_external_samples

    def compute_empirical_ch_weight(self, num_input_samples=None, num_external_samples=None,
                                    split_by_max_weight=None, split_by_rounds=False,
                                    seed=None, C_code=False, num_parallel_processes=None):
        """Compute and store the empirical weight.

        The main description of this method can be read from
        `abstractproperty.characteristic.Characteristic.compute_empirical_ch_weight`,
        simply by replacing `Property` by `Difference` and
        input-output pair by differential.

        The basic subroutine in this case consists of computing the
        fraction of right pairs for ``num_input_samples`` sampled input pairs.
        An input pair with difference `input_diff` is a right pair
        if `output_diff` is the difference of the output pair obtained
        from evaluating the input pair through the underlying bit-vector function.


            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateLeft
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.differential.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> rx_ch_model = ChModel(Speck32_KS, RXDiff, ["dmk0", "dmk1", "dmk2"])
            >>> zd = RXDiff(core.Constant(0, width=16))
            >>> td = RXDiff(core.Constant(3, width=16))
            >>> rx_ch = Characteristic([zd, zd, zd], [zd, zd, td], [zd, zd, zd, zd, td], rx_ch_model)
            >>> rx_ch.compute_empirical_ch_weight(seed=0)
            >>> rx_ch.empirical_ch_weight
            Decimal('3.033853610869555448012695123')
            >>> for data in rx_ch.empirical_data_list: print(data)  # doctest: +NORMALIZE_WHITESPACE
            EmpiricalWeightData(weight_avg_aux_prs=3.033853610869555448012695123,
                num_aux_weights=1, num_inf_aux_weights=0, num_input_samples=1466, seed=0, C_code=False)
            >>> rx_ch.compute_empirical_ch_weight(seed=0, C_code=True)
            >>> rx_ch.empirical_ch_weight
            Decimal('2.947813779802864030440499920')
            >>> rx_ch.compute_empirical_ch_weight(split_by_max_weight=1.5, seed=0)
            >>> rx_ch.empirical_ch_weight
            Decimal('2.682204698298087862793386808')
            >>> for data in rx_ch.empirical_data_list: print(data)  # doctest: +NORMALIZE_WHITESPACE
            EmpiricalWeightData(weight_avg_aux_prs=1.327361980937990421954710722,
                num_aux_weights=1, num_inf_aux_weights=0, num_input_samples=266, seed=0, C_code=False)
            EmpiricalWeightData(weight_avg_aux_prs=1.354842717360097440838676086,
                num_aux_weights=1, num_inf_aux_weights=0, num_input_samples=266, seed=0, C_code=False)

        """
        super().compute_empirical_ch_weight(
            num_input_samples=num_input_samples, num_external_samples=num_external_samples,
            split_by_max_weight=split_by_max_weight, split_by_rounds=split_by_rounds,
            seed=seed, C_code=C_code, num_parallel_processes=num_parallel_processes)

    @classmethod
    def random(cls, ch_model, seed, external_diffs=None):
        """Return a random `Characteristic` with given `differential.chmodel.ChModel`.

        See also `abstractproperty.characteristic.Characteristic.random`.

            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> from cascada.differential.characteristic import Characteristic
            >>> from cascada.primitives import speck
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(2)
            >>> xor_ch_model = ChModel(Speck32_KS, XorDiff, ["dmk0", "dmk1", "dmk2"])
            >>> Characteristic.random(xor_ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=25, assignment_weights=[13, 12, 0, 0, 0],
                input_diff=[0xc53e, 0xd755, 0x14ba], output_diff=[0x14ba, 0xa6ec, 0x2ce0],
                assign_outdiff_list=[0xf404, 0xb752, 0x14ba, 0xa6ec, 0x2ce0])

        """
        return super().random(ch_model, seed, external_diffs)


class EncryptionCharacteristic(abstractproperty.characteristic.EncryptionCharacteristic, Characteristic):
    """Represent differential characteristics over encryption functions.

    Given a `Cipher`, an `EncryptionCharacteristic` is an XOR
    differential characteristic  (see `Characteristic`) over
    the `Cipher.encryption` in the single-key setting
    (where the `Cipher.key_schedule` is ignored and round key differences
    are set to zero).

    .. note::

        `EncryptionCharacteristic` only supports `XorDiff`
        (see `differential.chmodel.EncryptionChModel`).

    The propagation probability of an `EncryptionCharacteristic` is also
    called in the literature the expected differential probability (EDP)
    of a characteristic (see https://eprint.iacr.org/2005/212).

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateRight, RotateLeft
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import EncryptionChModel
        >>> from cascada.differential.characteristic import EncryptionCharacteristic
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = EncryptionChModel(Speck32, XorDiff)
        >>> zd = core.Constant(0, width=16)
        >>> rr1 = RotateRight(Constant(1, width=16), 1)  # rr1 == 0b100···0
        >>> rr2 = RotateRight(Constant(1, width=16), 2)  # rr2 == 0b010···0
        >>> dp0 = RotateLeft(rr1, 7)  # dp0 >>> 7 == rr1
        >>> dp1 = rr1
        >>> dx6 = Constant(0x0002, width=16)
        >>> dx11 = Constant(0x000a, width=16)
        >>> ch = EncryptionCharacteristic([dp0, dp1], [dx6, dx11], [zd, dx6, dx6, dx11], ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        EncryptionCharacteristic(ch_weight=1, assignment_weights=[0, 1, 0, 0],
            input_diff=[0x0040, 0x8000], output_diff=[0x0002, 0x000a], external_diffs=[0x0000, 0x0000],
            assign_outdiff_list=[0x0000, 0x0002, 0x0002, 0x000a])
        >>> list(zip(ch.assignment_weights, ch.tuple_assign_outdiff2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (XorDiff(0x0000), XorModelBvAdd([XorDiff(0x8000), XorDiff(0x8000)]))),
        (Decimal('1'), (XorDiff(0x0002), XorModelBvAdd([XorDiff(0x0000), XorDiff(0x0002)]))),
        (Decimal('0'), (XorDiff(0x0002), XorModelId(XorDiff(0x0002)))),
        (Decimal('0'), (XorDiff(0x000a), XorModelId(XorDiff(0x000a))))]


    .. Implementation details:

        __init__ contains the argument ``external_diffs`` to match the signature
        of `abstractproperty.characteristic.EncryptionCharacteristic`, but
        the round key differences are always set to zero.

    """

    def __init__(self, input_diff, output_diff, assign_outdiff_list, ch_model, external_diffs=None,
                 free_diffs=None, empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        assert isinstance(ch_model, cascada_chmodel.EncryptionChModel)
        assert all(d.val == 0 for d in ch_model.external_var2diff.values())
        assert ch_model.diff_type == difference.XorDiff
        assert external_diffs is None or \
               (all(d.val == 0 for d in external_diffs) and
                len(external_diffs) == len(ch_model.external_var2diff))
        external_diffs = tuple(ch_model.external_var2diff.values())
        super().__init__(
            input_diff, output_diff, assign_outdiff_list, ch_model, external_props=external_diffs, free_props=free_diffs,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid)

    @classmethod
    def random(cls, ch_model, seed):
        """Return a random `EncryptionCharacteristic` with given `differential.chmodel.EncryptionChModel`.

        The round key differences are set to zero.

        See also `abstractproperty.characteristic.EncryptionCharacteristic.random`.

            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import EncryptionChModel
            >>> from cascada.differential.characteristic import EncryptionCharacteristic
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = EncryptionChModel(Speck32, XorDiff)
            >>> EncryptionCharacteristic.random(ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            EncryptionCharacteristic(ch_weight=24, assignment_weights=[13, 11, 0, 0],
                input_diff=[0xc53e, 0xd755], output_diff=[0xf43a, 0xc051], external_diffs=[0x0000, 0x0000],
                assign_outdiff_list=[0x904d, 0xf43a, 0xf43a, 0xc051])

        """
        return super().random(ch_model, seed)


class CipherCharacteristic(abstractproperty.characteristic.CipherCharacteristic):
    """Represent related-key differential characteristics over ciphers.

    A `CipherCharacteristic` is a related-key differential characteristic of a
    block cipher, given by one `Characteristic` over the `Cipher.key_schedule`
    and another `Characteristic` over the `Cipher.encryption`.

    The related-key setting means that the round key differences in the encryption
    characteristic (the differences of the external variables of the encryption `SSA`)
    are set to the output differences of the key-schedule characteristic.

        >>> from cascada.bitvector.core import Constant
        >>> from cascada.bitvector.operation import RotateRight, RotateLeft
        >>> from cascada.differential.difference import XorDiff, RXDiff
        >>> from cascada.differential.chmodel import CipherChModel
        >>> from cascada.differential.characteristic import CipherCharacteristic
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> xor_ch_model = CipherChModel(Speck32, XorDiff)
        >>> zd = core.Constant(0, width=16)
        >>> rr1 = RotateRight(Constant(1, width=16), 1)  # rr1 == 0b100···0
        >>> rr2 = RotateRight(Constant(1, width=16), 2)  # rr2 == 0b010···0
        >>> mk0 = RotateLeft(Constant(1, width=16), 5)  # mk0 >>> 7 == 0b010···0
        >>> dp0 = RotateLeft(rr1, 7)  # dp0 >>> 7 == rr1
        >>> dp1 = rr1
        >>> dx6 = Constant(0x0002, width=16)
        >>> dx11 = Constant(0x000a, width=16)
        >>> ch = CipherCharacteristic(
        ...     [mk0, zd], [zd, rr2], [rr2, zd, rr2],
        ...     [dp0, dp1], [dx6^rr2, dx11^rr2], [zd, dx6, dx6^rr2, dx11^rr2], xor_ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=1,
            assignment_weights=[1, 0, 0],
            input_diff=[0x0020, 0x0000], output_diff=[0x0000, 0x4000],
            assign_outdiff_list=[0x4000, 0x0000, 0x4000]),
        enc_characteristic=Characteristic(ch_weight=1,
            assignment_weights=[0, 1, 0, 0],
            input_diff=[0x0040, 0x8000], output_diff=[0x4002, 0x400a], external_diffs=[0x0000, 0x4000],
            assign_outdiff_list=[0x0000, 0x0002, 0x4002, 0x400a]))
        >>> ks_ch, enc_ch = ch.ks_characteristic, ch.enc_characteristic
        >>> list(zip(ks_ch.assignment_weights, ks_ch.tuple_assign_outdiff2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('1'), (XorDiff(0x4000), XorModelBvAdd([XorDiff(0x4000), XorDiff(0x0000)]))),
        (Decimal('0'), (XorDiff(0x0000), XorModelId(XorDiff(0x0000)))),
        (Decimal('0'), (XorDiff(0x4000), XorModelId(XorDiff(0x4000))))]
        >>> list(zip(enc_ch.assignment_weights, enc_ch.tuple_assign_outdiff2op_model))  # doctest: +NORMALIZE_WHITESPACE
        [(Decimal('0'), (XorDiff(0x0000), XorModelBvAdd([XorDiff(0x8000), XorDiff(0x8000)]))),
        (Decimal('1'), (XorDiff(0x0002), XorModelBvAdd([XorDiff(0x0000), XorDiff(0x0002)]))),
        (Decimal('0'), (XorDiff(0x4002), XorModelId(XorDiff(0x4002)))),
        (Decimal('0'), (XorDiff(0x400a), XorModelId(XorDiff(0x400a))))]
        >>> cipher_ch_model = CipherChModel(Speck32, RXDiff)
        >>> zd = RXDiff(core.Constant(0, width=16))
        >>> ch = CipherCharacteristic(
        ...     [zd, zd], [zd, zd], [zd, zd, zd],
        ...     [zd, zd], [zd, zd], [zd, zd, zd, zd], cipher_ch_model)
        >>> ch  # doctest: +NORMALIZE_WHITESPACE
        CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=1.414993472392016501828616947,
            assignment_weights=[1.414993472392016501828616947, 0, 0],
            input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000],
            assign_outdiff_list=[0x0000, 0x0000, 0x0000]),
        enc_characteristic=Characteristic(ch_weight=2.829986944784033003657233894,
            assignment_weights=[1.414993472392016501828616947, 1.414993472392016501828616947, 0, 0],
            input_diff=[0x0000, 0x0000], output_diff=[0x0000, 0x0000], external_diffs=[0x0000, 0x0000],
            assign_outdiff_list=[0x0000, 0x0000, 0x0000, 0x0000]))
        >>> ch.srepr()  # doctest: +NORMALIZE_WHITESPACE
        'Ch(ks_ch=Ch(w=1.415, id=0000 0000, od=0000 0000), enc_ch=Ch(w=2.830, id=0000 0000, od=0000 0000))'

    """
    _Characteristic_cls = Characteristic

    def __init__(
            self, ks_input_diff, ks_output_diff, ks_assign_outdiff_list,
            enc_input_diff, enc_output_diff, enc_assign_outdiff_list,
            cipher_ch_model, ks_free_diffs=None, enc_free_diffs=None,
            ks_empirical_ch_weight=None, ks_empirical_data_list=None,
            enc_empirical_ch_weight=None, enc_empirical_data_list=None,
            ks_is_valid=True, enc_is_valid=True
    ):
        # avoid *_props=*_props (super might not abstract)
        super().__init__(
            ks_input_diff, ks_output_diff, ks_assign_outdiff_list,
            enc_input_diff, enc_output_diff, enc_assign_outdiff_list,
            cipher_ch_model, ks_free_diffs, enc_free_diffs,
            ks_empirical_ch_weight=ks_empirical_ch_weight, ks_empirical_data_list=ks_empirical_data_list,
            enc_empirical_ch_weight=enc_empirical_ch_weight, enc_empirical_data_list=enc_empirical_data_list,
            ks_is_valid=ks_is_valid, enc_is_valid=enc_is_valid
        )

    def vrepr(self):
        """Return an executable string representation.

        See also `abstractproperty.characteristic.CipherCharacteristic.vrepr`.

            >>> from cascada.bitvector.core import Constant
            >>> from cascada.bitvector.operation import RotateRight, RotateLeft
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.differential.characteristic import CipherCharacteristic
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> xor_ch_model = CipherChModel(Speck32, XorDiff)
            >>> zd = core.Constant(0, width=16)
            >>> rr1 = RotateRight(Constant(1, width=16), 1)  # rr1 == 0b100···0
            >>> rr2 = RotateRight(Constant(1, width=16), 2)  # rr2 == 0b010···0
            >>> mk0 = RotateLeft(Constant(1, width=16), 5)  # mk0 >>> 7 == 0b010···0
            >>> dp0 = RotateLeft(rr1, 7)  # dp0 >>> 7 == rr1
            >>> dp1 = rr1
            >>> dx6 = Constant(0x0002, width=16)
            >>> dx11 = Constant(0x000a, width=16)
            >>> ch = CipherCharacteristic(
            ...     [mk0, zd], [zd, rr2], [rr2, zd, rr2],
            ...     [dp0, dp1], [dx6^rr2, dx11^rr2], [zd, dx6, dx6^rr2, dx11^rr2], xor_ch_model)
            >>> ch.vrepr()  # doctest: +NORMALIZE_WHITESPACE
            'CipherCharacteristic(ks_input_diff=[Constant(0x0020, width=16), Constant(0x0000, width=16)],
                ks_output_diff=[Constant(0x0000, width=16), Constant(0x4000, width=16)],
                ks_assign_outdiff_list=[Constant(0x4000, width=16), Constant(0x0000, width=16), Constant(0x4000, width=16)],
                enc_input_diff=[Constant(0x0040, width=16), Constant(0x8000, width=16)],
                enc_output_diff=[Constant(0x4002, width=16), Constant(0x400a, width=16)],
                enc_assign_outdiff_list=[Constant(0x0000, width=16), Constant(0x0002, width=16),
                    Constant(0x4002, width=16), Constant(0x400a, width=16)],
                cipher_ch_model=CipherChModel(cipher=SpeckCipher.set_num_rounds_and_return(2), diff_type=XorDiff))'
            >>> ch.srepr()
            'Ch(ks_ch=Ch(w=1, id=0020 0000, od=0000 4000), enc_ch=Ch(w=1, id=0040 8000, od=4002 400a))'

        """
        return super().vrepr()

    @classmethod
    def random(cls, cipher_ch_model, seed):
        """Return a random `CipherCharacteristic` with given `differential.chmodel.CipherChModel`.

        See also `abstractproperty.characteristic.CipherCharacteristic.random`.

            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.differential.characteristic import CipherCharacteristic
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> xor_ch_model = CipherChModel(Speck32, XorDiff)
            >>> CipherCharacteristic.random(xor_ch_model, 0)  # doctest: +NORMALIZE_WHITESPACE
            CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=13,
                assignment_weights=[13, 0, 0],
                input_diff=[0xc53e, 0xd755], output_diff=[0xd755, 0xcd1a],
                assign_outdiff_list=[0x904d, 0xd755, 0xcd1a]),
            enc_characteristic=Characteristic(ch_weight=24,
                assignment_weights=[11, 13, 0, 0],
                input_diff=[0xfff4, 0xaa87],
                output_diff=[0x4523, 0xb896],
                external_diffs=[0xd755, 0xcd1a],
                 assign_outdiff_list=[0x0226, 0x8839, 0x4523, 0xb896]))

        """
        return super().random(cipher_ch_model, seed)


def _run_C_code_get_num_right_inputs(my_seed, my_module_name, my_lib_path, my_num_external_samples):
    from importlib import util as my_importlib_util
    from random import Random as my_Random

    my_spec = my_importlib_util.spec_from_file_location(my_module_name, my_lib_path)
    my_pymod = my_importlib_util.module_from_spec(my_spec)
    my_spec.loader.exec_module(my_pymod)

    my_PRNG = my_Random()
    my_PRNG.seed(my_seed)

    list_my_num_right_inputs = []
    for _ in range(my_num_external_samples):
        # get_num_right_inputs hardcoded
        my_num_right_inputs = my_pymod.lib.get_num_right_inputs(my_PRNG.randrange(0, 2 ** 64))
        list_my_num_right_inputs.append(my_num_right_inputs)

    return list_my_num_right_inputs
