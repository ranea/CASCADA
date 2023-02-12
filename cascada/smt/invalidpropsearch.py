"""Search for zero-probability (invalid) properties
(e.g., impossible differentials or zero-correlation hulls)
by modeling the search as a sequence of SMT problems.

.. autosummary::
   :nosignatures:

    ActiveBitMode
    InvalidPropFinder
    InvalidCipherPropFinder
    round_based_invalidprop_search
    round_based_invalidcipherprop_search
"""
import collections
import enum
import functools
import itertools
import warnings

from pysmt import logics

from cascada.bitvector import core
from cascada.bitvector import operation
from cascada.bitvector import context
from cascada.bitvector import ssa as cascada_ssa
from cascada import abstractproperty
from cascada.primitives import blockcipher
from cascada.smt import pysmttypes
from cascada.smt import chsearch

from cascada.smt.wrappedchmodel import get_wrapped_chmodel, get_wrapped_cipher_chmodel  # needed


zip = functools.partial(zip, strict=True)
PrintingMode = chsearch.PrintingMode
INCREMENT_NUM_ROUNDS = chsearch.INCREMENT_NUM_ROUNDS


class ActiveBitMode(enum.Enum):
    """Represent the subsets of bit-vectors available depending on
    which bits are activated (set to 1) for ``find_next_invalidprop_activebitmode``.

    Attributes:
        Default: all bit-vectors
        SingleBit: bit-vectors with up to one bit activated (zero included)
        MSBbit: bit-vectors with up to the most significant bit activated (zero included)
        Zero: the zero bit-vector

    """
    Default = enum.auto()
    SingleBit = enum.auto()
    MSBit = enum.auto()
    Zero = enum.auto()


def _generate_bitvectors(widths, total_num_active_bits, active_bits_mode):
    """Generate lists of bit-vectors.

    Given ``widths`` as a list ``[w_1, ..., w_t]`` of t integers,
    this method generate all lists ``[bv_1, ..., bv_t]`` of t bit-vectors,
    where:

    * the `ActiveBitMode` of each ``bv_i`` is ``active_bits_mode``
    * the sum of active bits of  ``[bv_1, ..., bv_t]`` is ``total_num_active_bits``
    * ``bv_i`` has width ``w_i``.

    ::

        >>> list(_generate_bitvectors([2, 2], 1, ActiveBitMode.Default))
        [[0b01, 0b00], [0b10, 0b00], [0b00, 0b01], [0b00, 0b10]]
        >>> list(_generate_bitvectors([2, 2], 1, ActiveBitMode.SingleBit))
        [[0b01, 0b00], [0b00, 0b01]]
        >>> list(_generate_bitvectors([2, 2], 1, ActiveBitMode.MSBit))
        [[0b10, 0b00], [0b00, 0b10]]
        >>> list(_generate_bitvectors([2, 2], 0, ActiveBitMode.Zero))
        [[0b00, 0b00]]

    """
    if active_bits_mode == ActiveBitMode.Zero or total_num_active_bits == 0:
        if total_num_active_bits != 0:
            raise ValueError("total_num_active_bits != 0 but active_bits_mode=Zero")
        yield [core.Constant(0, w) for w in widths]

    elif active_bits_mode in [ActiveBitMode.SingleBit, ActiveBitMode.MSBit]:
        for combination in itertools.combinations(range(len(widths)), total_num_active_bits):
            if active_bits_mode == ActiveBitMode.MSBit:
                iterables = [[w_i - 1] for i, w_i in enumerate(widths) if i in combination]
            else:
                # active_bits_mode == SingleBit
                iterables = [range(w_i - 1) for i, w_i in enumerate(widths) if i in combination]
            for w_combination in itertools.product(*iterables):
                bv_list = []
                counter_w_c = 0
                for index_w, w in enumerate(widths):
                    if index_w in combination:
                        bv_list.append(core.Constant(1 << w_combination[counter_w_c], w))
                        counter_w_c += 1
                    else:
                        bv_list.append(core.Constant(0, w))
                yield bv_list

    elif active_bits_mode == ActiveBitMode.Default:
        # Source: https://stackoverflow.com/a/10838990 and
        #   https://en.wikipedia.org/wiki/Combinatorial_number_system#Applications.
        assert total_num_active_bits > 0
        total_width = sum(widths)

        n = total_width
        k = total_num_active_bits

        def next_combination(x):
            u = (x & -x)
            v = u + x
            return v + (((v ^ x) // u) >> 2)

        x = (1 << k) - 1  # smallest number with k active bits
        while (x >> n) == 0:
            bv = core.Constant(x, n)
            bv_list = []
            sum_w = 0
            for w in widths:
                bv_list.append(bv[sum_w + w - 1:sum_w])
                sum_w += w
            yield bv_list
            x = next_combination(x)

    else:
        raise ValueError("invalid active_bits_mode")


class InvalidPropFinder(chsearch.ChFinder):
    """Search for zero-probability (invalid) property pairs by modeling the search as a sequence of SMT problems.

    Given a characteristic model
    defined for a particular `Property` (e.g., `XorDiff` or `LinearMask`),
    this class finds *universally-invalid* characteristics
    following the characteristic model by modelling the search as a sequence
    of SMT problems in the bit-vector theory.

    A *universally-invalid* characteristic is a characteristic
    where the characteristic input property :math:`\\alpha`
    propagates to the characteristic output property :math:`\\beta`
    with probability zero regardless of the intermediate properties
    (i.e., for all assignments of the intermediate properties).
    In other words, the input-output property pair
    :math:`(\\alpha, \\beta)` has zero propagation probability.

    .. note::
        For the `Difference` (resp. `LinearMask`) property,
        a universally-invalid characteristic is actually an impossible
        differential (resp. a zero-correlation hull).

        Search for universally-invalid algebraic characteristic is not supported.

    Consider the SMT problem :math:`\Omega` of whether there exists a
    valid characteristic with constant input property :math:`\\alpha`
    and constant output property :math:`\\beta`
    (and where the intermediate properties are not specified).
    The main idea of the SMT-based search is that one can check whether
    :math:`\\alpha` propagates to :math:`\\beta` with probability zero
    by checking whether :math:`\Omega` is unsatisfiable (UNSAT).
    Note that only the validity constraints are needed to build :math:`\Omega`;
    the weight constraints are ignored when searching for universally-invalid characteristics.

    The initialization argument ``ch_model`` must be a subclass of
    `abstractproperty.chmodel.ChModel` with up to one non-trivial transitions
    (`abstractproperty.opmodel.OpModel` excluding `ModelIdentity`),
    since a zero-probability characteristic with up to one non-trivial transitions
    is a universally-invalid characteristic.
    For a characteristic model with more than one non-trivial transitions,
    the function `get_wrapped_chmodel` can be used to wrap the characteristic
    model into an equivalent characteristic model with one non-trivial transition.

    An `InvalidPropFinder` object is also an instance of `ChFinder` where
    `assert_type` is `Validity` and with the given initialization arguments
    ``ch_model``,  ``solver_name``, ``printing_mode``, ``filename``, ``solver_seed``
    and ``env=env``. See also `ChFinder`.

    Similar as `ChFinder`,  the methods of `InvalidPropFinder` that search for
    universally-invalid characteristics are Python `generator` functions,
    returning an `iterator` that yields the universally-invalid characteristics
    found in the search.
    If initialization argument ``ch_model`` is a `abstractproperty.chmodel.ChModel`
    (resp. `abstractproperty.chmodel.EncryptionChModel`),
    then these methods yield
    `abstractproperty.characteristic.Characteristic`
    (resp. `abstractproperty.characteristic.EncryptionCharacteristic`) objects.

    If the initialization argument ``check_universally_invalid_ch_found`` is ``True``,
    all universally-invalid characteristics found in the search are checked by searching
    for a valid characteristic with the same input and output property with
    `ChFinder.find_next_ch`.

        >>> # example of SMT problem of universally-invalid LinearMask-EncryptionCharacteristic of (wrapped) Speck32
        >>> from cascada.linear.mask import LinearMask
        >>> from cascada.linear.chmodel import EncryptionChModel
        >>> from cascada.smt.invalidpropsearch import InvalidPropFinder
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> wrapped_ch_model = get_wrapped_chmodel(EncryptionChModel(Speck32, LinearMask))
        >>> invalid_prop_finder = InvalidPropFinder(wrapped_ch_model, "z3", solver_seed=0)
        >>> invalid_prop_finder.formula_size()
        133
        >>> print(invalid_prop_finder.hrepr(full_repr=True))  # doctest: +NORMALIZE_WHITESPACE
        ; characteristic model assertions
        assert (_mx0 == (mx9_out :: mx7_out)) &
            ((~((mx1 ^ (_mp0 >>> 7)) | (mx1 ^ mp1__0)) | _tmp20affb7ca27930ce775156bcc0ecaf20) == 0xffff) &
            ((_tmp20affb7ca27930ce775156bcc0ecaf20 ^ (_tmp20affb7ca27930ce775156bcc0ecaf20 >> 0x0001) ^
                ((mx1 ^ (_mp0 >>> 7) ^ mp1__0) >> 0x0001)) == 0x0000) &
            (mx1 == _mk0) & (mx1 == mx2) & (((_mp1 ^ mp1__0) <<< 2) == mx2__0) & (((_mp1 ^ mp1__0) <<< 2) == mx4) &
            ((~((mx6 ^ ((mx2 ^ mx2__0) >>> 7)) | (mx6 ^ mx4__0)) | _tmp824d7e7c80d9889507eb4e5d5c7be280) == 0xffff) &
            ((_tmp824d7e7c80d9889507eb4e5d5c7be280 ^ (_tmp824d7e7c80d9889507eb4e5d5c7be280 >> 0x0001) ^
                ((mx6 ^ ((mx2 ^ mx2__0) >>> 7) ^ mx4__0) >> 0x0001)) == 0x0000) &
            (mx6 == _mk1) & (mx6 == mx7) & (((mx4 ^ mx4__0) <<< 2) == mx7__0) &
            (((mx4 ^ mx4__0) <<< 2) == mx9) & ((mx7 ^ mx7__0) == mx7_out) & (mx9 == mx9_out)
        assert PropExtract_{·, 15, 0}(_mx0) == _mx1_out
        assert PropExtract_{·, 31, 16}(_mx0) == _mx2_out

    ::

        >>> # example of SMT problem of universally-invalid XorDiff-Characteristic of Speck32-KeySchedule
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import ChModel
        >>> from cascada.smt.invalidpropsearch import InvalidPropFinder
        >>> from cascada.primitives import speck
        >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> Speck32_KS.set_num_rounds(1)
        >>> ch_model = ChModel(Speck32_KS, XorDiff, ["mk0", "mk1"])
        >>> invalid_prop_finder = InvalidPropFinder(ch_model, "z3", solver_seed=0)
        >>> invalid_prop_finder.formula_size()
        42
        >>> print(invalid_prop_finder.hrepr(full_repr=True))  # doctest: +NORMALIZE_WHITESPACE
        ; characteristic model assertions
        assert ((~((mk0 >>> 7) << 0x0001) ^ (mk1 << 0x0001)) & (~((mk0 >>> 7) << 0x0001) ^ (dx1 << 0x0001)) &
            ((mk0 >>> 7) ^ mk1 ^ dx1 ^ ((mk0 >>> 7) << 0x0001))) == 0x0000
        assert mk1 == mk1_out
        assert ((mk1 <<< 2) ^ dx1) == dx3_out

    """
    def __init__(self, ch_model, solver_name, check_universally_invalid_ch_found=True,
                 # initial_constraints=None, var_prop2ct_prop=None,  exclude_zero_input_prop=None,
                 printing_mode=PrintingMode.Silent, filename=None,
                 solver_seed=None, env=None):  # weight_prefix="w",
        assert isinstance(ch_model, abstractproperty.chmodel.ChModel)

        non_id_opmodels = []
        for op_model in ch_model.assign_outprop2op_model.values():
            # PropConcat/PropExtract don't create OpModel objects
            if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                non_id_opmodels.append(op_model)
        if len(non_id_opmodels) >= 2:
            raise ValueError("characteristic model has more than 1 OpModel (excluding Identity-based ones)"
                             f"\nnon-trivial OpModel ({len(non_id_opmodels)}) = {non_id_opmodels}")

        super().__init__(
            ch_model, assert_type=chsearch.ChModelAssertType.Validity, solver_name=solver_name,
            # initial_constraints=initial_constraints, var_prop2ct_prop=var_prop2ct_prop,
            raise_exception_missing_var=True,
            printing_mode=printing_mode, filename=filename, solver_seed=solver_seed, env=env
        )

        self.check_universally_invalid_ch_found = check_universally_invalid_ch_found

    # ch_model_* properties abstracted for InvalidCipherProp

    @property
    def _ch_model_input_prop(self):
        return self.ch_model.input_prop

    @property
    def _ch_model_output_prop(self):
        return self.ch_model.output_prop

    @property
    def _ch_model_assign_outprop2op_model(self):
        return self.ch_model.assign_outprop2op_model

    @property
    def _ch_model_prop_label(self):
        return self.ch_model._prop_label

    def _get_uni_inv_ch(self, ct_inputs=None, ct_outputs=None, solution_var2ct=None):
        """Get the characteristic object from the constant input and output properties."""
        if solution_var2ct is None:
            assert ct_inputs is not None and ct_outputs is not None
            solution_var2ct = collections.OrderedDict()
        else:
            assert ct_inputs is None and ct_outputs is None
        if ct_inputs is not None:
            for var_prop, ct in zip(self._ch_model_input_prop, ct_inputs):
                solution_var2ct[var_prop.val] = ct
        if ct_outputs is not None:
            for var_prop, ct in zip(self._ch_model_output_prop, ct_outputs):
                solution_var2ct[var_prop.val] = ct

        for var_prop in itertools.chain(self._ch_model_input_prop, self._ch_model_output_prop):
            assert var_prop.val in solution_var2ct

        # get_properties_for_initialization finds all intermediate properties
        # (from ct_inputs and starting from the beginning) up to
        # the output property OUTP of the non-Identity transition.
        # Since OUTP only depends on the output properties of the ch. model,
        # OUTP is obtained through backward propagation using an SMT-solver
        # (get_properties_for_initialization only does forward propagation)

        constraints = []
        for out_prop, op_model in self._ch_model_assign_outprop2op_model.items():
            constraints.append(op_model.validity_constraint(out_prop))

        extra_constraint = True
        for v, c in solution_var2ct.items():
            extra_constraint &= operation.BvComp(v, c)

        # # debugging
        # print("\n_get_uni_inv_ch")
        # print("ch model:", self.ch_model)
        # if hasattr(self.ch_model, "_unwrapped_ch_model"):
        #     print("ch model unwrapped:", self.ch_model._unwrapped_ch_model)
        # if hasattr(self.ch_model, "_unwrapped_cipher_ch_model"):
        #     print("ch model unwrapped:", self.ch_model._unwrapped_cipher_ch_model)
        # print("ct_inputs:", ct_inputs)
        # print("ct_outputs:", ct_outputs)
        # print("solution_var2ct:", solution_var2ct)
        # print("constraints:")
        # for c in constraints:
        #     print("\t", c)
        # print("extra_constraint:", extra_constraint)
        # print()
        #

        chsearch.environment.push_env()
        env = chsearch.environment.get_env()
        psr = True if self.solver_name == "btor" else False
        bv2pysmt = functools.partial(pysmttypes.bv2pysmt, env=env, parse_shifts_rotations=psr)

        found_unique_extended_solution = False
        for r in range(1, len(constraints) + 1):  # r = num constraints to remove
            for constraint_indices in itertools.combinations(range(len(constraints)), r):
                and_constraint = True
                with context.Simplification(False):
                    for i in range(len(constraints)):
                        if i not in constraint_indices:
                            and_constraint &= constraints[i]
                    and_constraint &= extra_constraint

                pysmt_formula = bv2pysmt(and_constraint, boolean=True)
                pysmt_model = env.factory.get_model(pysmt_formula, logic=logics.QF_BV)
                if pysmt_model is None:
                    # # debugging
                    # print(f"_get_uni_inv_ch | no solution found without constraints {constraint_indices}")
                    #
                    continue

                extended_solution_var2ct = pysmttypes.pysmt_model2bv_model(pysmt_model)
                exclude_last_solution = False
                for model_var, model_val in extended_solution_var2ct.items():
                    exclude_last_solution |= ~operation.BvComp(model_var, model_val)
                pysmt_formula = bv2pysmt(and_constraint & exclude_last_solution, boolean=True)
                if env.factory.is_sat(pysmt_formula, logic=logics.QF_BV):
                    # # debugging
                    # second_sol = pysmttypes.pysmt_model2bv_model(env.factory.get_model(pysmt_formula, logic=logics.QF_BV))
                    # print(f"_get_uni_inv_ch | found 2 solutions without constraints {constraint_indices}: ",
                    #       f"{extended_solution_var2ct}, {second_sol}")
                    #
                    continue

                found_unique_extended_solution = True
                break

            if found_unique_extended_solution:
                break

        assert found_unique_extended_solution is True

        if self.printing_mode != PrintingMode.Silent:
            contradictions = []
            for i, (out_prop, op_model) in enumerate(self._ch_model_assign_outprop2op_model.items()):
                if i in constraint_indices:
                    contradictions.append((out_prop, op_model))
            smart_print = chsearch._get_smart_print(self.filename)
            smart_print(f"Contradiction found in transitions {contradictions}")

        chsearch.environment.pop_env()
        assert chsearch.environment.get_env() == self._env

        for sol_var, sol_val in solution_var2ct.items():
            assert extended_solution_var2ct[sol_var] == sol_val

        # extra checks done in _pysmt_model2ch
        return self._pysmt_model2ch(extended_solution_var2ct, is_pysmt_model=False, is_sat=False)

    def _check(self, uni_inv_ch_found, external_var2ct=None):
        assert isinstance(self.ch_model, abstractproperty.chmodel.ChModel)
        assert self.ch_model == uni_inv_ch_found.ch_model
        if hasattr(self.ch_model, "_unwrapped_ch_model"):
            list_ch_model = [self.ch_model, self.ch_model._unwrapped_ch_model]
            for v1, v2 in zip(
                    self.ch_model.external_var2prop.values(),
                    self.ch_model._unwrapped_ch_model.external_var2prop.values()
            ):
                if isinstance(v1.val, core.Constant) or isinstance(v2.val, core.Constant):
                    assert v1 == v2
        else:
            list_ch_model = [self.ch_model]
        for ch_model in list_ch_model:
            var_prop2ct_prop = collections.OrderedDict()
            for vp, cp in zip(ch_model.input_prop, uni_inv_ch_found.input_prop):
                var_prop2ct_prop[vp] = cp
            for vp, cp in zip(ch_model.output_prop, uni_inv_ch_found.output_prop):
                var_prop2ct_prop[vp] = cp
            if external_var2ct is not None:
                for (var, prop), (other_var, other_ct) in zip(
                        ch_model.external_var2prop.items(), external_var2ct.items()
                ):
                    assert var == other_var
                    if isinstance(prop.val, core.Constant):
                        assert prop.val == other_ct
                    var_prop2ct_prop[ch_model.prop_type(var)] = ch_model.prop_type(other_ct)
            ch_finder = chsearch.ChFinder(
                ch_model, assert_type=self.assert_type, solver_name=self.solver_name,
                var_prop2ct_prop=var_prop2ct_prop, raise_exception_missing_var=False,
                printing_mode=self.printing_mode, filename=self.filename, solver_seed=self.solver_seed
            )
            for valid_ch_found in ch_finder.find_next_ch():
                raise ValueError(
                    f"last characteristic found:"
                    f"\n - {uni_inv_ch_found}, {uni_inv_ch_found.ch_model} "
                    f"\nin the search is not universally-invalid; found compatible valid characteristic:"
                    f"\n - {valid_ch_found}"
                    f"\nChFinder:\n - ch_model: {ch_model}"
                    f"\n - var_prop2ct_prop: {var_prop2ct_prop}"
                    f"\n - assertions: {ch_finder.initial_constraints+list(ch_finder.chmodel_asserts)}")
            del ch_finder
        assert self._env == chsearch.environment.get_env()

    def find_next_invalidprop_activebitmode(self, initial_num_active_bits, input_prop_activebitmode, output_prop_activebitmode):
        """Return an iterator that yields the universally-invalid characteristics found in the SMT-based search
        with given `ActiveBitMode`.

        This method searches for universally-invalid characteristic using SMT solvers by checking
        one-by-one all input and output properties with given `ActiveBitMode`.

        Given a particular input and output properties :math:`(\\alpha, \\beta)`,
        the main subroutine of this method (herein call the *check subroutine*)
        checks whether :math:`\\alpha`
        propagates to :math:`\\beta` with probability zero by checking
        with an SMT solver whether the SMT problem, of whether there exists
        a valid characteristic with input property :math:`\\alpha` and output property
        :math:`\\beta`, is unsatisfiable (UNSAT).
        If the problem is UNSAT, the universally-invalid
        `abstractproperty.characteristic.Characteristic` object with
        input and output properties :math:`(\\alpha, \\beta)`
        is created and *yielded*.

        The check subroutine is repeated for all input and output properties where
        the `ActiveBitMode` of each word in the input (resp. output) property is
        ``input_prop_activebitmode`` (resp. ``output_prop_activebitmode``).
        The search starts considering input and output properties where
        the total number of active bits is ``initial_num_active_bits``,
        and the total number of active bits is incremented when
        all the input and output properties are checked.

            >>> # example of search for universally-invalid LinearMask-EncryptionCharacteristic of (wrapped) Speck32
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import EncryptionChModel
            >>> from cascada.smt.invalidpropsearch import InvalidPropFinder, ActiveBitMode
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> wrapped_ch_model = get_wrapped_chmodel(EncryptionChModel(Speck32, LinearMask))
            >>> invalid_prop_finder = InvalidPropFinder(wrapped_ch_model, "btor", solver_seed=0)
            >>> inab, ipabm, opabm = 1, ActiveBitMode.MSBit, ActiveBitMode.MSBit
            >>> for ch in invalid_prop_finder.find_next_invalidprop_activebitmode(inab, ipabm, opabm):
            ...     print(ch.srepr())
            Ch(w=Infinity, id=8000 0000, od=8000 0000)
            Ch(w=Infinity, id=8000 0000, od=0000 8000)
            Ch(w=Infinity, id=0000 8000, od=8000 0000)
            Ch(w=Infinity, id=0000 8000, od=0000 8000)
            Ch(w=Infinity, id=8000 0000, od=8000 8000)
            Ch(w=Infinity, id=0000 8000, od=8000 8000)
            Ch(w=Infinity, id=8000 8000, od=8000 0000)
            Ch(w=Infinity, id=8000 8000, od=0000 8000)
            Ch(w=Infinity, id=8000 8000, od=8000 8000)
            >>> # example of SMT problem of universally-invalid XorDiff-Characteristic of Speck32-KeySchedule
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(1)
            >>> ch_model = ChModel(Speck32_KS, XorDiff, ["mk0", "mk1"])
            >>> invalid_prop_finder = InvalidPropFinder(ch_model, "btor", solver_seed=0)
            >>> inab, ipabm, opabm = 1, ActiveBitMode.SingleBit, ActiveBitMode.SingleBit
            >>> ch = next(invalid_prop_finder.find_next_invalidprop_activebitmode(inab, ipabm, opabm))
            >>> print(ch)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=inf, assignment_weights=[inf, inf, 0],
                input_diff=[0x0001, 0x0000], output_diff=[0x0001, 0x0000],
                assign_outdiff_list=[0x0000, 0x0001, 0x0000])

        """
        smart_print = chsearch._get_smart_print(self.filename)

        # initializing the solver

        parse_shifts_rotations = True if self.solver_name == "btor" else False
        bv2pysmt = functools.partial(
            pysmttypes.bv2pysmt, env=self.env, parse_shifts_rotations=parse_shifts_rotations)

        solver_kwargs = {}
        if self.solver_seed is not None:
            if self.solver_name == "btor":
                solver_kwargs = {"solver_options": {"seed": int(self.solver_seed) % 2**32}}  # btor seed uint32
            else:
                solver_kwargs = {"random_seed": self.solver_seed}
        solver = self.env.factory.Solver(name=self.solver_name, logic=logics.QF_BV, **solver_kwargs)

        assert not self.initial_constraints
        for c in itertools.chain(self.initial_constraints, self.chmodel_asserts):
            solver.add_assertion(bv2pysmt(c, boolean=True))

        # setting max/min in/out/ num/mode active bit

        in_abmode = input_prop_activebitmode  # all variables using in/out
        out_abmode = output_prop_activebitmode

        if ActiveBitMode.Zero not in [in_abmode, out_abmode] and initial_num_active_bits == 0:
            initial_num_active_bits = 1

        in_widths = [p.val.width for p in self._ch_model_input_prop]
        out_widths = [p.val.width for p in self._ch_model_output_prop]

        def abmode2max_min_num_active_bits(my_abmode, my_list_widths):
            my_min_num_active_bits = 1
            if my_abmode == ActiveBitMode.Default:
                my_max_num_active_bits = sum(my_list_widths)
            elif my_abmode in [ActiveBitMode.SingleBit, ActiveBitMode.MSBit]:
                my_max_num_active_bits = len(my_list_widths)
            elif my_abmode == ActiveBitMode.Zero:
                my_max_num_active_bits = my_min_num_active_bits = 0
            else:
                raise ValueError("invalid mode")
            return my_max_num_active_bits, my_min_num_active_bits

        in_max_num_ab, in_min_num_ab = abmode2max_min_num_active_bits(in_abmode, in_widths)
        out_max_num_ab, out_min_num_ab = abmode2max_min_num_active_bits(out_abmode, out_widths)
        # max_in_out_active_bits = max_in_active_bits + max_out_active_bits

        prop_label = self._ch_model_prop_label  # e.g., diff, mask

        #

        for in_out_num_ab in range(initial_num_active_bits, in_max_num_ab + out_max_num_ab + 1):
            for in_num_ab in range(in_min_num_ab, in_max_num_ab + 1):
                out_num_ab = in_out_num_ab - in_num_ab
                if out_num_ab < out_min_num_ab or out_num_ab > out_max_num_ab:
                    continue

                if self.printing_mode == PrintingMode.Debug:
                    smart_print(f"Finding input/output {prop_label} with {in_num_ab} input"
                                f" and {out_num_ab} output active bits", prepend_time=True)

                for in_ct_words in _generate_bitvectors(in_widths, in_num_ab, in_abmode):
                    solver.push()

                    for var_prop, ct in zip(self._ch_model_input_prop, in_ct_words):
                        constraint = operation.BvComp(var_prop.val, ct)
                        solver.add_assertion(bv2pysmt(constraint, boolean=True))

                    if self.printing_mode == PrintingMode.Debug:
                        smart_print(f"Fixed input {prop_label} to {in_ct_words}", prepend_time=True)

                    for out_ct_words in _generate_bitvectors(out_widths, out_num_ab, out_abmode):
                        solver.push()

                        for var_prop, ct in zip(self._ch_model_output_prop, out_ct_words):
                            constraint = operation.BvComp(var_prop.val, ct)
                            solver.add_assertion(bv2pysmt(constraint, boolean=True))

                        if self.printing_mode == PrintingMode.Debug:
                            smart_print(f"Fixed output {prop_label} to {out_ct_words}", prepend_time=True)

                        satisfiable = solver.solve()

                        if not satisfiable:
                            last_ch_found = self._get_uni_inv_ch(in_ct_words, out_ct_words)
                            if self.check_universally_invalid_ch_found:
                                self._check(last_ch_found)
                            yield last_ch_found

                        solver.pop()
                    solver.pop()
        solver.exit()

    def find_next_invalidprop_miss_in_the_middle(
            self, ch_model_E0, ch_model_E2,
            ch_model_E=None, ch_model_external_E=None,
            exclude_zero_input_prop_E0=True,
            exclude_zero_input_prop_E2=True,
            exclude_zero_input_prop_external_E=None,
    ):
        """Return an iterator that yields the universally-invalid characteristics found in the SMT+MitM-based search.

        This method searches for universally-invalid characteristic using SMT problems
        and the miss-in-the-middle approach.

        Let :math:`E` be a function split into three functions
        :math:`E = E_2 \circ E_1 \circ E_0`.
        Let :math:`((p_0, p_1), (p_2, p_3))` denote a *partial* characteristic
        over :math:`E`, that is, a characteristic over :math:`E` where:

        * :math:`(p_0, p_1)` are the non-zero input and output properties of a
          characteristic with probability 1 over :math:`E_0`
        * :math:`(p_2, p_3)` are the non-zero input and output properties of a
          characteristic with probability 1 over :math:`E_2`
        * no relation is imposed between :math:`(p_1, p_2)`, the input and output
          properties of :math:`E_1`.

        The underlying function of ``self.ch_model`` corresponds to :math:`E_1`,
        the underlying function of the `abstractproperty.chmodel.ChModel`
        ``ch_model_E0`` corresponds to :math:`E_0`,
        and the underlying function of the `abstractproperty.chmodel.ChModel`
        ``ch_model_E2`` corresponds to :math:`E_2`.
        The underlying function of the `abstractproperty.chmodel.ChModel`
        ``ch_model_E`` corresponds to :math:`E`,
        but this argument is optional (more on that later).

        By default the input properties of ``ch_model_E0`` and
        ``ch_model_E2`` are excluded to be zero, but this can be
        changed with the optional arguments ``exclude_zero_input_prop_*``.

        .. note::
            This method requires that for any probability-one characteristic
            over :math:`E_0` with input-output property :math:`(p_0, p_1)`,
            there is no other probability-one characteristic over :math:`E_0`
            with input property :math:`p_0` but output property :math:`\\neq p_1`.

            Similarly, for any probability-one characteristic
            over :math:`E_2` with input-output property :math:`(p_2, p_3)`,
            there is no other probability-one characteristic over :math:`E_2`
            with output property :math:`p3` but input property :math:`\\neq p_2`.

            If :math:`E_0` and :math:`E_2` are permutations, then these two
            requirements are satisfied for `Difference` and `LinearMask`
            properties.

        If the optional argument ``ch_model_external_E`` is given as a
        `abstractproperty.chmodel.ChModel` with input and output properties
        :math:`(q_0, q_1)`, the definition of a partial characteristic is
        extended to :math:`((p_0, p_1), (p_2, p_3), (q_0, q_1)`
        such that :math:`(q_0, q_1)` are the input and output properties of a
        characteristic with probability 1 where :math:`q_1` is the list of
        external variables of :math:`E` (see `SSA`).
        If ``ch_model_external_E`` is given,
        the argument  ``exclude_zero_input_prop_external_E``
        that determines whether to exclude non-zero :math:`q_0`
        must also be given.

        .. note::
            The functions :math:`(E_0, E_1, E_2)` can be easily obtained
            from a `RoundBasedFunction` :math:`E` that includes
            `add_round_outputs` calls in its ``eval``.

            For example, obtaining :math:`E_0` from the round ``ns`` to
            ``ns+ne0`` (``ns`` denoting the initial number of skipped rounds),
            :math:`E_1` as the next ``ne1`` rounds, and :math:`E_2`
            as the next ``ne2`` rounds can be done as follows:

            .. code:: python

                [...]
                ns, ne0, ne1, ne2 = ...
                MyRoundBasedFunction.set_num_rounds(ns+ne0+ne1+ne2)
                ch_model_E = ChModel(MyRoundBasedFunction, ...)
                rs = ch_model.get_round_separators()
                # the end of the i-th round (i=1,2,...) is rs[i-1]
                e0_rs, e1_rs = rs[ns+ne0-1], rs[ns+ne0+ne1-1]
                ch_model_E0, ch_model_E1, ch_model_E2 = ch_model_E.split([e0_rs, e1_rs])
                ch_model_E1 = get_wrapped_chmodel(ch_model_E1)  # in case ch_model_E1 2+ non-trivial transitions
                invalid_prop_finder = InvalidPropFinder(ch_model_E1, ...)
                invalid_prop_finder.find_next_invalidprop_miss_in_the_middle(
                    ch_model_E0=ch_model_E0, ch_model_E2=ch_model_E2, ch_model_E=ch_model_E)

            Alternatively, one can use the function `round_based_invalidprop_search`
            which automates the generation of :math:`(E_0, E_1, E_2)`
            and applies this method iteratively on the number of rounds.

        This method finds universally-invalid characteristics by searching for all
        partial characteristics over :math:`E` using `ChFinder.find_next_ch`,
        and for each partial characteristic we apply the *check subroutine*
        to check whether :math:`p_1` propagates to :math:`p_2` with
        zero probability over :math:`E_1`.
        The check subroutine is explained in `find_next_invalidprop_activebitmode`.

        For each partial characteristic :math:`((p_0, p_1), (p_2, p_3))` found,
        if the check subroutine finds that :math:`p_1` propagates to :math:`p_2`
        with zero probability, a tuple of 3
        `abstractproperty.characteristic.Characteristic` is  *yielded*:

        * the first characteristic corresponds to the characteristic with probability 1
          over :math:`E_0` with input and output properties :math:`(p_0, p_1)`
        * the second characteristic corresponds to the universally-invalid characteristic over :math:`E_1`
          with input and output properties :math:`(p_1, p_2)`
        * the third characteristic corresponds to the characteristic with probability 1
          over :math:`E_2` with input and output properties :math:`(p_2, p_3)`

        Since the first and third characteristics have probability one,
        the concatenation of these three characteristics is a universally-invalid
        characteristic over :math:`E` (regardless of the external variables of :math:`E`)

        If the optional argument ``ch_model_external_E`` is given,
        instead a tuple of 4 characteristic is yieled; the 4-th
        characteristic corresponds to the characteristic with probability 1
        with input and output properties :math:`(q_0, q_1)`.
        In this case, the concatenation of the first 3 characteristics is a universally-invalid
        characteristic over :math:`E` *for* the external properties
        given by the outputs of the 4-th characteristic.

        If the initialization argument ``check_universally_invalid_ch_found`` is ``True``,
        all universally-invalid characteristics found over :math:`E_1` in the search
        are checked by searching for a valid characteristic with the same
        input and output property with `ChFinder.find_next_ch`.
        In addition, if the optional argument ``ch_model_E`` is given,
        then the universally-invalid characteristic over :math:`E` (the concatenation
        of the characteristic founds over  :math:`E_0`, :math:`E_1`
        and :math:`E_2`) is also checked in a similar way.

            >>> # example of search for universally-invalid LinearMask-EncryptionCharacteristic of (wrapped) Speck32
            >>> from cascada.linear.mask import LinearMask
            >>> from cascada.linear.chmodel import EncryptionChModel
            >>> from cascada.smt.invalidpropsearch import InvalidPropFinder
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(3)
            >>> ch_model_E = EncryptionChModel(Speck32, LinearMask)
            >>> ch_model_E0, ch_model_E1, ch_model_E2 = ch_model_E.split(ch_model_E.get_round_separators())
            >>> ch_model_E1 = get_wrapped_chmodel(ch_model_E1)
            >>> invalid_prop_finder = InvalidPropFinder(ch_model_E1, "btor", solver_seed=0)
            >>> tuple_iterator = invalid_prop_finder.find_next_invalidprop_miss_in_the_middle(ch_model_E0, ch_model_E2)
            >>> for i, (pr1_ch_E0, uni_inv_ch_E1, pr1_ch_E2) in enumerate(tuple_iterator):
            ...     print(pr1_ch_E0.srepr(), uni_inv_ch_E1.srepr(), pr1_ch_E2.srepr())
            ...     if i == 2: break
            Ch(w=0, id=0000 0001, od=0004 0004) Ch(w=Infinity, id=0004 0004, od=0000 0001) Ch(w=0, id=0000 0001, od=0004 0004)
            Ch(w=0, id=0000 0001, od=0004 0004) Ch(w=Infinity, id=0004 0004, od=0080 e001) Ch(w=0, id=0080 e001, od=8002 8003)
            Ch(w=0, id=0000 0001, od=0004 0004) Ch(w=Infinity, id=0004 0004, od=0080 f001) Ch(w=0, id=0080 f001, od=c002 c003)
            >>> # example of SMT problem of universally-invalid XorDiff-Characteristic of Speck32-KeySchedule
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(3)
            >>> ch_model_E = ChModel(Speck32_KS, XorDiff, ["mk0", "mk1", "mk2", "mk3"])
            >>> ch_model_E0, ch_model_E1, ch_model_E2 = ch_model_E.split(ch_model_E.get_round_separators())
            >>> invalid_prop_finder = InvalidPropFinder(ch_model_E1, "btor", solver_seed=0)
            >>> ti = invalid_prop_finder.find_next_invalidprop_miss_in_the_middle(ch_model_E0, ch_model_E2, ch_model_E=ch_model_E)
            >>> pr1_ch_E0, uni_inv_ch_E1, pr1_ch_E2 = next(ti)
            >>> print(pr1_ch_E0)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0, 0],
                input_diff=[0x0001, 0x0001, 0x0000, 0x0000], output_diff=[0x0001, 0x0000, 0x0001, 0x0000],
                assign_outdiff_list=[0x0000, 0x0001, 0x0000, 0x0001, 0x0000])
            >>> print(uni_inv_ch_E1)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=inf, assignment_weights=[inf, inf, 0, inf, inf],
                input_diff=[0x0001, 0x0000, 0x0001, 0x0000], output_diff=[0x0000, 0x8000, 0x0001, 0x0001],
                assign_outdiff_list=[0x8000, 0x0000, 0x8000, 0x0001, 0x0001])
            >>> print(pr1_ch_E2)  # doctest: +NORMALIZE_WHITESPACE
            Characteristic(ch_weight=0, assignment_weights=[0, 0, 0, 0, 0],
                input_diff=[0x0000, 0x8000, 0x0001, 0x0001], output_diff=[0x0001, 0x0001, 0x8000, 0x8002],
                assign_outdiff_list=[0x8000, 0x0001, 0x0001, 0x8000, 0x8002])

        """
        #  0. Preliminary checks

        assert not (self.check_universally_invalid_ch_found is False and ch_model_E is not None)
        assert not (ch_model_external_E is not None and exclude_zero_input_prop_external_E is None)

        # no cipher ch model allowed (no need to use self._ch_model_*)
        list_aux_ch_model = [self.ch_model, ch_model_E0, ch_model_E2]
        list_aux_prone_ch_model = [ch_model_E0, ch_model_E2]
        if ch_model_E is not None:
            list_aux_ch_model.append(ch_model_E)
        if ch_model_external_E is not None:
            list_aux_ch_model.append(ch_model_external_E)
            list_aux_prone_ch_model.append(ch_model_external_E)
        for aux_ch_model in list_aux_ch_model:
            if not isinstance(aux_ch_model, abstractproperty.chmodel.ChModel):
                raise ValueError("found non-ChModel input")
        for i, aux_prone_ch_model in enumerate(list_aux_prone_ch_model):
            if i == 0:
                name_prone_ch_model = "ch_model_E0"
            elif i == 1:
                name_prone_ch_model = "ch_model_E2"
            else:
                assert i == 2
                name_prone_ch_model = "ch_model_external_E"
            if aux_prone_ch_model.pr_one_assertions() == (core.Constant(0, 1), ):
                raise ValueError(f"{name_prone_ch_model}.pr_one_assertions() == False\n{aux_prone_ch_model}")
            if aux_prone_ch_model.max_weight(truncate=True) == 0:
                warnings.warn(f"{name_prone_ch_model} might contain too many characteristics with probability 1 "
                              f"since {name_prone_ch_model}.max_weight() is 0 \n{aux_prone_ch_model}")

        if ch_model_E0.func.output_widths != self.ch_model.func.input_widths:
            raise ValueError(f"outputs widths of ch_model_E0 {ch_model_E0.func.output_widths}"
                             f" != {self.ch_model.func.input_widths} input widths of self.ch_model")
        if ch_model_E2.func.input_widths != self.ch_model.func.output_widths:
            raise ValueError(f"input widths of ch_model_E2 {ch_model_E2.func.input_widths}"
                             f" != {self.ch_model.func.output_widths} output widths of self.ch_model")

        # for all Pr.1 ch over E0, there must be a unique output property for each input property
        # for all Pr.1 ch over E2, there must be a unique input property for each output property

        from cascada.differential.difference import Difference
        from cascada.linear.mask import LinearMask
        if issubclass(self.ch_model.prop_type, Difference):
            # for differentials with Pr. 1, an input property propagates to a unique output property
            # E0 automatically valid, E2 needs an inverse
            sum_iw, sum_ow = sum(ch_model_E2.func.input_widths), sum(ch_model_E2.func.output_widths)
            if sum_iw != sum_ow:
                raise ValueError("with the Difference property, E2 needs to be a permutation"
                                 f"but input size = {sum_iw} != {sum_ow} = output size")
            pass
        if issubclass(self.ch_model.prop_type, LinearMask):
            # for hulls with Pr. 1, an output property propagates (backwards) to a unique input property
            # E2 automatically valid, E0 needs an inverse
            sum_iw, sum_ow = sum(ch_model_E0.func.input_widths), sum(ch_model_E0.func.output_widths)
            if sum_iw != sum_ow:
                raise ValueError("with the LinearMask property, E0 needs to be a permutation"
                                 f"but input size = {sum_iw} != {sum_ow} = output size")

        if ch_model_external_E is not None:
            external_props_E0 = set(ch_model_E0.prop_type(v) for v in ch_model_E0.external_var2prop)
            external_props_E1 = set(self.ch_model.prop_type(v) for v in self.ch_model.external_var2prop)
            external_props_E2 = set(ch_model_E2.prop_type(v) for v in ch_model_E2.external_var2prop)
            output_props_external_E = set(ch_model_external_E.output_prop)
            if any(isinstance(p.val, core.Constant) for p in self.ch_model.external_var2prop.values()):
                raise ValueError(f"ch_model_external_E contains a constant external property"
                                 f"\nch_model_external_E: {ch_model_external_E}")
            if not external_props_E1.issubset(output_props_external_E):
                raise ValueError(f"E1 contains an external variable not included in ch_model_external_E outputs"
                                 f"\nch. model of E1: {self.ch_model}\nch_model_external_E: {ch_model_external_E}")
            external_props_E0_E1_E2 = external_props_E0 | external_props_E1 | external_props_E2
            if not set(output_props_external_E).issubset(external_props_E0_E1_E2):
                raise ValueError(f"ch_model_external_E contains an output that is not an external property of E"
                                 f"ch_model_external_E: {ch_model_external_E}\nexternal properties of E: {external_props_E0_E1_E2}")

        #  1. Initialization of the ChFinder objects

        # zero input prop excluded by default in E0 since ch_model_E
        # with input/output = (0, non-zero) is always uni-inv for *Diff and LinearMask
        # zero input prop also excluded by default in E2 since (non-zero, 0)
        # is always uni-inv for permutations with either *Diff or LinearMask

        chfinder_E0 = chsearch.ChFinder(
            ch_model_E0, chsearch.ChModelAssertType.ProbabilityOne, self.solver_name,
            exclude_zero_input_prop=exclude_zero_input_prop_E0,
            raise_exception_missing_var=False,
            printing_mode=self.printing_mode, filename=self.filename,
            solver_seed=self.solver_seed, env=self.env
        )
        # don't delete chfinder_E2 to avoid destructing shared env
        chfinder_E2 = chsearch.ChFinder(
            ch_model_E2, chsearch.ChModelAssertType.ProbabilityOne, self.solver_name,
            exclude_zero_input_prop=exclude_zero_input_prop_E2,
            raise_exception_missing_var=False,
            printing_mode=self.printing_mode, filename=self.filename,
            solver_seed=self.solver_seed, env=self.env
        )

        if ch_model_external_E is not None:
            chfinder_external_E = chsearch.ChFinder(
                ch_model_external_E, chsearch.ChModelAssertType.ProbabilityOne, self.solver_name,
                exclude_zero_input_prop=exclude_zero_input_prop_external_E,
                raise_exception_missing_var=False,
                printing_mode=self.printing_mode, filename=self.filename,
                solver_seed=self.solver_seed, env=self.env
            )

        #  2. Initialization of the solver for the universally-invalid ch E1

        bv2pysmt_E1 = functools.partial(
            pysmttypes.bv2pysmt, env=self.env,
            parse_shifts_rotations=True if self.solver_name == "btor" else False)

        solver_E1_kwargs = {}
        if self.solver_seed is not None:
            if self.solver_name == "btor":
                solver_E1_kwargs = {"solver_options": {"seed": int(self.solver_seed) % 2**32}}  # btor seed uint32
            else:
                solver_E1_kwargs = {"random_seed": self.solver_seed}
        solver_E1 = self.env.factory.Solver(name=self.solver_name, logic=logics.QF_BV, **solver_E1_kwargs)

        assert not self.initial_constraints
        for c in itertools.chain(self.initial_constraints, self.chmodel_asserts):
            solver_E1.add_assertion(bv2pysmt_E1(c, boolean=True))

        #  3. Auxiliary functions

        stored_prone_ch_assignment_E0 = []
        stored_prone_ch_assignment_E2 = []

        def get_next_prone_ch_assignment_E0(my_var2ct):
            if len(stored_prone_ch_assignment_E0) > 0:
                for my_prone_ch_assignment in stored_prone_ch_assignment_E0:
                    yield my_prone_ch_assignment
            else:
                if my_var2ct is not None:
                    original_initial_constraints = chfinder_E0.initial_constraints[:]
                    for ext_v in chfinder_E0.ch_model.external_var2prop:
                        chfinder_E0.initial_constraints.append(operation.BvComp(ext_v, my_var2ct[ext_v]))
                for my_prone_ch_assignment in chfinder_E0.find_next_ch(yield_assignment=True):
                    if my_var2ct is None:
                        stored_prone_ch_assignment_E0.append(my_prone_ch_assignment)
                    yield my_prone_ch_assignment
                if my_var2ct is not None:
                    chfinder_E0.initial_constraints = original_initial_constraints

        def get_next_prone_ch_assignment_E2(my_var2ct):
            if len(stored_prone_ch_assignment_E2) > 0:
                for my_prone_ch_assignment in stored_prone_ch_assignment_E2:
                    yield my_prone_ch_assignment
            else:
                if my_var2ct is not None:
                    original_initial_constraints = chfinder_E2.initial_constraints[:]
                    for ext_v in chfinder_E2.ch_model.external_var2prop:
                        chfinder_E2.initial_constraints.append(operation.BvComp(ext_v, my_var2ct[ext_v]))
                for my_prone_ch_assignment in chfinder_E2.find_next_ch(yield_assignment=True):
                    if my_var2ct is None:
                        stored_prone_ch_assignment_E2.append(my_prone_ch_assignment)
                    yield my_prone_ch_assignment
                if my_var2ct is not None:
                    chfinder_E2.initial_constraints = original_initial_constraints

        def get_next_prone_ch_assignment_external_E():
            if ch_model_external_E is None:
                yield None
            else:
                for my_prone_ch_assignment in chfinder_external_E.find_next_ch(yield_assignment=True):
                    yield my_prone_ch_assignment

        def check_concatenated_ch(my_prone_ch_E0, my_uni_inv_ch_E1, my_prone_ch_E2, my_prone_ch_external_E):
            my_var_prop2ct_prop_E = collections.OrderedDict()
            for my_vp_E, my_cp_E0 in zip(ch_model_E.input_prop, my_prone_ch_E0.input_prop):
                my_var_prop2ct_prop_E[my_vp_E] = my_cp_E0
            for my_vp_E, my_cp_E2 in zip(ch_model_E.output_prop, my_prone_ch_E2.output_prop):
                my_var_prop2ct_prop_E[my_vp_E] = my_cp_E2
            my_chfinder_E = chsearch.ChFinder(
                ch_model_E, assert_type=self.assert_type, solver_name=self.solver_name,
                var_prop2ct_prop=my_var_prop2ct_prop_E, raise_exception_missing_var=False,
                printing_mode=self.printing_mode, filename=self.filename, solver_seed=self.solver_seed
            )
            if ch_model_external_E is not None:
                for vp_external_E, cp_external_E in zip(ch_model_external_E.output_prop, my_prone_ch_external_E.output_prop):
                    if vp_external_E.val in my_chfinder_E._vars_in_constraints:
                        my_chfinder_E.initial_constraints.append(operation.BvComp(vp_external_E.val, cp_external_E.val))
            for valid_ch_found_E in my_chfinder_E.find_next_ch():
                if ch_model_external_E is not None:
                    aux_str = f"\n - prone_ch_external_E: {my_prone_ch_external_E}, " \
                              f"{my_prone_ch_external_E.ch_model}"
                else:
                    aux_str = ""
                raise ValueError(
                    "the concatenation of the last characteristic tuple found,"
                    f"\n - prone_ch_E0: {my_prone_ch_E0}, {my_prone_ch_E0.ch_model}"
                    f"\n - uni_inv_ch_E1: {my_uni_inv_ch_E1}, {my_uni_inv_ch_E1.ch_model}"
                    f"\n - prone_ch_E2: {my_prone_ch_E2}, {my_prone_ch_E2.ch_model}{aux_str}"
                    f"\n - ch_model E: {ch_model_E}\n - var_prop2ct_prop: {my_var_prop2ct_prop_E}"
                    f"\n - ch_finder E: {my_chfinder_E.initial_constraints + list(my_chfinder_E.chmodel_asserts)},"
                    f"\n is not universally-invalid (found compatible valid characteristic over E {valid_ch_found_E})")
            del my_chfinder_E
            assert self._env == chsearch.environment.get_env()

        #  4. Search for probability-one characteristics

        smart_print = chsearch._get_smart_print(self.filename)

        for prone_ch_assignment_external_E in get_next_prone_ch_assignment_external_E():
            assert (ch_model_external_E is None) == (prone_ch_assignment_external_E is None)

            if ch_model_external_E is None:
                aux_str_E = "", ""
            else:
                aux_str_E = " (and external E)", f", {prone_ch_assignment_external_E}"

                output_var2ct_external_E = collections.OrderedDict()
                for out_var_eE in ch_model_external_E.ssa.output_vars:
                    ct_val_eE = ch_model_external_E.var2prop[out_var_eE].val.xreplace(prone_ch_assignment_external_E)
                    assert isinstance(ct_val_eE, core.Constant)
                    output_var2ct_external_E[out_var_eE] = ct_val_eE

                constraint_for_E1_from_external_E = True
                external_var2ct_E1 = collections.OrderedDict()
                for ext_var_E1 in self.ch_model.external_var2prop:
                    constraint_for_E1_from_external_E &= operation.BvComp(ext_var_E1, output_var2ct_external_E[ext_var_E1])
                    external_var2ct_E1[ext_var_E1] = output_var2ct_external_E[ext_var_E1]

            for prone_ch_assignment_E0 in get_next_prone_ch_assignment_E0(
                    None if ch_model_external_E is None else output_var2ct_external_E):
                ct_outputs_E0 = []
                for out_var_E0 in ch_model_E0.ssa.output_vars:
                    ct_val_E0 = ch_model_E0.var2prop[out_var_E0].val.xreplace(prone_ch_assignment_E0)
                    assert isinstance(ct_val_E0, core.Constant)
                    ct_outputs_E0.append(ct_val_E0)

                for prone_ch_assignment_E2 in get_next_prone_ch_assignment_E2(
                        None if ch_model_external_E is None else output_var2ct_external_E):
                    if self.printing_mode == PrintingMode.Debug:
                        smart_print(f"Found probability-one characteristics over E0 and E2{aux_str_E[0]}: "
                                    f"{prone_ch_assignment_E0}, {prone_ch_assignment_E2}{aux_str_E[1]}", prepend_time=True)

                    ct_inputs_E2 = []
                    for in_var_E2 in ch_model_E2.ssa.input_vars:
                        ct_val_E2 = ch_model_E2.var2prop[in_var_E2].val.xreplace(prone_ch_assignment_E2)
                        assert isinstance(ct_val_E2, core.Constant)
                        ct_inputs_E2.append(ct_val_E2)

                    constraint_for_E1 = True if ch_model_external_E is None else constraint_for_E1_from_external_E
                    solution_var2ct_E1 = collections.OrderedDict() if ch_model_external_E is None else external_var2ct_E1.copy()
                    for var_prop_E1, ct_val_E0 in zip(self.ch_model.input_prop, ct_outputs_E0):
                        constraint_for_E1 &= operation.BvComp(var_prop_E1.val, ct_val_E0)
                        solution_var2ct_E1[var_prop_E1.val] = ct_val_E0
                    for var_prop_E1, ct_val_E2 in zip(self.ch_model.output_prop, ct_inputs_E2):
                        constraint_for_E1 &= operation.BvComp(var_prop_E1.val, ct_val_E2)
                        solution_var2ct_E1[var_prop_E1.val] = ct_val_E2

                    # # debugging
                    # print("\nfind_next_invalidprop_miss_in_the_middle")
                    # print("ch_model_E0", ch_model_E0)
                    # print("ch model E1", self.ch_model)
                    # if hasattr(self.ch_model, "_unwrapped_ch_model"):
                    #     print("unwrapped ch model E1", self.ch_model._unwrapped_ch_model)
                    # print("ch_model_E2", ch_model_E2)
                    # if ch_model_external_E:
                    #     print("ch_model_external_E", ch_model_external_E)
                    # if ch_model_E:
                    #     print("ch_model_E", ch_model_E)
                    # print("self.chmodel_asserts:", self.chmodel_asserts)
                    # print("output_var2ct_external_E:", output_var2ct_external_E)
                    # print("external_var2ct_E1:", external_var2ct_E1)
                    # print("constraint_for_E1:", constraint_for_E1, "\n")
                    #

                    if not solver_E1.solve([bv2pysmt_E1(constraint_for_E1, boolean=True)]):
                        uni_inv_ch_E1 = self._get_uni_inv_ch(solution_var2ct=solution_var2ct_E1)

                        prone_ch_E0 = chfinder_E0._pysmt_model2ch(prone_ch_assignment_E0, is_pysmt_model=False)
                        prone_ch_E2 = chfinder_E2._pysmt_model2ch(prone_ch_assignment_E2, is_pysmt_model=False)
                        assert prone_ch_E0.ch_weight == 0, f"{prone_ch_E0}"
                        assert prone_ch_E2.ch_weight == 0, f"{prone_ch_E2}"
                        if ch_model_external_E is not None:
                            prone_ch_external_E = chfinder_external_E._pysmt_model2ch(prone_ch_assignment_external_E, is_pysmt_model=False)
                            assert prone_ch_external_E.ch_weight == 0, f"{prone_ch_external_E}"

                        if self.check_universally_invalid_ch_found:
                            self._check(uni_inv_ch_E1, external_var2ct=None if ch_model_external_E is None else external_var2ct_E1)
                            if ch_model_E is not None:
                                check_concatenated_ch(prone_ch_E0, uni_inv_ch_E1, prone_ch_E2,
                                                      None if ch_model_external_E is None else prone_ch_external_E)

                        if ch_model_external_E is not None:
                            yield prone_ch_E0, uni_inv_ch_E1, prone_ch_E2, prone_ch_external_E
                        else:
                            yield prone_ch_E0, uni_inv_ch_E1, prone_ch_E2
                else:
                    # no pr-one ch. found for E2, no need to find another E0
                    break

        solver_E1.exit()

    def find_next_invalidprop_quantified_logic(self):
        """Return an iterator that yields the universally-invalid characteristics found in the quantified SMT-based search.

        This method searches for universally-invalid characteristic using SMT problems
        in the quantified bit-vector logic (with the *ForAll* quantifier).

        Let :math:`P(\\alpha, \gamma_1, \dots, \gamma_t, \\beta)` be the
        underlying bit-vector formula of the decision problem
        of whether there exists a characteristic following the
        characteristic model ``ch_model`` with non-zero probability,
        where :math:`(\\alpha, \\beta)` is the input and output properties
        and :math:`(\gamma_1, \dots, \gamma_t)` are the intermediate properties.

        First, this method creates the decision problem of whether there exists
        an assignment of the input and output properties :math:`(\\alpha, \\beta)`
        such that for all intermediate properties :math:`(\gamma_1, \dots, \gamma_t)`
        the negation of :math:`P` is True; in other words, the decision problem
        given by the underlying quantified formula
        :math:`\exists \\alpha, \\beta, \\forall \gamma_1, \dots, \gamma_t : \  \\neg
        P(\\alpha, \gamma_1, \dots, \gamma_t, \\beta)`

        If the SMT solver finds the first problem satisfiable,
        an assignment of the input and output properties :math:`(\\alpha, \\beta)`
        that makes :math:`\\neg P(\\alpha, \gamma_1, \dots, \gamma_t, \\beta) = True` is
        obtained, and a universally-invalid `abstractproperty.characteristic.Characteristic`
        object is created and *yielded*.

        Afterwards, an additional constraint is added to the SMT problem
        to exclude the characteristic yielded and this procedure is repeated
        until all characteristics are found.

        This method requires that the SMT solver given in ``solver_name``
        supports the bit-vector logic with quantifiers.
        Although the recent version of boolector supports the bit-vector logic
        with quantifiers, pySMT does not support yet this recent feature
        of boolector.

            >>> # example of search for universally-invalid XorDiff-EncryptionCharacteristic of (wrapped) Speck32
            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import EncryptionChModel
            >>> from cascada.smt.invalidpropsearch import InvalidPropFinder
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> wrapped_ch_model = get_wrapped_chmodel(EncryptionChModel(Speck32, XorDiff))
            >>> invalid_prop_finder = InvalidPropFinder(wrapped_ch_model, "z3", solver_seed=0)
            >>> for i, ch in enumerate(invalid_prop_finder.find_next_invalidprop_quantified_logic()):
            ...     print(ch.srepr())
            ...     if i == 2: break  # doctest: +ELLIPSIS
            Ch(w=Infinity, id=..., od=...)
            Ch(w=Infinity, id=..., od=...)
            Ch(w=Infinity, id=..., od=...)
            >>> # example of SMT problem of universally-invalid RXDiff-Characteristic of Speck32-KeySchedule
            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.chmodel import ChModel
            >>> Speck32_KS = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
            >>> Speck32_KS.set_num_rounds(1)
            >>> ch_model = ChModel(Speck32_KS, RXDiff, ["mk0", "mk1"])
            >>> invalid_prop_finder = InvalidPropFinder(ch_model, "z3", solver_seed=0)
            >>> for i, ch in enumerate(invalid_prop_finder.find_next_invalidprop_quantified_logic()):
            ...     print(ch.srepr())
            ...     if i == 2: break  # doctest: +ELLIPSIS
            Ch(w=Infinity, id=..., od=...)
            Ch(w=Infinity, id=..., od=...)
            Ch(w=Infinity, id=..., od=...)

        """
        smart_print = chsearch._get_smart_print(self.filename)

        # InputOutput contains _input_vars_not_used
        in_out_sig_type = abstractproperty.chmodel.ChModelSigType.InputOutput
        symbolic_sig = self.ch_model.signature(in_out_sig_type)

        # initializing the solver

        parse_shifts_rotations = True if self.solver_name == "btor" else False
        bv2pysmt = functools.partial(
            pysmttypes.bv2pysmt, env=self.env, parse_shifts_rotations=parse_shifts_rotations)

        solver_kwargs = {}
        if self.solver_seed is not None:
            if self.solver_name == "btor":
                solver_kwargs = {"solver_options": {"seed": int(self.solver_seed) % 2 ** 32}}  # btor seed uint32
            else:
                solver_kwargs = {"random_seed": self.solver_seed}
        solver = self.env.factory.Solver(name=self.solver_name, logic=logics.BV, **solver_kwargs)

        #

        compact_constraint = True
        assert not self.initial_constraints
        for c in itertools.chain(self.initial_constraints, self.chmodel_asserts):
            compact_constraint &= c

        in_out_vars = [p.val for p in itertools.chain(self._ch_model_input_prop, self._ch_model_output_prop)]
        forall_vars = [v for v in self._vars_in_constraints if v not in in_out_vars]

        pysmt_formula = self.env.formula_manager.ForAll(
            [bv2pysmt(v) for v in forall_vars],
            bv2pysmt(operation.BvNot(compact_constraint), boolean=True)
        )

        # # debugging
        # print("\nfind_next_invalidprop_quantified_logic")
        # print("ch model:", self.ch_model)
        # print("compact_constraint:", compact_constraint)
        # print("in_out_vars:", in_out_vars)
        # print("forall_vars:", forall_vars)
        # print("pysmt_formula:", pysmt_formula.serialize(), "\n")
        #

        solver.add_assertion(pysmt_formula)

        last_ch_found = None
        while True:
            if last_ch_found is not None:
                if len(symbolic_sig) == 0:
                    warnings.warn(f"empty signature of {self.ch_model}")
                    break
                last_ch_sig = last_ch_found.signature(in_out_sig_type)
                # disable simplification due to recursion error
                with context.Simplification(False):
                    exclude_last_ch = functools.reduce(
                        operation.BvOr,
                        [~operation.BvComp(ss, ls) for ss, ls in zip(symbolic_sig, last_ch_sig)]
                    )
                solver.add_assertion(bv2pysmt(exclude_last_ch, boolean=True))
                if self.printing_mode == PrintingMode.Debug:
                    smart_print(f"exclude_last_ch: {exclude_last_ch}", prepend_time=True)

            satisfiable = solver.solve()

            if satisfiable:
                solution_var2ct = pysmttypes.pysmt_model2bv_model(solver.get_model())

                # add missing input-output vars
                in_out_vars = [p.val for p in self._ch_model_input_prop + self._ch_model_output_prop]
                missing_in_out_vars = []
                for v in in_out_vars:
                    if v not in solution_var2ct:
                        missing_in_out_vars.append(v)
                        solution_var2ct[v] = core.Constant(0, v.width)
                if missing_in_out_vars and (self.printing_mode != PrintingMode.Silent):
                    smart_print(f"Found solution of quantified SMT problem for all values of {missing_in_out_vars}; "
                                f"setting {self.ch_model.prop_type.__name__} of {missing_in_out_vars} "
                                f"to 0 in yielded universally-invalid characteristic")

                last_ch_found = self._get_uni_inv_ch(solution_var2ct=solution_var2ct)
                if self.check_universally_invalid_ch_found:
                    self._check(last_ch_found)
                yield last_ch_found

            else:
                break

        solver.exit()

    def hrepr(self, full_repr=False):
        """Return a human-readable representation of the base SMT problem.

        The base SMT problem is the decision problem of whether there exists
        a valid characteristic for an input-output property pair.
        In other words, it contains the validity assertions
        of the underlying characteristic model.

        The methods `InvalidPropFinder.find_next_invalidprop_activebitmode` and
        `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle` check
        for the unsatisfiability of this base SMT problem
        (with some additional constraints),
        while `InvalidPropFinder.find_next_invalidprop_quantified_logic`
        uses this base SMT problem to create a quantified
        bit-vector formula.

        If ``full_repr`` is False, the short string representation srepr is used.
        """
        return super().hrepr(full_repr=full_repr)


class InvalidCipherPropFinder(InvalidPropFinder):
    """Search for invalid properties of ciphers by modeling the search as a sequence of SMT problems.

    Given a characteristic model of a `Cipher`
    (`abstractproperty.chmodel.CipherChModel`)
    defined for a particular `Property` (e.g., `XorDiff` or `RXDiff`),
    this class finds *universally-invalid* cipher characteristics
    (`abstractproperty.characteristic.CipherCharacteristic`)
    following the characteristic model by modelling the search
    as a sequence of SMT problems in the bit-vector theory.

    Given a cipher characteristic, let :math:`\\alpha_{KS}` be the input
    property of the underlying key-schedule characteristic
    and :math:`(\\alpha_{ENC}, \\beta_{ENC})` be the input and output
    properties of the underlying encryption characteristic.
    A universally-invalid characteristic
    over a cipher is a characteristic where
    :math:`(\\alpha_{KS}, \\alpha_{ENC})` propagates to :math:`\\beta_{ENC}`
    with probability zero regardless of the intermediate properties.
    In other words, the input-output property pair
    :math:`((\\alpha_{KS}, \\alpha_{ENC}), \\beta_{ENC})`
    has zero propagation probability.

    .. note::
        For the `Difference` property,
        a universally-invalid characteristic over a cipher is actually a
        related-key impossible differential.

    To initialize an `InvalidCipherPropFinder` object, first two auxiliary
    instances of `InvalidPropFinder` are created:

    - ``ks_finder`` an `InvalidPropFinder` with characteristic model
      ``ch_model.ks_ch_model``
    - ``enc_finder`` an `InvalidPropFinder` with characteristic model
      ``ch_model.enc_ch_model``

    Both ``ks_finder`` and ``enc_finder`` (together with the
    `InvalidCipherPropFinder` object) share the arguments `solver_name`,
    `printing_mode`, `filename`,  `solver_seed` and `env`.

    Then, these two auxiliary `InvalidPropFinder` objects are merged into an
    `InvalidCipherPropFinder` (which is also an instance of `InvalidPropFinder`)
    as follows:

    - ``solver_name``, ``printing_mode``, ``filename``,  ``solver_seed``
      ``env`` are the same as the ones from ``ks_finder`` and ``enc_finder``
    - ``ch_model`` is set to the characteristic model of the cipher
      (a subclass of `abstractproperty.chmodel.CipherChModel`)
    - ``chmodel_asserts`` is the union of `chmodel_asserts` of
      ``ks_finder`` and ``enc_finder``

    See also `ChFinder`.

    ::

        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.differential.chmodel import CipherChModel
        >>> from cascada.smt.invalidpropsearch import InvalidCipherPropFinder, get_wrapped_cipher_chmodel
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_num_rounds(2)
        >>> ch_model = get_wrapped_cipher_chmodel(CipherChModel(Speck32, XorDiff))
        >>> invalid_prop_finder = InvalidCipherPropFinder(ch_model, "btor", solver_seed=0)
        >>> invalid_prop_finder.formula_size()
        177
        >>> print(invalid_prop_finder.hrepr(full_repr=False))  # doctest: +NORMALIZE_WHITESPACE
        ; characteristic model assertions
        assert (_dk0 == (dk3_out :: dmk1_out)) & ((... & ... & (... ^ ...)) == 0x0000) &
            (_dmk1 == dmk1_out) & (((_dmk1 <<< 2) ^ dk1) == dk3_out)
        assert PropExtract_{·, 15, 0}(_dk0) == _dk1_out
        assert PropExtract_{·, 31, 16}(_dk0) == _dk2_out
        assert (_dx0 == (dx9_out :: dx7_out)) & ((... & ...) == 0x0000) & ((... & ...) == 0x0000) &
            ((dx6 ^ _dk2_out) == dx7_out) & ((((... ^ ...) <<< 2) ^ dx6 ^ _dk2_out) == dx9_out)
        assert PropExtract_{·, 15, 0}(_dx0) == _dx1_out
        assert PropExtract_{·, 31, 16}(_dx0) == _dx2_out

    """

    def __init__(self, ch_model, solver_name, check_universally_invalid_ch_found=True,
                 printing_mode=PrintingMode.Silent, filename=None,
                 solver_seed=None, env=None):
        assert isinstance(ch_model, abstractproperty.chmodel.CipherChModel)

        ks_finder = InvalidPropFinder(
            ch_model.ks_ch_model, solver_name=solver_name,
            check_universally_invalid_ch_found=check_universally_invalid_ch_found,
            printing_mode=PrintingMode.Silent, filename=None, env=env,
        )

        enc_finder = InvalidPropFinder(
            ch_model.enc_ch_model, solver_name=solver_name,
            check_universally_invalid_ch_found=check_universally_invalid_ch_found,
            printing_mode=PrintingMode.Silent, filename=None, env=ks_finder.env,
        )

        assert ks_finder.env == enc_finder.env
        assert ks_finder.assert_type == chsearch.ChModelAssertType.Validity
        assert enc_finder.assert_type == chsearch.ChModelAssertType.Validity
        assert ks_finder._ch_weight is None and ks_finder._error == 0
        assert enc_finder._ch_weight is None and enc_finder._error == 0
        assert ks_finder._exclude_zero_input_prop is False
        assert enc_finder._exclude_zero_input_prop is False

        ch_weight = None
        error = 0
        assert_type = chsearch.ChModelAssertType.Validity
        exclude_zero_input_prop = False

        chmodel_asserts = ks_finder.chmodel_asserts + enc_finder.chmodel_asserts
        vars_in_constraints = ks_finder._vars_in_constraints | enc_finder._vars_in_constraints

        self.ks_finder = ks_finder
        self.enc_finder = enc_finder

        self.ch_model = ch_model
        self.assert_type = assert_type
        self.solver_name = solver_name
        self.initial_constraints = []
        self.printing_mode = printing_mode
        self.filename = filename
        self.weight_prefix = None
        self.solver_seed = solver_seed
        self._env = ks_finder.env
        self.chmodel_asserts = chmodel_asserts

        # variables not added in docstring (private variables)
        self._exclude_zero_input_prop = exclude_zero_input_prop
        self._var_prop2ct_prop = {}
        self._ch_weight = ch_weight
        self._error = error
        self._vars_in_constraints = vars_in_constraints

        self.check_universally_invalid_ch_found = check_universally_invalid_ch_found

    @property
    def _ch_model_input_prop(self):
        return self.ch_model.ks_ch_model.input_prop + self.ch_model.enc_ch_model.input_prop

    @property
    def _ch_model_output_prop(self):
        return self.ch_model.enc_ch_model.output_prop

    @property
    def _ch_model_assign_outprop2op_model(self):
        return collections.OrderedDict(itertools.chain(
            self.ch_model.ks_ch_model.assign_outprop2op_model.items(),
            self.ch_model.enc_ch_model.assign_outprop2op_model.items()))

    @property
    def _ch_model_prop_label(self):
        assert self.ch_model.ks_ch_model._prop_label == self.ch_model.enc_ch_model._prop_label
        return self.ch_model.ks_ch_model._prop_label

    def _pysmt_model2ch(self, solution_var2ct, target_weight=None, is_pysmt_model=True, is_sat=True):
        assert target_weight is None
        assert is_sat is False

        if is_pysmt_model:
            solution_var2ct = pysmttypes.pysmt_model2bv_model(solution_var2ct)
        else:
            solution_var2ct = solution_var2ct.copy()

        def _get_needed_vars(my_ch_model):
            var_needed = [p.val for p in my_ch_model.input_prop if p.val not in my_ch_model.ssa._input_vars_not_used]
            # # ks_ch_model has no external vars and enc_ch_model gets those from ks_ch_model.output
            # for ext_var, prop in my_ch_model.external_var2prop.items():
            #     if not isinstance(prop.val, core.Constant):
            #         var_needed.append(ext_var)
            for outprop, op_model in my_ch_model.assign_outprop2op_model.items():
                # if op_model.max_weight() != 0:
                if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                    var_needed.append(outprop.val)
            return var_needed

        def get_needed_vars(my_cipher_ch_model):
            return _get_needed_vars(my_cipher_ch_model.ks_ch_model) + _get_needed_vars(my_cipher_ch_model.enc_ch_model)

        missing_signature_vars = []
        for v in get_needed_vars(self.ch_model):
            if v not in solution_var2ct:
                missing_signature_vars.append(v)
                solution_var2ct[v] = core.Constant(0, v.width)

        # universally-invalid characteristics are invalid regardless of non-input non-output properties
        in_out_vars = [p.val for p in self._ch_model_input_prop + self._ch_model_output_prop]
        if missing_signature_vars and (
                self.printing_mode != PrintingMode.Silent or
                any(v in in_out_vars for v in missing_signature_vars)
        ):
            smart_print = chsearch._get_smart_print(self.filename)
            smart_print(f"Found {'satisfiable' if is_sat else 'unsatisfiable'} assignment "
                        f"of SMT problem for all values of {missing_signature_vars}; "
                        f"setting {self.ch_model.prop_type.__name__} of {missing_signature_vars} "
                        f"to 0 in yielded characteristic")

        # if target_weight is not None and \
        #    [...]

        CipherCharacteristic_cls = self.ch_model.__class__._get_CipherCharacteristic_cls()

        init_props = CipherCharacteristic_cls.get_properties_for_initialization(self.ch_model, solution_var2ct)
        assert len(init_props) == 6
        ks_input_prop, ks_output_prop, ks_assign_outprop_list = init_props[:3]
        enc_input_prop, enc_output_prop, enc_assign_outprop_list = init_props[-3:]

        # # debugging
        # print("InvalidCipherProp._pysmt_model2ch")
        # print("ch model:", self.ch_model)
        # print("ks ssa:", self.ch_model.ks_ch_model.ssa)
        # print("enc ssa:", self.ch_model.enc_ch_model.ssa)
        # print("solution_var2ct:", solution_var2ct)
        # print("needed vars:", get_needed_vars(self.ch_model))
        # print("get_properties_for_initialization():", init_props, "\n")
        #

        # avoid *_props=*_props (super might not abstract)
        last_ch_found = None
        for ks_is_sat, enc_is_sat in [[False, False], [False, True], [True, False], [True, True]]:
            try:
                last_ch_found = CipherCharacteristic_cls(
                    ks_input_prop,
                    ks_output_prop,
                    ks_assign_outprop_list,
                    enc_input_prop,
                    enc_output_prop,
                    enc_assign_outprop_list,
                    self.ch_model,
                    # ks_free_props,
                    # enc_free_props,
                    ks_empirical_ch_weight=None,
                    ks_empirical_data_list=None,
                    enc_empirical_ch_weight=None,
                    enc_empirical_data_list=None,
                    ks_is_valid=ks_is_sat,
                    enc_is_valid=enc_is_sat,
                )
            except (ValueError,  abstractproperty.opmodel.InvalidOpModelError) as e:
                if "is_valid" in str(e):
                    continue
                else:
                    raise e
            else:
                break

        if last_ch_found is not None and [ks_is_sat, enc_is_sat] == [True, True]:
            raise ValueError(f"SMT solution {solution_var2ct} leads to a valid characteristic"
                             f"\n{last_ch_found}")
        elif last_ch_found is None:
            raise ValueError(f"no characteristic can be built from SMT solution {solution_var2ct}")

        assert isinstance(last_ch_found, CipherCharacteristic_cls), f"{last_ch_found}"

        # assert not (self.ks_finder.assert_type == chsearch.ChModelAssertType.ProbabilityOne
        #     [...]

        # for i, aux_finder in enumerate([self.ks_finder, self.enc_finder]):
        #     [...]  # _exclude_zero_input_prop

        # for var_prop, ct_prop in itertools.chain(
        #     [...]

        # if self._var_prop2ct_prop:
        #     [...]

        # # ignored due to new solution_var2ct
        # with context.Simplification(False):
        #     chmodel_asserts = [a.xreplace(solution_var2ct) for a in self.chmodel_asserts]

        # if target_weight is not None:
        #     [...]

        return last_ch_found

    def _check(self, invalid_cipher_ch_found):
        assert self.ch_model == invalid_cipher_ch_found.cipher_ch_model
        if hasattr(self.ch_model, "_unwrapped_cipher_ch_model"):
            list_cipher_ch_model = [self.ch_model, self.ch_model._unwrapped_cipher_ch_model]
            for v1, v2 in zip(
                    self.ch_model.enc_ch_model.external_var2prop.values(),
                    self.ch_model._unwrapped_cipher_ch_model.enc_ch_model.external_var2prop.values()
            ):
                if isinstance(v1.val, core.Constant) or isinstance(v2.val, core.Constant):
                    assert v1 == v2
        else:
            list_cipher_ch_model = [self.ch_model]

        # avoid self._ch_model_input_prop since we also have self.ch_model._unwrapped_cipher_ch_model

        def get_input_prop(ch_or_ch_model):
            if isinstance(ch_or_ch_model, abstractproperty.characteristic.CipherCharacteristic):
                return ch_or_ch_model.ks_characteristic.input_prop + \
                       ch_or_ch_model.enc_characteristic.input_prop
            else:
                assert isinstance(ch_or_ch_model, abstractproperty.chmodel.CipherChModel)
                return ch_or_ch_model.ks_ch_model.input_prop + \
                       ch_or_ch_model.enc_ch_model.input_prop

        def get_output_prop(ch_or_ch_model):
            if isinstance(ch_or_ch_model, abstractproperty.characteristic.CipherCharacteristic):
                return ch_or_ch_model.ks_characteristic.output_prop + \
                       ch_or_ch_model.enc_characteristic.output_prop
            else:
                assert isinstance(ch_or_ch_model, abstractproperty.chmodel.CipherChModel)
                return ch_or_ch_model.ks_ch_model.output_prop + \
                       ch_or_ch_model.enc_ch_model.output_prop

        for cipher_ch_model in list_cipher_ch_model:
            var_prop2ct_prop = collections.OrderedDict()
            for vp, cp in zip(get_input_prop(cipher_ch_model), get_input_prop(invalid_cipher_ch_found)):
                var_prop2ct_prop[vp] = cp
            for vp, cp in zip(get_output_prop(cipher_ch_model), get_output_prop(invalid_cipher_ch_found)):
                var_prop2ct_prop[vp] = cp
            cipher_ch_finder = chsearch.CipherChFinder(
                cipher_ch_model, ks_assert_type=self.assert_type, enc_assert_type=self.assert_type,
                solver_name=self.solver_name, var_prop2ct_prop=var_prop2ct_prop, raise_exception_missing_var=False,
                printing_mode=self.printing_mode, filename=self.filename, solver_seed=self.solver_seed
            )
            for valid_cipher_ch_found in cipher_ch_finder.find_next_ch():
                raise ValueError(f"last characteristic found {invalid_cipher_ch_found} in the search is not "
                                 f"universally-invalid (found compatible valid characteristic {valid_cipher_ch_found})")
            del cipher_ch_finder
        assert self._env == chsearch.environment.get_env()

    def find_next_invalidprop_activebitmode(self, initial_num_active_bits, input_prop_activebitmode, output_prop_activebitmode):
        """Return an iterator that yields the universally-invalid characteristics found in the SMT-based search
        with given `ActiveBitMode`.

        This method is similar to `InvalidPropFinder.find_next_invalidprop_activebitmode`;
        the only difference is that the input property considered by this method
        is the concatenation of the input property of the underlying key-schedule
        characteristic and the input property of the underlying encryption
        characteristic, and the output property considered by this method
        is the output property of the encryption characteristic.
        In other words, ``output_prop_activebitmode`` only affects
        to the output property of the encryption characteristic
        and not to the output property of the key-schedule characteristic.

            >>> from cascada.differential.difference import XorDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.smt.invalidpropsearch import InvalidCipherPropFinder, ActiveBitMode, get_wrapped_cipher_chmodel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = get_wrapped_cipher_chmodel(CipherChModel(Speck32, XorDiff))
            >>> invalid_prop_finder = InvalidCipherPropFinder(ch_model, "btor", solver_seed=0)
            >>> inab, ipabm, opabm = 1, ActiveBitMode.MSBit, ActiveBitMode.Zero
            >>> for ch in invalid_prop_finder.find_next_invalidprop_activebitmode(inab, ipabm, opabm):
            ...     print(ch.srepr())
            Ch(ks_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000), enc_ch=Ch(w=0, id=0000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=0000 8000, od=0000 0000), enc_ch=Ch(w=0, id=0000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=0000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000), enc_ch=Ch(w=0, id=0000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=0000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 8000, od=8000 8002), enc_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 8000, od=8000 8002), enc_ch=Ch(w=Infinity, id=0000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000), enc_ch=Ch(w=Infinity, id=0000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 0000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=0, id=0000 8000, od=8000 8002), enc_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000))
            Ch(ks_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000), enc_ch=Ch(w=Infinity, id=8000 8000, od=0000 0000))

        """
        return super().find_next_invalidprop_activebitmode(
            initial_num_active_bits, input_prop_activebitmode, output_prop_activebitmode
        )

    def find_next_invalidprop_miss_in_the_middle(self, *args, **kargs):
        """This method is disabled, see `round_based_invalidcipherprop_search` for an alternative."""
        raise NotImplementedError("find_next_invalidprop_miss_in_the_middle is disabled in InvalidCipherPropFinder,"
                                  "see round_based_invalidcipherprop_search")

    def find_next_invalidprop_quantified_logic(self):
        """Return an iterator that yields the universally-invalid characteristics found in the quantified SMT-based search.

        See also `InvalidPropFinder.find_next_invalidprop_quantified_logic`.

            >>> from cascada.differential.difference import RXDiff
            >>> from cascada.differential.chmodel import CipherChModel
            >>> from cascada.smt.invalidpropsearch import InvalidCipherPropFinder, get_wrapped_cipher_chmodel
            >>> from cascada.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_num_rounds(2)
            >>> ch_model = get_wrapped_cipher_chmodel(CipherChModel(Speck32, RXDiff))
            >>> invalid_prop_finder = InvalidCipherPropFinder(ch_model, "z3", solver_seed=0)
            >>> uni_inv_ch = next(invalid_prop_finder.find_next_invalidprop_quantified_logic())
            >>> print(uni_inv_ch)  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
            CipherCharacteristic(ks_characteristic=Characteristic(ch_weight=...,
                assignment_weights=[..., ..., ...],
                input_diff=[..., ...], output_diff=[..., ...],
                assign_outdiff_list=[..., ..., ...]),
            enc_characteristic=Characteristic(ch_weight=...,
                assignment_weights=[..., ..., ...],
                input_diff=[..., ...], output_diff=[..., ...], external_diffs=[..., ...],
                assign_outdiff_list=[..., ..., ...]))

        """
        return super().find_next_invalidprop_quantified_logic()


def round_based_invalidprop_search(
        func, initial_num_rounds, final_num_rounds, prop_type, solver_name,
        max_num_skipped_rounds=0, min_num_E0_rounds=1, min_num_E2_rounds=1,
        extra_chmodel_args=None,  # op_model_class2options,
        extra_invalidpropfinder_args=None,  # pm, fn, env, solver_seed
        exclude_zero_input_prop_E0=True,
        exclude_zero_input_prop_E2=True,
        **kwargs  # exclude_zero_input_prop_external_E, find_cipher_invalid_prop
        ):
    """Search for zero-probability (invalid) property pairs of round-based functions over multiple number of rounds.

    This function searches for universally-invalid characteristics
    (leading to invalid properties, see `InvalidPropFinder`)
    of a `RoundBasedFunction` ``func``
    by modelling the search as a sequence of SMT problems
    (using `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`),
    but the search is perfomed iteratively over the number of rounds of ``func``.
    That is, first universally-invalid characteristics covering ``initial_num_rounds`` rounds
    are searched, then ``initial_num_rounds + 1``, until ``final_num_rounds``.
    See also `round_based_ch_search`.

    .. note::
        The `RoundBasedFunction` ``func`` must include `add_round_outputs`
        calls in its ``eval``.

        While `InvalidPropFinder` requires wrapping the characteristic model
        if it has more than one non-trivial transition, this method does
        require the function ``func`` to be not wrapped.

        This method also requires that for all the round functions :math:`f_i`
        of ``func`` (generated through `SSA.split` with
        `SSA.get_round_separators`), given any probability-one
        characteristic over :math:`f` with input-output property
        :math:`(\\alpha, \\beta)`,  then there is no other probability-one
        characteristic with input property  :math:`\\alpha`
        (resp. output property :math:`\\beta`) but output property
        :math:`\\neq \\beta` (resp. input property :math:`\\neq \\alpha`).
        If all the round functions are permutations, then this is satisfied
        for `Difference` and `LinearMask` properties.
        See also `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`.

    This function proceed as follows:

    1. Set the current number of rounds of the universally-invalid characteristics to search
       for to ``initial_num_rounds``.
    2. Set the current number of initial rounds to skip  to ``0``.
    3. Set the number of rounds of ``func`` to the sum of the number of rounds
       of step 1 and step 2, and split ``func`` into :math:`E \circ S`
       (:math:`S` denotes the skipped rounds and :math:`E` the target function
       of the universally-invalid characteristics to search for).
    4. Create a `abstractproperty.chmodel.ChModel`
       (resp. `abstractproperty.chmodel.EncryptionChModel`) object
       of :math:`E` using as arguments ``prop_type`` and ``extra_chmodel_args``.
    5. Split :math:`E` into :math:`E = E_2 \circ E_1 \circ E_0`
       taking into account ``min_num_E0_rounds, min_num_E2_rounds``
       and generate the  characteristic models of :math:`(E_0, E_1, E_2)`
       using `abstractproperty.chmodel.ChModel.get_round_separators`
       and `abstractproperty.chmodel.ChModel.split`.
       See `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`
       for more details about :math:`(E_0, E_1, E_2)`.
    6. Create an `InvalidPropFinder` object with arguments
       the characteristic model over :math:`E_1`,
       ``solver_name`` and ``extra_invalidpropfinder_args``.
    7. Loop over the generator `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`
       (with arguments ``exclude_zero_input_prop_E0``
       and ``exclude_zero_input_prop_E2``)
       and yield all the 3-length tuples of characteristics from the
       generator (together with the current number of rounds).
    8. After the generator is exhausted, go to step 5 but splitting :math:`E`
       into antoher another partition :math:`(E_0, E_1, E_2)`.

       a. If all partitions has been exhausted,
          instead increase the current number of initial rounds to skip
          (up to ``max_num_skipped_rounds``) and go to step 3.
       b. If the current number of skipped rounds was ``max_num_skipped_rounds``,
          instead increase the current number of rounds of
          the universally-invalid characteristics to search for and go to step 2.
       c. If this number was ``final_num_rounds``, instead the search is finished.

    This function is a Python `generator` function
    (see `InvalidPropFinder`), returning an `iterator` that yields
    2-length tuples:

    * The first element in the tuple is a 4-length tuple containing
      the number of initial skipped rounds, the number of rounds
      of :math:`E_0`, the number of rounds of :math:`E_1`
      and the number of rounds of :math:`E_2`.
    * The second element in the tuple is a 3-length tuple containing
      the characteristics over :math:`E_0`, :math:`E_1` and :math:`E_2`
      respectively (i.e., the outputs of
      `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`).
      Note that these characteristics are
      `abstractproperty.characteristic.Characteristic` objects
      if ``func`` is a `RoundBasedFunction` object, or
      `abstractproperty.characteristic.EncryptionCharacteristic` objects
      if ``func`` is a `RoundBasedFunction`-encryption function of a `Cipher`.

    The argument ``prop_type`` is a particular `Property` such as `XorDiff`
    or `LinearMask`. For ``solver_name``, see `InvalidPropFinder`.
    The optional arguments ``extra_chmodel_args`` and ``extra_invalidpropfinder_args``
    can be given as dictionaries (in the form of ``**kwargs``) containing
    additional arguments for ``ChModel/EncryptionChModel`` and `InvalidPropFinder`
    respectively.

    It is possible to abort the current search for the current number of rounds
    and start the search with one more round by passing the
    value `INCREMENT_NUM_ROUNDS`
    to the generator iterator with `generator.send`
    (see `round_based_ch_search`).

    This function reuses information from previous partitions :math:`(E_0', E_1', E_2')`
    to directly avoid some new partitions :math:`(E_0, E_1, E_2)` that don't contain
    universally-invalid characteristics.
    Assume that no universally-invalid characteristic was found for the partition
    :math:`(E_0', E_1', E_2')`,
    where :math:`E_0'` covers from the :math:`a'`-th round to the :math:`b'`-th
    round (i.e., ``a'-›b'``) and :math:`E_2'` covers ``c'-›d'``.
    Then it holds that no universally-invalid characteristic can be found
    using `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle` from any partition
    :math:`(E_0, E_1, E_2)` where :math:`E_0` covers ``a-›a'-›b'-›b`` and
    :math:`E_2` covers ``c-›c'-›d'-›d``, that is,
    from any partition :math:`(E_0, E_1, E_2)`
    where :math:`E_0` covers ``a-›b`` and :math:`E_2` covers ``c-›d``
    such that :math:`a \le a', b' \le b, c \le c` and :math:`d' \le d`.

    .. note::
        Note that `InvalidPropFinder` contains other methods to search
        for universally-invalid characteristics (e.g.,
        `InvalidPropFinder.find_next_invalidprop_activebitmode` or
        `InvalidPropFinder.find_next_invalidprop_quantified_logic`)
        which might find universally-invalid characteristics faster.

    ::

        >>> # example of searching for XorDiff universally-invalid Characteristic over a BvFunction
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.smt.invalidpropsearch import round_based_invalidprop_search, INCREMENT_NUM_ROUNDS
        >>> from cascada.primitives import speck
        >>> Speck32_ks = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64).key_schedule
        >>> iterator = round_based_invalidprop_search(Speck32_ks, 3, 3, XorDiff, "btor",
        ...     extra_invalidpropfinder_args={"solver_seed":0})
        >>> for i, (tuple_rounds, tuple_chs) in enumerate(iterator):
        ...     print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))
        ...     if i == 2: break  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        (0, 1, 1, 1) : Ch(w=0, id=..., od=...), Ch(w=Infinity, id=..., od=...), Ch(w=0, id=..., od=...)
        (0, 1, 1, 1) : Ch(w=0, id=..., od=...), Ch(w=Infinity, id=..., od=...), Ch(w=0, id=..., od=...)
        (0, 1, 1, 1) : Ch(w=0, id=..., od=...), Ch(w=Infinity, id=..., od=...), Ch(w=0, id=..., od=...)
        >>> # example of searching for LinearMask universally-invalid EncryptionCharacteristic over a Cipher
        >>> from cascada.linear.mask import LinearMask
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> iterator = round_based_invalidprop_search(Speck32, 4, 5, LinearMask, "btor",
        ...     min_num_E0_rounds=2, extra_invalidpropfinder_args={"solver_seed":0})
        >>> tuple_rounds, tuple_chs = next(iterator)
        >>> print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))
        (0, 2, 1, 1) : Ch(w=0, id=0080 4021, od=0201 0200), Ch(w=Infinity, id=0201 0200, od=0000 0001), Ch(w=0, id=0000 0001, od=0004 0004)
        >>> tuple_rounds, tuple_chs = next(iterator)
        >>> print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))
        (0, 2, 1, 1) : Ch(w=0, id=0080 4021, od=0201 0200), Ch(w=Infinity, id=0201 0200, od=0080 e001), Ch(w=0, id=0080 e001, od=8002 8003)
        >>> iterator.send(INCREMENT_NUM_ROUNDS)  # stop current num_rounds and increment by 1
        >>> tuple_rounds, tuple_chs = next(iterator)
        >>> print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))
        (0, 2, 1, 2) : Ch(w=0, id=0080 4021, od=0201 0200), Ch(w=Infinity, id=0201 0200, od=0080 4021), Ch(w=0, id=0080 4021, od=0201 0200)

    .. Implementation details:

        Example 1.
        If there is no           2-›7 ID over E with E0 covering [2-›3] and E2  [6-›7]
        then there are no bigger *-›* ID over E with E0       [*-›2-›3] and E2  [6-›7-›*] for any *
        This is because in the 2nd case the Pr1-E0 outputs and the Pr-E2 inputs
        are a subset of the 1st case.

        Example 2.
        If there is no ID over E with E0 covering   [2-›3] and E2       [6-›7]
        then there is no ID over E with E0 covering [2-›3-›4] and E2 [5-›6-›7]
        This is because in the 2nd case E0 and E2 fully contains
        the E0 and E2 of the 1st case, (1st case contains more Pr.1 than 2nd).

    """
    if not (issubclass(func, cascada_ssa.RoundBasedFunction) or
            (issubclass(func, blockcipher.Cipher) and
             issubclass(func.encryption, cascada_ssa.RoundBasedFunction))):
        raise ValueError(f"{func} is not a RoundBasedFunction or a Cipher")

    if initial_num_rounds <= 2:
        raise ValueError(f"initial_num_rounds ({initial_num_rounds}) must be at least 3")

    assert all(isinstance(aux_nr, int) for aux_nr in
               [initial_num_rounds, final_num_rounds, max_num_skipped_rounds,
                min_num_E0_rounds, min_num_E2_rounds])
    assert initial_num_rounds <= final_num_rounds

    if extra_chmodel_args is None:
        extra_chmodel_args = {}
    else:
        extra_chmodel_args = extra_chmodel_args.copy()
    if extra_invalidpropfinder_args is None:
        extra_invalidpropfinder_args = {}
    else:
        extra_invalidpropfinder_args = extra_invalidpropfinder_args.copy()

    printing_mode = extra_invalidpropfinder_args.get("printing_mode", PrintingMode.Silent)
    filename = extra_invalidpropfinder_args.get("filename", None)
    smart_print = chsearch._get_smart_print(filename)

    find_cipher_invalid_prop = kwargs.pop("find_cipher_invalid_prop", False)
    exclude_zero_input_prop_external_E = kwargs.pop("exclude_zero_input_prop_external_E", None)
    # ignore_trivial_E1 = kwargs.pop("ignore_trivial_E1", False)

    assert not (exclude_zero_input_prop_external_E is not None and find_cipher_invalid_prop is False)

    if kwargs:
        raise ValueError(f"invalid arguments: {kwargs}")

    from cascada.differential.difference import XorDiff, RXDiff
    from cascada.linear.mask import LinearMask
    # from cascada.algebraic.value import BitValue, WordValue

    if prop_type in [XorDiff, RXDiff]:
        from cascada.differential.chmodel import ChModel, EncryptionChModel, CipherChModel
    elif prop_type == LinearMask:
        assert find_cipher_invalid_prop is False
        from cascada.linear.chmodel import ChModel, EncryptionChModel
    # elif prop_type in [BitValue, WordValue]:
    #     from cascada.algebraic.chmodel import ChModel, EncryptionChModel, CipherChModel
    else:
        raise ValueError(f"prop_type not in {[XorDiff, RXDiff, LinearMask]}")

    #

    bad_partitions = []

    def get_a_b_c_d_partition(my_num_S_rounds, my_num_E0_rounds, my_num_E1_rounds, my_num_E2_rounds):
        """Get the tuple (a, b, c, d) where a/b is the start/end rounds of E0 and similar for c/d and E2."""
        a = my_num_S_rounds
        b = my_num_S_rounds + my_num_E0_rounds
        c = my_num_S_rounds + my_num_E0_rounds + my_num_E1_rounds
        d = my_num_S_rounds + my_num_E0_rounds + my_num_E1_rounds + my_num_E2_rounds
        return a, b, c, d, [my_num_S_rounds, my_num_E0_rounds, my_num_E1_rounds, my_num_E2_rounds]

    #

    num_E_rounds = initial_num_rounds
    while True:  # loop over num_E_rounds (not a for loop due to INCREMENT_NUM_ROUNDS)
        found_invalidprop = False
        found_INCREMENT_NUM_ROUNDS = False

        for num_S_rounds in range(0, max_num_skipped_rounds + 1):
            func.set_num_rounds(num_S_rounds + num_E_rounds)

            external_chmodel = None
            if issubclass(func, blockcipher.Cipher):
                if find_cipher_invalid_prop:
                    aux_SE_ch_model = CipherChModel(func, prop_type, **extra_chmodel_args)
                    external_chmodel, SE_ch_model = aux_SE_ch_model.ks_ch_model, aux_SE_ch_model.enc_ch_model
                else:
                    SE_ch_model = EncryptionChModel(func, prop_type, **extra_chmodel_args)
            else:
                prefix = EncryptionChModel._prefix
                input_prop_names = [f"{prefix}p{i}" for i in range(len(func.input_widths))]
                SE_ch_model = ChModel(func, prop_type, input_prop_names, **extra_chmodel_args)

            if num_S_rounds == 0:
                S_ch_model, E_ch_model = None, SE_ch_model
            else:
                SE_all_rs = SE_ch_model.get_round_separators()
                if SE_all_rs is None or len(SE_all_rs) < 2:
                    raise ValueError(f"{SE_all_rs.func.get_name()} cannot be decomposed in 3 or more rounds")
                assert len(SE_all_rs) == num_S_rounds + num_E_rounds - 1
                # the end of the i-th round (i=1,2,...) is round_separators[i-1]
                S_ch_model, E_ch_model = SE_ch_model.split([SE_all_rs[num_S_rounds - 1]])
                # S/E_ch_model.func is not a RoundBasedFunction (no num_rounds, no get_round_separators)

            if printing_mode != PrintingMode.Silent:
                if num_E_rounds != initial_num_rounds:
                    smart_print("")
                    if printing_mode == PrintingMode.Debug:
                        smart_print("")
                smart_print(f"Current number of rounds of (S, E): "
                            f"({num_S_rounds}, {num_E_rounds})", prepend_time=True)

            if printing_mode == PrintingMode.Debug:
                if num_S_rounds > 0:
                    smart_print(f"Characteristic model over E \circ S: {SE_ch_model}")
                    smart_print(f"Output of S: {SE_all_rs[num_S_rounds - 1]}")
                    smart_print(f"Characteristic model over S: {S_ch_model}")
                smart_print(f"Characteristic model over E: {E_ch_model}")
                if external_chmodel:
                    smart_print(f"External characteristic model (over the key schedule): {external_chmodel}")

            for num_E1_rounds in range(1, num_E_rounds - 2 + 1):  # - 2 to reserve 1 round for E0 and 1 for E2
                aux_num_E0_rounds_num_E2_rounds = []
                for num_E0_rounds in range(1, num_E_rounds - num_E1_rounds - 1 + 1):  # - 1 to reserve 1 for E2
                    num_E2_rounds = num_E_rounds - num_E0_rounds - num_E1_rounds
                    assert num_E2_rounds > 0 and num_E0_rounds + num_E1_rounds + num_E2_rounds == num_E_rounds
                    if num_E0_rounds < min_num_E0_rounds or num_E2_rounds < min_num_E2_rounds:
                        continue
                    aux_num_E0_rounds_num_E2_rounds.append([num_E0_rounds, num_E2_rounds])
                # sorting ensure first (E0,E2) pair where each Ei has roughly half the rounds of E0+E2
                aux_num_E0_rounds_num_E2_rounds.sort(key=lambda x: (abs(x[0] - x[1]), x[1]))

                for num_E0_rounds, num_E2_rounds in aux_num_E0_rounds_num_E2_rounds:
                    a, b, c, d, _ = get_a_b_c_d_partition(num_S_rounds, num_E0_rounds, num_E1_rounds, num_E2_rounds)
                    for a_prime, b_prime, c_prime, d_prime, bad_partition in bad_partitions:
                        if a <= a_prime and b_prime <= b and c <= c_prime and d_prime <= d:
                            if printing_mode != PrintingMode.Silent:
                                if printing_mode == PrintingMode.Debug:
                                    smart_print("")
                                # EX_ch_model.func.get_name() doesn't give useful information
                                smart_print(f"Ignoring current number of rounds "
                                            f"({num_S_rounds}, {num_E0_rounds}, {num_E1_rounds}, {num_E2_rounds}) ",
                                            f"of (S, E0, E1, E2) due to previous (S', E0', E1', E2') "
                                            f"with number of rounds {tuple(bad_partition)} "
                                            f"that did not contain any universally-invalid characteristics",  prepend_time=True)
                            continue

                    E_all_rs = E_ch_model.get_round_separators()
                    if E_all_rs is None or len(E_all_rs) < 2:
                        raise ValueError(f"{E_ch_model.func.get_name()} cannot be decomposed in 3 or more rounds")
                    assert len(E_all_rs) == num_E_rounds - 1
                    E0_rs = E_all_rs[num_E0_rounds - 1]
                    E1_rs = E_all_rs[num_E0_rounds + num_E1_rounds - 1]
                    E0_ch_model, E1_ch_model, E2_ch_model = E_ch_model.split([E0_rs, E1_rs])

                    E1_non_id_opmodels = []
                    for op_model in E1_ch_model.assign_outprop2op_model.values():
                        if not isinstance(op_model, abstractproperty.opmodel.ModelIdentity):
                            E1_non_id_opmodels.append(op_model)
                    # if ignore_trivial_E1 and len(E1_non_id_opmodels) == 0:
                    #     continue
                    if len(E1_non_id_opmodels) > 1:
                        E1_ch_model = get_wrapped_chmodel(E1_ch_model)

                    if printing_mode != PrintingMode.Silent:
                        if printing_mode == PrintingMode.Debug:
                            smart_print(f"\nCharacteristic model over E0: {E0_ch_model}")
                            smart_print(f"Characteristic model over E1: {E1_ch_model}")
                            smart_print(f"Characteristic model over E2: {E2_ch_model}")
                        # EX_ch_model.func.get_name() doesn't give useful information
                        smart_print(f"Current number of rounds of (E0, E1, E2): "
                                    f"({num_E0_rounds}, {num_E1_rounds}, {num_E2_rounds})", prepend_time=True)

                    invalid_prop_finder = InvalidPropFinder(E1_ch_model, solver_name, **extra_invalidpropfinder_args)

                    if printing_mode == PrintingMode.Debug:
                        smart_print("Size of the base SMT problem:", invalid_prop_finder.formula_size())
                        smart_print(f"Base SMT problem:\n{invalid_prop_finder.hrepr()}")

                    iterator = invalid_prop_finder.find_next_invalidprop_miss_in_the_middle
                    for tuple_chs in iterator(E0_ch_model, E2_ch_model, ch_model_E=E_ch_model, ch_model_external_E=external_chmodel,
                                              exclude_zero_input_prop_E0=exclude_zero_input_prop_E0,
                                              exclude_zero_input_prop_E2=exclude_zero_input_prop_E2,
                                              exclude_zero_input_prop_external_E=exclude_zero_input_prop_external_E):
                        if num_S_rounds == 0 and find_cipher_invalid_prop:
                            prone_ch_E0, uni_inv_ch_E1, prone_ch_E2, external_prone_ch_E = tuple_chs
                            var_prop2ct_prop = collections.OrderedDict()
                            for vp, cp in itertools.chain(
                                    zip(aux_SE_ch_model.enc_ch_model.input_prop, prone_ch_E0.input_prop),
                                    zip(aux_SE_ch_model.enc_ch_model.output_prop, prone_ch_E2.output_prop),
                                    zip(aux_SE_ch_model.ks_ch_model.input_prop, external_prone_ch_E.input_prop)
                            ):
                                var_prop2ct_prop[vp] = cp
                            cipher_ch_finder = chsearch.CipherChFinder(
                                aux_SE_ch_model, ks_assert_type=chsearch.ChModelAssertType.Validity,
                                enc_assert_type=chsearch.ChModelAssertType.Validity, solver_name=solver_name,
                                var_prop2ct_prop=var_prop2ct_prop, raise_exception_missing_var=False,
                                printing_mode=printing_mode, filename=filename
                            )
                            for valid_cipher_ch_found in cipher_ch_finder.find_next_ch():
                                raise ValueError(
                                    "the concatenation of the last characteristic tuple found,"
                                    f"\n - prone_ch_E0: {prone_ch_E0}, {prone_ch_E0.ch_model}"
                                    f"\n - uni_inv_ch_E1: {uni_inv_ch_E1}, {uni_inv_ch_E1.ch_model}"
                                    f"\n - prone_ch_E2: {prone_ch_E2}, {prone_ch_E2.ch_model}"
                                    f"\n - external_prone_ch_E: {external_prone_ch_E}, {external_prone_ch_E.ch_model}"
                                    f"\n - cipher ch model: {aux_SE_ch_model}\n - var_prop2ct_prop: {var_prop2ct_prop}"
                                    f"\n - cipher_ch_finder: {cipher_ch_finder.initial_constraints+list(cipher_ch_finder.chmodel_asserts)},"
                                    f"\n is not universally-invalid (found compatible valid cipher characteristic {valid_cipher_ch_found})")
                            del cipher_ch_finder

                        found_invalidprop = True
                        tuple_rounds = (num_S_rounds, num_E0_rounds, num_E1_rounds, num_E2_rounds)
                        sent_value = (yield (tuple_rounds, tuple_chs))
                        if sent_value is not None:
                            if sent_value == INCREMENT_NUM_ROUNDS:
                                found_INCREMENT_NUM_ROUNDS = True
                                yield None
                                break
                            else:
                                warnings.warn(f"value {sent_value} is sent to the generator "
                                              f"but only sending INCREMENT_NUM_ROUNDS"
                                              f" affects the generator")

                    if found_INCREMENT_NUM_ROUNDS:
                        assert found_invalidprop
                        break

                    if not found_invalidprop:
                        assert found_INCREMENT_NUM_ROUNDS is False
                        bad_partitions.append(get_a_b_c_d_partition(num_S_rounds, num_E0_rounds, num_E1_rounds, num_E2_rounds))
                        if printing_mode == PrintingMode.Debug:
                            smart_print("No universally-invalid characteristic found for number of rounds "
                                        f"({num_E0_rounds}, {num_E1_rounds}, {num_E2_rounds}) of (E0, E1, E2)",
                                        prepend_time=True)

                if found_INCREMENT_NUM_ROUNDS:
                    assert found_invalidprop is True
                    break

            if found_INCREMENT_NUM_ROUNDS:
                assert found_invalidprop is True
                break

        if not found_invalidprop or num_E_rounds == final_num_rounds:
            break
        else:
            num_E_rounds += 1
            assert num_E_rounds <= final_num_rounds


def round_based_invalidcipherprop_search(
        cipher, initial_num_rounds, final_num_rounds, prop_type, solver_name,
        max_num_skipped_rounds=0, min_num_E0_rounds=1, min_num_E2_rounds=1,
        extra_cipherchmodel_args=None,
        extra_invalidcipherpropfinder_args=None,
        exclude_zero_input_prop_E0=True,
        exclude_zero_input_prop_E2=True,
        exclude_zero_input_prop_external_E=True,
        # **kwargs
    ):
    """Search for zero-probability (invalid) properties of iterated ciphers over multiple number of rounds.

    .. note::
        The `Cipher.encryption` of ``cipher`` must be a
        `RoundBasedFunction` including `add_round_outputs`
        calls in its ``eval``.

    This function is similar to `round_based_invalidprop_search`.
    The only differences are:

    - The function ``func`` (i.e., :math:`E \circ S`) is the
      `Cipher.encryption` of the given ``cipher``.
      Thus, :math:`S` denote the skipped rounds of the encryption function.
    - Let :math:`K` denote the `Cipher.key_schedule` of ``cipher``, that is,
      the function whose outputs are the round keys used in :math:`E \circ S`.
      The generator `InvalidPropFinder.find_next_invalidprop_miss_in_the_middle`
      is called with the argument ``ch_model_external_E`` given as the
      characteristic model over :math:`K` and with the argument
      ``exclude_zero_input_prop_external_E``.
    - This function yields 2-length tuples where the 2nd element is a
      4-length tuple; the last characteristic is the characteristic
      with probability 1 over :math:`K`.
      Thus, the concatenation of the first 3 characteristics is a universally-invalid
      characteristic over :math:`E` where the round key properties
      are given by the outputs of the probability-one characteristic
      over :math:`K`.

    Note that initial rounds are only skipped in the encryption function
    and not in the key-schedule function.

    .. note::
        Let ``(tuple_nr, tuple_ch)`` be an element
        yielded by `round_based_invalidcipherprop_search`.

        Let :math:`\\alpha_{K}` be the input property of the
        4-th characteristic  in ``tuple_ch``,
        and let :math:`(\\alpha_{E}, \\beta_{E})` be the input-output
        property pair of the concatenation of the first three characteristic
        in ``tuple_ch``.

        If ``tuple_nr[0]`` is 0, no initial rounds are skipped,
        and :math:`(\\alpha_{K}, \\alpha_{E}) \mapsto \\beta_{E}`
        is a universally-invalid cipher characteristic
        (as defined in `InvalidCipherPropFinder`) of
        ``cipher.set_num_rounds_and_return(tuple_nr[1]+tuple_nr[2]+tuple_nr[3])``,
        that is, the ``cipher`` with number of rounds
        ``tuple_nr[1]+tuple_nr[2]+tuple_nr[3]``.

        If ``tuple_nr[0]`` is not zero then a universally-invalid cipher characteristic
        is also obtained but the underlying cipher is more difficult
        to generate due to the skipped initial rounds.

    ::

        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.smt.invalidpropsearch import round_based_invalidcipherprop_search
        >>> from cascada.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> iterator = round_based_invalidcipherprop_search(Speck32, 3, 3, XorDiff, "btor",
        ...     extra_invalidcipherpropfinder_args={"solver_seed":0})
        >>> for i, (tuple_rounds, tuple_chs) in enumerate(iterator):
        ...     print(tuple_rounds, ":", ', '.join([ch.srepr() for ch in tuple_chs]))
        ...     if i == 1: break  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        (0, 1, 1, 1) : Ch(w=0, id=0000 8000, od=8000 8002),
                       Ch(w=Infinity, id=8000 8002, od=0000 8000),
                       Ch(w=0, id=0000 8000, od=0002 0000),
                       Ch(w=0, id=0000 0040 0000, od=0000 8000 8002)
        (0, 1, 1, 1) : Ch(w=0, id=0000 8000, od=8000 8002),
                       Ch(w=Infinity, id=8000 8002, od=0040 8000),
                       Ch(w=0, id=0040 8000, od=8002 8000),
                       Ch(w=0, id=0000 0040 0000, od=0000 8000 8002)


    .. Implementation details:

         A universally-invalid cipher characteristic cannot be returned since
         the underlying cipher cannot be generated due to the
         skipped rounds.

         Initial rounds are not skipped in the key-schedule function
         since ``split`` changes the names of the variables and afterwards
         these names don't match the round key names in the
         encryption characteristics.

         The key-schedule and the encryption computations cannot be
         easily merged into a single bit-vector function with the ``eval``
         method because in that case first the whole key-schedule
         would be computed (and thus the whole key-schedule would
         be part of the first split characteristic).

    """
    return round_based_invalidprop_search(
        cipher, initial_num_rounds, final_num_rounds, prop_type, solver_name,
        max_num_skipped_rounds=max_num_skipped_rounds,
        min_num_E0_rounds=min_num_E0_rounds,
        min_num_E2_rounds=min_num_E2_rounds,
        extra_chmodel_args=extra_cipherchmodel_args,
        extra_invalidpropfinder_args=extra_invalidcipherpropfinder_args,
        exclude_zero_input_prop_E0=exclude_zero_input_prop_E0,
        exclude_zero_input_prop_E2=exclude_zero_input_prop_E2,
        exclude_zero_input_prop_external_E=exclude_zero_input_prop_external_E,
        find_cipher_invalid_prop=True,
        # **kwargs
    )
