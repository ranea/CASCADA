"""Manage non-symbolic characteristics w.r.t an abstract property."""
import decimal
import collections
import functools
import itertools
import math
import random
import warnings

from cascada.bitvector import core
from cascada.bitvector import context
from cascada.bitvector import ssa as cascada_ssa
from cascada.bitvector import operation
from cascada.abstractproperty import property
from cascada.abstractproperty import chmodel as cascada_chmodel
from cascada.abstractproperty import opmodel as cascada_opmodel


zip = functools.partial(zip, strict=True)


class EmpiricalWeightData(object):
    """Represent the auxiliary data of `empirical_ch_weight`.

    This class stores the auxiliary data (the auxiliary empirical weights
    and the input parameters) of characteristic empirical weights
    computed through the method `Characteristic.compute_empirical_ch_weight`.

    Attributes:
        weight_avg_aux_prs: the negative binary logarithm (weight)
            of the average of the underlying probabilities of the
            auxiliary empirical weights
        num_aux_weights: the total number of auxiliary weights computed
            (including weights with value `math.inf`)
        num_inf_aux_weights: the number of auxiliary weights with value `math.inf`
        num_input_samples: the number of inputs sampled
        num_external_samples: the number of external samples used
            (equal to the number of auxiliary weights if there are two or more)
        seed: the seed used to sample bit-vectors
        C_code: whether the on-the-fly C implementations was used
        aux_weights: the list of auxiliary empirical weights

    """
    def __init__(self, aux_weights, num_input_samples, seed, C_code, num_external_samples=None):
        if len(aux_weights) == 1:
            self.weight_avg_aux_prs = aux_weights[0]
        else:
            # self.weight_avg_aux_prs is dependent of the order of aux_weights
            aux_weights = sorted(aux_weights)

            zero, two = decimal.Decimal(0), decimal.Decimal(2)
            aux_prs = [zero if w == math.inf else two**(-w) for w in aux_weights]

            avg_aux_prs = sum(aux_prs) / decimal.Decimal(len(aux_prs))
            if avg_aux_prs == 0:
                self.weight_avg_aux_prs = math.inf
            else:
                self.weight_avg_aux_prs = - cascada_opmodel.log2_decimal(avg_aux_prs)

        self.num_aux_weights = len(aux_weights)
        self.num_inf_aux_weights = aux_weights.count(math.inf)

        self.num_input_samples = num_input_samples
        self.num_external_samples = 0 if num_external_samples is None else num_external_samples

        assert self.num_aux_weights == max(1, self.num_external_samples)

        self.seed = seed
        self.C_code = C_code

        if self.weight_avg_aux_prs == math.inf:
            assert all(w == math.inf for w in aux_weights)
            self.aux_weights = aux_weights
        else:
            key_lambda = lambda x: math.inf if x == math.inf else (x - self.weight_avg_aux_prs).copy_abs()
            self.aux_weights = sorted(aux_weights, key=key_lambda)

    def __str__(self):
        return "{}(weight_avg_aux_prs={}, num_aux_weights={}, num_inf_aux_weights={}, " \
               "num_input_samples={}, seed={}, C_code={})".format(
            type(self).__name__,
            self.weight_avg_aux_prs,
            self.num_aux_weights,
            self.num_inf_aux_weights,
            self.num_input_samples,
            self.seed,
            self.C_code,
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `EmpiricalWeightData` object with the same content.
        """
        return "{}(num_input_samples={}{}, seed={}, C_code={}, aux_weights={})".format(
            type(self).__name__,
            self.num_input_samples,
            "" if not self.num_external_samples else f", num_external_samples={self.num_external_samples}",
            self.seed,
            self.C_code,
            self.aux_weights,
        )


class Characteristic(object):
    """Represent characteristics over bit-vector functions w.r.t some property.

    Given a `BvFunction` :math:`f` and its `SSA` decomposition into a list of
    *simple* assignments :math:`x_{i+1} \leftarrow f_i(x_i)`,
    a characteristic is a trail of properties
    :math:`(\Delta_{x_0} \mapsto \Delta_{x_1} \mapsto \dots \mapsto \Delta_{x_r})`
    containing the input and output `Property` pair
    :math:`(\Delta_{x_{i}}, \Delta_{x_{i+1}})` of each
    assignment :math:`x_{i+1} \leftarrow f_i(x_i)`,
    where :math:`f_i` is a bit-vector `Operation`.
    The initial property :math:`\Delta_{x_0}` is called the input property
    of the characteristic, and the last property :math:`\Delta_{x_r}`
    is called the output property of the characteristic.

    If a `BvFunction` :math:`f` contains external variables,
    recall the external variables appear in the `SSA` as input operands in some
    assignments. Thus, the properties associated to the external variables
    appear in the characteristic trail as input properties of those
    assignments.

    .. note::

        For example, given the function :math:`f(x) = (x \oplus 1) \\boxplus z`
        with external variable :math:`z`, a characteristic over :math:`f` is
        a trail of properties given by 2 property pairs:

        - The first property pair :math:`(\Delta_x, \Delta_{x'})`
          is over the assignment :math:`x' \leftarrow f_0(x) = x \oplus 1`.
        - The second property pair :math:`((\Delta_{x'}, \Delta_z), \Delta_{x''})``
          is over the assignment :math:`x'' \leftarrow f_1(x', z) = x' \\boxplus z`.

    The propagation probability of a characteristic,
    or simply the characteristic probability, is the product
    of the propagation probabilities of the `Property` pairs
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` over :math:`f_i`.

    The propagation weight of a characteristic is the negative binary
    logarithm of the characteristic probability, that is, the sum
    of the propagation weights of the `Property` pairs
    :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})`.

    .. note::

        For a function :math:`f(x) = g(x, x)` with duplicated input :math:`x`,
        if the property model of :math:`f` is not implemented but the model of
        :math:`g` is, then the property model of :math:`f` is approximated
        with the model of :math:`g`.

        For example, the `XorDiff` model of the unary operation
        :math:`f(x) = x \\boxplus x` is approximated
        by `XorModelBvAdd` with the same difference for the first and
        second operands.

    The characteristic probability is an approximation of the
    propagation probability of the property pair
    :math:`(\Delta_{x_0}, \Delta_{x_r})` over :math:`f`, but the accuracy
    of this approximation varies depending on the `Property` type,
    the function :math:`f` and the `SSA` decomposition of :math:`f`.
    For some properties such as  `Difference` and `LinearMask`,
    if :math:`f` has external variables then the characteristic probability
    approximates the propagation probability of :math:`(\Delta_{x_0}, \Delta_{x_r})`
    averaged over the set of all values of the external variables.

    A `Characteristic` is defined for a `abstractproperty.chmodel.ChModel`
    (providing the type of `Property`, the `BvFunction` :math:`f`,
    the `SSA` decomposition, and the (symbolic) `Property` propagations over
    the non-trivial assignments as `abstractproperty.opmodel.OpModel` objects).

    .. note::

        A `Characteristic` object represents an instance of characteristic
        (where all the properties and weights are constant values),
        whereas a `abstractproperty.chmodel.ChModel` object represents
        the (symbolic) model of a characteristic.

    A `Characteristic` also requires the values of the properties
    :math:`\Delta_{x_i}` in the trail. The property values can be given
    as `Constant` objects or `Property` of `Constant` objects.

    .. note::

        To initialize a `Characteristic`, all the constant values
        of the input, output, external and assignment-output properties
        must be given, even for properties not affecting the trail.
        These free symbolic properties can be given in the ``free_props``
        list to denote that their value does not affect the characteristic.

        If the underlying bit-vector function does not have external variables,
        or the attribute ``external_var2prop`` of the underlying characteristic
        model maps all external variables to constant `Property` objects,
        then the initialization argument ``external_props`` can be omitted.

    If the bit-vector function :math:`f` of the given
    `abstractproperty.chmodel.ChModel` is a
    `RoundBasedFunction` including `add_round_outputs` calls in its ``eval``,
    then the characteristic over each round can be obtaining by combining
    `split` and `get_round_separators`.

    To initialize an invalid `Characteristic` (with zero characteristic
    probability), the initialization argument ``is_valid=False``
    must be provided (by default, ``is_valid=True``).

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic w.r.t some property,
    (see `differential.characteristic.Characteristic`,
    `linear.characteristic.Characteristic`
    or `algebraic.characteristic.Characteristic` for some examples).

    Attributes:
        ch_model: the underlying `abstractproperty.chmodel.ChModel`.
        ch_weight: the decimal weight of the characteristic computed
            as the sum of `assignment_weights`.
        assignment_weights: the decimal weight of the `Property`
            :math:`(\Delta_{x_{i}} \mapsto \Delta_{x_{i+1}})` over
            each non-trivial assignment :math:`x_{i+1} \leftarrow f_i(x_i)`,
            computed from the `abstractproperty.opmodel.OpModel.decimal_weight`
            method of the `abstractproperty.opmodel.OpModel` objects stored in
            `tuple_assign_outprop2op_model`.
        empirical_ch_weight: the empirical weight of the characteristic
            (available after calling `compute_empirical_ch_weight`).
        empirical_data_list: a list of `EmpiricalWeightData` objects
            containing the auxiliary data of the empirical weight
            (see also `compute_empirical_ch_weight`).
        input_prop: a list of `Property` objects containing
            the (constant) input property.
        output_prop: a list of `Property` objects containing
            the (constant) output property.
        external_props: a list containing the (constant) `Property` of
            the external variables of the function.
        tuple_assign_outprop2op_model:  a tuple where each element is a pair
            containing: (1) the output (constant) `Property` :math:`\Delta_{x_{i+1}}`
            of the non-trivial assignment  :math:`x_{i+1} \leftarrow f_i(x_i)`
            and (2) the `abstractproperty.opmodel.OpModel` of this assignment with
            a (constant) input `Property` :math:`\Delta_{x_{i}}`.
        free_props: a list of (symbolic) `Property` objects of
            the `ch_model`, whose values do not affect the characteristic,
            and were replaced by constant properties in `input_prop`,
            `output_prop`, `external_props` or `tuple_assign_outprop2op_model`.
        var_prop2ct_prop: a `collections.OrderedDict` mapping each
            symbolic `Property` in the trail to its constant property.

    """
    _prop_label = None

    @staticmethod
    def get_properties_for_initialization(ch_model, var2ct, free_props=None):
        """Return the properties needed to initialize a `Characteristic` object.

        Given a `abstractproperty.chmodel.ChModel`
        and a dictionary mapping `Variable` objects
        (representing the symbolic properties) to their `Constant` values,
        this method returns the following objects: ``input_prop``,
        ``output_prop``, ``external_props`` and ``assign_outprop_list``
        for the ``Characteristic.__init__`` method.

        Symbolic properties not affecting the characteristics can be
        given in ``free_props`` and they will be set to 0.
        """
        assert isinstance(ch_model, cascada_chmodel.ChModel)

        var2ct = var2ct.copy()
        for v, c in var2ct.items():
            if not(isinstance(v, core.Variable) and isinstance(c, core.Constant)):
                raise ValueError(f"invalid item ({v.vrepr()}, {c.vrepr()}) in var2ct = {var2ct}")

        if free_props is None:
            free_props = []
        else:
            free_props = free_props[:]
        for var in ch_model.ssa._input_vars_not_used:
            if var not in var2ct:
                var_prop = ch_model.prop_type(var)
                if var_prop not in free_props:
                    free_props.append(var_prop)

        aux_intersection = set(var2ct.keys()).intersection(set([p.val for p in free_props]))
        assert len(aux_intersection) == 0, f"{aux_intersection}"

        prop_str = ch_model.prop_type.__name__

        input_prop = []
        for var_prop in ch_model.input_prop:
            if var_prop.val not in var2ct:
                if var_prop not in free_props:
                    raise ValueError(f"input {prop_str} {var_prop} not included in var2ct {var2ct}")
                ct_prop = type(var_prop)(core.Constant(0, var_prop.val.width))
            else:
                val = var2ct[var_prop.val]
                assert isinstance(val, core.Constant)
                ct_prop = type(var_prop)(val)
            input_prop.append(ct_prop)

        external_props = []
        for var in ch_model.external_var2prop:
            var_prop = ch_model.prop_type(var)
            if isinstance(ch_model.external_var2prop[var].val, core.Constant):
                assert var_prop not in free_props
                ct_prop = ch_model.external_var2prop[var]
                assert var2ct.get(var_prop.val, ct_prop.val) == ct_prop.val
                var2ct[var_prop.val] = ct_prop.val
            if var_prop.val not in var2ct:
                if var_prop not in free_props:
                    raise ValueError(f"external {prop_str} {var_prop} not included in var2ct {var2ct}")
                ct_prop = type(var_prop)(core.Constant(0, var_prop.val.width))
            else:
                val = var2ct[var_prop.val]
                assert isinstance(val, core.Constant)
                ct_prop = type(var_prop)(val)
            external_props.append(ct_prop)

        assign_outprop_list = []
        for var_prop, om in ch_model.assign_outprop2op_model.items():
            if var_prop.val not in var2ct:
                if isinstance(om, cascada_opmodel.ModelIdentity):
                    assert var_prop not in free_props or var_prop.val or ch_model.ssa._input_vars_not_used
                    ct_prop = type(var_prop)(om.input_prop[0].val.xreplace(var2ct))
                else:
                    if var_prop not in free_props:
                        raise ValueError(f"assignment output {prop_str} {var_prop} not included in var2ct {var2ct}")
                    ct_prop = type(var_prop)(core.Constant(0, var_prop.val.width))
                assert isinstance(ct_prop.val, core.Constant)
                assert var2ct.get(var_prop.val, ct_prop.val) == ct_prop.val
                var2ct[var_prop.val] = ct_prop.val
            else:
                val = var2ct[var_prop.val]
                assert isinstance(val, core.Constant)
                ct_prop = type(var_prop)(val)
            assign_outprop_list.append(ct_prop)

        output_prop = []
        for var_prop in ch_model.output_prop:
            if var_prop.val not in var2ct:
                if var_prop not in free_props:
                    raise ValueError(f"output {prop_str} {var_prop} not included in var2ct {var2ct}")
                ct_prop = type(var_prop)(core.Constant(0, var_prop.val.width))
            else:
                assert var_prop not in free_props
                val = var2ct[var_prop.val]
                assert isinstance(val, core.Constant)
                ct_prop = type(var_prop)(val)
            output_prop.append(ct_prop)

        return input_prop, output_prop, external_props, assign_outprop_list

    def _prop_init(self, input_prop, output_prop, assign_outprop_list,
                   ch_model, external_props, free_props,
                   empirical_ch_weight, empirical_data_list, is_valid):
        # auxiliary method called by __init__ and _prop_new (with "prop" labels in arguments)
        assert isinstance(ch_model, cascada_chmodel.ChModel)

        input_prop = input_prop[:]  # [:] also works on tuples (.copy() not)
        output_prop = output_prop[:]
        assign_outprop_list = assign_outprop_list[:]

        assert len(input_prop) == len(ch_model.input_prop), \
            f"len({input_prop}) == len({ch_model.input_prop})"
        for i, (ct_prop, var_prop) in enumerate(zip(input_prop, ch_model.input_prop)):
            if not isinstance(ct_prop, ch_model.prop_type):
                assert isinstance(ct_prop, core.Constant)
                ct_prop = ch_model.prop_type(ct_prop)
                input_prop[i] = ct_prop
            assert ct_prop.val.width == var_prop.val.width
        assert len(output_prop) == len(ch_model.output_prop), \
            f"len({output_prop}) == len({ch_model.output_prop})"
        for i, (ct_prop, var_prop) in enumerate(zip(output_prop, ch_model.output_prop)):
            if not isinstance(ct_prop, ch_model.prop_type):
                assert isinstance(ct_prop, core.Constant)
                ct_prop = ch_model.prop_type(ct_prop)
                output_prop[i] = ct_prop
            assert ct_prop.val.width == var_prop.val.width
        assert len(assign_outprop_list) == len(ch_model.assign_outprop2op_model), \
            f"{len(assign_outprop_list)} == {len(ch_model.assign_outprop2op_model)}" \
            f"\nlen({assign_outprop_list}) == len({ch_model.assign_outprop2op_model})"
        for i, (ct_prop, var_prop) in enumerate(zip(assign_outprop_list, ch_model.assign_outprop2op_model)):
            if not isinstance(ct_prop, ch_model.prop_type):
                assert isinstance(ct_prop, core.Constant)
                ct_prop = ch_model.prop_type(ct_prop)
                assign_outprop_list[i] = ct_prop
            assert ct_prop.val.width == var_prop.val.width

        if external_props is not None:
            external_props = external_props[:]
        else:
            external_props = list(ch_model.external_var2prop.values())
        assert len(external_props) == len(ch_model.external_var2prop), \
            f"len({external_props}) == len({ch_model.external_var2prop})"
        for i, (prop, var) in enumerate(zip(external_props, ch_model.external_var2prop)):
            if not isinstance(prop, ch_model.prop_type):
                assert isinstance(prop, core.Constant)
                prop = ch_model.prop_type(prop)
                external_props[i] = prop
            if not isinstance(prop.val, core.Constant):
                raise ValueError(f"external_{self.__class__._prop_label}s[i] = {prop} is not constant")
            aux_prop = ch_model.external_var2prop[var]
            if isinstance(aux_prop.val, core.Constant) and prop != aux_prop:
                raise ValueError(f"external_{self.__class__._prop_label}s[i] = {prop} != "
                                 f"{aux_prop} = external_var2{self.__class__._prop_label}[{var}]")

        if free_props is None:
            free_props = []
        else:
            free_props = free_props[:]
        for i, prop in enumerate(free_props):
            if not isinstance(prop, ch_model.prop_type):
                assert isinstance(prop, core.Variable)
                free_props[i] = ch_model.prop_type(prop)
            else:
                assert isinstance(prop, property.Property)
                assert isinstance(prop.val, core.Variable)
        aux_iv_fp = [ch_model.prop_type(var) for var in ch_model.ssa._input_vars_not_used]
        free_props = [p for p in aux_iv_fp if p not in free_props] + free_props
        free_props_set = set(free_props)

        self.ch_model = ch_model

        self.input_prop = input_prop
        self.output_prop = output_prop
        self.external_props = external_props

        self.free_props = free_props

        self.var_prop2ct_prop = self._get_var_prop2ct_prop(assign_outprop_list)
        self.tuple_assign_outprop2op_model = self._get_assign_outprop2op_model()

        self._is_valid = is_valid  # store before _get_ch_assignment_weights_and_check

        self.ch_weight, self.assignment_weights = self._get_ch_assignment_weights_and_check()

        with context.Simplification(False):
            var2ct = {v.val: d.val for v, d in self.var_prop2ct_prop.items()}
            validity = self.ch_model.validity_assertions()
            assert all(not free_props_set.intersection(assertion.atoms(core.Variable)) for assertion in validity)
            validity = [assertion.xreplace(var2ct) for assertion in validity]
            validity = functools.reduce(operation.BvAnd, validity)
            if validity == core.Constant(int(not is_valid), 1):
                raise cascada_opmodel.InvalidOpModelError(
                    f"is_valid = {is_valid} but ch_model.validity_assertions() = {validity}")
            elif validity not in [core.Constant(0, 1), core.Constant(1, 1)]:
                set_evs = set(self.ch_model.external_vars_validity_assertions())
                for v in validity.atoms(core.Variable):
                    if v not in set_evs:
                        raise ValueError(f"unknown variable {v} in validity_assertions\nch_model: "
                                         f"{self.ch_model}\nch_model.validity_assertions(): {validity}"
                                         f"\nknown assertion external variables: {set_evs}")

            pr_one = self.ch_model.pr_one_assertions()
            assert all(not free_props_set.intersection(assertion.atoms(core.Variable)) for assertion in pr_one)
            pr_one = [assertion.xreplace(var2ct) for assertion in pr_one]
            pr_one = functools.reduce(operation.BvAnd, pr_one)
            has_pr_one = self.ch_weight == 0
            if pr_one == core.Constant(int(not has_pr_one), 1):
                raise ValueError(f"{not has_pr_one} == ch_model.pr_one_assertions() but weight = {self.ch_weight}")
            elif pr_one != core.Constant(1, 1) and self.ch_weight == 0:
                set_evs = set(self.ch_model.external_vars_pr_one_assertions())
                for v in pr_one.atoms(core.Variable):
                    if v not in set_evs:
                        raise ValueError(f"unknown variable {v} in pr_one_assertions\nch_model: "
                                         f"{self.ch_model}\nch_model.pr_one_assertions(): {pr_one}"
                                         f"\nknown assertion external variables: {set_evs}")

        assert (empirical_ch_weight is None) == (empirical_data_list is None)
        # only checks involving both empirical_ch_weight and empirical_data_list
        if empirical_ch_weight is not None:
            assert all(isinstance(ewd, EmpiricalWeightData) for ewd in empirical_data_list)
        self.empirical_ch_weight = empirical_ch_weight
        self.empirical_data_list = empirical_data_list

    def __init__(self, input_prop, output_prop, assign_outprop_list,
                 ch_model, external_props=None, free_props=None,
                 empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        self._prop_init(
            input_prop=input_prop, output_prop=output_prop, assign_outprop_list=assign_outprop_list,
            ch_model=ch_model, external_props=external_props, free_props=free_props,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid
        )

    @classmethod
    def _prop_new(cls, input_prop, output_prop, assign_outprop_list,
                  ch_model, external_props, free_props,
                  empirical_ch_weight, empirical_data_list, is_valid=True):
        # create a new object with "prop" labels in arguments
        # (__init__ argument names might be overridden)
        obj = cls.__new__(cls)  # does not call __init__
        obj._prop_init(
            input_prop=input_prop, output_prop=output_prop, assign_outprop_list=assign_outprop_list,
            ch_model=ch_model, external_props=external_props, free_props=free_props,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid
        )
        return obj

    def _get_var_prop2ct_prop(self, assign_outprop_list):
        var_prop2ct_prop = collections.OrderedDict()
        for ct_prop, var_prop in zip(self.input_prop, self.ch_model.input_prop):
            var_prop2ct_prop[var_prop] = ct_prop
        for ct_prop, var in zip(self.external_props, self.ch_model.external_var2prop):
            var_prop2ct_prop[self.ch_model.prop_type(var)] = ct_prop

        extra_var_prop2ct_prop = {}
        for ct_prop, var_prop in zip(assign_outprop_list, self.ch_model.assign_outprop2op_model):
            extra_var_prop2ct_prop[var_prop] = ct_prop

        for var, var_prop in self.ch_model.var2prop.items():
            if self.ch_model.prop_type(var) in var_prop2ct_prop:
                # ignore input and external prop already processed
                continue
            # second dictionary's values overwriting those from the first.
            ct_prop = var_prop.xreplace({**extra_var_prop2ct_prop, **var_prop2ct_prop})
            # for v in ct_prop.val.atoms(core.Variable):
            #     assert v in self.ch_model.ssa.external_vars
            var_prop2ct_prop[self.ch_model.prop_type(var)] = ct_prop

        # check var_prop2ct_prop is consistent with input/output_prop and assign_outprop_list
        my_zip = zip([self.input_prop, self.output_prop, assign_outprop_list],
                     [self.ch_model.input_prop, self.ch_model.output_prop, self.ch_model.assign_outprop2op_model])
        for ct_prop_list, var_prop_list in my_zip:
            for ct_prop, var_prop in zip(ct_prop_list, var_prop_list):
                assert var_prop2ct_prop[var_prop] == ct_prop, f"{var_prop2ct_prop[var_prop]} != {ct_prop}"

        assert all(isinstance(k.val, core.Variable) and isinstance(v.val, core.Constant)
                   for k, v in var_prop2ct_prop.items())

        return var_prop2ct_prop

    def _get_assign_outprop2op_model(self):
        assign_outprop_op_model = []
        for outprop_var, op_model_symbolic in self.ch_model.assign_outprop2op_model.items():
            inprop_ct = [d.xreplace(self.var_prop2ct_prop) for d in op_model_symbolic.input_prop]
            op_model_ct = type(op_model_symbolic)(inprop_ct)
            outprop_ct = outprop_var.xreplace(self.var_prop2ct_prop)
            assign_outprop_op_model.append((outprop_ct, op_model_ct))
        return tuple(assign_outprop_op_model)

    def _get_ch_assignment_weights_and_check(self):
        ch_weight = decimal.Decimal(0)
        assignment_weights = []
        for i, (outprop, op_model) in enumerate(self.tuple_assign_outprop2op_model):
            with context.Simplification(False), context.Cache(False):
                validity = op_model.validity_constraint(outprop)
                pr_one = op_model.pr_one_constraint(outprop)

            # quick check for validity and pr_one
            set_evs = set(op_model.external_vars_validity_constraint(outprop))
            assert all(v in set_evs for v in validity.atoms(core.Variable))
            set_evs = set(op_model.external_vars_pr_one_constraint(outprop))
            assert all(v in set_evs for v in pr_one.atoms(core.Variable))
            assert not(pr_one == core.Constant(1, 1) and validity == core.Constant(0, 1))

            invalid_msg = f"the characteristic was initialized with is_valid=True but it "\
                          f"contains an invalid propagation {outprop} <- {op_model} " \
                          f"(False == validity_constraint())"

            if validity == core.Constant(0, 1):
                if self._is_valid is True:
                    raise cascada_opmodel.InvalidOpModelError(invalid_msg)
                aw = math.inf
            else:
                try:
                    aw = op_model.decimal_weight(outprop)
                except cascada_opmodel.InvalidOpModelError as e:
                    if self._is_valid is True:
                        raise cascada_opmodel.InvalidOpModelError(f"{str(e)}\n{invalid_msg}")
                    aw = math.inf

                assert not (aw == 0 and pr_one == core.Constant(0, 1))
                assert not (aw == math.inf and pr_one == core.Constant(1, 1))

            if math.inf in [aw, ch_weight]:
                # id adding math.inf (float) and aw (decimal)
                ch_weight = math.inf
            else:
                ch_weight += aw

            # print(f" - i, aw, error, fb, (outprop, op_model): {i}, {aw:.2f}, "
            #       f"{op_model.error():.2f}, {op_model.num_frac_bits()}, {outprop, op_model}")

            assignment_weights.append(aw)

        if self._is_valid != (ch_weight != math.inf):
            raise ValueError(f"weight = {ch_weight} but is_valid = {self._is_valid} was given")

        return ch_weight, assignment_weights

    def _check_bv_weights(self, bv_ch_weight, bv_assignment_weights, truncate=True):
        # if some OpModel is not valid, its bv_assignment_weight
        # (and bv_ch_weight) must be math.inf or None

        assert len(bv_assignment_weights) == len(self.tuple_assign_outprop2op_model)

        for i, (outprop, op_model) in enumerate(self.tuple_assign_outprop2op_model):
            bv_aw = bv_assignment_weights[i]
            if self.assignment_weights[i] == math.inf:
                if not(bv_aw is None or bv_aw == math.inf):
                    raise ValueError(f"assignment_weights[{i}] == math.inf but"
                                     f"bv_assignment_weights[{i}] == {bv_aw}")
            else:
                assert isinstance(bv_aw, core.Constant)
                assert bv_aw.width == op_model.weight_width()
                assert int(bv_aw) <= op_model.max_weight()
                bv_aw = int(bv_aw) / decimal.Decimal(2**op_model.num_frac_bits())
                decimal_aw = self.assignment_weights[i]
                abs_error = (bv_aw - decimal_aw).copy_abs()
                max_abs_error = op_model.error()
                max_abs_error = decimal.Decimal(max_abs_error).quantize(
                    decimal.Decimal("1." + "0" * (decimal.getcontext().prec // 2)),
                    rounding=decimal.ROUND_UP)
                if abs_error > max_abs_error:
                    raise ValueError(f"absolute error for {i}-th model {op_model} between "
                                     f"bit-vector assignment weight {bv_aw} (w/o frac bits) and "
                                     f"decimal assignment weight {decimal_aw} is {abs_error}, "
                                     f"which is greater than maximum absolute error "
                                     f"given by op_model.error()={max_abs_error}\n")

        if not self._is_valid:
            if not (bv_ch_weight is None or bv_ch_weight == math.inf):
                raise ValueError(f"ch_weight == {self.ch_weight} but bv_ch_weight is {bv_ch_weight}"
                                 f" and is_valid is {self._is_valid}")
            return

        # rest of method assume self._is_valid

        max_fb = self.ch_model.num_frac_bits()
        abs_error = (int(bv_ch_weight) - self.ch_weight).copy_abs()
        max_abs_error = self.ch_model.error()
        if truncate:
            # extra error due to truncate=True
            # e.g., if fb=2, max error = 0.11b (= 0.75 decimal) = 1 - 2**2
            # last factor to avoid Python decimal error
            extra_error = 1 - (decimal.Decimal(2) ** (-max_fb))
            assert not (max_fb == 0 and extra_error != 0)
            max_abs_error += extra_error
        max_abs_error = decimal.Decimal(max_abs_error).quantize(
            decimal.Decimal("1." + "0" * (decimal.getcontext().prec // 2)), rounding=decimal.ROUND_UP)
        if abs_error > max_abs_error:
            raise ValueError(f"absolute error between bit-vector ch. weight {bv_ch_weight} "
                             f"(truncate={truncate}) and decimal weight {self.ch_weight} is {abs_error}, "
                             f"which is greater than maximum absolute error given by "
                             f"ch_model.error()={max_abs_error}\n")

        def zero_ext_right(var, num_zeros):
            """Expand with zeros to the right."""
            return var if num_zeros == 0 else operation.Concat(var, core.Constant(0, num_zeros))

        # extracted from weight_assertions
        # max_fb = self.ch_model.num_frac_bits()
        max_width_wo_truncate = self.ch_model.weight_width(truncate=False)
        aw_with_max_fb_and_width = []
        for i, (_, op_model) in enumerate(self.tuple_assign_outprop2op_model):
            bv_aw = bv_assignment_weights[i]
            bv_aw = zero_ext_right(bv_aw, max_fb - op_model.num_frac_bits())
            bv_aw = operation.zero_extend(bv_aw, max_width_wo_truncate - bv_aw.width)
            aw_with_max_fb_and_width.append(bv_aw)
        if len(aw_with_max_fb_and_width) > 0:
            sum_aw = sum(aw_with_max_fb_and_width)
        else:
            assert max_fb == 0
            sum_aw = core.Constant(0, 1)
        if truncate:
            if max_fb >= sum_aw.width:
                if sum_aw != 0:
                    aux_max_weight = self.ch_model.max_weight(truncate=False) / decimal.Decimal(2**max_fb)
                    assert 0 < aux_max_weight < 1
                    raise ValueError(f"_check_bv_weights(..., truncate=True) cannot truncate characteristic weight"
                                     f" with 0 < {aux_max_weight}=(max_weight(truncate=False) / num_frac_bits()) < 1")
            else:
                sum_aw = sum_aw[:max_fb]

        assert bv_ch_weight == sum_aw
        assert bv_ch_weight.width == self.ch_model.weight_width(truncate=truncate)
        assert int(bv_ch_weight) <= self.ch_model.max_weight(truncate=truncate)

        free_props_set = set(self.free_props)
        with context.Simplification(False):
            var2ct = {v.val: d.val for v, d in self.var_prop2ct_prop.items()}
            wa = self.ch_model.weight_assertions(bv_ch_weight, bv_assignment_weights, truncate=True)
            assert all(not free_props_set.intersection(assertion.atoms(core.Variable)) for assertion in wa)
            wa = [assertion.xreplace(var2ct).xreplace(var2ct) for assertion in wa]
            wa = functools.reduce(operation.BvAnd, wa)
            if wa == core.Constant(0, 1):
                raise ValueError(f"is_valid = {self._is_valid} but False == "
                                 f"ch_model.weight_assertions({bv_ch_weight}, {bv_assignment_weights})")
            elif wa != core.Constant(1, 1):
                set_evs = set(self.ch_model.external_vars_weight_assertions(bv_assignment_weights))
                for v in wa.atoms(core.Variable):
                    if v not in set_evs:
                        raise ValueError(f"unknown variable {v} in weight_assertions\nch_model: "
                                         f"{self.ch_model}\nch_model.weight_assertions(): {wa}"
                                         f"\nknown assertion external variables: {set_evs}")

    def __str__(self):
        if self.empirical_data_list is not None:
            edl_str = f"[{', '.join([str(e) for e in self.empirical_data_list])}]"
        prop = self.ch_model.__class__._prop_label
        format_str = f"{{}}(ch_weight={{}}{{}}, assignment_weights={{}}, " \
                     f"input_{prop}={{}}, output_{prop}={{}}{{}}, assign_out{prop}_list={{}}{{}})"
        return format_str.format(
            type(self).__name__,
            self.ch_weight,
            "" if self.empirical_ch_weight is None else f", empirical_ch_weight={self.empirical_ch_weight}",
            # str(self.input_prop) ignored to print it in a list-like way
            f"[{', '.join([str(v) for v in self.assignment_weights])}]",
            f"[{', '.join([str(v.val) for v in self.input_prop])}]",
            f"[{', '.join([str(v.val) for v in self.output_prop])}]",
            "" if not self.external_props else f", external_{prop}s=[{', '.join([str(v.val) for v in self.external_props])}]",
            f"[{', '.join([str(v.val) for v, _ in self.tuple_assign_outprop2op_model])}]",
            "" if not self.free_props else f", free_{prop}s=[{', '.join([str(v.val) for v in self.free_props])}]",
            "" if self.empirical_data_list is None else f", empirical_data_list={edl_str}",
        )

    __repr__ = __str__

    def vrepr(self, ignore_external_props=False):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `Characteristic` object with the same content.
        """
        if self.empirical_data_list is not None:
            edl_vrepr = f"[{', '.join([e.vrepr() for e in self.empirical_data_list])}]"
        # using v.val.vrepr() instead of v.vrepr() since __init__ allows Constant
        # or Property(Constant) and the former is easier
        prop = self.ch_model.__class__._prop_label
        format_str = f"{{}}(input_{prop}={{}}, output_{prop}={{}}, " \
                     f"assign_out{prop}_list={{}}, ch_model={{}}{{}}{{}}{{}}{{}}{{}})"
        return format_str.format(
            type(self).__name__,
            f"[{', '.join([v.val.vrepr() for v in self.input_prop])}]",
            f"[{', '.join([v.val.vrepr() for v in self.output_prop])}]",
            f"[{', '.join([v.val.vrepr() for v, _ in self.tuple_assign_outprop2op_model])}]",
            self.ch_model.vrepr(),
            "" if ignore_external_props or not self.external_props else
                f", external_{prop}s=[{', '.join([v.val.vrepr() for v in self.external_props])}]",
            "" if not self.free_props else f", free_{prop}s=[{', '.join([v.val.vrepr() for v in self.free_props])}]",
            "" if self.empirical_ch_weight is None else f", empirical_ch_weight={self.empirical_ch_weight}",
            "" if self.empirical_data_list is None else f", empirical_data_list={edl_vrepr}",
            "" if self._is_valid else f", is_valid={self._is_valid}"
        )

    def srepr(self):
        """Return a short string representation of the characteristic.

        The short representation includes the characteristic weight rounded up
        to 3 fractional digits (and similar for the empirical weight if defined)
        and the input and output properties, printed in hexadecimal
        but omitting the prefix ``0x``.

        Doctest included in `vrepr`.
        """
        def to_hex_or_bin(my_bv):
            my_bv = my_bv.val  # my_bv is a Property object
            if my_bv.width % 4 != 0:
                my_bv = operation.Concat(core.Constant(0, 4 - (my_bv.width % 4)), my_bv)
            return my_bv.hex()[2:]  # remove 0x

        num_frac_digits = 3
        my_context = decimal.Context(prec=num_frac_digits+1)
        ch_weight = my_context.create_decimal(self.ch_weight)
        if self.empirical_ch_weight is not None:
            if self.empirical_ch_weight == math.inf:
                empirical_ch_weight = math.inf
            else:
                empirical_ch_weight = my_context.create_decimal(self.empirical_ch_weight)
        else:
            empirical_ch_weight = None
        input_prop = ' '.join([to_hex_or_bin(x) for x in self.input_prop])
        output_prop = ' '.join([to_hex_or_bin(x) for x in self.output_prop])
        return "Ch(w={}{}, id={}, od={})".format(
            ch_weight,
            "" if self.empirical_ch_weight is None else f", ew={empirical_ch_weight}",
            input_prop,
            output_prop)

    def signature(self, ch_signature_type):
        """Return the signature of the characteristic.

        This method is similar that `abstractproperty.chmodel.ChModel.signature`
        but for non-symbolic characteristic.
        """
        symbolic_sig = self.ch_model.signature(ch_signature_type)
        ct_sig = []
        for var in symbolic_sig:
            ct_sig.append(self.ch_model.prop_type(var).xreplace(self.var_prop2ct_prop).val)
        return ct_sig

    def split(self, prop_separators):
        """Split into multiple `Characteristic` objects given the list of property separators.

        Given the `Characteristic` :math:`c`, this method returns a list of characteristics
        :math:`c_1, c_2, ..., c_n`, such that their composition is equal to
        the main characteristic :math:`c`.

        The argument ``prop_separators`` is a list containing lists of symbolic properties.
        The :math:`i`-th property list denote the last properties of the
        :math:`i`-th `abstractproperty.chmodel.ChModel`.
        In other words, the :math:`(i+1)`-th characteristic
        immediately starts after the last property in ``prop_separators[i]``.

        To split into :math:`n` characteristics, ``prop_separators`` must contain
        :math:`n-1` lists, as the property list of the last characteristic is not given
        (its last properties are the output properties of the main characteristic).

        This method internally calls `abstractproperty.chmodel.ChModel.split` to
        split the underlying characteristic model `ch_model`.
        """
        sub_chmodel_list = self.ch_model.split(prop_separators)

        class SubCh(object):
            def __init__(self_ch, sub_chmodel):
                self_ch.input_prop = []
                self_ch.output_prop = []
                self_ch.assign_outprop_list = []
                self_ch.ch_model = sub_chmodel
                self_ch.external_props = []
                self_ch.free_props = []

            def __str__(self_ch):
                msg = "SubCh:"
                msg += f"\n - input_prop: {self_ch.input_prop}"
                msg += f"\n - output_prop: {self_ch.output_prop}"
                msg += f"\n - assign_outprop_list: {self_ch.assign_outprop_list}"
                msg += f"\n - external_props: {self_ch.external_props}"
                msg += f"\n - free_props: {self_ch.free_props}"
                msg += f"\n - ch_model: {self_ch.ch_model}"
                return msg

        assert len(sub_chmodel_list) == len(prop_separators) + 1
        sub_ch_list = [SubCh(sub_chmodel_list[i]) for i in range(len(prop_separators) + 1)]

        # old_om2ct_op_om maps OpModel from the parent ChModel to (ct) OpModel from the parent Characteristic
        # (old) only refers to OpModel from the parent ChModel (!= from OpModel from SubChModels)
        # (ct OpModel from the parent Characteristic == ct OpModel from SubCh)
        old_om2ct_op_om = collections.OrderedDict()
        for i, (old_om) in enumerate(self.ch_model.assign_outprop2op_model.values()):
            if not isinstance(old_om, cascada_opmodel.ModelIdentity):
                old_om2ct_op_om[old_om] = self.tuple_assign_outprop2op_model[i]

        # new_om2old_om maps OpModel from the SubChModel to OpModel from the parent ChModel
        old_om_list = list(old_om2ct_op_om.keys())
        new_om2old_om = collections.OrderedDict()
        for sub_ch in sub_ch_list:
            for new_om in sub_ch.ch_model.assign_outprop2op_model.values():
                if not isinstance(new_om, cascada_opmodel.ModelIdentity):
                    assert new_om not in new_om2old_om
                    old_om = old_om_list[len(new_om2old_om)]
                    assert type(new_om) == type(old_om), f"{old_om} != {new_om}"
                    new_om2old_om[new_om] = old_om
        assert len(old_om2ct_op_om) == len(new_om2old_om)

        aux_free_props = set()
        for i, sub_ch in enumerate(sub_ch_list):
            # if i == 0:
            #     print("# Main Characteristic and ChModel")
            #     print(self)
            #     print(self.ch_model)
            #     print(self.ch_model.ssa)
            # print(f"\n# Computing sub_ch_list[{i}]")
            # print("ch_model:", sub_ch.ch_model)
            # print("ssa:", sub_ch.ch_model.ssa)

            var_prop2ct_prop = collections.OrderedDict()
            for var_prop, ct_prop in zip(sub_ch.ch_model.input_prop,
                                         self.input_prop if i == 0 else sub_ch_list[i-1].output_prop):
                assert ct_prop not in var_prop2ct_prop
                var_prop2ct_prop[var_prop] = ct_prop

            # print("var_prop2ct_prop (+ input_prop):", var_prop2ct_prop)

            # ChModel.split ensures external props are the same
            for var, ct_prop in zip(self.ch_model.external_var2prop, self.external_props):
                if var in sub_ch.ch_model.external_var2prop:
                    assert ct_prop not in var_prop2ct_prop
                    var_prop2ct_prop[type(ct_prop)(var)] = ct_prop

            # print("var_prop2ct_prop (+ external_props):", var_prop2ct_prop)

            found_invalid_om = False
            for var_out_prop, new_om in sub_ch.ch_model.assign_outprop2op_model.items():
                if isinstance(new_om, cascada_opmodel.ModelIdentity):
                    ct_in_prop = new_om.input_prop[0].xreplace(var_prop2ct_prop)
                    assert isinstance(ct_in_prop.val, core.Constant)
                    assert var_prop2ct_prop.get(var_out_prop, ct_in_prop) == ct_in_prop
                    var_prop2ct_prop[var_out_prop] = ct_in_prop
                else:
                    ct_out_prop, ct_old_om = old_om2ct_op_om[new_om2old_om[new_om]]

                    index_assign = self.tuple_assign_outprop2op_model.index((ct_out_prop, ct_old_om))
                    if self.assignment_weights[index_assign] == math.inf:
                        found_invalid_om = True

                    for var_in_prop, ct_in_prop in zip(new_om.input_prop, ct_old_om.input_prop):
                        var_in_prop = var_in_prop.xreplace(var_prop2ct_prop)
                        if isinstance(var_in_prop.val, core.Constant):
                            continue
                        assert isinstance(var_in_prop.val, core.Variable)
                        assert var_prop2ct_prop.get(var_in_prop, ct_in_prop) == ct_in_prop
                        var_prop2ct_prop[var_in_prop] = ct_in_prop

                    assert var_prop2ct_prop.get(var_out_prop, ct_out_prop) == ct_out_prop
                    var_prop2ct_prop[var_out_prop] = ct_out_prop

            # print("var_prop2ct_prop (+ assignment):", var_prop2ct_prop)

            var2ct = collections.OrderedDict(
                [(var_prop.val, ct_prop.val) for var_prop, ct_prop in var_prop2ct_prop.items()]
            )
            init_props = self.__class__.get_properties_for_initialization(sub_ch.ch_model, var2ct)
            sub_ch.input_prop, sub_ch.output_prop, \
            sub_ch.external_props, sub_ch.assign_outprop_list = init_props

            if i == len(sub_ch_list) - 1:
                assert sub_ch.output_prop == self.output_prop

            # print("sub_ch (SubCh):", sub_ch_list[i])

            # ensuring sub_ch_list[i] is an instance of Ch and not of EncCh
            if issubclass(self.__class__, EncryptionCharacteristic):
                for ch_cls in self.__class__.__bases__:
                    if not issubclass(ch_cls, EncryptionCharacteristic):
                        break
                else:
                    raise ValueError(f"Characteristic not in parents of {self.__class__} = "
                                     f"{self.__class__.__bases__}")
            else:
                ch_cls = self.__class__

            sub_ch_list[i] = ch_cls._prop_new(
                input_prop=sub_ch.input_prop,
                output_prop=sub_ch.output_prop,
                assign_outprop_list=sub_ch.assign_outprop_list,
                ch_model=sub_ch.ch_model,
                external_props=sub_ch.external_props,
                free_props=sub_ch.free_props,
                empirical_ch_weight=None,
                empirical_data_list=None,
                is_valid=not found_invalid_om
            )

            # print("sub_ch (Characteristic):", sub_ch_list[i])

            aux_free_props |= set(sub_ch_list[i].free_props)

        assert len(aux_free_props) == len(self.free_props)

        return sub_ch_list

    def _get_empirical_ch_weights_C(self, num_input_samples, num_external_samples, seed, num_parallel_processes, verbose=False):
        """Return a list of empirical weights (one for each ``num_external_samples``)
        by compiling and executing C code."""
        raise NotImplementedError("subclasses must override this method")

    def _get_empirical_ch_weights_Python(self, num_input_samples, num_external_samples, seed):
        """Return a list of empirical weights (one for each ``num_external_samples``)."""
        raise NotImplementedError("subclasses must override this method")

    def _get_empirical_data_complexity(self, num_input_samples, num_external_samples):
        """Return num_input_samples and num_external_samples depending on the weight."""
        raise NotImplementedError("subclasses must override this method")

    def compute_empirical_ch_weight(self, num_input_samples=None, num_external_samples=None,
                                    split_by_max_weight=None, split_by_rounds=False,
                                    seed=None, C_code=False, num_parallel_processes=None):
        """Compute and store the empirical weight.

        Given the characteristic
        :math:`(\Delta_{x_0} \mapsto \Delta_{x_1} \mapsto \dots \mapsto \Delta_{x_r})`
        over :math:`f`,
        the empirical weight is an estimation of the propagation weight
        of :math:`(\Delta_{x_0}, \Delta_{x_r})` empirically obtained by evaluating
        :math:`f` for many sampled inputs and external samples
        (if :math:`f` has external variables).

        .. note::

            The empirical weight only takes into account
            the input and output properties;
            intermediate properties in the trail are ignored.

            Note it is possible for the empirical weight to be smaller
            than `ch_weight` in some cases and to be larger in others.

        If :math:`f` does not have external variables,
        in general (depends on  the `Property`)
        the empirical weight is computed by :math:`f`-evaluating
        ``num_input_samples`` random inputs satisfying `input_prop`
        and counting the number of outputs satisfying `output_prop`.
        This procedure is called here the basic subroutine, and it varies
        from `Property` to `Property`.

        If :math:`f` contains external variables,
        the basic subroutine is repeated
        ``num_external_samples`` times, where the external variables
        are fixed in each iteration to random values satisfying `external_props`.
        As a result, ``num_external_samples`` auxiliary empirical weights are
        obtained, and the final empirical weight is taken as
        the negative binary logarithm of the average of the underlying
        probabilities of the auxiliary empirical weights.

        .. note::

            The average is taken on the underlying probabilities since
            these probabilities might be zero in some cases
            (and their corresponding weights `math.inf`).

        This method computes the empirical weight and stores it
        in `empirical_ch_weight`. Moreover, the input parameters
        and the auxiliary weights is stored in
        `empirical_data_list` as a list of `EmpiricalWeightData` objects.
        This list contains multiple objects only if the characteristic
        was split into multiple ones (see below); in that case, this list
        contains an `EmpiricalWeightData` object for each characteristic
        the main one was split into.

        If one of the arguments ``split_by_max_weight`` or ``split_by_rounds``
        is enabled, the characteristic is first `split` into multiple characteristics
        and then the empirical weights of the new characteristics are computed
        (by calling `compute_empirical_ch_weight` on each new characteristic).
        The final empirical weight is taken as the sum of
        the empirical weights of the new characteristics.
        If the empirical weight of one of the new characteristics is
        `math.inf`, the ongoing computations are aborted and the unfinished
        empirical weights are stored as `EmpiricalWeightData` objects
        with ``math.inf`` weight, ``num_input_samples=0`` and
        ``num_external_samples=0``.

        .. note::

            When the characteristic is split, the final empirical weight is more
            an approximation of the characteristic weight that the weight of
            the input-output pair, as now the empirical weight computation
            also takes into account the intermediate properties
            that are part of the input and output properties of the new
            characteristics are also taken into account.

        Args:
            num_input_samples: the number of inputs to sample.
            num_external_samples: the number of external samples
                (set to 0 if the underlying bit-vector function does not
                contain external variables).
            split_by_max_weight: an optional number; if given, computes
                the empirical weight by splitting the characteristic
                into multiple characteristics (each one with maximum
                weight less than
                ``max(split_by_max_weight, max(self.assignment_weights))``
                ).
            split_by_rounds: if ``True``, computes the empirical weight
                by splitting the characteristic into multiple
                characteristics, each one covering one round (only available
                if the underlying function of the characteristic model is
                a `RoundBasedFunction` including `add_round_outputs` calls
                in its ``eval``)
            seed: the seed for sampling random bit-vectors
            C_code: whether to use a faster C implementation generated on-the-fly
            num_parallel_processes: if not ``None`` and ``C_code=True``,
                the auxiliary empirical weights are computed in parallel
                (using the given number of worker processes)

        """
        if num_parallel_processes is not None and C_code is False:
            raise ValueError("num_parallel_processes is only supported for C_code=True")
        assert num_parallel_processes is None or 2 <= num_parallel_processes

        if C_code:
            for prop in itertools.chain(self.input_prop, self.output_prop, self.external_props):
                if prop.val.width > 64:
                    warnings.warn("disabling C_code and num_parallel_processes as "
                                  "bit-vector with more than 64 bits found")
                    C_code = False
                    num_parallel_processes = None
                    break

        if num_input_samples is None and self.ch_weight == math.inf:
            raise ValueError("num_input_samples must be provided if ch_weight == math.inf")

        prop_separators = None

        if split_by_max_weight is not None and split_by_max_weight < self.ch_weight:
            if split_by_max_weight < max(self.assignment_weights):
                split_by_max_weight = max(self.assignment_weights)
                warnings.warn(f"setting split_by_max_weight to max(assignment_weights)={split_by_max_weight}")
            prop_separators = [[]]
            max_subch_weigh = split_by_max_weight
            sum_aw = 0
            for aw, outprop in zip(self.assignment_weights, self.ch_model.assign_outprop2op_model):
                if sum_aw + aw <= max_subch_weigh:  # or len(prop_separators[-1]) == 0
                    sum_aw += aw
                    prop_separators[-1].append(outprop)
                else:
                    assert len(prop_separators[-1]) > 0
                    sum_aw = 0
                    prop_separators.append([])
            # remove last item so that the new last item is a separator
            # (and not a terminator) between the 2nd-to-last round and the last round
            del prop_separators[-1]
            prop_separators = tuple(tuple(ds) for ds in prop_separators)
        elif split_by_rounds:
            if not issubclass(self.ch_model.func, cascada_ssa.RoundBasedFunction):
                raise ValueError("the underlying function of the characteristic "
                                 "model is not a RoundBasedFunction")
            if self.ch_model._rounds_outputs is None or len(self.ch_model._rounds_outputs) == 0:
                raise ValueError("the underlying function of the characteristic "
                                 "model does not include add_round_outputs calls")
            prop_separators = self.get_round_separators()

        if prop_separators is not None and prop_separators != (((),)) and len(prop_separators) > 0:
            sub_ch_list = self.split(prop_separators)

            total_ew = decimal.Decimal(0)
            for i, sub_ch in enumerate(sub_ch_list):
                sub_ch.compute_empirical_ch_weight(
                    num_input_samples=num_input_samples,
                    num_external_samples=num_external_samples,
                    C_code=C_code, seed=seed)
                assert len(sub_ch.empirical_data_list) == 1

                if sub_ch.empirical_ch_weight != math.inf:
                    total_ew += sub_ch.empirical_ch_weight
                else:
                    total_ew = math.inf
                    for j in range(i + 1, len(sub_ch_list)):
                        sub_ch_list[j].empirical_data_list = [EmpiricalWeightData(
                            aux_weights=[math.inf], num_input_samples=0,
                            num_external_samples=0, seed=seed, C_code=C_code)]
                    break

            self.empirical_ch_weight = total_ew
            self.empirical_data_list = [sub_ch.empirical_data_list[0] for sub_ch in sub_ch_list]
        else:
            num_input_samples, num_external_samples = self._get_empirical_data_complexity(
                num_input_samples, num_external_samples
            )

            if C_code:
                if num_input_samples * num_external_samples > 2**30:
                    warnings.warn("calling compute_empirical_ch_weight() with more than 2**30 samples "
                                  f"({num_input_samples * num_external_samples})")
                ew_list = self._get_empirical_ch_weights_C(num_input_samples, num_external_samples, seed, num_parallel_processes)
            else:
                if num_input_samples * num_external_samples > 2**20:
                    warnings.warn("calling compute_empirical_ch_weight() with C_code=False and with more than "
                                  f"2**20 samples ({num_input_samples * num_external_samples})")
                ew_list = self._get_empirical_ch_weights_Python(num_input_samples, num_external_samples, seed)

            data = EmpiricalWeightData(
                aux_weights=ew_list, num_input_samples=num_input_samples,
                num_external_samples=num_external_samples, seed=seed, C_code=C_code)

            self.empirical_ch_weight = data.weight_avg_aux_prs
            self.empirical_data_list = [data]

    @classmethod
    def _sample_bv(cls, myPRNG, width):
        return core.Constant(myPRNG.randrange(2 ** width), width)

    @classmethod
    def _sample_outprop_opmodel(cls, prop_type, ct_op_model, outprop_width, get_random_bv):
        zero_one_set = {core.Constant(0, 1), core.Constant(1, 1)}
        with context.Simplification(False):
            for i in range(2**min(outprop_width, 16)):
                ct_prop = prop_type(get_random_bv(outprop_width))
                validity = ct_op_model.validity_constraint(ct_prop)
                if validity not in zero_one_set:
                    evs = ct_op_model.external_vars_validity_constraint(ct_prop)
                    assert len(evs) > 0
                    validity = validity.xreplace({v: get_random_bv(v.width) for v in evs})
                    assert validity in zero_one_set
                if bool(validity):
                    return ct_prop
            else:
                return None

    @classmethod
    def _sample_var2ct(cls, ch_model, seed, external_props):
        # external_props needed for CipherCharacteristic
        PRNG = random.Random()
        PRNG.seed(seed)

        MAX_TRIES_PER_OM = min(2**(sum(p.val.width for p in itertools.chain(
            ch_model.input_prop, ch_model.assign_outprop2op_model
        ))), 2**16)

        def get_random_bv(width):
            return cls._sample_bv(PRNG, width)

        # print("# Computing get_random_characteristic")
        # print("ch_model:", ch_model)
        # print()

        index_try = -1
        while True:
            index_try += 1
            if index_try == MAX_TRIES_PER_OM:
                raise ValueError(f"no random characteristic found in {MAX_TRIES_PER_OM} tries for {ch_model}")

            var_prop2ct_prop = collections.OrderedDict()
            # input_prop = []
            for var_prop in ch_model.input_prop:
                ct_prop = type(var_prop)(get_random_bv(var_prop.val.width))
                # input_prop.append(ct_prop)
                var_prop2ct_prop[var_prop] = ct_prop

            # print("var_prop2ct_prop (+ input_prop):", var_prop2ct_prop)

            if external_props is not None:
                assert all(isinstance(p.val, core.Constant) for p in external_props)
                external_props = external_props[:]
            else:
                external_props = list(ch_model.external_var2prop.values())
            assert len(external_props) == len(ch_model.external_var2prop), \
                f"len({external_props}) == len({ch_model.external_var2prop})"
            for i, var in enumerate(ch_model.external_var2prop):
                model_prop = ch_model.external_var2prop[var]
                if isinstance(model_prop.val, core.Constant):
                    assert external_props[i] == model_prop
                elif not isinstance(external_props[i].val, core.Constant):
                    external_props[i] = ch_model.prop_type(get_random_bv(model_prop.val.width))
                var_prop2ct_prop[ch_model.prop_type(var)] = external_props[i]

            # print("var_prop2ct_prop (+ external_props):", var_prop2ct_prop)

            # assign_outprop_list = []
            found_invalid_opmodel = False
            for outprop, om in ch_model.assign_outprop2op_model.items():
                if isinstance(om, cascada_opmodel.ModelIdentity):
                    ct_prop = om.input_prop[0].xreplace(var_prop2ct_prop)
                    # assign_outprop_list.append(ct_prop)
                    var_prop2ct_prop[outprop] = ct_prop
                else:
                    ct_om = type(om)([d.xreplace(var_prop2ct_prop) for d in om.input_prop])
                    ct_prop = cls._sample_outprop_opmodel(
                        type(outprop), ct_om, outprop.val.width, get_random_bv)
                    if ct_prop is None:
                        found_invalid_opmodel = True
                        break
                    var_prop2ct_prop[outprop] = ct_prop
                if found_invalid_opmodel:
                    break
            if not found_invalid_opmodel:
                break

        # print("var_prop2ct_prop (+ assignment):", var_prop2ct_prop)

        var2ct = collections.OrderedDict(
            [(var_prop.val, ct_prop.val) for var_prop, ct_prop in var_prop2ct_prop.items()]
        )

        return var2ct

    @classmethod
    def random(cls, ch_model, seed, external_props=None):
        """Return a random `Characteristic` with given `abstractproperty.chmodel.ChModel`.

        Args:
            ch_model: the underlying characteristic model
            seed: the seed used to sample bit-vectors
            external_props: a list containing the (constant) `Property` of
                the external variables or ``None`` to sample random properties
                for the external variables.

        """
        assert isinstance(ch_model, cascada_chmodel.ChModel)

        var2ct = cls._sample_var2ct(ch_model, seed, external_props)

        init_props = Characteristic.get_properties_for_initialization(ch_model, var2ct)
        input_prop, output_prop, external_props, assign_outprop_list = init_props

        ch = cls._prop_new(
            input_prop=input_prop, output_prop=output_prop, assign_outprop_list=assign_outprop_list,
            ch_model=ch_model, external_props=external_props, free_props=None,
            empirical_ch_weight=None, empirical_data_list=None, is_valid=True)

        return ch

    def get_formatted_logged_msgs(self):
        """Return the list of logged messages.

        If ``self.ch_model.func`` includes `log_msg` calls in its ``eval``,
        this method return the list of messages logged with the format field
        objects applied. Otherwise, an empty list is returned.

        In the first case, the `Variable` objects of ``self.ch_model.ssa``
        appearing in the format field objects are replaced with their
        associated constant  `Property` objects.
        """
        if self.ch_model._logger is None:
            return []
        list_msgs = []
        replacements = {vp.val: cp.val for vp, cp in self.var_prop2ct_prop.items()}
        for format_string, format_field_objects in self.ch_model._logger:
            for i in range(len(format_field_objects)):
                expr = format_field_objects[i]
                if isinstance(expr, core.Term) and not isinstance(expr, core.Constant):
                    format_field_objects[i] = self.ch_model.prop_type(expr.xreplace(replacements))
            list_msgs.append(format_string.format(*format_field_objects))
        return list_msgs

    def get_round_separators(self):
        """Return the round separators if ``self.ch_model.func`` is a `RoundBasedFunction`.

        If ``self.ch_model.func`` includes `add_round_outputs` calls in its ``eval``,
        this method returns a list with the (symbolic) round `Property`
        outputs delimiting the rounds. Otherwise, ``None`` is returned.

        In the first case, this list contains ``num_rounds - 1`` entries,
        where the ``i``-th entry is the list of (symbolic) `Property`
        outputs of the ``i``-th round. In particular, the output properties
        of the last round are not included in this list.

        The list returned by this method is meant to be used as the argument
        of `Characteristic.split` to get the `Characteristic` object of each round.
        """
        if getattr(self.ch_model, "_rounds_outputs", None) is None:
            return None
        if len(self.ch_model._rounds_outputs) == 0:
            return None
        return tuple(tuple(self.ch_model.prop_type(v) for v in lv)
                     for lv in self.ch_model._rounds_outputs[:-1])

    # dotprinting last method
    def dotprinting(self, repeat=True, vrepr_label=False, **kwargs):
        """Return the DOT description of the expression tree of `tuple_assign_outprop2op_model`.

        See also `printing.dotprinting`.

        Args:
            repeat: whether to use different nodes for common subexpressions
                (default True)
            vrepr_label: whether to use the verbose representation (`Term.vrepr`)
                to label the nodes (default False)
            kwargs: additional arguments passed to `printing.dotprinting`

        """
        # similar as ChModel.dotprinting
        from cascada.bitvector.printing import dotprinting
        from sympy.core.containers import Tuple
        from sympy.core.basic import Basic
        # sympy_property = type(self.prop_type.__name__, (Basic,), {})
        op_model_class2sympy_op_model = {}
        expr = []
        for out_prop, op_model in self.tuple_assign_outprop2op_model:
            if type(op_model) not in op_model_class2sympy_op_model:
                sympy_op_model = type(type(op_model).__name__, (Basic,), {})
                op_model_class2sympy_op_model[type(op_model)] = sympy_op_model
            else:
                sympy_op_model = op_model_class2sympy_op_model[type(op_model)]
            expr.append(Tuple(
                # sympy_property(out_prop.val),
                out_prop.val,
                # sympy_op_model(Tuple(*[sympy_property(d.val) for d in op_model.input_prop]))
                sympy_op_model(*[d.val for d in op_model.input_prop])
            ))
        expr = Tuple(*expr)
        return dotprinting(expr, repeat=repeat, vrepr_label=vrepr_label, **kwargs)


class EncryptionCharacteristic(object):
    """Represent characteristics over encryption functions w.r.t some property.

    Given a `Cipher`, an `EncryptionCharacteristic` is a characteristic
    (see `Characteristic`) for a particular `Property` over
    the `Cipher.encryption` (where the `Cipher.key_schedule`
    is ignored).

    As opposed to `Characteristic`, an `EncryptionCharacteristic` is defined
    for a `abstractproperty.chmodel.EncryptionChModel`.

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic over encryption functions,
    (see `differential.characteristic.EncryptionCharacteristic`,
    `linear.characteristic.EncryptionCharacteristic` or
    `algebraic.characteristic.EncryptionCharacteristic` for some examples).

    .. Implementation details:

        This class does not subclass `Characteristic`, but subclasses
        of this class defined for a particular `Property`
        must subclass the corresponding ``Characteristic``.

        Some subclasses might need to provide ``external_props``.

    """

    def __init__(self, input_prop, output_prop, assign_outprop_list, ch_model, external_props=None,
                 free_props=None, empirical_ch_weight=None, empirical_data_list=None, is_valid=True):
        assert isinstance(ch_model, cascada_chmodel.EncryptionChModel)
        # ch_model instead of enc_ch_model as self has ch_model attribute
        # avoid external_props=external_props and free_props=free_props (super might not abstract)
        super().__init__(
            input_prop, output_prop, assign_outprop_list, ch_model, external_props, free_props,
            empirical_ch_weight=empirical_ch_weight, empirical_data_list=empirical_data_list, is_valid=is_valid)

    def split(self, prop_separators):
        """Split into multiple `Characteristic` objects given the list of property separators.

        Given an `EncryptionCharacteristic`, this method calls `Characteristic.split` to
        split the characteristic.

        .. note::

            The new split characteristics are instances of `Characteristic`
            and not of `EncryptionCharacteristic`.

        """
        return super().split(prop_separators)

    @classmethod
    def random(cls, ch_model, seed):
        """Return a random `EncryptionCharacteristic` with given `abstractproperty.chmodel.EncryptionChModel`.

        Args:
            ch_model: the underlying characteristic model of the encryption function
            seed: the seed used to sample bit-vectors

        """
        assert isinstance(ch_model, cascada_chmodel.EncryptionChModel)

        var2ct = cls._sample_var2ct(ch_model, seed, None)

        init_props = Characteristic.get_properties_for_initialization(ch_model, var2ct)
        input_prop, output_prop, external_props, assign_outprop_list = init_props

        ch = cls(
            input_prop, output_prop, assign_outprop_list, ch_model, external_props,
            empirical_ch_weight=None, empirical_data_list=None, is_valid=True)

        return ch


class CipherCharacteristic(object):
    """Represent characteristics over ciphers w.r.t some property.

    A `CipherCharacteristic` is a pair of `Characteristic` objects
    where one covers the `Cipher.key_schedule`
    and the other one covers the `Cipher.encryption`.

    .. note::

        The characteristic over the `Cipher.encryption` is an instance of
        `Characteristic` and not an instance of `EncryptionCharacteristic`.

    The round key properties in the encryption characteristic
    (the properties of the external variables of the encryption `SSA`)
    are set to the output properties of the key-schedule characteristic.

    This class is not meant to be instantiated but to provide a base
    class for creating characteristic over ciphers,
    (see `differential.characteristic.CipherCharacteristic` or
    `algebraic.characteristic.CipherCharacteristic` for some examples).

    Attributes:
        cipher_ch_model: the underlying `abstractproperty.chmodel.CipherChModel`
        ks_characteristic: the `Characteristic` over the key schedule
        enc_characteristic: the `Characteristic` over the encryption function

    """
    _Characteristic_cls = None

    @staticmethod
    def get_properties_for_initialization(cipher_ch_model, var2ct, ks_free_props=None, enc_free_props=None):
        """Return the properties needed to initialize a `CipherCharacteristic` object.

        Given a `abstractproperty.chmodel.CipherChModel`
        and a dictionary mapping `Variable` objects
        (representing the symbolic properties) to their `Constant` values,
        this method returns the following objects:
        ``ks_input_prop``, ``ks_output_prop``, ``ks_assign_outprop_list``,
        ``enc_input_prop``, ``enc_output_prop`` and ``enc_assign_outprop_list``
        for the ``CipherCharacteristic.__init__`` method.

        Symbolic properties not affecting the characteristics can be
        given in ``ks_free_props`` and ``enc_free_props``,
        and they will be set to 0.
        """
        # var2ct and *_free compatibility are checked in Characteristic.get_properties_for_initialization
        var2ct = var2ct.copy()

        ks_init_props = Characteristic.get_properties_for_initialization(
            cipher_ch_model.ks_ch_model, var2ct, free_props=ks_free_props)
        ks_input_prop, ks_output_prop, \
        ks_external_props, ks_assign_outprop_list = ks_init_props

        assert len(ks_external_props) == 0

        if enc_free_props is None:
            enc_free_props = []

        for v, p in cipher_ch_model.enc_ch_model.external_var2prop.items():
            index_v = cipher_ch_model.ks_ch_model.ssa.output_vars.index(v)
            assert v == cipher_ch_model.ks_ch_model.output_prop[index_v].val
            if isinstance(p.val, core.Constant):
                assert p.val == ks_output_prop[index_v].val == var2ct.get(v, p.val)
            else:
                if v not in var2ct and cipher_ch_model.prop_type(v) not in enc_free_props:
                    var2ct[v] = ks_output_prop[index_v].val

        enc_init_props = Characteristic.get_properties_for_initialization(
            cipher_ch_model.enc_ch_model, var2ct, free_props=enc_free_props)
        enc_input_prop, enc_output_prop, \
        enc_external_props, enc_assign_outprop_list = enc_init_props

        return ks_input_prop, ks_output_prop, ks_assign_outprop_list, \
               enc_input_prop, enc_output_prop, enc_assign_outprop_list

    def __init__(
            self, ks_input_prop, ks_output_prop, ks_assign_outprop_list,
            enc_input_prop, enc_output_prop, enc_assign_outprop_list,
            cipher_ch_model, ks_free_props=None, enc_free_props=None,
            ks_empirical_ch_weight=None, ks_empirical_data_list=None,
            enc_empirical_ch_weight=None, enc_empirical_data_list=None,
            ks_is_valid=True, enc_is_valid=True,
    ):
        assert isinstance(cipher_ch_model, cascada_chmodel.CipherChModel)

        self.cipher_ch_model = cipher_ch_model

        # ks_output_prop ordered as round_keys but enc_external_props ordered as ssa.external_vars
        enc_external_props = []
        for v in cipher_ch_model.enc_ch_model.ssa.external_vars:
            index_v = cipher_ch_model.ks_ch_model.ssa.output_vars.index(v)
            assert v == cipher_ch_model.ks_ch_model.output_prop[index_v].val
            enc_external_props.append(ks_output_prop[index_v])

        myCharacteristic = self.__class__._Characteristic_cls

        # _prop_new to use free_props
        self.ks_characteristic = myCharacteristic._prop_new(
            ks_input_prop, ks_output_prop, ks_assign_outprop_list,
            ch_model=cipher_ch_model.ks_ch_model, external_props=None, free_props=ks_free_props,
            empirical_ch_weight=ks_empirical_ch_weight, empirical_data_list=ks_empirical_data_list, is_valid=ks_is_valid)
        self.enc_characteristic = myCharacteristic._prop_new(
            enc_input_prop, enc_output_prop, enc_assign_outprop_list,
            ch_model=cipher_ch_model.enc_ch_model, external_props=enc_external_props, free_props=enc_free_props,
            empirical_ch_weight=enc_empirical_ch_weight, empirical_data_list=enc_empirical_data_list, is_valid=enc_is_valid)

        assert id(self.ks_characteristic.ch_model) == id(cipher_ch_model.ks_ch_model)
        assert id(self.enc_characteristic.ch_model) == id(cipher_ch_model.enc_ch_model)

        aux_dict = self.ks_characteristic.var_prop2ct_prop.keys() & \
                   self.enc_characteristic.var_prop2ct_prop.keys()
        for v in aux_dict:
            assert v in self.cipher_ch_model.ks_ch_model.output_prop
            assert self.ks_characteristic.var_prop2ct_prop[v] == \
                   self.enc_characteristic.var_prop2ct_prop[v]

    def __str__(self):
        return "{}(ks_characteristic={}, enc_characteristic={})".format(
            type(self).__name__,
            self.ks_characteristic.__str__(),
            self.enc_characteristic.__str__(),
        )

    __repr__ = __str__

    def vrepr(self):
        """Return an executable string representation.

        This method returns a string so that ``eval(self.vrepr())``
        returns a new `CipherCharacteristic` object with the same content.
        """
        ks_ch = self.ks_characteristic
        enc_ch = self.enc_characteristic

        # using v.val.vrepr() instead of v.vrepr() since __init__ allows Constant
        # or Difference(Constant) and the former is easier

        if ks_ch.free_props:
            kfd_vrepr = f"[{', '.join([v.val.vrepr() for v in ks_ch.free_props])}]"
        if enc_ch.free_props:
            efd_vrepr = f"[{', '.join([v.val.vrepr() for v in enc_ch.free_props])}]"

        if ks_ch.empirical_data_list is not None:
            kedl_vrepr = f"[{', '.join([e.vrepr() for e in ks_ch.empirical_data_list])}]"
        if enc_ch.empirical_data_list is not None:
            eedl_vrepr = f"[{', '.join([e.vrepr() for e in enc_ch.empirical_data_list])}]"

        prop = self.ks_characteristic.ch_model._prop_label
        assert prop == self.enc_characteristic.ch_model._prop_label

        format_str = f"{{}}(ks_input_{prop}={{}}, ks_output_{prop}={{}}, ks_assign_out{prop}_list={{}}, " \
                     f"enc_input_{prop}={{}}, enc_output_{prop}={{}}, enc_assign_out{prop}_list={{}}, " \
                     "cipher_ch_model={}{}{}{}{})"
        return format_str.format(
            type(self).__name__,
            f"[{', '.join([v.val.vrepr() for v in ks_ch.input_prop])}]",
            f"[{', '.join([v.val.vrepr() for v in ks_ch.output_prop])}]",
            f"[{', '.join([v.val.vrepr() for v, _ in ks_ch.tuple_assign_outprop2op_model])}]",
            f"[{', '.join([v.val.vrepr() for v in enc_ch.input_prop])}]",
            f"[{', '.join([v.val.vrepr() for v in enc_ch.output_prop])}]",
            f"[{', '.join([v.val.vrepr() for v, _ in enc_ch.tuple_assign_outprop2op_model])}]",
            self.cipher_ch_model.vrepr(),
            "" if not ks_ch.free_props else f", ks_free_{prop}s={kfd_vrepr}",
            "" if not enc_ch.free_props else f", enc_free_{prop}s={efd_vrepr}",
            "" if ks_ch.empirical_ch_weight is None else f", ks_empirical_ch_weight={ks_ch.empirical_ch_weight}",
            "" if ks_ch.empirical_data_list is None else f", ks_empirical_data_list={kedl_vrepr}",
            "" if enc_ch.empirical_ch_weight is None else f", enc_empirical_ch_weight={enc_ch.empirical_ch_weight}",
            "" if enc_ch.empirical_data_list is None else f", enc_empirical_data_list={eedl_vrepr}",
            "" if ks_ch._is_valid else f", is_valid={ks_ch._is_valid}"
            "" if enc_ch._is_valid else f", is_valid={enc_ch._is_valid}"
        )

    def srepr(self):
        """Return a short string representation of the cipher characteristic.

        See also `Characteristic.srepr`.
        """
        return "Ch(ks_ch={}, enc_ch={})".format(
            self.ks_characteristic.srepr(),
            self.enc_characteristic.srepr(),
        )

    def signature(self, ch_signature_type):
        """Return the signature of the cipher characteristic.

        See also `Characteristic.signature`.
        """
        return self.ks_characteristic.signature(ch_signature_type) + \
               self.enc_characteristic.signature(ch_signature_type)

    # def compute_empirical_ch_weight(self, num_input_samples=None, num_external_samples=None,
    #                                 split_by_max_weight=None, split_by_rounds=False,
    #                                 seed=None, C_code=False, num_parallel_processes=None):
    #     """Compute and store the empirical weight.
    #
    #     Compute the empirical weight of the key schedule characteristic
    #     and the empirical weight of the encryption characteristic
    #     with two independent calls to `Characteristic.compute_empirical_ch_weight`.
    #
    #     The empirical weights and auxiliary data are stored in
    #     ``.empirical_ch_weight`` and ``.empirical_data_list`` of
    #     `ks_characteristic` and `enc_characteristic`.
    #
    #     See also `Characteristic.compute_empirical_ch_weight`.
    #     """
    #     self.ks_characteristic.compute_empirical_ch_weight(
    #         num_input_samples=num_input_samples, num_external_samples=num_external_samples,
    #         split_by_max_weight=split_by_max_weight, split_by_rounds=split_by_rounds,
    #         C_code=C_code, num_parallel_processes=num_parallel_processes, seed=seed)
    #     self.enc_characteristic.compute_empirical_ch_weight(
    #         num_input_samples=num_input_samples, num_external_samples=num_external_samples,
    #         split_by_max_weight=split_by_max_weight, split_by_rounds=split_by_rounds,
    #         C_code=C_code, num_parallel_processes=num_parallel_processes, seed=seed)

    @classmethod
    def random(cls, cipher_ch_model, seed):
        """Return a random `CipherCharacteristic` with given `abstractproperty.chmodel.CipherChModel`.

        Args:
            cipher_ch_model: the underlying cipher characteristic model
            seed: the seed used to sample bit-vectors

        """
        assert isinstance(cipher_ch_model, cascada_chmodel.CipherChModel)
        myCharacteristic = cls._Characteristic_cls

        ks_var2ct = myCharacteristic._sample_var2ct(
            cipher_ch_model.ks_ch_model, seed, None)

        new_seed = repr(seed) + repr(ks_var2ct)

        # ks_output_prop ordered as round_keys but enc_external_props ordered as ssa.external_vars
        pt = cipher_ch_model.enc_ch_model.prop_type
        enc_external_props_vars = []
        enc_external_props_cts = []
        for v in cipher_ch_model.enc_ch_model.ssa.external_vars:
            enc_external_props_vars.append(v)
            enc_external_props_cts.append(pt(ks_var2ct[v]))

        enc_var2ct = myCharacteristic._sample_var2ct(
            cipher_ch_model.enc_ch_model, new_seed, enc_external_props_cts)

        assert all(v in enc_external_props_vars for v in ks_var2ct.keys() & enc_var2ct.keys()), \
            f"{ks_var2ct.keys() & enc_var2ct.keys()}\n{ks_var2ct}\n{enc_var2ct}"

        init_props = CipherCharacteristic.get_properties_for_initialization(
            cipher_ch_model, {**ks_var2ct, **enc_var2ct})
        assert len(init_props) == 6
        ks_input_prop, ks_output_prop, ks_assign_outprop_list = init_props[:3]
        enc_input_prop, enc_output_prop, enc_assign_outprop_list = init_props[-3:]

        # avoid *_props=*_props (super might not abstract)
        cipher_ch = cls(
            ks_input_prop,
            ks_output_prop,
            ks_assign_outprop_list,
            enc_input_prop,
            enc_output_prop,
            enc_assign_outprop_list,
            cipher_ch_model,
            ks_empirical_ch_weight=None,
            ks_empirical_data_list=None,
            enc_empirical_ch_weight=None,
            enc_empirical_data_list=None,
            ks_is_valid=True,
            enc_is_valid=True,
        )
        return cipher_ch

    def get_formatted_logged_msgs(self):
        """Return the list of logged messages.

        See also `Characteristic.get_formatted_logged_msgs`.
        """
        return self.ks_characteristic.get_formatted_logged_msgs() + \
               self.enc_characteristic.get_formatted_logged_msgs()

