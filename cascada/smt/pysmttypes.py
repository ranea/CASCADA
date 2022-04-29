"""Convert between pySMT_ types and `bitvector` types.

.. _pySMT: https://github.com/pysmt/pysmt
"""

from pysmt import environment

from cascada.bitvector import core
from cascada.bitvector import operation

from cascada.abstractproperty import property


def _is_power_of_2(x):
    # http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    return (x & (x - 1)) == 0


def bv2pysmt(bv, boolean=False, parse_shifts_rotations=False, env=None):
    """Convert a bit-vector type to a pySMT type.

    The SMT solver Boolector_ only supports shifts and rotations
    where the width of the first bit-vector operand is a power of two.
    If ``parse_shifts_rotations=True``, shifts and rotations are preprocessed:
    shifts by extending, shifting and then shrinking back and rotations by
    extracting and concatenating bits.

    .. _Boolector: https://boolector.github.io/docs/boolector.html?highlight=power#pyboolector.Boolector.Rol

    Args:
        bv: the bit-vector `Term` to convert
        boolean: if True, boolean pySMT types (e.g., `pysmt.shortcuts.Bool`) are used instead of
            bit-vector pySMT types (e.g., `pysmt.shortcuts.BV`).
        parse_shifts_rotations: if `True`, shifts and rotation are preprocessed as above.
        env: a `pysmt.environment.Environment`; if not specified, a new pySMT environment is created.

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.secondaryop import BvMaj
        >>> from cascada.smt.pysmttypes import bv2pysmt
        >>> s = bv2pysmt(Constant(0b00000001, 8), boolean=False)
        >>> s, s.get_type()
        (1_8, BV{8})
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> s = bv2pysmt(x)
        >>> s, s.get_type()
        (x, BV{8})
        >>> s = bv2pysmt(x +  y)
        >>> s, s.get_type()
        ((x + y), BV{8})
        >>> s = bv2pysmt(x <=  y)
        >>> s, s.get_type()
        ((x u<= y), Bool)
        >>> s = bv2pysmt(x[4: 2])
        >>> s, s.get_type()
        (x[2:4], BV{3})
        >>> s = bv2pysmt(BvMaj(x, y, x + y))
        >>> s, s.get_type()
        ((((x & y) | (x & (x + y))) | (y & (x + y))), BV{8})

    """
    msg = "unknown conversion of {} to a pySMT type".format(type(bv).__name__)

    if not hasattr(bv2pysmt, "_counter"):
        bv2pysmt._counter = -1
    bv2pysmt._counter += 1

    ## debugging
    # print(f"{'  '*bv2pysmt._counter}bv2pysmt({bv}, boolean={boolean}, parse_shifts_rotations={parse_shifts_rotations})")

    if env is None:
        env = environment.reset_env()
    fm = env.formula_manager

    # preprocessing bv

    if isinstance(bv, int):
        bv2pysmt._counter -= 1
        return bv
    if isinstance(bv, property.Property):
        bv = bv.val

    while True:
        # _get_base_op_expr/doit/BvIdentity might return a Variable or Constant
        if isinstance(bv, operation.PartialOperation):
            bv = bv._get_base_op_expr()
        elif isinstance(bv, operation.SecondaryOperation):
            bv = bv.doit(eval_sec_ops=True)
        elif isinstance(bv, operation.BvIdentity):
            bv = bv.args[0]
        else:
            break

    assert isinstance(bv, core.Term)

    pysmt_bv = None

    if isinstance(bv, core.Variable):
        if boolean:
            assert bv.width == 1
            pysmt_bv = fm.Symbol(bv.name, env.type_manager.BOOL())
        else:
            pysmt_bv = fm.Symbol(bv.name, env.type_manager.BVType(bv.width))

    elif isinstance(bv, core.Constant):
        if boolean:
            assert bv.width == 1
            pysmt_bv = fm.Bool(bool(bv))
        else:
            pysmt_bv = fm.BV(bv.val, bv.width)

    elif isinstance(bv, operation.Operation):
        if boolean:
            assert bv.width == 1

        if type(bv) in [operation.BvNot, operation.BvAnd, operation.BvOr, operation.BvXor]:
            # -- Operations that requires boolean arguments to output a boolean---
            args = [bv2pysmt(a, boolean, parse_shifts_rotations, env) for a in bv.args]

            if type(bv) == operation.BvNot:
                if boolean:
                    pysmt_bv = fm.Not(*args)
                else:
                    pysmt_bv = fm.BVNot(*args)
            elif type(bv) == operation.BvAnd:
                if boolean:
                    pysmt_bv = fm.And(*args)
                else:
                    pysmt_bv = fm.BVAnd(*args)
            elif type(bv) == operation.BvOr:
                if boolean:
                    pysmt_bv = fm.Or(*args)
                else:
                    pysmt_bv = fm.BVOr(*args)
            else:
                assert type(bv) == operation.BvXor
                if boolean:
                    pysmt_bv = fm.Xor(*args)
                else:
                    pysmt_bv = fm.BVXor(*args)

        elif type(bv) == operation.Ite:
            # fm.Ite can either output a boolean or a BV, but
            # fm.Ite always requires a Boolean type for args[0] and
            # bv2pysmt(bv.args[0], boolean=True, ...)  might cause an error
            args = [None for _ in range(len(bv.args))]
            try:
                args[0] = bv2pysmt(bv.args[0], True, parse_shifts_rotations, env)
            except Exception as e:
                raise e
                # args[0] = bv2pysmt(bv.args[0], False, parse_shifts_rotations, env)
                # if args[0].get_type().is_bv_type():
                #    args[0] = fm.Equals(args[0], fm.BV(1, 1))
            args[1:] = [bv2pysmt(a, boolean, parse_shifts_rotations, env) for a in bv.args[1:]]
            pysmt_bv = fm.Ite(*args)

        else:
            # -- Operations that don't require boolean arguments to output a boolean ---

            args = [bv2pysmt(a, False, parse_shifts_rotations, env) for a in bv.args]

            if isinstance(bv, operation.BvComp):  # for PropConcat
                if boolean:
                    pysmt_bv = fm.EqualsOrIff(*args)
                else:
                    pysmt_bv = fm.BVComp(*args)

            elif type(bv) == operation.BvUlt:
                pysmt_bv = fm.BVULT(*args)
            elif type(bv) == operation.BvUle:
                pysmt_bv = fm.BVULE(*args)
            elif type(bv) == operation.BvUgt:
                pysmt_bv = fm.BVUGT(*args)
            elif type(bv) == operation.BvUge:
                pysmt_bv = fm.BVUGE(*args)

            else:
                # -- Operations that don't support boolean arguments or boolean outputs ---

                if type(bv) in [operation.BvShl, operation.BvLshr]:
                    if not parse_shifts_rotations or _is_power_of_2(args[0].bv_width()):
                        if type(bv) == operation.BvShl:
                            pysmt_bv = fm.BVLShl(*args)
                        elif type(bv) == operation.BvLshr:
                            pysmt_bv = fm.BVLShr(*args)
                    else:
                        x, r = bv.args
                        offset = 0
                        while not _is_power_of_2(x.width):
                            x = operation.zero_extend(x, 1)
                            r = operation.zero_extend(r, 1)
                            offset += 1
                        shift = bv2pysmt(type(bv)(x, r), False, parse_shifts_rotations, env)
                        pysmt_bv = fm.BVExtract(shift, end=shift.bv_width() - offset - 1)
                elif type(bv) == operation.RotateLeft:
                    if not parse_shifts_rotations or _is_power_of_2(args[0].bv_width()):
                        pysmt_bv = fm.BVRol(*args)
                    else:
                        x, r = bv.args
                        n = x.width
                        rol = operation.Concat(x[n - r - 1:], x[n - 1: n - r])
                        pysmt_bv = bv2pysmt(rol, False, parse_shifts_rotations, env)
                elif type(bv) == operation.RotateRight:
                    if not parse_shifts_rotations or _is_power_of_2(args[0].bv_width()):
                        pysmt_bv = fm.BVRor(*args)
                    else:
                        x, r = bv.args
                        n = x.width
                        rot = operation.Concat(x[r - 1:], x[n - 1: r])
                        pysmt_bv = bv2pysmt(rot, False, parse_shifts_rotations, env)

                elif isinstance(bv, operation.Extract):  # for PropExtract
                    # pySMT Extract(bv, start, end)
                    pysmt_bv = fm.BVExtract(args[0], args[2], args[1])
                elif type(bv) == operation.Concat:
                    pysmt_bv = fm.BVConcat(*args)

                elif type(bv) == operation.BvNeg:
                    pysmt_bv = fm.BVNeg(*args)
                elif type(bv) == operation.BvAdd:
                    pysmt_bv = fm.BVAdd(*args)
                elif type(bv) == operation.BvSub:
                    pysmt_bv = fm.BVSub(*args)
                elif type(bv) == operation.BvMul:
                    pysmt_bv = fm.BVMul(*args)
                elif type(bv) == operation.BvUdiv:
                    pysmt_bv = fm.BVUDiv(*args)
                elif type(bv) == operation.BvUrem:
                    pysmt_bv = fm.BVURem(*args)

                if pysmt_bv is not None:
                    if boolean:
                        pysmt_bv = fm.EqualsOrIff(pysmt_bv, fm.BV(1, 1))
                else:
                    raise ValueError(f"invalid primary operation {bv.vrepr()}")

    if pysmt_bv is not None:
        try:
            pysmt_bv_width = pysmt_bv.bv_width()
        except (AssertionError, TypeError):
            pysmt_bv_width = 1  # boolean type

        assert bv.width == pysmt_bv_width
        bv2pysmt._counter -= 1
        return pysmt_bv
    else:
        raise NotImplementedError(msg)


def pysmt2bv(ps):
    """Convert a pySMT type to a bit-vector type.

    Currently, only conversion from `pysmt.shortcuts.BV`
    and `pysmt.shortcuts.Symbol` is supported.

        >>> from pysmt import shortcuts, typing  # pySMT shortcuts and typing modules
        >>> from cascada.smt.pysmttypes import pysmt2bv
        >>> env = shortcuts.reset_env()
        >>> pysmt2bv(env.formula_manager.Symbol("x", env.type_manager.BVType(8))).vrepr()
        "Variable('x', width=8)"
        >>> pysmt2bv(env.formula_manager.BV(1, 8)).vrepr()
        'Constant(0b00000001, width=8)'

    """
    class_name = type(ps).__name__
    msg = "unknown conversion of {} ({} {}) to a bit-vector type".format(ps, ps.get_type(), class_name)

    if ps.is_symbol():
        if str(ps.get_type()) == "Bool":
            return core.Variable(ps.symbol_name(), 1)
        else:
            return core.Variable(ps.symbol_name(), ps.bv_width())
    elif ps.is_bv_constant():
        return core.Constant(int(ps.constant_value()), ps.bv_width())
    elif ps.is_false():
        return core.Constant(0, 1)
    elif ps.is_true():
        return core.Constant(1, 1)
    else:
        raise NotImplementedError(msg)


def pysmt_model2bv_model(model, properties=None):
    """Convert a `pysmt.solvers.solver.Model` into a `dict` of bit-vector types.

    To return `Property` values instead of `Term` values,
    a list of symbolic properties can be passed as argument.
    In that case, variables in the model also present in ``properties``
    will be added to the bit-vector model as `Property` objects.

        >>> from pysmt import shortcuts
        >>> from cascada.bitvector.core import Variable
        >>> from cascada.differential.difference import XorDiff
        >>> from cascada.smt.pysmttypes import pysmt_model2bv_model  # pySMT shortcuts
        >>> env = shortcuts.reset_env()
        >>> fm = env.formula_manager
        >>> formula = fm.Equals(fm.BV(0, 8), fm.Symbol("x", env.type_manager.BVType(8)))
        >>> pysmt_model = env.factory.get_model(formula)
        >>> for var, val in pysmt_model: print(var, val, var.get_type(), val.get_type())
        x 0_8 BV{8} BV{8}
        >>> bv_model = pysmt_model2bv_model(pysmt_model)
        >>> for var, val in bv_model.items(): print(var.vrepr(), ":", val.vrepr())
        Variable('x', width=8) : Constant(0b00000000, width=8)
        >>> bv_model = pysmt_model2bv_model(pysmt_model, properties=[XorDiff(Variable("x", 8))])
        >>> for var, val in bv_model.items(): print(var.vrepr(), ":", val.vrepr())
        XorDiff(Variable('x', width=8)) : XorDiff(Constant(0b00000000, width=8))

    """
    if properties is not None:
        name2prop = {}
        for prop in properties:
            name2prop[prop.val.name] = prop
    else:
        name2prop = {}

    bv_model = {}
    for var, value in model:
        bv_var = pysmt2bv(var)
        bv_value = pysmt2bv(value)
        if isinstance(bv_var, core.Variable):
            if bv_var.name in name2prop:
                bv_var = name2prop[bv_var.name]
                bv_value = type(bv_var)(bv_value)
        bv_model[bv_var] = bv_value
    return bv_model
