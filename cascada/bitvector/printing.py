"""Manage the representation of bit-vector expressions."""
from sympy.printing import repr as sympy_repr
from sympy.printing import str as sympy_str


# noinspection PyPep8Naming,PyMethodMayBeStatic
class BvStrPrinter(sympy_str.StrPrinter):
    """Printing class that handles the `str` method of `Term`."""

    @staticmethod
    def _need_parentheses(bv, parent):
        """Return true if bv need parenthesis when used in infix notation."""
        from cascada.bitvector import core

        # assert isinstance(bv, (core.Term, int))

        if isinstance(bv, int):
            return False
        elif isinstance(bv, core.Term) and len(bv.args) in [0, 1]:
            return False
        elif type(bv) == type(parent):
            from cascada.bitvector import operation
            if type(parent) == operation.BvComp:
                # printing (a == b) == (c == d) instead of a == b == c == d
                return True
            else:
                # printing a + b + c + d instead of (a + b) + (c + d)
                return False
        else:
            return True

    def _print_Term(self, bv):
        assert False
        # args = [self._print(a) for a in bv.args]
        # return "{}({})".format(type(bv).__name__, ", ".join(args))

    def _print_Constant(self, bv):
        if bv.width % 4 == 0:
            return bv.hex()
        else:
            return bv.bin()

    def _print_Variable(self, bv):
        return bv.name

    def _print_Operation(self, bv):
        from cascada.bitvector.operation import PartialOperation, Extract
        if isinstance(bv, PartialOperation) and \
                (hasattr(bv.__class__.base_op, "infix_symbol") or bv.__class__.base_op == Extract):
            # infix_symbol not stored in the partial operation and
            # Extract has a custom method in BvStrPrinter
            return self._print(bv._get_base_op_expr())
        else:
            has_symbol = hasattr(bv, "unary_symbol") or hasattr(bv, "infix_symbol")
            op_name = getattr(bv, "alt_name", type(bv).__name__)

            args = []
            for a in bv.args:
                if has_symbol and BvStrPrinter._need_parentheses(a, bv):
                    args.append("({})".format(self._print(a)))
                else:
                    args.append(self._print(a))

            if has_symbol:
                assert sum(bv.arity) in [1, 2]
                if sum(bv.arity) == 1:
                    return "{}{}".format(bv.unary_symbol, args[0])
                elif sum(bv.arity) == 2:
                    return "{} {} {}".format(args[0], bv.infix_symbol, args[1])
            else:
                return "{}({})".format(op_name, ", ".join(args))

    def _print_Extract(self, bv):
        x, i, j = bv.args
        delimiter = ":"

        if i == j:
            delimiter = i = ""
        else:
            if j == 0:
                j = ""

            if i == x.width - 1:
                i = ""

        if BvStrPrinter._need_parentheses(x, bv):
            x = "({})".format(self._print(x))
        else:
            x = self._print(x)

        return "{}[{}{}{}]".format(x, i, delimiter, j)


class BvShortPrinter(BvStrPrinter):
    """Printing class that handles the `Term.srepr` method of `Term`."""

    lvl = 0

    max_lvl = 5

    def _print_Term(self, bv):
        assert False
        # args = [self._print(a) for a in bv.args]
        # return "{}({})".format(type(bv).__name__, ", ".join(args))

    def _print_Operation(self, bv):
        has_symbol = hasattr(bv, "unary_symbol") or hasattr(bv, "infix_symbol")
        op_name = getattr(bv, "alt_name", type(bv).__name__)

        from cascada.bitvector import secondaryop
        # if all(isinstance(a, (int, core.Constant, core.Variable))
        #        or type(a) == type(bv) for a in bv.args):
        if all(isinstance(a, (secondaryop.Reverse, secondaryop.PopCount, secondaryop.PopCountSum2,
                              secondaryop.PopCountSum3, secondaryop.PopCountDiff)) or
               type(a) == type(bv) for a in bv.args):
            next_lvl = False
        else:
            next_lvl = True
        if next_lvl:
            BvShortPrinter.lvl += 1

        args = []
        for a in bv.args:
            if BvShortPrinter.lvl > BvShortPrinter.max_lvl:
                args.append("...")
            elif has_symbol and BvStrPrinter._need_parentheses(a, bv):
                args.append("({})".format(self._print(a)))
            else:
                args.append(self._print(a))

        if next_lvl:
            BvShortPrinter.lvl -= 1

        if has_symbol:
            assert sum(bv.arity) in [1, 2]
            if sum(bv.arity) == 1:
                return "{}{}".format(bv.unary_symbol, args[0])
            elif sum(bv.arity) == 2:
                return "{} {} {}".format(args[0], bv.infix_symbol, args[1])
        else:
            return "{}({})".format(op_name, ", ".join(args))


# noinspection PyPep8Naming,PyMethodMayBeStatic
class BvReprPrinter(sympy_repr.ReprPrinter):
    """Printing class that handles the `Term.vrepr` method."""

    def _print_Term(self, bv):
        assert False
        # if bv.args:
        #     args = ', '.join([self._print(a) for a in bv.args])
        #     return "{}({}, width={})".format(type(bv).__name__, args, bv.width)
        # else:
        #     return "{}(width={})".format(type(bv).__name__, bv.width)

    def _print_Constant(self, bv):
        return "{}({}, width={})".format(
            type(bv).__name__,
            # prevent a 128-bit vector to be fully printed
            bv.bin() if bv.width <= 8 or bv.width % 4 != 0 else bv.hex(),
            bv.width)

    def _print_Variable(self, bv):
        name = self._print(bv.name)
        return "{}({}, width={})".format(type(bv).__name__, name, bv.width)

    def _print_Operation(self, bv):
        from cascada.bitvector.operation import PartialOperation, make_partial_operation
        if isinstance(bv, PartialOperation):
            # nfa_v similar as args below
            nfa_v = ', '.join([self._print(a) for a in bv.args])

            # fa_v needs to be a tuple so that it is hashable for lru_cache
            # bv.fixed_args has always length 2+
            fa_v = ', '.join([self._print(a) for a in bv.fixed_args])
            fa_v = f"({fa_v})"

            foo_name = make_partial_operation.__name__

            return f"{foo_name}({bv.__class__.base_op.__name__}, {fa_v})({nfa_v})"
        else:
            args = ', '.join([self._print(a) for a in bv.args])
            return "{}({})".format(type(bv).__name__, args)


class BvWrapPrinter(BvStrPrinter):
    """Printing class similar to `BvStrPrinter` but prints bit-vector expressions in multiple compact lines."""

    len_prefix = 0
    max_line_width = 100

    use_symbols = False
    new_line_right_parenthesis = True

    def _print_Operation(self, bv):
        standard_repr = super()._print_Operation(bv)
        if len(standard_repr) + self.__class__.len_prefix < self.__class__.max_line_width:
            return standard_repr

        if self.use_symbols and hasattr(bv, "unary_symbol"):
            op_name = bv.unary_symbol
        elif self.use_symbols and hasattr(bv, "infix_symbol"):
            op_name = bv.infix_symbol
        else:
            op_name = getattr(bv, "alt_name", type(bv).__name__)

        old_len_prefix = self.__class__.len_prefix
        self.__class__.len_prefix += len(op_name) + 1

        args = [self._print(bv.args[0])]
        for a in bv.args[1:]:
            args.append("\n" + " "*self.__class__.len_prefix + self._print(a))

        # end = "\n" + " "*old_len_prefix + ")"

        self.__class__.len_prefix = old_len_prefix

        if self.new_line_right_parenthesis:
            return "{}({}{}".format(op_name, ",".join(args),
                                    "\n" + " "*(self.__class__.len_prefix) + ")")
        else:
            return "{}({})".format(op_name, ",".join(args))


class BvCCodePrinter(sympy_str.StrPrinter):
    """Printing class that handles the `Term.crepr` method.

    .. Implementation details:

        Since the C code of the secondary operations is automatically
        obtained from the C code of the primary operations
        (and the set of primary ones is fixed), the C code of the
        primary operations is implemented here and not in the
        class of the operations.

        See also https://en.cppreference.com/w/c/types/integer ,
        https://en.cppreference.com/w/c/language/integer_constant and
        https://en.cppreference.com/w/c/language/arithmetic_types .

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import Concat
        >>> for w in [1, 63, 64, 65, 128]: print(Constant(1, w).crepr())
        0x1U
        0x1ULL
        0x1ULL
        ((__uint128)(0x1ULL)) & (((__uint128)(0x1ULL) << 64) | ((__uint128)(0xffffffffffffffffULL)))
        (__uint128)(0x1ULL)
        >>> for w in [7, 8]: print((~Variable("a", w)).crepr())
        (~a) & 0x7fU
        (~a) & 0xffU
        >>> for w in [7, 8]: print((Variable("a", w) & Variable("b", w)).crepr())
        a & b
        a & b
        >>> for w in [7, 8]: print((Variable("a", w) * Variable("b", w)).crepr())
        (a * b) & 0x7fU
        (a * b) & 0xffU
        >>> for w in [7, 8]: print((Variable("a", w) < Variable("b", w)).crepr())
        (_Bool)((a < b) & 0x1U)
        (_Bool)((a < b) & 0x1U)
        >>> for w in [7, 8]: print(Concat(Variable("a", w), Variable("b", w)).crepr())
        (((uint16_t)(a) << 0x7U) | ((uint16_t)(b))) & 0x3fffU
        (((uint16_t)(a) << 0x8U) | ((uint16_t)(b))) & 0xffffU
        >>> print(Variable('a', 8)[2:1].crepr(), ",", Variable('a', 8)[0:0].crepr())
        (a >> 0x1U) & 0x3U , a & 0x1U

    """

    @classmethod
    def _width2C_type(cls, width):
        """Return the integer C type associated to the given bit-vector width."""
        if width == 1:
            return "_Bool"
        elif width <= 8:
            return "uint8_t"
        elif width <= 16:
            return "uint16_t"
        elif width <= 32:
            return "uint32_t"
        elif width <= 64:
            return "uint64_t"
        elif width <= 128:
            return "__uint128"
        else:
            raise ValueError("bit-vectors with more than 128-bits cannot be converted to C code")

    @classmethod
    def _get_integer_suffix(cls, width):
        """Return the integer C suffix associated to the given bit-vector width."""
        if width <= 8:
            return "U"
        elif width <= 16:
            return "U"
        elif width <= 32:
            return "UL"
        elif width <= 64:
            return "ULL"
        else:
            return "U"

    @classmethod
    def _get_big_mask_C_code(cls, width):
        """Return the mask for bit-vectors with more than 64 bits."""
        from cascada.bitvector.core import Constant
        assert 64 < width <= 128
        mask = (2 ** width) - 1
        iu_64 = cls._get_integer_suffix(width=64)
        mask_64_MS = hex(int((Constant(int(Constant(mask, 128) >> 64), 64))))
        mask_64_LS = hex(int((Constant(mask & ((2 ** 64) - 1), 64))))
        return f"((__uint128)({mask_64_MS}{iu_64}) << 64) | ((__uint128)({mask_64_LS}{iu_64}))"

    @classmethod
    def _get_and_mask_C_code(cls, width):
        """Return the ``& mask`` operation."""
        # widths [1, 8, 16, 32, 64, 128] also need a mask
        if width <= 64:
            iu = cls._get_integer_suffix(width)
            return f" & {hex((2 ** width) - 1)}{iu}"
        elif width <= 128:
            return f" & ({cls._get_big_mask_C_code(width)})"
        else:
            raise ValueError("bit-vectors with more than 128-bits cannot be converted to C code")

    def _print_int(self, expr):
        # _get_integer_suffix cannot be used since original width is not known
        return f"{hex(expr)}U"

    def _print_Term(self, bv):
        assert False

    def _print_Constant(self, bv):
        from cascada.bitvector.core import Constant
        if bv.width <= 64:
            iu = self.__class__._get_integer_suffix(bv.width)
            c_code = f"{hex(int(bv))}{iu}"
            return c_code  # mask not needed
        elif bv.width <= 128:
            iu_64 = self.__class__._get_integer_suffix(width=64)
            bv_64_MS_Cc = hex(int(Constant(int(Constant(int(bv), 128) >> 64), 64)))
            bv_64_LS_Cc = hex(int(Constant(int(bv) & ((2 ** 64) - 1), 64)))
            if int(bv) >> 64 == 0:  # bv_64_MS_Cc
                c_code = f"(__uint128)({bv_64_LS_Cc}{iu_64})"
            else:
                c_code = f"((__uint128)({bv_64_MS_Cc}{iu_64}) << 64) | ((__uint128)({bv_64_LS_Cc}{iu_64}))"
            if bv.width == 128:
                return c_code
            else:
                return f"({c_code}) & ({self.__class__._get_big_mask_C_code(bv.width)})"
        else:
            raise ValueError("bit-vectors with more than 128-bits cannot be converted to C code")

    def _print_Variable(self, bv):
        return str(bv)

    def _print_Operation(self, bv, short_ccode=False):
        from cascada.bitvector import operation, secondaryop
        if isinstance(bv, operation.PartialOperation):
            return self._print(bv._get_base_op_expr())
        elif isinstance(bv, operation.SecondaryOperation):
            if isinstance(bv, secondaryop.LutOperation):
                # no need and_out_mask
                arg = self._print(bv.args[0])
                lut_ccode = ','.join([self._print(x_i) for x_i in bv.lut])
                return f"{{{lut_ccode}}}[{arg}]"  # e,g, {0x0U,0x1U,0x2U,0x3U}[a]
            else:
                return self._print(bv.doit(eval_sec_ops=True))
        else:
            args = []
            for index_a, a in enumerate(bv.args):
                if isinstance(bv, operation.Ite) and index_a == 0 and \
                        isinstance(a, (operation.BvUlt, operation.BvUle, operation.BvUgt, operation.BvUge,
                                       operation.BvComp)):  # for LutOperation
                    a_ccode = self._print_Operation(a, short_ccode=True)
                else:
                    a_ccode = self._print(a)
                a_ccode = f"({a_ccode})" if " " in a_ccode else a_ccode
                args.append(a_ccode)

            and_out_mask = self.__class__._get_and_mask_C_code(bv.width)

            # - For each op, non-used bits are ensured to be zero
            # - Only (_Bool) results are cast, rest are done when assigning to variables.
            # - Unsigned integer arithmetic is always performed modulo 2n
            # https://en.cppreference.com/w/c/language/operator_arithmetic

            lb = "(" if and_out_mask != "" else ""
            rb = ")" if and_out_mask != "" else ""

            # unary
            if isinstance(bv, (operation.BvNot, operation.BvNeg)):
                return f"{lb}{bv.unary_symbol}{args[0]}{rb}{and_out_mask}"
            elif isinstance(bv, operation.BvIdentity):
                return args[0]

            # binary
            elif isinstance(bv, (operation.BvAnd, operation.BvOr, operation.BvXor, operation.BvLshr)):
                return f"{args[0]} {bv.infix_symbol} {args[1]}"
            elif isinstance(bv, (operation.BvShl, operation.BvAdd, operation.BvSub, operation.BvMul,
                                 operation.BvUdiv, operation.BvUrem)):
                return f"{lb}{args[0]} {bv.infix_symbol} {args[1]}{rb}{and_out_mask}"
            elif isinstance(bv, (operation.BvUlt, operation.BvUle, operation.BvUgt, operation.BvUge,
                                 operation.BvComp)):
                if short_ccode:
                    # ignoring (_Bool) and and_out_mask
                    return f"{args[0]} {bv.infix_symbol} {args[1]}"
                else:
                    return f"(_Bool)({lb}{args[0]} {bv.infix_symbol} {args[1]}{rb}{and_out_mask})"
            #
            elif isinstance(bv, operation.RotateLeft):
                x, r = args[0], args[1]
                offset = self._print(bv.args[0].width - bv.args[1])
                return f"{lb}({x} << {r}) | ({x} >> {offset}){rb}{and_out_mask}"
            elif isinstance(bv, operation.RotateRight):
                x, r = args[0], args[1]
                offset = self._print(bv.args[0].width - bv.args[1])
                return f"{lb}({x} >> {r}) | ({x} << {offset}){rb}{and_out_mask}"

            # ternary
            elif isinstance(bv, operation.Ite):
                # https://en.cppreference.com/w/c/language/operator_other
                return f"{args[0]} ? {args[1]} : {args[2]}"  # and_out_mask not needed
                # return f"{lb}{args[0]} ? {args[1]} : {args[2]}{rb}{and_out_mask}"

            elif isinstance(bv, operation.Concat):
                x, y = args
                y_width = bv.args[1].width
                output_C_type = self.__class__._width2C_type(bv.width)
                return f"{lb}(({output_C_type})({x}) << {self._print(y_width)}) | (({output_C_type})({y})){rb}{and_out_mask}"

            elif isinstance(bv, operation.Extract):
                x, i, j = args  # j == 0 means take LSB, i == j means take 1 bit
                extract_mask = self._print(2**(bv.args[1] - bv.args[2] + 1) - 1)
                if bv.args[2] == 0:  # (j is a string)
                    return f"{x} & {extract_mask}"  # no need and_out_mask
                else:
                    return f"({x} >> {j}) & {extract_mask}"  # no need and_out_mask

            else:
                raise NotImplementedError(f"C code of operation {type(bv)} has not been implemented")


def dotprinting(bv, repeat=True, vrepr_label=False, **kwargs):
    """Return the DOT description of a bit-vector `Term` expression tree.

    For more information see
    `SymPy's dotprinting method <https://docs.sympy.org/latest/modules/printing.html#dotprint>`_.

    Args:
        bv: a bit-vector `Term`
        repeat: whether to use different nodes for common subexpressions
            (default True)
        vrepr_label: wheter to use the verbose representation (`Term.vrepr`)
            to label the nodes (default False)
        kwargs: additional arguments passed to SymPy's dotprinting method

    ::

        >>> from cascada.bitvector.core import Constant, Variable
        >>> from cascada.bitvector.operation import make_partial_operation, RotateLeft, Extract
        >>> LSB = make_partial_operation(Extract, (None, 0, 0))
        >>> expr = LSB(RotateLeft((Constant(1, 8) + Variable("x", 8)), 1))
        >>> print(dotprinting(expr))  # doctest:+NORMALIZE_WHITESPACE
        digraph{
        # Graph style
        "ordering"="out"
        "rankdir"="TD"
        #########
        # Nodes #
        #########
        "Extract_{·, 0, 0}(RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1))_()" ["color"="black", "label"="Extract_{·, 0, 0}", "shape"="ellipse"];
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)_(0,)" ["color"="black", "label"="RotateLeft", "shape"="ellipse"];
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))_(0, 0)" ["color"="black", "label"="BvAdd", "shape"="ellipse"];
        "Constant(0b00000001, width=8)_(0, 0, 0)" ["color"="blue1", "label"="0x01", "shape"="box"];
        "Variable('x', width=8)_(0, 0, 1)" ["color"="aquamarine3", "label"="x", "shape"="box"];
        "1_(0, 1)" ["color"="blue4", "label"="1", "shape"="box"];
        #########
        # Edges #
        #########
        "Extract_{·, 0, 0}(RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1))_()" -> "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)_(0,)";
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)_(0,)" -> "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))_(0, 0)";
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)_(0,)" -> "1_(0, 1)";
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))_(0, 0)" -> "Constant(0b00000001, width=8)_(0, 0, 0)";
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))_(0, 0)" -> "Variable('x', width=8)_(0, 0, 1)";
        }
        >>> print(dotprinting(expr, repeat=False, vrepr_label=True))  # doctest:+NORMALIZE_WHITESPACE
        digraph{
        # Graph style
        "ordering"="out"
        "rankdir"="TD"
        #########
        # Nodes #
        #########
        "Extract_{·, 0, 0}(RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1))" ["color"="black", "label"="Extract_{·, 0, 0}", "shape"="ellipse"];
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)" ["color"="black", "label"="RotateLeft", "shape"="ellipse"];
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))" ["color"="black", "label"="BvAdd", "shape"="ellipse"];
        "Constant(0b00000001, width=8)" ["color"="blue1", "label"="Constant(0b00000001, width=8)", "shape"="box"];
        "Variable('x', width=8)" ["color"="aquamarine3", "label"="Variable('x', width=8)", "shape"="box"];
        "1" ["color"="blue4", "label"="int(1)", "shape"="box"];
        #########
        # Edges #
        #########
        "Extract_{·, 0, 0}(RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1))" -> "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)";
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)" -> "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))";
        "RotateLeft(BvAdd(Constant(0b00000001, width=8), Variable('x', width=8)), 1)" -> "1";
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))" -> "Constant(0b00000001, width=8)";
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8))" -> "Variable('x', width=8)";
        }

    Note:
         This method requires `graphviz <https://www.graphviz.org/>`_
         and its `python interface <https://pypi.org/project/graphviz/>`_
         to be installed.

    .. Useful links

        * http://matthiaseisen.com/articles/graphviz/
        * https://graphviz.readthedocs.io/

    """
    from cascada.bitvector.core import Term
    from cascada.bitvector.dot import dotprint
    if not vrepr_label:
        labelfunc = str
    else:
        def labelfunc(x):
            if isinstance(x, Term):
                return x.vrepr()
            else:
                return f"{x.__class__.__name__}({x})"
    return dotprint(bv, repeat=repeat, labelfunc=labelfunc)
