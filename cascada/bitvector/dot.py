"""Adaptation of sympy.printing.dot to support `Term` objects.

.. autosummary::
   :nosignatures:

    purestr
    dotnode
    dotedges
    dotprint
"""
from cascada.bitvector.core import Constant, Variable
from cascada.bitvector.operation import Operation
from sympy.printing.dot import (
    Basic, srepr, styleof, attrprint, _graphstyle, template
)

default_styles = (
    (Basic, {'color': 'grey', 'shape': 'ellipse'}),
    (int, {'color': 'blue4', 'shape': 'box'}),
    (Constant, {'color': 'blue1', 'shape': 'box'}),
    (Variable, {'color': 'aquamarine3', 'shape': 'box'}),
    (Operation, {'color': 'black', 'shape': 'ellipse'}),
    # (Expr,  {'color': 'black'})
)


def purestr(x, with_args=False):
    """A string that follows ```obj = type(obj)(*obj.args)``` exactly.

    See also ``SymPy.purestr``.
    """
    sargs = ()
    if not isinstance(x, Basic):
        rv = str(x)
    elif isinstance(x, (Constant, Variable)):
        rv = x.vrepr()  # label must include width and val
    elif not x.args:
        rv = srepr(x)
    else:
        args = x.args
        sargs = tuple(map(purestr, args))
        rv = "%s(%s)"%(type(x).__name__, ', '.join(sargs))
    if with_args:
        rv = rv, sargs
    return rv


def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """String defining a node.

    Copy of ``SymPy.dotnode``.
    """
    style = styleof(expr, styles)

    if isinstance(expr, Basic) and not expr.is_Atom:
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label
    expr_str = purestr(expr)
    if repeat:
        expr_str += '_%s' % str(pos)
    return '"%s" [%s];' % (expr_str, attrprint(style))


def dotedges(expr, atom=lambda x: not isinstance(x, Basic), pos=(), repeat=True):
    """List of strings for all expr->expr.arg pairs

    Copy of ``SymPy.dotedges``.
    """
    if atom(expr):
        return []
    else:
        expr_str, arg_strs = purestr(expr, with_args=True)
        if repeat:
            expr_str += '_%s' % str(pos)
            arg_strs = ['%s_%s' % (a, str(pos + (i,)))
                for i, a in enumerate(arg_strs)]
        return ['"%s" -> "%s";' % (expr_str, a) for a in arg_strs]


def dotprint(expr,
    styles=default_styles, atom=lambda x: not isinstance(x, Basic),
    maxdepth=None, repeat=True, labelfunc=str, **kwargs):
    """DOT description of a SymPy expression tree

    Copy of ``SymPy.dotprint``.
    """
    graphstyle = _graphstyle.copy()
    graphstyle.update(kwargs)

    nodes = []
    edges = []
    def traverse(e, depth, pos=()):
        new_node = dotnode(e, styles, labelfunc=labelfunc, pos=pos, repeat=repeat)
        if new_node not in nodes:
            nodes.append(new_node)
        if maxdepth and depth >= maxdepth:
            return
        for new_edge in dotedges(e, atom=atom, pos=pos, repeat=repeat):
            if new_edge not in edges:
                edges.append(new_edge)
        if not isinstance(e, int):
            for i, arg in enumerate(e.args):
                if not atom(arg) or isinstance(arg, int):
                    traverse(arg, depth + 1, pos + (i,))
    traverse(expr, 0)

    return template%{'graphstyle': attrprint(graphstyle, delimiter='\n'),
                     'nodes': '\n'.join(nodes),
                     'edges': '\n'.join(edges)}
