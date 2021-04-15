# Copyright (c) 2019-2021, Jonas Eschle, Jim Pivarski, Eduardo Rodrigues, and Henry Schreiner.
#
# Distributed under the 3-clause BSD license, see accompanying file LICENSE
# or https://github.com/scikit-hep/vector for details.

"""
Ensures that new or modified vector.compute.* functions don't break any existing
or future backends by using unsupportable Python language features.

Compute functions are highly restricted, a least common denominator for all the
backends we *ever* want to support. The functions themselves are duck-typed:
arguments could be numbers, NumPy arrays, Awkward Arrays, and potentially
TensorFlow/Torch/JAX/etc. An ``if`` statement on individual numbers would have
to be ``np.where` or a masked assignment in NumPy, so ``if`` is not allowed.
JAX traces a function for JIT-compilation and autodifferentiation by passing a
"tracer" object through it, and that object can only follow one code path, another
reason to exclude ``if`` statements. Loops are even more problematic.

This suite of tests statically analyzes all of the compute functions by decompiling
their bytecode with uncompyle6 (on Python 3.8; will have to be modified slightly
every few years). Some compute functions are dynamically generated, so they don't
all have an AST to inspect.

The sieve has been defined narrowly: compute functions can use more functions,
binary operators, and possibly more language features than are allowed here.
Expanding this set of rules is therefore allowed and encouraged. The test failure
and requirement to expand the rules is intended to force you to think about
new features, to ask yourself if they can be supported by all current and hoped-for
backends, and whether a (formally) simpler implementation is possible.
"""

import collections
import inspect
import sys

import pytest

import vector._compute.lorentz
import vector._compute.planar
import vector._compute.spatial

uncompyle6 = pytest.importorskip("uncompyle6")
spark_parser = pytest.importorskip("spark_parser")
pytestmark = pytest.mark.dis


Context = collections.namedtuple("Context", ["name", "closure"])


functions = dict(
    [
        (
            f'{y.__name__}({", ".join(repr(v) if isinstance(v, str) else v.__name__ for v in w)})',
            z[0],
        )
        for x, y in inspect.getmembers(
            vector._compute.planar, predicate=inspect.ismodule
        )
        if hasattr(y, "dispatch_map")
        for w, z in y.dispatch_map.items()
    ]
    + [
        (
            f'{y.__name__}({", ".join(repr(v) if isinstance(v, str) else v.__name__ for v in w)})',
            z[0],
        )
        for x, y in inspect.getmembers(
            vector._compute.spatial, predicate=inspect.ismodule
        )
        if hasattr(y, "dispatch_map")
        for w, z in y.dispatch_map.items()
    ]
    + [
        (
            f'{y.__name__}({", ".join(repr(v) if isinstance(v, str) else v.__name__ for v in w)})',
            z[0],
        )
        for x, y in inspect.getmembers(
            vector._compute.lorentz, predicate=inspect.ismodule
        )
        if hasattr(y, "dispatch_map")
        for w, z in y.dispatch_map.items()
    ]
)


@pytest.mark.slow
@pytest.mark.parametrize("signature", functions.keys())
def test(signature):
    analyze_function(functions[signature])


# def test():
#     for signature, function in functions.items():
#         print(signature)
#         analyze_function(function)


def analyze_function(function):
    if function not in analyze_function.done:
        # print(function.__module__ + "." + function.__name__)

        closure = dict(function.__globals__)
        if function.__closure__ is not None:
            for var, cell in zip(function.__code__.co_freevars, function.__closure__):
                try:
                    closure[var] = cell.cell_contents
                except ValueError:
                    pass  # the cell has not been filled yet, so ignore it

        analyze_code(function.__code__, Context(function.__name__, closure))
        analyze_function.done.add(function)


analyze_function.done = set()


def analyze_code(code, context):
    # this block is all uncompyle6
    python_version = float(sys.version[0:3])
    is_pypy = "__pypy__" in sys.builtin_module_names
    parser = uncompyle6.parser.get_python_parser(
        python_version,
        debug_parser=dict(spark_parser.DEFAULT_DEBUG),
        compile_mode="exec",
        is_pypy=is_pypy,
    )
    scanner = uncompyle6.scanner.get_scanner(python_version, is_pypy=is_pypy)
    tokens, customize = scanner.ingest(code, code_objects={}, show_asm=False)
    parsed = uncompyle6.parser.parse(parser, tokens, customize, code)

    # now the disassembled bytecodes have been parsed into a tree for us to walk
    analyze_body(parsed, context)


def analyze_body(node, context):
    assert node.kind == "stmts"
    assert len(node) >= 1

    for statement in node[:-1]:
        analyze_assignment(statement, context)
    analyze_return(node[-1], context)


def analyze_assignment(node, context):
    assert node.kind == "sstmt"
    assert len(node) == 1

    assert (
        node[0].kind == "assign"
    ), "only assignments and a final 'return' are allowed (and not tuple-assignment)"
    assert len(node[0]) == 2
    assert node[0][1].kind == "store"

    if node[0][1][0].kind == "STORE_FAST":
        analyze_expression(expr(node[0][0]), context)

    elif node[0][1][0].kind == "unpack":
        assert len(node[0][1][0]) >= 2
        assert node[0][1][0][0].kind.startswith("UNPACK_SEQUENCE")
        for item in node[0][1][0][1:]:
            assert item.kind == "store"
            assert len(item) == 1
            assert item[0].kind == "STORE_FAST"

    else:
        print(node[0][1][0])
        raise AssertionError("what is this?")


def expr(node):
    assert node.kind == "expr"
    assert len(node) == 1
    return node[0]


def is_pi(node):
    return (
        node.kind == "attribute"
        and len(node) == 2
        and expr(node[0]).kind == "LOAD_FAST"
        and expr(node[0]).attr == "lib"
        and node[1].kind == "LOAD_ATTR"
        and node[1].attr == "pi"
    )


def is_nan_to_num(node):
    if node.kind != "call_kw36" or len(node) < 3:
        return False

    function = expr(node[0])
    return (
        function.kind == "attribute"
        and expr(function[0]).attr == "lib"
        and function[1].attr == "nan_to_num"
    )


def analyze_return(node, context):
    assert node.kind == "sstmt"
    assert len(node) == 1

    assert node[0].kind == "return", "compute function must end with a 'return'"
    assert len(node[0]) == 2
    assert node[0][0].kind == "ret_expr"
    assert len(node[0][0]) == 1
    expr(node[0][0][0])
    assert node[0][1].kind == "RETURN_VALUE"

    if node[0][0][0][0].kind == "tuple":
        assert len(node[0][0][0][0]) >= 2, "returning an empty tuple?"
        assert node[0][0][0][0][-1].kind.startswith("BUILD_TUPLE")
        for item in node[0][0][0][0][:-1]:
            analyze_expression(expr(item), context)

    else:
        analyze_expression(node[0][0][0][0], context)


def analyze_expression(node, context):
    if node.kind == "LOAD_FAST":
        # Don't bother checking to see if this variable has been defined.
        # Unit checks test that if the coverage is complete.
        pass

    elif node.kind == "LOAD_CONST":
        assert isinstance(node.attr, (int, float))

    elif is_pi(node):
        pass

    elif node.kind == "unary_op":
        assert len(node) == 2
        analyze_expression(expr(node[0]), context)
        assert node[1].kind == "unary_operator"
        assert len(node[1]) == 1
        analyze_unary_operator(node[1][0], context)

    elif node.kind == "bin_op":
        assert len(node) == 3
        analyze_expression(expr(node[0]), context)
        analyze_expression(expr(node[1]), context)
        assert node[2].kind == "binary_operator"
        assert len(node[2]) == 1
        analyze_binary_operator(node[2][0], context)

    elif node.kind == "compare":
        assert len(node) == 1
        assert node[0].kind == "compare_single", "only do single comparisons"
        assert len(node[0]) == 3
        analyze_expression(expr(node[0][0]), context)
        analyze_expression(expr(node[0][1]), context)
        assert node[0][2].kind == "COMPARE_OP"
        assert (
            node[0][2].attr in allowed_comparisons
        ), f"add {repr(node[0][2].attr)} to allowed_comparisons"

    elif node.kind == "call":
        assert len(node) >= 2

        assert node[-1].kind.startswith("CALL_METHOD") or node[-1].kind.startswith(
            "CALL_FUNCTION"
        )
        analyze_callable(expr(node[0]), context)

        for argument in node[1:-1]:
            assert argument.kind == "pos_arg", "only positional arguments"
            analyze_expression(expr(argument[0]), context)

    elif is_nan_to_num(node):
        analyze_expression(expr(node[1]), context)

    else:
        print(node)
        raise AssertionError("what is this?")


def analyze_unary_operator(node, context):
    assert (
        node.kind in allowed_unary_operators
    ), f"add {repr(node.kind)} to allowed_unary_operators"


def analyze_binary_operator(node, context):
    assert (
        node.kind in allowed_binary_operators
    ), f"add {repr(node.kind)} to allowed_binary_operators"


def analyze_callable(node, context):
    if node.kind == "attribute37":
        assert len(node) == 2
        module = expr(node[0])
        assert module.kind in {"LOAD_FAST", "LOAD_GLOBAL"}
        assert node[1].kind == "LOAD_METHOD"

        if module.attr == "lib":
            assert (
                node[1].attr in allowed_lib_functions
            ), f"add {repr(node[1].attr)} to allowed_lib_functions"

        else:
            module_name = ".".join(
                context.closure.get(module.attr).__name__.split(".")[:-1]
            )
            assert module_name in (
                "vector._compute.planar",
                "vector._compute.spatial",
                "vector._compute.lorentz",
            )

    elif node.kind in {"LOAD_GLOBAL", "LOAD_DEREF"}:
        function = context.closure.get(node.attr)
        assert (
            function is not None
        ), f"unrecognized function in scope: {repr(node.attr)}"
        analyze_function(function)

    else:
        print(node)
        raise AssertionError("what is this?")


allowed_unary_operators = [
    "UNARY_NEGATIVE",
]

allowed_binary_operators = [
    "BINARY_ADD",
    "BINARY_SUBTRACT",
    "BINARY_MULTIPLY",
    "BINARY_TRUE_DIVIDE",
    "BINARY_MODULO",
    "BINARY_POWER",
    "BINARY_AND",
]

allowed_comparisons = [
    "==",
    "!=",
    "<",
    ">",
]

allowed_lib_functions = [
    "absolute",
    "sign",
    "copysign",
    "maximum",
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "sinh",
    "cosh",
    "arctanh",
    "isclose",
]
