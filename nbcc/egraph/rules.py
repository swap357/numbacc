# mypy: disable-error-code="empty-body"
from __future__ import annotations

import warnings

import egglog
import sealir.eqsat.py_eqsat as py
import sealir.eqsat.rvsdg_eqsat as rvsdg
import sealir.rvsdg.grammar as rg
from egglog import union
from sealir.ase import SExpr

from ..frontend import grammar as sg
from nbcc.developer import TODO

Term = rvsdg.Term
TermList = rvsdg.TermList
_w = rvsdg.wildcard

i64 = egglog.i64


def egraph_optimize(egraph: egglog.EGraph):
    rule_schedule = make_schedule()
    egraph.run(rule_schedule)


def make_schedule(enable_constant_folding: bool = False) -> egglog.Schedule:
    """
    Build the optimization schedule for e-graph rewriting.

    The schedule runs in three phases:
    1. Lowering: Transform Python ops to typed ops (Py_Call → Op_i32_*)
    2. Algebraic: Apply algebraic rewrites to create alternatives
    3. Constant Folding: Evaluate operations on constant values

    Args:
        enable_constant_folding: If True, include constant folding rules
    """
    # Phase 1: Lower Python operations to typed operations
    lowering = (
        ruleset_simplify_builtin_arith
        | ruleset_simplify_builtin_print
        | ruleset_typing
        | ruleset_call_fqn
    )

    # Phase 2: Algebraic rewrites that create optimization alternatives
    # These run AFTER lowering so we have typed ops to work with
    algebraic = (
        ruleset_commutativity
        | ruleset_associativity
        | ruleset_identity
        | ruleset_strength_reduction
        | ruleset_distributivity
        | ruleset_comparison_flip
    )

    # Base schedule: lowering then algebraic
    schedule = lowering.saturate() + algebraic.saturate()

    # Phase 3: Constant folding (optional)
    # This evaluates operations on literal values at compile time
    if enable_constant_folding:
        from nbcc.experiments.constant_folding import (
            ruleset_constant_fold,
            ruleset_constant_fold_division,
            ruleset_constant_fold_unary,
        )

        constant_folding = (
            ruleset_constant_fold
            | ruleset_constant_fold_unary
            | ruleset_constant_fold_division
        )

        # Run constant folding after algebraic rewrites have created
        # opportunities (e.g., associativity exposing constant subexpressions)
        schedule = schedule + constant_folding.saturate()

    return schedule


def egraph_convert_metadata(mdlist: list[SExpr], memo) -> egglog.Vec[Metadata]:
    def gen(md):
        from sealir.eqsat.rvsdg_convert import WrapTerm

        match md:
            case rg.DbgValue(name, value, srcloc, interloc):
                TODO("skip DbgValue")
            case sg.TypeInfo(value=value, type_expr=typexpr):
                if value in memo:
                    anchor = memo[value]
                    if isinstance(anchor, WrapTerm):
                        anchor = anchor.term
                    return Metadata.typeinfo(anchor, type_expr=gen(typexpr))
            case sg.TypeExpr(str(name), tuple(args)):
                match name:
                    case ".function":
                        return TypeExpr.function(list(map(gen, args)))
                    case _ if not args:
                        if not args:
                            return TypeExpr.simple(name)

                raise NotImplementedError(md)
            case sg.IRTagData(key=str(key), value=str(value)):
                return IRTagData(key=key, value=value)
            case sg.IRTag(value=value, tag=str(tag), data=tuple(data)):
                if data:
                    data_node = egglog.Vec[IRTagData](*map(gen, data))
                else:
                    data_node = egglog.Vec[IRTagData].empty()
                anchor = memo[value]
                if isinstance(anchor, WrapTerm):
                    anchor = anchor.term
                return Metadata.irtag(value=anchor, tag=tag, data=data_node)
            case _:
                raise NotImplementedError(md)

    return egglog.Vec[Metadata](
        *filter(lambda x: x is not None, map(gen, mdlist))
    )


class TypeExpr(egglog.Expr):
    @classmethod
    def simple(cls, name: egglog.StringLike) -> TypeExpr: ...

    # @classmethod
    # def compound(cls, name: egglog.StringLike, *args: TypeExpr) -> TypeExpr:
    #     ...

    @classmethod
    def function(cls, args: egglog.Vec[TypeExpr]) -> TypeExpr: ...


class IRTagData(egglog.Expr):
    def __init__(self, key: egglog.StringLike, value: egglog.StringLike): ...


class Metadata(egglog.Expr):
    @classmethod
    def typeinfo(cls, value: Term, type_expr: TypeExpr) -> Metadata: ...

    @classmethod
    def irtag(
        cls, value: Term, tag: egglog.StringLike, data: egglog.Vec[IRTagData]
    ) -> Metadata: ...


@egglog.function
def Md_type_info(value: Term, typename: egglog.StringLike) -> Term: ...


# -----------------------------------------------------------------------------
# Arithmetic Operations
# -----------------------------------------------------------------------------

@egglog.function
def Op_i32_add(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_sub(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_mul(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_div(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_mod(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_neg(operand: Term) -> Term: ...


@egglog.function
def Op_i32_shl(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_shr(lhs: Term, rhs: Term) -> Term: ...


# -----------------------------------------------------------------------------
# Comparison Operations
# -----------------------------------------------------------------------------

@egglog.function
def Op_i32_lt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_le(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_gt(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_ge(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_eq(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_ne(lhs: Term, rhs: Term) -> Term: ...


@egglog.function
def Op_i32_not(operand: Term) -> Term: ...


# -----------------------------------------------------------------------------
# Literal Constants (for algebraic rules)
# -----------------------------------------------------------------------------

@egglog.function
def Literal_i32(value: egglog.i64) -> Term: ...


@egglog.function
def Builtin_print_i32(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_print_str(io: Term, arg: Term) -> Term: ...


@egglog.function
def Builtin_struct__make__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__lift__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__unlift__(args: TermList) -> Term: ...


@egglog.function
def Builtin_struct__get_field__(struct: Term, pos: egglog.i64) -> Term: ...


class FQN(egglog.Expr):
    @classmethod
    def function(cls, fullname: egglog.StringLike) -> Term: ...


@egglog.function
def CallFQN(fqn: Term, io: Term, args: TermList) -> Term: ...


@egglog.function
def Load_FQN(fqn) -> Term: ...


@egglog.ruleset
def ruleset_simplify_builtin_arith(
    io: Term,
    operand: Term,
    lhs: Term,
    rhs: Term,
    argvec: egglog.Vec[Term],
    call: Term,
):
    # Binary arithmetic and comparison operations
    BINOPS = {
        # Arithmetic
        "operator::i32_add": Op_i32_add,
        "operator::i32_sub": Op_i32_sub,
        "operator::i32_mul": Op_i32_mul,
        "operator::i32_div": Op_i32_div,
        "operator::i32_floordiv": Op_i32_div,  # Python // maps to div for i32
        "operator::i32_mod": Op_i32_mod,
        # Comparisons
        "operator::i32_gt": Op_i32_gt,
        "operator::i32_lt": Op_i32_lt,
        "operator::i32_ge": Op_i32_ge,
        "operator::i32_le": Op_i32_le,
        "operator::i32_eq": Op_i32_eq,
        "operator::i32_ne": Op_i32_ne,
        # Bitwise
        "operator::i32_lshift": Op_i32_shl,
        "operator::i32_rshift": Op_i32_shr,
    }
    for fname, ctor in BINOPS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == lhs,
            argvec[1] == rhs,
            egglog.eq(argvec.length()).to(i64(2)),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(ctor(lhs, rhs)),
        )

    # Handle PyUnaryOp not
    yield egglog.rule(
        call
        == py.Py_NotIO(
            io=io,
            term=operand,
        ),
    ).then(
        union(call.getPort(0)).with_(io),
        union(call.getPort(1)).with_(Op_i32_not(operand)),
    )


@egglog.ruleset
def ruleset_simplify_builtin_print(
    io: Term, printee: Term, argvec: egglog.Vec[Term], call: Term
):
    KNOWN_PRINTS = {
        "builtins::print_i32": Builtin_print_i32,
        "builtins::print_str": Builtin_print_str,
    }

    for fname, builtin_ctor in KNOWN_PRINTS.items():
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=rvsdg.TermList(argvec),
            ),
            argvec[0] == printee,
            egglog.eq(argvec.length()).to(i64(1)),
        ).then(
            union(call.getPort(0)).with_(builtin_ctor(io, printee)),
            union(call.getPort(0)).with_(call.getPort(1)),
        )


def create_ruleset_struct__make__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__make__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__make__(args)),
        )

    ruleset_struct__make__.__name__ += fname

    return egglog.ruleset(ruleset_struct__make__)


def create_ruleset_struct__lift__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__lift__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__lift__(args)),
        )

    ruleset_struct__lift__.__name__ += fname

    return egglog.ruleset(ruleset_struct__lift__)


def create_ruleset_struct__unlift__(w_obj):
    fname = w_obj.fqn.fullname

    def ruleset_struct__unlift__(io: Term, args: TermList, call: Term):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=args,
            ),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(Builtin_struct__unlift__(args)),
        )

    ruleset_struct__unlift__.__name__ += fname

    return egglog.ruleset(ruleset_struct__unlift__)


def create_ruleset_struct__get_field__(w_obj, field_pos: int):
    fname = w_obj.fqn.fullname

    def ruleset_struct__get_field__(
        io: Term,
        args: TermList,
        call: Term,
        argvec: egglog.Vec[Term],
        struct: Term,
    ):
        yield egglog.rule(
            call
            == py.Py_Call(
                io=io,
                func=py.Py_LoadGlobal(io=_w(Term), name=fname),
                args=TermList(argvec),
            ),
            argvec[0] == struct,
            egglog.eq(argvec.length()).to(i64(1)),
        ).then(
            union(call.getPort(0)).with_(io),
            union(call.getPort(1)).with_(
                Builtin_struct__get_field__(struct, i64(field_pos))
            ),
        )

    ruleset_struct__get_field__.__name__ += fname
    return egglog.ruleset(ruleset_struct__get_field__)


@egglog.ruleset
def ruleset_typing(x: Term):
    if False:
        yield


@egglog.ruleset
def ruleset_call_fqn(
    io: Term,
    args: TermList,
    call: Term,
    fqn: egglog.String,
    callee: Term,
    functype: TypeExpr,
):
    yield egglog.rule(
        call
        == py.Py_Call(
            io=io,
            func=callee,
            args=args,
        ),
        callee == py.Py_LoadGlobal(io=_w(Term), name=fqn),
        Metadata.typeinfo(callee, functype),
    ).then(
        union(call).with_(CallFQN(fqn=callee, io=io, args=args)),
        union(callee).with_(FQN.function(fqn)),
    )


# =============================================================================
# Algebraic Rewrite Rules
#
# These rules create meaningful alternatives for e-graph extraction.
# The cost model determines which alternative is chosen.
# =============================================================================


@egglog.ruleset
def ruleset_commutativity(a: Term, b: Term):
    """
    Commutativity rules: a + b ↔ b + a, a * b ↔ b * a

    These create equivalent representations - cost model picks based on
    operand costs (e.g., if 'a' is more expensive to compute than 'b').
    """
    # Addition is commutative
    yield egglog.rule(
        Op_i32_add(a, b),
    ).then(
        union(Op_i32_add(a, b)).with_(Op_i32_add(b, a)),
    )

    # Multiplication is commutative
    yield egglog.rule(
        Op_i32_mul(a, b),
    ).then(
        union(Op_i32_mul(a, b)).with_(Op_i32_mul(b, a)),
    )


@egglog.ruleset
def ruleset_associativity(a: Term, b: Term, c: Term):
    """
    Associativity rules: (a + b) + c ↔ a + (b + c)

    These can expose optimization opportunities:
    - If b and c are constants, (b + c) can be folded
    - Different register allocation patterns
    """
    # Addition associativity: (a + b) + c ↔ a + (b + c)
    yield egglog.rule(
        Op_i32_add(Op_i32_add(a, b), c),
    ).then(
        union(Op_i32_add(Op_i32_add(a, b), c)).with_(
            Op_i32_add(a, Op_i32_add(b, c))
        ),
    )

    # Multiplication associativity: (a * b) * c ↔ a * (b * c)
    yield egglog.rule(
        Op_i32_mul(Op_i32_mul(a, b), c),
    ).then(
        union(Op_i32_mul(Op_i32_mul(a, b), c)).with_(
            Op_i32_mul(a, Op_i32_mul(b, c))
        ),
    )


@egglog.ruleset
def ruleset_identity(a: Term):
    """
    Identity elimination: a + 0 → a, a * 1 → a, a * 0 → 0

    These are clear wins - removing unnecessary operations.
    """
    # Use the actual literal representation from sealir
    zero = Term.LiteralI64(i64(0))
    one = Term.LiteralI64(i64(1))

    # Additive identity: a + 0 = a
    yield egglog.rule(
        Op_i32_add(a, zero),
    ).then(
        union(Op_i32_add(a, zero)).with_(a),
    )

    yield egglog.rule(
        Op_i32_add(zero, a),
    ).then(
        union(Op_i32_add(zero, a)).with_(a),
    )

    # Multiplicative identity: a * 1 = a
    yield egglog.rule(
        Op_i32_mul(a, one),
    ).then(
        union(Op_i32_mul(a, one)).with_(a),
    )

    yield egglog.rule(
        Op_i32_mul(one, a),
    ).then(
        union(Op_i32_mul(one, a)).with_(a),
    )

    # Multiplication by zero: a * 0 = 0
    yield egglog.rule(
        Op_i32_mul(a, zero),
    ).then(
        union(Op_i32_mul(a, zero)).with_(zero),
    )

    yield egglog.rule(
        Op_i32_mul(zero, a),
    ).then(
        union(Op_i32_mul(zero, a)).with_(zero),
    )

    # Subtractive identity: a - 0 = a
    yield egglog.rule(
        Op_i32_sub(a, zero),
    ).then(
        union(Op_i32_sub(a, zero)).with_(a),
    )


@egglog.ruleset
def ruleset_strength_reduction(a: Term):
    """
    Strength reduction: replace expensive ops with cheaper equivalents.

    These create ALTERNATIVES - cost model decides which is cheaper:
    - a * 2 ↔ a + a ↔ a << 1
    - a * 4 ↔ a << 2
    - a / 2 ↔ a >> 1 (for unsigned, careful with signed!)

    Costs differ:
    - mul: ~3-4 cycles latency
    - add: ~1 cycle latency
    - shl: ~1 cycle latency
    """
    # Use the actual literal representation from sealir
    two = Term.LiteralI64(i64(2))
    four = Term.LiteralI64(i64(4))
    eight = Term.LiteralI64(i64(8))
    one = Term.LiteralI64(i64(1))
    three = Term.LiteralI64(i64(3))

    # a * 2 ↔ a + a
    yield egglog.rule(
        Op_i32_mul(a, two),
    ).then(
        union(Op_i32_mul(a, two)).with_(Op_i32_add(a, a)),
    )

    # a * 2 ↔ a << 1
    yield egglog.rule(
        Op_i32_mul(a, two),
    ).then(
        union(Op_i32_mul(a, two)).with_(Op_i32_shl(a, one)),
    )

    # a * 4 ↔ a << 2
    yield egglog.rule(
        Op_i32_mul(a, four),
    ).then(
        union(Op_i32_mul(a, four)).with_(Op_i32_shl(a, two)),
    )

    # a * 8 ↔ a << 3
    yield egglog.rule(
        Op_i32_mul(a, eight),
    ).then(
        union(Op_i32_mul(a, eight)).with_(Op_i32_shl(a, three)),
    )

    # a + a ↔ a << 1 (connect the other direction too)
    yield egglog.rule(
        Op_i32_add(a, a),
    ).then(
        union(Op_i32_add(a, a)).with_(Op_i32_shl(a, one)),
    )


@egglog.ruleset
def ruleset_distributivity(a: Term, b: Term, c: Term):
    """
    Distributivity: a * (b + c) ↔ (a * b) + (a * c)

    Factoring vs expanding - cost model decides:
    - Factored: 1 mul + 1 add
    - Expanded: 2 mul + 1 add

    Factoring is usually better, but not always (e.g., if a is complex).
    """
    # a * (b + c) → (a * b) + (a * c)
    yield egglog.rule(
        Op_i32_mul(a, Op_i32_add(b, c)),
    ).then(
        union(Op_i32_mul(a, Op_i32_add(b, c))).with_(
            Op_i32_add(Op_i32_mul(a, b), Op_i32_mul(a, c))
        ),
    )

    # (a * b) + (a * c) → a * (b + c)  [factoring]
    yield egglog.rule(
        Op_i32_add(Op_i32_mul(a, b), Op_i32_mul(a, c)),
    ).then(
        union(Op_i32_add(Op_i32_mul(a, b), Op_i32_mul(a, c))).with_(
            Op_i32_mul(a, Op_i32_add(b, c))
        ),
    )


@egglog.ruleset
def ruleset_comparison_flip(a: Term, b: Term):
    """
    Comparison flipping: a < b ↔ b > a

    Creates equivalent comparisons - may enable other optimizations.
    """
    # a < b ↔ b > a
    yield egglog.rule(
        Op_i32_lt(a, b),
    ).then(
        union(Op_i32_lt(a, b)).with_(Op_i32_gt(b, a)),
    )

    # a <= b ↔ b >= a
    yield egglog.rule(
        Op_i32_le(a, b),
    ).then(
        union(Op_i32_le(a, b)).with_(Op_i32_ge(b, a)),
    )

    # a > b ↔ b < a
    yield egglog.rule(
        Op_i32_gt(a, b),
    ).then(
        union(Op_i32_gt(a, b)).with_(Op_i32_lt(b, a)),
    )

    # a >= b ↔ b <= a
    yield egglog.rule(
        Op_i32_ge(a, b),
    ).then(
        union(Op_i32_ge(a, b)).with_(Op_i32_le(b, a)),
    )
