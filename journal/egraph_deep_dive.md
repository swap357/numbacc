# E-graph Deep Dive: How numbacc Uses Equality Saturation

**Date:** 2026-01-18

---

## What is an E-graph?

An **e-graph** (equivalence graph) is a data structure that efficiently represents many equivalent programs simultaneously. Unlike traditional compilers that destructively rewrite code, e-graphs **add** equivalences while preserving all versions.

```
Traditional compiler:    E-graph compiler:
   A → B → C                A ≡ B ≡ C
   (lose A, B)              (keep all, pick best)
```

## Why numbacc Uses E-graphs

numbacc compiles typed Python (SPy) to native code. The challenge:

1. SPy uses generic Python operations: `Py_Call(operator::i32_add, a, b)`
2. We want specific, efficient operations: `Op_i32_add(a, b)`
3. Many such simplifications exist, and they interact

E-graphs let us:
- Apply ALL simplifications without worrying about order
- Keep all alternatives in one structure
- Pick the cheapest final program

---

## The Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  .spy source                                                     │
│    def main():                                                   │
│        a = 1; b = 2                                              │
│        if a < b: ...                                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   SPy VM (redshift)   │
                    │   Parse & type check  │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  RVSDG S-expression                                              │
│    Func("main",                                                  │
│      Region(                                                     │
│        PyCall(PyLoadGlobal("operator::i32_lt"), io, [a, b]),    │
│        IfElse(...)))                                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  egraph_conversion()  │
                    │  S-expr → E-graph     │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  E-graph (initial)                                               │
│    [Py_Call] ─── [PyLoadGlobal("i32_lt")] ─── [a] ─── [b]       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   egraph_optimize()   │
                    │   Apply rewrite rules │
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  E-graph (saturated)                                             │
│    [Py_Call ≡ Op_i32_lt] ─── [a] ─── [b]                        │
│     └─ Both in same e-class (equivalent)                         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  egraph_extraction()  │
                    │  Pick cheapest version│
                    └───────────┬───────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  Optimized RVSDG                                                 │
│    Func("main",                                                  │
│      Region(                                                     │
│        Op_i32_lt(a, b),    ← Replaced Py_Call                   │
│        IfElse(...)))                                             │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   MLIR Backend        │
                    │   → LLVM → Native     │
                    └───────────────────────┘
```

---

## Key Components

### 1. RVSDG (Regionalized Value State Dependence Graph)

The intermediate representation. Key features:
- **Regions**: Scoped blocks (functions, if-branches, loops)
- **Value dependencies**: Explicit data flow
- **State dependencies**: IO threading for side effects

```
Region[in: io] {
    $a = PyInt(1)
    $b = PyInt(2)
    $cond = Py_Call(Py_LoadGlobal("operator::i32_lt"), io, [$a, $b])
    IfElse($cond, then_region, else_region)
} -> [out: io, result]
```

### 2. E-graph Terms (sealir/eqsat/rvsdg_eqsat.py)

RVSDG nodes become e-graph terms:

```python
class Term(Expr):
    # Structural
    Func(uid, fname, body) -> Term
    Region(uid, inports) -> Region
    IfElse(cond, then, else, operands) -> Term

    # Python operations (with IO threading)
    Py_Call(func, io, args) -> Term
    Py_LoadGlobal(io, name) -> Term
    Py_AddIO(io, a, b) -> Term

    # Literals
    LiteralI64(val) -> Term
    LiteralStr(val) -> Term
```

### 3. Rewrite Rules (nbcc/egraph/rules.py)

Rules that add equivalences:

```python
@egglog.ruleset
def ruleset_simplify_builtin_arith():
    # Pattern: Py_Call to operator::i32_add
    # Adds equivalent: Op_i32_add

    yield rewrite(
        Py_Call(Py_LoadGlobal(io, "operator::i32_add"), io2, args)
    ).to(
        Op_i32_add(args[0], args[1])  # Direct operation
    )
```

**Available rulesets:**
| Ruleset | What it does |
|---------|--------------|
| `ruleset_simplify_builtin_arith` | `Py_Call(i32_add)` → `Op_i32_add` |
| `ruleset_simplify_builtin_print` | `Py_Call(print_i32)` → `Builtin_print_i32` |
| `ruleset_call_fqn` | `Py_Call(Py_LoadGlobal(f))` → `CallFQN(f)` |
| `ruleset_typing` | Propagate type metadata |
| `create_ruleset_struct__*` | Dynamic rules for struct ops |

### 4. Cost Model (nbcc/compiler.py)

Guides extraction to pick the "best" equivalent:

```python
class CostModel:
    def get_cost_function(self, nodename, op, ...):
        if op in ["Py_Call", "Py_LoadGlobal"]:
            return 10000  # Expensive - avoid
        elif op in ["CallFQN"]:
            return 1      # Cheap - prefer
        elif op.startswith("Op_"):
            return 1      # Direct ops are cheap
        # Default: inherited cost
```

### 5. Extraction

After saturation, extract the minimum-cost program:

```python
extraction = egraph_extraction(egraph, cost_model=CostModel())
extraction.compute()
result = extraction.extract_common_root()
print(f"Total cost: {result.cost}")
```

---

## Concrete Example

**Input (.spy):**
```python
def main() -> None:
    a: i32 = 1
    b: i32 = 2
    if a < b:
        print(b)
    else:
        print(a)
```

**RVSDG (before):**
```
Func("main")
  Region[io]
    $a = PyInt(1)
    $b = PyInt(2)
    $lt_fn = Py_LoadGlobal(io, "operator::i32_lt")
    $cond = Py_Call($lt_fn, io, [$a, $b])
    IfElse($cond[1],
      then: Region[io, a, b]
        $print = Py_LoadGlobal(io, "builtins::print_i32")
        Py_Call($print, io, [$b])
      else: Region[io, a, b]
        $print = Py_LoadGlobal(io, "builtins::print_i32")
        Py_Call($print, io, [$a])
    )
```

**After e-graph optimization + extraction:**
```
Func("main")
  Region[io]
    $a = PyInt(1)
    $b = PyInt(2)
    $cond = Op_i32_lt($a, $b)      ← Simplified!
    IfElse($cond,
      then: Region[io, a, b]
        Builtin_print_i32(io, $b)  ← Simplified!
      else: Region[io, a, b]
        Builtin_print_i32(io, $a)  ← Simplified!
    )
```

**What happened:**
1. `Py_Call(Py_LoadGlobal("operator::i32_lt"), ...)` → `Op_i32_lt(...)`
2. `Py_Call(Py_LoadGlobal("builtins::print_i32"), ...)` → `Builtin_print_i32(...)`

Both versions exist in the e-graph. Extraction chose the cheaper ones.

---

## Why E-graphs Over Traditional Rewriting?

### Problem with Traditional Rewriting

```
Rule 1: A → B
Rule 2: A → C
Rule 3: B + C → D

If we apply Rule 1 first, we lose A, so Rule 2 can't fire.
Result depends on rule order!
```

### E-graph Solution

```
Apply Rule 1: A ≡ B (both in same e-class)
Apply Rule 2: A ≡ B ≡ C (all equivalent)
Apply Rule 3: Since B and C are both available, can derive D

No rule order dependency!
```

---

## The egglog Library

numbacc uses [egglog](https://github.com/egraphs-good/egglog-python), a Python binding for equality saturation with Datalog.

**Key features:**
- Declarative rewrite rules
- Efficient e-graph operations
- Cost-based extraction
- Supports complex patterns

```python
from egglog import EGraph, rewrite, ruleset

@ruleset
def my_rules():
    x, y = vars("x y", Term)
    # Pattern matching and rewriting
    yield rewrite(Add(x, Zero())).to(x)
    yield rewrite(Mul(x, One())).to(x)

egraph = EGraph()
egraph.let("root", my_program)
egraph.run(my_rules.saturate())
```

---

## Visualization

Run with `VIZ=1` to dump e-graph state:

```bash
VIZ=1 python examples/component_demos/05_full_lifecycle.py
```

Outputs to `/tmp/nbcc_viz/`:
- `*_egraph_initial.egraph_json` - Before optimization
- `*_egraph_saturated.egraph_json` - After saturation

View with model-explorer:
```bash
pip install model-explorer
model-explorer /tmp/nbcc_viz/03_egraph_saturated.egraph_json
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `nbcc/egraph/rules.py` | Rewrite rules and optimization schedule |
| `nbcc/egraph/conversion.py` | E-graph → RVSDG back-conversion |
| `nbcc/compiler.py` | Pipeline orchestration, cost model |
| `deps/sealir/sealir/eqsat/rvsdg_eqsat.py` | E-graph term definitions |
| `deps/sealir/sealir/eqsat/rvsdg_convert.py` | RVSDG → E-graph conversion |
| `deps/sealir/sealir/eqsat/rvsdg_extract.py` | Extraction algorithm |
| `deps/sealir/sealir/model_explorer/core.py` | Visualization helpers |

---

## Potential Improvements

1. **More rewrite rules**: Constant folding, algebraic simplifications
2. **Better cost model**: Consider operation latency, not just presence
3. **Partial evaluation**: Evaluate pure computations at compile time
4. **Loop optimizations**: Unrolling, vectorization as rewrites
5. **Visualization**: Real-time e-graph growth animation

---

## Further Reading

- [egg: Fast and Extensible E-graphs](https://dl.acm.org/doi/10.1145/3434304) - The foundational paper
- [egglog](https://github.com/egraphs-good/egglog) - Datalog + e-graphs
- [RVSDG](https://dl.acm.org/doi/10.1145/3391902) - The IR used by numbacc
