# Cost Model Experiments for E-Graph Extraction

**Date**: 2026-01-18
**Branch**: `feature/cost-model-experiments`

## Overview

Implemented an experiment framework to compare different cost models for e-graph extraction. The goal is to find cost models that better predict actual runtime, leading to better extraction decisions and faster generated code.

## Problem Statement

The current cost model is trivial:
```python
if op in ["Py_Call", "Py_LoadGlobal"]: return 10000
elif op in ["CallFQN"]: return 1
```

While this effectively guides extraction away from generic Python operations toward specialized ones, it doesn't correlate with actual runtime performance. Better cost models could improve extraction quality.

## Implementation

### File Structure
```
nbcc/experiments/
├── __init__.py             # Public API exports
├── cost_models.py          # 4 cost model implementations
├── features.py             # ML feature extraction
├── benchmark.py            # Experiment runner
├── programs/               # Test SPy programs
│   ├── arithmetic.spy      # Pure computation
│   ├── branches.spy        # Control flow
│   ├── loops.spy           # Iteration
│   └── mixed.spy           # Combination
└── tests/
    └── test_cost_models.py # Unit tests (18 tests)
```

### Cost Models Implemented

| Model | Strategy | Use Case |
|-------|----------|----------|
| `BaselineCostModel` | Static costs: Py_Call=10000, Op_*=1 | Current approach |
| `PerfCostModel` | Hardware counter measurements via `perf stat` | Ground truth for primitives |
| `MLCostModel` | XGBoost predictions from node features | Learned cost prediction |
| `HybridCostModel` | 0.7*perf + 0.3*ml weighted combination | Best of both |

### Features for ML Model

```python
@dataclass
class NodeFeatures:
    op_category: int      # 0=arith, 1=cmp, 2=ctrl, 3=io, 4=call, 5=literal
    num_children: int     # Fan-in
    has_io: bool          # Touches IO state
    is_pure: bool         # No side effects
    type_weight: float    # Type-based weight
    is_python_generic: bool
    is_lowered: bool
```

### Quality Metric

**Spearman correlation** between predicted cost and actual runtime:
- Correlation = 1.0: Perfect ranking (extraction always picks fastest)
- Correlation = 0.0: Random (cost model is useless)

## Benchmark Results

All 4 programs compiled and benchmarked successfully:

| Program | Baseline Cost | ML/Hybrid Cost | Nodes |
|---------|--------------|----------------|-------|
| arithmetic | 26,670 | 194.5 | 57 |
| branches | 3.84M | 137.5 | 79 |
| loops | 3.51B | 223.0 | 122 |
| mixed | 8.62B | 266.5 | 143 |

### Key Observations

1. **Same extraction structure**: All models extract identical node counts because they extract from the same e-graph.

2. **Cost differences are intentional**:
   - Baseline/Perf: High costs penalize Python ops, guiding toward specialized ops
   - ML/Hybrid: Lower costs based on feature heuristics

3. **Extraction speed**: ~3-4ms across all models

## Usage

```python
from nbcc.experiments import BenchmarkRunner

runner = BenchmarkRunner()
runner.add_programs_from_dir("nbcc/experiments/programs")
runner.run()
runner.print_results()
```

Or use individual models:
```python
from nbcc.experiments import MLCostModel
from nbcc.compiler import middle_end

# Use ML cost model for extraction
model = MLCostModel()
# ... pass to egraph_extraction()
```

## Next Steps

1. **Calibrate PerfCostModel**: Run microbenchmarks to get real cycle counts
2. **Train MLCostModel**: Collect training data from real compilations
3. **Measure actual runtime**: Compile to WASM and measure execution time
4. **Compute correlations**: Compare predicted vs actual costs

## Phase 2: Principled Cost Models

### New Modules Added

| Module | Purpose |
|--------|---------|
| `principled_costs.py` | CPU instruction-based costs (Agner Fog tables) |
| `synthetic_data.py` | Synthetic program generation for training |
| `ml_training.py` | ML training pipeline with proper validation |

### Principled Cost Hierarchy

Based on real CPU microarchitecture (Skylake-era x86-64):

| Operation Type | Cycles | Rationale |
|---------------|--------|-----------|
| Add/Sub/Compare | 0.5 | Single cycle, 4-wide dispatch |
| Multiply | 1.0 | 3-4 cycle latency, 1 throughput |
| Division | 25.0 | 20-100 cycles (data dependent) |
| Branch (predicted) | 0.5 | Well-predicted |
| Branch (mispredicted) | 15.0 | Pipeline flush |
| Direct call | 3.0 | Stack + jump |
| Python dispatch | 100.0 | Dynamic lookup |
| IO/syscall | 500.0 | Kernel transition |

### ML Model Training

Enhanced features for ML prediction:
- `op_category`: Operation type (0-6)
- `is_division`: Critical for cost (25x vs 0.5x)
- `is_multiplication`: Moderate cost (1x)
- `is_python_generic`: High cost indicator
- `is_call`: Function call overhead

Training results on analytical cost data:
- **R²: 0.9997** - Nearly perfect fit
- **Spearman: 0.77** - Good ranking preservation
- **MAE: 1.25 cycles** - Low prediction error

### Benchmark Results

| Program | Analytical | Instruction | Baseline |
|---------|-----------|-------------|----------|
| arithmetic | 10.88 | 26,588 | 26,670 |
| branches | 10.75 | 3.83M | 3.84M |
| loops | 16.12 | 3.50B | 3.51B |
| mixed | 18.75 | 8.61B | 8.62B |

Key insight: Instruction and Baseline models give similar relative costs
but different scales. Analytical model normalizes for ILP parallelism.

## Technical Notes

### SPy Control Flow Limitations

Some control flow patterns cause assertion errors in the restructuring pass:
- Early returns inside conditionals with else branches
- Nested if-else with multiple return points

Workaround: Use single return point with result variable:
```python
# Works
def abs_val(x: i32) -> i32:
    result = x
    if x < 0:
        result = 0 - x
    return result

# Fails (assertion error)
def abs_val(x: i32) -> i32:
    if x < 0:
        return 0 - x
    else:
        return x
```

### Division Types

SPy `/` returns `f64`, use `//` for integer division:
```python
q = a // b  # i32 result
r = a / b   # f64 result
```

## Phase 3: Algebraic Rewrite Rules

### Problem: Cost Models Had No Effect

Initial validation revealed that all cost models produced **identical extraction results**:
- All 4 programs extracted the same number of nodes
- Only 12 of 65 e-classes had multiple alternatives
- All alternatives were trivial (Py_Call vs Op_*) where every model agreed

**Root cause**: Without algebraic rewrites, the e-graph only contained the original program structure plus lowered operations.

### Solution: Add Meaningful Rewrite Rules

Added 6 new rulesets to `nbcc/egraph/rules.py`:

| Ruleset | Examples | Purpose |
|---------|----------|---------|
| `ruleset_commutativity` | `a + b ↔ b + a` | Create symmetric alternatives |
| `ruleset_associativity` | `(a+b)+c ↔ a+(b+c)` | Enable constant folding |
| `ruleset_identity` | `a + 0 → a`, `a * 1 → a` | Remove redundant operations |
| `ruleset_strength_reduction` | `a * 2 ↔ a + a ↔ a << 1` | Replace expensive with cheap |
| `ruleset_distributivity` | `a*(b+c) ↔ a*b + a*c` | Factor/expand expressions |
| `ruleset_comparison_flip` | `a < b ↔ b > a` | Normalize comparisons |

### New Operations Added

Extended the IR with:
- `Op_i32_mul`, `Op_i32_div`, `Op_i32_mod`
- `Op_i32_le`, `Op_i32_ge`, `Op_i32_eq`, `Op_i32_ne`
- `Op_i32_shl`, `Op_i32_shr` (for strength reduction)
- `Op_i32_neg`

### Validation: Cost Models Now Matter

With divergent cost models (artificially designed to disagree):

| Model | Arithmetic Nodes | Strategy |
|-------|-----------------|----------|
| prefer_shift | 53 | shift < add < mul |
| prefer_mul | 54 | mul < add < shift |
| prefer_add | 56 | add < shift < mul |

**Key insight**: Standard models (baseline, instruction, analytical) still produce same results because they all agree on relative ordering (shift < add < mul).

### Rule Firing Statistics (arithmetic.spy)

```
Op_i32_add commutativity: 95 matches
Op_i32_mul commutativity: 28 matches
Associativity (add): 147 matches
Associativity (mul): 48 matches
Distributivity: 37 matches
Strength reduction (mul→shl): 3 matches
```

### Implementation Details

1. **Literal representation**: SPy uses `Term.LiteralI64` from sealir, not a custom literal type. Rules must match against this.

2. **Schedule structure**:
   ```python
   lowering.saturate() + algebraic.saturate()
   ```
   - Phase 1: Lower Py_Call → Op_* (runs to saturation)
   - Phase 2: Apply algebraic rewrites (runs to saturation)

3. **E-graph blowup**: Algebraic rules can cause exponential growth. Current approach uses saturation but may need iteration limits for complex programs.
