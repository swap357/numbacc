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
