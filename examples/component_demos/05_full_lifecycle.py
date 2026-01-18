#!/usr/bin/env python3
"""
Demo 5: Full Lifecycle Trace
============================
Traces a function from source to optimized IR, showing every step.

Run with VIZ=1 to dump visualization files:
    VIZ=1 python examples/component_demos/05_full_lifecycle.py

This demonstrates the complete pipeline:
    .spy source
        ↓ [SPy VM redshift]
    SPy AST
        ↓ [restructure]
    SCFG (Structured Control Flow Graph)
        ↓ [translate]
    RVSDG S-expression
        ↓ [egraph_conversion]
    E-graph terms
        ↓ [egraph_optimize]
    Saturated E-graph
        ↓ [egraph_extraction]
    Optimal E-graph subgraph
        ↓ [ExtendEGraphToRVSDG.convert]
    Optimized RVSDG
        ↓ [Backend - requires MLIR]
    MLIR → LLVM IR → Native code
"""

import json
import os
import sys
sys.path.insert(0, ".")

from typing import cast
from egglog import EGraph
from sealir.ase import SExpr
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as BaseCostModel, egraph_extraction
from sealir.rvsdg import format_rvsdg
import sealir.rvsdg.grammar as rg

from nbcc.frontend import frontend
from nbcc.egraph.rules import egraph_convert_metadata, egraph_optimize
from nbcc.egraph.conversion import ExtendEGraphToRVSDG

# Check for VIZ mode
VIZ = os.environ.get("VIZ", "0") == "1"
VIZ_DIR = "/tmp/nbcc_viz"

SOURCE = "examples/basic_if.spy"

class CostModel(BaseCostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        if op in ["Py_Call", "Py_LoadGlobal"]:
            return self.get_simple(10000)
        elif op in ["CallFQN", "Op_i32_lt", "Op_i32_gt", "Op_i32_add", "Op_i32_sub"]:
            return self.get_simple(1)
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)

def save_viz(name: str, content: str, ext: str = "txt"):
    """Save visualization artifact if VIZ=1."""
    if not VIZ:
        return
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = f"{VIZ_DIR}/{name}.{ext}"
    with open(path, "w") as f:
        f.write(content)
    print(f"  [VIZ] Saved: {path}")

def save_egraph_viz(name: str, egraph: EGraph):
    """Save egraph JSON for visualization."""
    if not VIZ:
        return
    os.makedirs(VIZ_DIR, exist_ok=True)
    path = f"{VIZ_DIR}/{name}.egraph_json"
    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    with open(path, "w") as f:
        f.write(serialized.to_json())
    print(f"  [VIZ] Saved: {path}")

def stage_banner(num: int, title: str):
    print(f"\n{'═' * 70}")
    print(f"  STAGE {num}: {title}")
    print(f"{'═' * 70}")

def main():
    if VIZ:
        print(f"VIZ mode enabled. Artifacts will be saved to {VIZ_DIR}/")
        os.makedirs(VIZ_DIR, exist_ok=True)

    # Read source
    with open(SOURCE) as f:
        source_code = f.read()

    print(f"\nSource file: {SOURCE}")
    print("─" * 70)
    print(source_code)
    save_viz("00_source", source_code, "spy")

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(1, "Frontend (SPy → RVSDG S-expression)")
    # ═══════════════════════════════════════════════════════════════════

    print("\nParsing with SPy VM (redshift)...")
    tu = frontend(SOURCE)

    fqn = tu.list_functions()[0]
    fi = tu.get_function(fqn)

    print(f"\nFunction: {fi.fqn}")
    print(f"Metadata entries: {len(fi.metadata)}")

    rvsdg_str = format_rvsdg(fi.region)
    print("\nRVSDG S-expression:")
    print("─" * 70)
    print(rvsdg_str)
    save_viz("01_rvsdg_initial", rvsdg_str)

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(2, "E-graph Conversion (RVSDG → E-graph Terms)")
    # ═══════════════════════════════════════════════════════════════════

    print("\nConverting RVSDG nodes to e-graph terms...")
    memo = egraph_conversion(fi.region)

    print(f"Memo table: {len(memo)} entries")

    # Show some conversions
    print("\nSample conversions (first 8):")
    for i, (sexpr, term) in enumerate(list(memo.items())[:8]):
        sexpr_name = type(sexpr).__name__
        print(f"  {sexpr_name:25} → {str(term)[:50]}...")

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(3, "E-graph Initialization")
    # ═══════════════════════════════════════════════════════════════════

    root_term = memo[fi.region]
    root = GraphRoot(root_term)

    egraph = EGraph()
    egraph.let("root", root)
    egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    json_str = serialized.to_json()
    json_data = json.loads(json_str)
    print(f"\nE-graph created:")
    print(f"  Nodes: {len(json_data.get('nodes', []))}")
    print(f"  E-classes: {len(json_data.get('class_data', {}))}")

    save_egraph_viz("02_egraph_initial", egraph)

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(4, "E-graph Optimization (Equality Saturation)")
    # ═══════════════════════════════════════════════════════════════════

    print("\nApplying rewrite rules...")
    print("""
Rules being applied:
  - ruleset_simplify_builtin_arith: Py_Call(i32_add) → Op_i32_add
  - ruleset_simplify_builtin_print: Py_Call(print_i32) → Builtin_print_i32
  - ruleset_call_fqn: Py_Call(Py_LoadGlobal(f)) → CallFQN(f)
  - ruleset_typing: Propagate type metadata
""")

    egraph_optimize(egraph)

    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    json_str = serialized.to_json()
    json_data = json.loads(json_str)
    print(f"E-graph after saturation:")
    print(f"  Nodes: {len(json_data.get('nodes', []))}")
    print(f"  E-classes: {len(json_data.get('class_data', {}))}")

    save_egraph_viz("03_egraph_saturated", egraph)

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(5, "E-graph Extraction (Find Optimal Program)")
    # ═══════════════════════════════════════════════════════════════════

    print("\nExtracting lowest-cost program...")
    print("""
Cost model:
  Py_Call, Py_LoadGlobal: 10000 (avoid - generic Python ops)
  CallFQN, Op_*:          1     (prefer - direct operations)
""")

    extraction = egraph_extraction(egraph, cost_model=CostModel())
    extraction.compute()
    extresult = extraction.extract_common_root()

    print(f"\nExtraction complete:")
    print(f"  Total cost: {extresult.cost}")

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(6, "Back-conversion (E-graph → Optimized RVSDG)")
    # ═══════════════════════════════════════════════════════════════════

    print("\nConverting extracted graph back to RVSDG...")
    converted_root: SExpr = extresult.convert(fi.region, ExtendEGraphToRVSDG)

    # Find the optimized Func node
    optimized_func = None
    for node in converted_root._args:
        if isinstance(node, rg.Func):
            optimized_func = node
            break

    if optimized_func:
        optimized_str = format_rvsdg(cast(SExpr, optimized_func))
        print("\nOptimized RVSDG:")
        print("─" * 70)
        print(optimized_str)
        save_viz("04_rvsdg_optimized", optimized_str)

    # ═══════════════════════════════════════════════════════════════════
    stage_banner(7, "Summary: What Changed")
    # ═══════════════════════════════════════════════════════════════════

    print("""
BEFORE (generic Python operations):
  $cond = Py_Call(Py_LoadGlobal("operator::i32_lt"), io, [$a, $b])

AFTER (direct operations):
  $cond = Op_i32_lt($a, $b)

The e-graph optimization:
1. Added rewrite: Py_Call(i32_lt) ≡ Op_i32_lt (same e-class)
2. Extraction chose Op_i32_lt (cost=1) over Py_Call (cost=10000)

This is equality saturation at work:
- Don't destructively rewrite
- Add equivalences, then choose the best
""")

    if VIZ:
        print(f"\n{'═' * 70}")
        print(f"  Visualization files saved to: {VIZ_DIR}/")
        print(f"{'═' * 70}")
        print(f"""
Files:
  00_source.spy          - Original source code
  01_rvsdg_initial.txt   - RVSDG before optimization
  02_egraph_initial.egraph_json  - E-graph before rules
  03_egraph_saturated.egraph_json - E-graph after saturation
  04_rvsdg_optimized.txt - RVSDG after extraction

View e-graphs with model-explorer (if installed):
  model-explorer {VIZ_DIR}/03_egraph_saturated.egraph_json
""")

if __name__ == "__main__":
    main()
