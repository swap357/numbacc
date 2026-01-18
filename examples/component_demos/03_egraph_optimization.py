#!/usr/bin/env python3
"""
Demo 3: E-graph Optimization (The Core of Equality Saturation)
==============================================================
Shows how egglog rules transform the e-graph.

This demonstrates:
- Ruleset application (simplify_builtin_arith, simplify_builtin_print, etc.)
- Equality saturation - adding equivalent representations
- Cost model for extraction
- Before/after comparison

Key insight: E-graphs don't replace nodes, they ADD equivalent alternatives.
Extraction then picks the lowest-cost version.
"""

import sys
sys.path.insert(0, ".")

from egglog import EGraph
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot

from nbcc.frontend import frontend
from nbcc.egraph.rules import (
    egraph_convert_metadata,
    egraph_optimize,
    ruleset_simplify_builtin_arith,
    ruleset_simplify_builtin_print,
    ruleset_call_fqn,
)

SOURCE = "examples/basic_if.spy"

def count_nodes(egraph):
    """Count nodes in egraph for statistics."""
    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    return len(serialized.nodes)

def main():
    print("=" * 60)
    print("STAGE 3: E-graph Optimization (Equality Saturation)")
    print("=" * 60)

    # Setup
    tu = frontend(SOURCE)
    fqn = tu.list_functions()[0]
    fi = tu.get_function(fqn)

    memo = egraph_conversion(fi.region)
    root = GraphRoot(memo[fi.region])

    egraph = EGraph()
    egraph.let("root", root)
    egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

    print(f"\nFunction: {fi.fqn}")
    print(f"Nodes BEFORE optimization: {count_nodes(egraph)}")

    # Save before state
    with open("/tmp/egraph_before_opt.json", "w") as f:
        f.write(egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False).to_json())

    # Apply optimization rulesets one by one to see effect
    print("\n" + "─" * 60)
    print("Applying rulesets incrementally...")
    print("─" * 60)

    print("\n1. ruleset_simplify_builtin_arith")
    print("   Rewrites: Py_Call(operator::i32_add, ...) → Op_i32_add(...)")
    egraph.run(ruleset_simplify_builtin_arith.saturate())
    print(f"   Nodes after: {count_nodes(egraph)}")

    print("\n2. ruleset_simplify_builtin_print")
    print("   Rewrites: Py_Call(builtins::print_i32, ...) → Builtin_print_i32(...)")
    egraph.run(ruleset_simplify_builtin_print.saturate())
    print(f"   Nodes after: {count_nodes(egraph)}")

    print("\n3. ruleset_call_fqn")
    print("   Rewrites: Py_Call(Py_LoadGlobal(fqn), ...) → CallFQN(fqn, ...)")
    egraph.run(ruleset_call_fqn.saturate())
    print(f"   Nodes after: {count_nodes(egraph)}")

    # Save after state
    with open("/tmp/egraph_after_opt.json", "w") as f:
        f.write(egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False).to_json())

    print("\n" + "─" * 60)
    print("Why does node count INCREASE?")
    print("─" * 60)
    print("""
E-graphs work by ADDING equivalent representations, not replacing.

Before: Py_Call(Py_LoadGlobal("operator::i32_add"), io, [a, b])
After:  Same node exists, BUT ALSO Op_i32_add(a, b) in same e-class

The e-graph now contains BOTH representations as "equal".
Extraction picks the cheaper one based on cost model.

Cost model in nbcc/compiler.py:
  - Py_Call, Py_LoadGlobal: cost = 10000 (expensive, avoid)
  - CallFQN: cost = 1 (cheap, preferred)
  - Op_i32_add: default cost (cheap)
""")

    print("\n" + "─" * 60)
    print("Saved e-graphs for visualization:")
    print("─" * 60)
    print("  Before: /tmp/egraph_before_opt.json")
    print("  After:  /tmp/egraph_after_opt.json")
    print("\nView with: model-explorer /tmp/egraph_after_opt.json")

if __name__ == "__main__":
    main()
