#!/usr/bin/env python3
"""
Demo 2: E-graph Conversion
==========================
Shows how RVSDG S-expressions are converted to e-graph representation.

This demonstrates:
- RVSDG to egraph term conversion
- GraphRoot initialization
- Metadata conversion to egraph form
"""

import sys
sys.path.insert(0, ".")

from egglog import EGraph
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot

from nbcc.frontend import frontend
from nbcc.egraph.rules import egraph_convert_metadata

SOURCE = "examples/basic_if.spy"

def main():
    print("=" * 60)
    print("STAGE 2: E-graph Conversion")
    print("=" * 60)

    # Get RVSDG from frontend
    tu = frontend(SOURCE)
    fqn = tu.list_functions()[0]
    fi = tu.get_function(fqn)

    print(f"\nFunction: {fi.fqn}")
    print(f"RVSDG root: {fi.region}")

    # Convert RVSDG S-expr to egraph terms
    print("\n" + "─" * 60)
    print("Converting RVSDG → E-graph terms...")
    print("─" * 60)

    memo = egraph_conversion(fi.region)

    print(f"\nMemo table has {len(memo)} entries")
    print("\nSample memo entries (S-expr → E-graph Term):")
    for i, (sexpr, term) in enumerate(list(memo.items())[:10]):
        print(f"  {i}: {type(sexpr).__name__:20} → {term}")

    # Create egraph and set root
    print("\n" + "─" * 60)
    print("Creating E-graph with GraphRoot")
    print("─" * 60)

    root_term = memo[fi.region]
    root = GraphRoot(root_term)

    egraph = EGraph()
    egraph.let("root", root)

    print(f"\nRoot term: {root_term}")
    print(f"GraphRoot: {root}")

    # Convert metadata
    print("\n" + "─" * 60)
    print("Converting Metadata → E-graph Metadata")
    print("─" * 60)

    md_vec = egraph_convert_metadata(fi.metadata, memo)
    egraph.let("mds", md_vec)

    print(f"\nMetadata vector: {md_vec}")

    # Serialize egraph to see structure
    print("\n" + "─" * 60)
    print("E-graph Statistics (before optimization)")
    print("─" * 60)

    # Get serialized form for stats
    serialized = egraph._serialize(n_inline_leaves=1, split_primitive_outputs=False)
    print(f"\nNodes: {len(serialized.nodes)}")
    print(f"Class data: {len(serialized.class_data)}")

    # Save for visualization
    import json
    with open("/tmp/egraph_before.json", "w") as f:
        f.write(serialized.to_json())
    print("\nSaved to /tmp/egraph_before.json for visualization")

if __name__ == "__main__":
    main()
