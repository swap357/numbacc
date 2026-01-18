#!/usr/bin/env python3
"""
Demo 4: E-graph Extraction
==========================
Shows how the optimal program is extracted from the saturated e-graph.

This demonstrates:
- Cost model definition
- Extraction algorithm
- Converting extracted graph back to RVSDG
- Before/after comparison of the IR
"""

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

SOURCE = "examples/basic_if.spy"

class CostModel(BaseCostModel):
    """
    Cost model that prefers simplified operations over Python calls.

    This is the key to making e-graph extraction useful:
    - High cost for generic Python operations (we want to eliminate these)
    - Low cost for specific, optimized operations
    """

    def get_cost_function(self, nodename, op, ty, cost, children):
        if op in ["Py_Call", "Py_LoadGlobal"]:
            # Python-level calls are expensive - avoid them
            return self.get_simple(10000)
        elif op in ["CallFQN"]:
            # Direct FQN calls are cheap
            return self.get_simple(1)
        elif op.startswith("Op_"):
            # Direct operations are very cheap
            return self.get_simple(1)
        elif op.startswith("Builtin_"):
            # Builtin operations are cheap
            return self.get_simple(1)
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)

def main():
    print("=" * 60)
    print("STAGE 4: E-graph Extraction")
    print("=" * 60)

    # Setup and optimize
    tu = frontend(SOURCE)
    fqn = tu.list_functions()[0]
    fi = tu.get_function(fqn)

    memo = egraph_conversion(fi.region)
    root = GraphRoot(memo[fi.region])

    egraph = EGraph()
    egraph.let("root", root)
    egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

    print(f"\nFunction: {fi.fqn}")

    # Show BEFORE
    print("\n" + "─" * 60)
    print("RVSDG BEFORE optimization:")
    print("─" * 60)
    print(format_rvsdg(fi.region))

    # Optimize
    egraph_optimize(egraph)

    # Extract
    print("\n" + "─" * 60)
    print("Extracting optimal program...")
    print("─" * 60)

    extraction = egraph_extraction(egraph, cost_model=CostModel())
    extraction.compute()
    extresult = extraction.extract_common_root()

    print(f"\nExtraction cost: {extresult.cost}")

    # Convert back to RVSDG
    converted_root: SExpr = extresult.convert(fi.region, ExtendEGraphToRVSDG)

    print("\n" + "─" * 60)
    print("RVSDG AFTER optimization:")
    print("─" * 60)

    # Find the Func node in the converted result
    for node in converted_root._args:
        if isinstance(node, rg.Func):
            print(format_rvsdg(cast(SExpr, node)))

    print("\n" + "─" * 60)
    print("What changed?")
    print("─" * 60)
    print("""
Before:
  Py_Call(Py_LoadGlobal("operator::i32_lt"), io, [a, b])

After:
  Op_i32_lt(a, b)

The e-graph found that these are equivalent, and extraction
chose Op_i32_lt because it has lower cost (1 vs 10000).

This is the power of equality saturation:
1. Add rewrite rules that prove equivalences
2. Let the e-graph saturate (find all equivalent forms)
3. Extract the cheapest representation
""")

if __name__ == "__main__":
    main()
