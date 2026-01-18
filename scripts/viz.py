#!/usr/bin/env python3
"""
Visualization wrapper for numbacc compilation.

Usage:
    VIZ=1 python -m nbcc source.spy
    python scripts/viz.py source.spy

This wraps the compilation pipeline and dumps visualization artifacts
at each stage, similar to tinygrad's VIZ=1.

Outputs to /tmp/nbcc_viz/:
    00_source.spy              - Original source
    01_rvsdg_input.txt         - RVSDG after frontend
    02_egraph_initial.json     - E-graph before optimization
    03_egraph_saturated.json   - E-graph after saturation
    04_rvsdg_optimized.txt     - RVSDG after extraction
    05_egraph_diff.txt         - Summary of what changed

Environment variables:
    VIZ=1           Enable visualization (default when using this script)
    VIZ_DIR=/path   Override output directory
    VIZ_OPEN=1      Auto-open visualization (requires model-explorer)
"""

import json
import os
import sys
from pathlib import Path
from typing import cast

# Ensure we can import nbcc
sys.path.insert(0, str(Path(__file__).parent.parent))

from egglog import EGraph
from sealir.ase import SExpr
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import CostModel as BaseCostModel, egraph_extraction
from sealir.rvsdg import format_rvsdg
import sealir.rvsdg.grammar as rg

from nbcc.frontend import frontend, TranslationUnit
from nbcc.egraph.rules import egraph_convert_metadata, egraph_optimize
from nbcc.egraph.conversion import ExtendEGraphToRVSDG


class VizContext:
    """Context manager for visualization artifacts."""

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.source_name = Path(source_path).stem
        self.viz_dir = Path(os.environ.get("VIZ_DIR", "/tmp/nbcc_viz"))
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.stage = 0
        self.artifacts: list[tuple[str, str]] = []

    def save(self, name: str, content: str, ext: str = "txt") -> Path:
        """Save an artifact and track it."""
        filename = f"{self.stage:02d}_{name}.{ext}"
        path = self.viz_dir / filename
        with open(path, "w") as f:
            f.write(content)
        self.artifacts.append((name, str(path)))
        self.stage += 1
        return path

    def save_egraph(self, name: str, egraph: EGraph) -> Path:
        """Save egraph as JSON for model-explorer."""
        serialized = egraph._serialize(
            n_inline_leaves=1, split_primitive_outputs=False
        )
        return self.save(name, serialized.to_json(), "egraph_json")

    def summary(self) -> str:
        """Generate summary of all artifacts."""
        lines = [
            f"Visualization artifacts for: {self.source_path}",
            f"Output directory: {self.viz_dir}",
            "",
            "Files:",
        ]
        for name, path in self.artifacts:
            lines.append(f"  {Path(path).name:40} - {name}")
        return "\n".join(lines)


class CostModel(BaseCostModel):
    """Cost model that tracks what it evaluates for visualization."""

    def __init__(self):
        super().__init__()
        self.evaluations: list[tuple[str, int]] = []

    def get_cost_function(self, nodename, op, ty, cost, children):
        if op in ["Py_Call", "Py_LoadGlobal"]:
            c = 10000
        elif op in ["CallFQN"]:
            c = 1
        elif op.startswith("Op_") or op.startswith("Builtin_"):
            c = 1
        else:
            return super().get_cost_function(nodename, op, ty, cost, children)

        self.evaluations.append((op, c))
        return self.get_simple(c)


def compile_with_viz(source_path: str) -> None:
    """Compile a .spy file with full visualization."""

    ctx = VizContext(source_path)

    # ─── Stage 0: Source ───────────────────────────────────────────────
    print(f"[VIZ] Compiling: {source_path}")

    with open(source_path) as f:
        source = f.read()
    ctx.save("source", source, "spy")

    # ─── Stage 1: Frontend ─────────────────────────────────────────────
    print("[VIZ] Stage 1: Frontend (SPy → RVSDG)")

    tu: TranslationUnit = frontend(source_path)

    for fqn in tu.list_functions():
        fi = tu.get_function(fqn)

        rvsdg_input = format_rvsdg(fi.region)
        ctx.save(f"rvsdg_input_{fi.fqn.replace('::', '_')}", rvsdg_input)

        # ─── Stage 2: E-graph conversion ───────────────────────────────
        print(f"[VIZ] Stage 2: E-graph conversion for {fi.fqn}")

        memo = egraph_conversion(fi.region)
        root = GraphRoot(memo[fi.region])

        egraph = EGraph()
        egraph.let("root", root)
        egraph.let("mds", egraph_convert_metadata(fi.metadata, memo))

        ctx.save_egraph(f"egraph_initial_{fi.fqn.replace('::', '_')}", egraph)

        # Stats before
        stats_before = egraph._serialize(
            n_inline_leaves=1, split_primitive_outputs=False
        )
        nodes_before = len(stats_before.nodes)

        # ─── Stage 3: Optimization ─────────────────────────────────────
        print(f"[VIZ] Stage 3: E-graph optimization for {fi.fqn}")

        egraph_optimize(egraph)

        ctx.save_egraph(f"egraph_saturated_{fi.fqn.replace('::', '_')}", egraph)

        # Stats after
        stats_after = egraph._serialize(
            n_inline_leaves=1, split_primitive_outputs=False
        )
        nodes_after = len(stats_after.nodes)

        # ─── Stage 4: Extraction ───────────────────────────────────────
        print(f"[VIZ] Stage 4: Extraction for {fi.fqn}")

        cost_model = CostModel()
        extraction = egraph_extraction(egraph, cost_model=cost_model)
        extraction.compute()
        extresult = extraction.extract_common_root()

        # ─── Stage 5: Back-conversion ──────────────────────────────────
        print(f"[VIZ] Stage 5: Back-conversion for {fi.fqn}")

        converted_root: SExpr = extresult.convert(fi.region, ExtendEGraphToRVSDG)

        for node in converted_root._args:
            if isinstance(node, rg.Func):
                rvsdg_output = format_rvsdg(cast(SExpr, node))
                ctx.save(f"rvsdg_optimized_{fi.fqn.replace('::', '_')}", rvsdg_output)

        # ─── Stage 6: Diff summary ─────────────────────────────────────
        diff_summary = f"""
E-graph Optimization Summary for {fi.fqn}
{'=' * 60}

Nodes before optimization: {nodes_before}
Nodes after optimization:  {nodes_after}
Node increase:             {nodes_after - nodes_before} (equivalences added)

Extraction cost: {extresult.cost}

Cost model evaluations (sample):
"""
        for op, cost in cost_model.evaluations[:20]:
            diff_summary += f"  {op:30} → cost={cost}\n"
        if len(cost_model.evaluations) > 20:
            diff_summary += f"  ... and {len(cost_model.evaluations) - 20} more\n"

        ctx.save(f"summary_{fi.fqn.replace('::', '_')}", diff_summary)

    # Final summary
    print()
    print("=" * 60)
    print(ctx.summary())
    print("=" * 60)

    # Auto-open if requested
    if os.environ.get("VIZ_OPEN") == "1":
        try:
            import subprocess
            # Find the saturated egraph
            for name, path in ctx.artifacts:
                if "saturated" in name and path.endswith(".egraph_json"):
                    print(f"\n[VIZ] Opening: {path}")
                    subprocess.run(["model-explorer", path])
                    break
        except Exception as e:
            print(f"[VIZ] Could not auto-open: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    source_path = sys.argv[1]
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found")
        sys.exit(1)

    compile_with_viz(source_path)


if __name__ == "__main__":
    main()
