#!/usr/bin/env python3
"""
Demo 1: Frontend Only
=====================
Shows how SPy source is parsed and converted to RVSDG S-expressions.

This demonstrates:
- SPy VM execution (redshift)
- AST to SCFG conversion
- SCFG to RVSDG conversion
"""

import sys
sys.path.insert(0, ".")

from pprint import pprint
from nbcc.frontend import frontend, TranslationUnit
from sealir.rvsdg import format_rvsdg

# Source file to compile
SOURCE = "examples/basic_if.spy"

def main():
    print("=" * 60)
    print("STAGE 1: Frontend - SPy to RVSDG S-expression")
    print("=" * 60)

    # Parse and convert to RVSDG
    tu: TranslationUnit = frontend(SOURCE)

    print(f"\nSource: {SOURCE}")
    print(f"Functions found: {tu.list_functions()}")

    for fqn in tu.list_functions():
        fi = tu.get_function(fqn)
        print(f"\n{'─' * 60}")
        print(f"Function: {fi.fqn}")
        print(f"{'─' * 60}")

        # Print the RVSDG S-expression
        print("\nRVSDG S-expression (raw):")
        print(fi.region)

        # Pretty-printed RVSDG
        print("\nRVSDG formatted:")
        print(format_rvsdg(fi.region))

        # Metadata (type info)
        print(f"\nMetadata ({len(fi.metadata)} entries):")
        for md in fi.metadata[:5]:  # First 5
            print(f"  {md}")
        if len(fi.metadata) > 5:
            print(f"  ... and {len(fi.metadata) - 5} more")

if __name__ == "__main__":
    main()
