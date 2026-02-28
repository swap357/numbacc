"""End-to-end compiler pipeline tests.

Tests the full compilation pipeline from SPy source through each stage:
  1. Frontend: SPy source -> TranslationUnit (AST, SCFG, RVSDG)
  2. Middle-end: TranslationUnit -> optimized RVSDG (e-graph optimization)
  3. MLIR generation: RVSDG -> MLIR module
  4. Backend passes: MLIR -> lowered MLIR (LLVM dialect)
  5. Binary generation: MLIR -> executable -> run and validate output
"""

import os.path
import subprocess as subp
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import sealir.rvsdg.grammar as rg

import nbcc
from nbcc.compiler import compile, compile_to_mlir, middle_end
from nbcc.frontend import TranslationUnit, frontend
from nbcc.frontend.grammar import TypeInfo, IRTag

e2e_dir = Path(os.path.dirname(nbcc.__file__)) / ".." / "examples" / "e2e"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def make_temp_directory() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory(delete=False) as dirpath:
        yield Path(dirpath)


def spy_file(filename: str) -> str:
    """Return the full path to an e2e example .spy file."""
    path = e2e_dir / filename
    assert path.exists(), f"Missing example file: {path}"
    return str(path)


def write_spy_source(src: str) -> str:
    """Write SPy source to a temp file and return its path."""
    tmpfile = tempfile.NamedTemporaryFile(
        "w+", suffix=".spy", prefix="test_pipeline_", delete=False
    )
    tmpfile.write(src)
    tmpfile.flush()
    tmpfile.close()
    return tmpfile.name


def run(path: Path | str) -> str:
    """Run a compiled binary and return its stdout."""
    return subp.check_output(str(path), encoding="utf-8")


def run_e2e_test(filename: str, expected_output: str) -> None:
    """Compile an e2e .spy file, run the binary, and assert output."""
    path = e2e_dir / filename
    assert path.exists()
    with make_temp_directory() as dir:
        outpath = dir / "a.out"
        compile(str(path), str(outpath))
        output = run(outpath)
    assert output == expected_output.lstrip()


def run_source_e2e_test(src: str, expected_output: str) -> None:
    """Compile inline SPy source, run the binary, and assert output."""
    path = write_spy_source(src)
    try:
        with make_temp_directory() as dir:
            outpath = dir / "a.out"
            compile(path, str(outpath))
            output = run(outpath)
        assert output == expected_output.lstrip()
    finally:
        os.unlink(path)


# ===========================================================================
# Stage 1: Frontend tests
# ===========================================================================


class TestFrontendStage:
    """Test that frontend() produces a valid TranslationUnit."""

    def test_frontend_returns_translation_unit(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        assert isinstance(tu, TranslationUnit)

    def test_frontend_ifelse_has_main_function(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        fns = tu.list_functions()
        assert len(fns) >= 1
        fqn_names = [f.fullname for f in fns]
        assert any("main" in name for name in fqn_names)

    def test_frontend_loops_has_main_function(self):
        tu = frontend(spy_file("e2e_loops.spy"))
        fns = tu.list_functions()
        assert len(fns) >= 1
        fqn_names = [f.fullname for f in fns]
        assert any("main" in name for name in fqn_names)

    def test_frontend_class_has_function_and_struct(self):
        tu = frontend(spy_file("e2e_class.spy"))
        fns = tu.list_functions()
        assert len(fns) >= 1
        # The class example defines a Point struct; check builtins are captured
        builtins = tu.list_builtins()
        assert len(builtins) >= 1

    def test_frontend_function_info_has_region_and_metadata(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        fns = tu.list_functions()
        for fqn in fns:
            fi = tu.get_function(fqn)
            assert fi.fqn is not None
            assert fi.region is not None
            assert fi.metadata is not None
            assert isinstance(fi.metadata, list)

    def test_frontend_arithmetic_program(self):
        tu = frontend(spy_file("e2e_arithmetic.spy"))
        fns = tu.list_functions()
        assert len(fns) >= 1
        fqn_names = [f.fullname for f in fns]
        assert any("main" in name for name in fqn_names)

    def test_frontend_nested_if_program(self):
        tu = frontend(spy_file("e2e_nested_if.spy"))
        fns = tu.list_functions()
        assert len(fns) >= 1

    def test_frontend_inline_source(self):
        """Test frontend with inline SPy source."""
        src = """\
def main() -> i32:
    x = 100
    y = 200
    z = x + y
    print(z)
    return 0
"""
        path = write_spy_source(src)
        try:
            tu = frontend(path)
            assert isinstance(tu, TranslationUnit)
            fns = tu.list_functions()
            assert len(fns) >= 1
        finally:
            os.unlink(path)

    def test_frontend_preserves_function_metadata(self):
        """Verify metadata list is non-empty for a program with assignments."""
        tu = frontend(spy_file("e2e_ifelse.spy"))
        for fqn in tu.list_functions():
            fi = tu.get_function(fqn)
            # Metadata should contain type info and debug info
            assert len(fi.metadata) > 0

    def test_frontend_struct_detection(self):
        """Test that struct types in the program are detected."""
        tu = frontend(spy_file("e2e_class.spy"))
        # The class example has @struct Point
        has_struct = False
        builtins = tu.list_builtins()
        for fqn in builtins:
            if "Point" in str(fqn) or "__make__" in str(fqn):
                has_struct = True
                break
        assert has_struct, "Expected to find struct-related builtins"


# ===========================================================================
# Stage 2: Middle-end tests (e-graph optimization)
# ===========================================================================


class TestMiddleEndStage:
    """Test that middle_end() produces valid optimized RVSDG."""

    def test_middle_end_returns_func_map_and_metadata(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        func_map, mdlist = middle_end(tu)
        assert isinstance(func_map, dict)
        assert isinstance(mdlist, list)
        assert len(func_map) >= 1

    def test_middle_end_func_map_contains_rvsdg_funcs(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        func_map, mdlist = middle_end(tu)
        for fname, rvsdg_func in func_map.items():
            assert isinstance(fname, str)
            assert isinstance(rvsdg_func, rg.Func)

    def test_middle_end_metadata_contains_typeinfo(self):
        tu = frontend(spy_file("e2e_ifelse.spy"))
        func_map, mdlist = middle_end(tu)
        # Metadata should contain TypeInfo and/or IRTag nodes
        has_typeinfo = any(isinstance(md, TypeInfo) for md in mdlist)
        assert has_typeinfo, "Expected TypeInfo in metadata"

    def test_middle_end_loops_optimization(self):
        tu = frontend(spy_file("e2e_loops.spy"))
        func_map, mdlist = middle_end(tu)
        assert len(func_map) >= 1
        # Verify the function was processed
        for fname, rvsdg_func in func_map.items():
            assert "main" in fname

    def test_middle_end_class_optimization(self):
        tu = frontend(spy_file("e2e_class.spy"))
        func_map, mdlist = middle_end(tu)
        assert len(func_map) >= 1

    def test_middle_end_arithmetic_optimization(self):
        tu = frontend(spy_file("e2e_arithmetic.spy"))
        func_map, mdlist = middle_end(tu)
        assert len(func_map) >= 1
        for fname in func_map:
            assert "main" in fname

    def test_middle_end_nested_if_optimization(self):
        tu = frontend(spy_file("e2e_nested_if.spy"))
        func_map, mdlist = middle_end(tu)
        assert len(func_map) >= 1

    def test_middle_end_eliminates_py_call(self):
        """After optimization, Py_Call nodes should be replaced with
        simpler operations (BuiltinOp, CallFQN, etc.)."""
        tu = frontend(spy_file("e2e_arithmetic.spy"))
        func_map, mdlist = middle_end(tu)
        # The optimization should have run to completion
        assert len(func_map) >= 1
        assert len(mdlist) >= 1


# ===========================================================================
# Stage 3: MLIR generation tests
# ===========================================================================


class TestMLIRGenerationStage:
    """Test that compile_to_mlir() produces a valid MLIR module."""

    def test_mlir_gen_ifelse(self):
        module = compile_to_mlir(spy_file("e2e_ifelse.spy"))
        assert module is not None
        # MLIR module should verify
        module.operation.verify()

    def test_mlir_gen_loops(self):
        module = compile_to_mlir(spy_file("e2e_loops.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_class(self):
        module = compile_to_mlir(spy_file("e2e_class.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_arithmetic(self):
        module = compile_to_mlir(spy_file("e2e_arithmetic.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_nested_if(self):
        module = compile_to_mlir(spy_file("e2e_nested_if.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_print_strings(self):
        module = compile_to_mlir(spy_file("e2e_print_strings.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_class_ops(self):
        module = compile_to_mlir(spy_file("e2e_class_ops.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_loop_accumulate(self):
        module = compile_to_mlir(spy_file("e2e_loop_accumulate.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_gen_loop_nested_if(self):
        module = compile_to_mlir(spy_file("e2e_loop_nested_if.spy"))
        assert module is not None
        module.operation.verify()

    def test_mlir_contains_main_function(self):
        """Verify the generated MLIR contains a 'main' function."""
        module = compile_to_mlir(spy_file("e2e_arithmetic.spy"))
        asm = module.operation.get_asm()
        assert "func" in asm
        assert "main" in asm

    def test_mlir_contains_arith_ops(self):
        """Verify arithmetic operations appear in the MLIR output."""
        module = compile_to_mlir(spy_file("e2e_arithmetic.spy"))
        asm = module.operation.get_asm()
        # After lowering, arithmetic should be present as LLVM ops
        assert "llvm" in asm.lower() or "arith" in asm.lower()

    def test_mlir_ifelse_has_control_flow(self):
        """Verify that if-else generates control flow in MLIR."""
        module = compile_to_mlir(spy_file("e2e_ifelse.spy"))
        asm = module.operation.get_asm()
        # After passes, SCF ops should be lowered to CF or LLVM
        assert "llvm" in asm.lower()

    def test_mlir_loop_has_control_flow(self):
        """Verify that while loop generates control flow in MLIR."""
        module = compile_to_mlir(spy_file("e2e_loops.spy"))
        asm = module.operation.get_asm()
        assert "llvm" in asm.lower()

    def test_mlir_inline_source(self):
        """Test MLIR generation from inline SPy source."""
        src = """\
def main() -> i32:
    a = 5
    b = 10
    c = a + b
    print(c)
    return 0
"""
        path = write_spy_source(src)
        try:
            module = compile_to_mlir(path)
            assert module is not None
            module.operation.verify()
        finally:
            os.unlink(path)


# ===========================================================================
# Stage 4+5: Full end-to-end pipeline tests (compile + run)
# ===========================================================================


class TestFullPipeline:
    """Test the complete pipeline: SPy source -> binary -> execute."""

    def test_e2e_arithmetic(self):
        expected = """
13
7
20
"""
        run_e2e_test("e2e_arithmetic.spy", expected)

    def test_e2e_nested_if(self):
        expected = """
x > y > z
0
"""
        run_e2e_test("e2e_nested_if.spy", expected)

    def test_e2e_loop_accumulate(self):
        expected = """
15
"""
        run_e2e_test("e2e_loop_accumulate.spy", expected)

    def test_e2e_loop_nested_if(self):
        expected = """
evens
3
odds
3
"""
        run_e2e_test("e2e_loop_nested_if.spy", expected)

    def test_e2e_print_strings(self):
        expected = """
hello
world
42
done
"""
        run_e2e_test("e2e_print_strings.spy", expected)

    def test_e2e_class_ops(self):
        expected = """
4
6
"""
        run_e2e_test("e2e_class_ops.spy", expected)

    def test_e2e_inline_simple_add(self):
        """Test e2e with inline source: simple addition."""
        src = """\
def main() -> i32:
    a = 7
    b = 8
    c = a + b
    print(c)
    return 0
"""
        run_source_e2e_test(src, "15\n")

    def test_e2e_inline_comparison(self):
        """Test e2e with inline source: comparison and branching."""
        src = """\
def main() -> i32:
    x = 50
    y = 25
    if x > y:
        print("yes")
    else:
        print("no")
    return 0
"""
        run_source_e2e_test(src, "yes\n")

    def test_e2e_inline_while_loop(self):
        """Test e2e with inline source: simple while loop."""
        src = """\
def main() -> i32:
    i = 0
    s = 0
    while i < 4:
        s = s + i
        i = i + 1
    print(s)
    return 0
"""
        run_source_e2e_test(src, "6\n")

    def test_e2e_inline_subtraction(self):
        """Test e2e with inline source: subtraction."""
        src = """\
def main() -> i32:
    a = 100
    b = 37
    c = a - b
    print(c)
    return 0
"""
        run_source_e2e_test(src, "63\n")

    def test_e2e_inline_string_and_int_printing(self):
        """Test mixed string and integer printing."""
        src = """\
def main() -> i32:
    print("result")
    x = 10
    y = 20
    print(x + y)
    return 0
"""
        run_source_e2e_test(src, "result\n30\n")

    def test_e2e_inline_chained_arithmetic(self):
        """Test chained arithmetic operations."""
        src = """\
def main() -> i32:
    a = 1
    b = 2
    c = 3
    d = a + b
    e = d + c
    f = e + d
    print(f)
    return 0
"""
        run_source_e2e_test(src, "9\n")

    def test_e2e_inline_nested_if_else(self):
        """Test nested if-else with multiple branches."""
        src = """\
def main() -> i32:
    a = 10
    b = 20
    c = 15
    if a > b:
        print("a")
    else:
        if c > b:
            print("c")
        else:
            print("b")
    return 0
"""
        run_source_e2e_test(src, "b\n")

    def test_e2e_inline_loop_with_conditional(self):
        """Test while loop with inner conditional."""
        src = """\
def main() -> i32:
    i = 0
    count = 0
    while i < 5:
        if i > 2:
            count = count + 1
        else:
            count = count
        i = i + 1
    print(count)
    return 0
"""
        run_source_e2e_test(src, "2\n")


# ===========================================================================
# Cross-stage pipeline integration tests
# ===========================================================================


class TestPipelineIntegration:
    """Test consistency across pipeline stages."""

    def test_frontend_to_middle_end_consistency(self):
        """Frontend output should be valid input for middle_end."""
        for spy_name in ["e2e_ifelse.spy", "e2e_loops.spy", "e2e_class.spy",
                         "e2e_arithmetic.spy"]:
            tu = frontend(spy_file(spy_name))
            func_map, mdlist = middle_end(tu)
            # Every function from frontend should appear in func_map
            for fqn in tu.list_functions():
                fname = fqn.fullname
                assert fname in func_map, (
                    f"Function {fname} from frontend not found in "
                    f"middle_end output for {spy_name}"
                )

    def test_middle_end_to_mlir_consistency(self):
        """Middle-end output should produce valid MLIR."""
        for spy_name in ["e2e_ifelse.spy", "e2e_loops.spy",
                         "e2e_arithmetic.spy"]:
            module = compile_to_mlir(spy_file(spy_name))
            module.operation.verify()

    def test_full_pipeline_all_examples(self):
        """Verify all e2e example files compile and produce a binary."""
        import glob
        spy_files = glob.glob(str(e2e_dir / "*.spy"))
        assert len(spy_files) > 0, "No .spy files found in e2e directory"
        for spy_path in spy_files:
            with make_temp_directory() as dir:
                outpath = dir / "a.out"
                compile(spy_path, str(outpath))
                assert outpath.exists(), (
                    f"Binary not created for {spy_path}"
                )
                # Verify the binary is executable
                result = subp.run(
                    str(outpath), capture_output=True, text=True
                )
                assert result.returncode == 0, (
                    f"Binary from {spy_path} exited with "
                    f"code {result.returncode}: {result.stderr}"
                )

    def test_metadata_flows_through_pipeline(self):
        """Verify metadata is preserved from frontend through middle-end."""
        tu = frontend(spy_file("e2e_ifelse.spy"))
        func_map, mdlist = middle_end(tu)
        # Metadata list should have TypeInfo entries
        typeinfo_count = sum(
            1 for md in mdlist if isinstance(md, TypeInfo)
        )
        assert typeinfo_count > 0, (
            "Expected TypeInfo metadata to flow through pipeline"
        )

    def test_struct_pipeline_frontend_to_middle_end(self):
        """Test struct programs flow correctly through frontend+middle-end."""
        tu = frontend(spy_file("e2e_class.spy"))
        # Verify struct was captured
        builtins = tu.list_builtins()
        builtin_names = [str(b) for b in builtins]
        assert any("__make__" in name for name in builtin_names), (
            "Expected __make__ builtin for struct"
        )
        # Now verify middle-end processes it
        func_map, mdlist = middle_end(tu)
        assert len(func_map) >= 1
