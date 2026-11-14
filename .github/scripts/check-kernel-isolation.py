"""Check that the SIMD kernel objects share no code with the library.

GCC and Clang objects must define no shared functions. MSVC objects may
define them with baseline instructions only.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

OBJECT_NAME = re.compile(
    r"CalculationKernel(SSE2|AVX2|AVX512|NEON)(?:\.cpp)?\.(?:o|obj)$"
)
COFF_FUNCTION = re.compile(
    r"SECT[0-9A-F]+\s+notype \(\)\s+External\s+\| (\S+)"
)
INSTRUCTION = re.compile(
    r"^\s+[0-9A-F]+:\s+((?:[0-9A-F]{2}\s+)+)([a-z?][a-z0-9?]*)(?:\s|$)"
)
# Instructions above the baseline that have no VEX prefix.
EXTRA_ISA = {"tzcnt", "lzcnt", "popcnt", "crc32"}


def run(*command):
    result = subprocess.run(
        command, capture_output=True, text=True, errors="replace"
    )
    if result.returncode != 0:
        raise SystemExit(
            f"{' '.join(command)} failed with exit status "
            f"{result.returncode}:\n{result.stderr}{result.stdout[:2000]}"
        )
    return result.stdout


def nm_command():
    """nm that takes the GNU options.  The cctools nm of the conda macOS
    compilers does not, use llvm-nm or the Xcode nm there."""
    if sys.platform != "darwin":
        return ("nm",)
    if shutil.which("llvm-nm"):
        return ("llvm-nm",)
    return ("xcrun", "nm")


def private_symbol(name, isa):
    namespace = f"xsimd_{isa.lower()}"
    if f"{namespace}::" in name or f"@{namespace}@@" in name:
        return True
    return "(anonymous namespace)" in name or "?A0x" in name


def kernel_entry(name, isa):
    entry = f"{isa.lower()}CalculationKernel"
    return name == f"diffpy::srreal::kernels::{entry}()" or name.startswith(
        f"?{entry}@kernels@srreal@diffpy@@"
    )


def check_nm(path, isa):
    output = run(
        *nm_command(),
        "--defined-only",
        "--extern-only",
        "--demangle",
        "--format=posix",
        str(path),
    )
    functions = []
    for line in output.splitlines():
        match = re.fullmatch(r"(.+) ([A-Za-z?]) [0-9a-fA-F]+(?: .*)?", line)
        if not match:
            raise ValueError(f"Unrecognized nm output: {line}")
        name, kind = match.groups()
        if kind in {"T", "W", "i", "I"}:
            functions.append(name)
    if not any(kernel_entry(name, isa) for name in functions):
        raise ValueError(f"Missing {isa} entry point in {path}")
    return [
        f"shared code symbol: {name}"
        for name in functions
        if not private_symbol(name, isa) and not kernel_entry(name, isa)
    ]


def check_coff(path, isa):
    symbols = run("dumpbin", "/nologo", "/symbols", str(path))
    functions = COFF_FUNCTION.findall(symbols)
    if not any(kernel_entry(name, isa) for name in functions):
        raise ValueError(f"Missing {isa} entry point in {path} (LTO object?)")
    shared = {
        name
        for name in functions
        if not private_symbol(name, isa) and not kernel_entry(name, isa)
    }
    output = run("dumpbin", "/nologo", "/disasm", str(path))
    bodies = {}
    current = None
    for line in output.splitlines():
        # A function starts with its decorated name.
        if line and not line[0].isspace() and line.endswith(":"):
            current = line.split(" ", 1)[0].rstrip(":")
            bodies[current] = []
        elif current is not None:
            match = INSTRUCTION.match(line)
            if match:
                bodies[current].append(match.groups())
    missing = [name for name in sorted(shared) if not bodies.get(name)]
    if missing:
        report_missing_disassembly(path, missing, shared, bodies, symbols, output)
        raise ValueError(
            f"No disassembly for {len(missing)} of {len(shared)} shared "
            f"functions in {path}, first {missing[0]}"
        )
    errors = []
    for name in sorted(shared):
        for raw, mnemonic in bodies[name]:
            # C4 and C5 start VEX instructions, 62 starts EVEX ones.
            if (
                raw.split()[0] in {"C4", "C5", "62"}
                or mnemonic.startswith("v")
                or mnemonic in EXTRA_ISA
                or "?" in mnemonic
            ):
                errors.append(f"non-baseline instruction {mnemonic}: {name}")
                break
    return errors


def report_missing_disassembly(path, missing, shared, bodies, symbols, disasm):
    """Print what dumpbin shows about functions without disassembly."""

    def show(title, lines, hits, before=3, after=8):
        print(f"  {title}: {len(hits)} matching lines", file=sys.stderr)
        for i in hits[:2]:
            for line in lines[max(0, i - before) : i + after]:
                print(f"    | {line}", file=sys.stderr)
            print("    | ...", file=sys.stderr)

    print(f"::group::Missing disassembly in {path}", file=sys.stderr)
    print(
        f"{len(shared)} shared functions, {len(missing)} without "
        f"disassembly, {len(bodies)} labels parsed",
        file=sys.stderr,
    )
    print(f"labels with instructions: {sum(map(bool, bodies.values()))}",
          file=sys.stderr)
    print("first parsed labels:", file=sys.stderr)
    for name in list(bodies)[:10]:
        print(f"  {name} ({len(bodies[name])} instructions)", file=sys.stderr)
    symlines = symbols.splitlines()
    dislines = disasm.splitlines()
    for name in missing[:5]:
        print(f"{name}:", file=sys.stderr)
        show("/symbols", symlines,
             [i for i, line in enumerate(symlines) if name in line], 2, 3)
        show("/disasm", dislines,
             [i for i, line in enumerate(dislines) if name in line])
    print("first 40 lines of /disasm:", file=sys.stderr)
    for line in dislines[:40]:
        print(f"    | {line}", file=sys.stderr)
    print("::endgroup::", file=sys.stderr)


def main():
    build_dir = Path(sys.argv[1])
    failures = []
    found = False
    for path in sorted(build_dir.rglob("CalculationKernel*")):
        match = OBJECT_NAME.fullmatch(path.name)
        if not match:
            continue
        found = True
        check = check_coff if path.suffix == ".obj" else check_nm
        errors = check(path, match[1])
        if errors:
            failures.extend(f"{path}: {error}" for error in errors)
        else:
            print(f"PASS {path}")
    if not found:
        failures.append(f"No SIMD kernel objects in {build_dir}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
