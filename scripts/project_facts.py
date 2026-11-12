"""Read-only inventory of the CyberNN project, so documentation states only verified facts."""

import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ["numpy", "pandas", "scikit-learn", "matplotlib", "torch", "streamlit",
            "pytest", "fastapi", "uvicorn", "pydantic"]


def installed_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def read_text_any(path):
    """Read a text file, noticing the UTF-16 encoding that PowerShell's > redirect can produce."""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore"), "utf-16 (pip may fail to read this)"
    return raw.decode("utf-8", errors="ignore"), "utf-8/ascii"


def show_dir(label, folder, pattern="*", recursive=False):
    print(f"\n== {label} ==")
    folder = ROOT / folder
    if not folder.exists():
        print("(folder not found)")
        return
    paths = folder.rglob(pattern) if recursive else folder.glob(pattern)
    files = sorted(p for p in paths if p.is_file() and "__pycache__" not in p.parts)
    if not files:
        print("(no files)")
    for p in files:
        print(f"{str(p.relative_to(ROOT)):70s} {p.stat().st_size / 1024:>12,.1f} KB")


print("== Python ==")
print(sys.version.split()[0], "|", sys.executable)

req_path = ROOT / "requirements.txt"
req_names = set()
print("\n== requirements.txt ==")
if req_path.exists():
    text, encoding = read_text_any(req_path)
    print("encoding:", encoding)
    for line in text.splitlines():
        name = line.split("==")[0].strip().lower().replace("_", "-")
        if name:
            req_names.add(name)
    print("entries:", len(req_names))
else:
    print("(requirements.txt not found)")

print("\n== Key packages: installed version | listed in requirements.txt? ==")
for pkg in PACKAGES:
    version = installed_version(pkg)
    listed = "yes" if pkg in req_names else "no"
    note = "   <-- installed but not listed" if version and pkg not in req_names else ""
    print(f"{pkg:14s} {version or 'not installed':16s} {listed}{note}")

print("\n== Git ==")
git = shutil.which("git")
print("git on PATH:", git or "not found")
if git:
    result = subprocess.run([git, "--version"], capture_output=True, text=True, timeout=10)
    print(result.stdout.strip())
print(".git folder present:", (ROOT / ".git").exists())

print("\n== Top-level documentation files ==")
for name in ["README.md", "LICENSE", ".gitignore", "requirements.txt", "docs"]:
    print(f"{name:18s} {'present' if (ROOT / name).exists() else 'missing'}")

total_lines = 0
print("\n== Source modules (lines) ==")
for folder in ["src/cybernn", "app"]:
    for p in sorted((ROOT / folder).glob("*.py")) if (ROOT / folder).exists() else []:
        n = len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
        total_lines += n
        print(f"{str(p.relative_to(ROOT)):45s} {n:>6} lines")
print(f"total in src and app: {total_lines} lines")

scripts = sorted(p.name for p in (ROOT / "scripts").glob("*.py")) if (ROOT / "scripts").exists() else []
print(f"\n== scripts ({len(scripts)}) ==")
print(", ".join(scripts))

show_dir("tests", "tests", recursive=True)
show_dir("models", "models")
show_dir("reports", "reports")
show_dir("data (raw, processed, test_uploads)", "data", recursive=True)