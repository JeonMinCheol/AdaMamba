from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess


def preload_ijit_stub() -> bool:
    base_dir = Path(__file__).resolve().parent.parent
    source_path = base_dir / "compat" / "ijit_stub.c"
    library_path = base_dir / "compat" / "libijit_stub.so"

    should_build = source_path.exists() and (
        not library_path.exists()
        or source_path.stat().st_mtime > library_path.stat().st_mtime
    )

    if should_build:
        compiler = shutil.which("gcc")
        if compiler is None:
            return False

        try:
            subprocess.run(
                [compiler, "-shared", "-fPIC", "-O2", str(source_path), "-o", str(library_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False

    if not library_path.exists():
        return False

    try:
        ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False

    return True
