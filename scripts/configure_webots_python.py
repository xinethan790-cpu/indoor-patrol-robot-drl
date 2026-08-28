#!/usr/bin/env python3
"""Bind every Webots Python controller to the active virtual environment."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repository_root = Path(__file__).absolute().parents[1]
    python_command = Path(sys.executable).absolute()
    virtual_environment = repository_root / ".venv"
    if virtual_environment not in python_command.parents:
        raise SystemExit(
            "Run this script with the project interpreter: "
            ".venv/bin/python scripts/configure_webots_python.py"
        )

    configured: list[Path] = []
    for controller_directory in sorted((repository_root / "controllers").iterdir()):
        entrypoint = controller_directory / f"{controller_directory.name}.py"
        if not entrypoint.is_file():
            continue
        runtime_file = controller_directory / "runtime.ini"
        runtime_file.write_text(
            f"[python]\nCOMMAND = {python_command}\n",
            encoding="utf-8",
        )
        configured.append(runtime_file.relative_to(repository_root))

    print(f"Configured {len(configured)} Webots controllers with {python_command}")
    for runtime_file in configured:
        print(f"- {runtime_file}")


if __name__ == "__main__":
    main()
