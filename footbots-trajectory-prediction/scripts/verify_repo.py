from __future__ import annotations

import py_compile
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "README.md",
    ROOT / ".gitignore",
    ROOT / "requirements.txt",
    ROOT / "pyproject.toml",
    ROOT / "src" / "__init__.py",
    ROOT / "configs" / "homography.example.json",
]
BLOCKED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".pt", ".pth", ".onnx"}
ABSOLUTE_WINDOWS_PATH = re.compile(
    r"(?<![A-Za-z0-9_])[A-Za-z]:\\(?:Users|Windows|Program Files|ProgramData|Projects|workspace)\\",
    re.IGNORECASE,
)


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED:
        if not path.exists():
            errors.append(f"必須ファイルがありません: {path.relative_to(ROOT)}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append(f"大容量・生成ファイルを削除してください: {rel}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"10MBを超えるファイルです: {rel}")
        if path.suffix.lower() in {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".cff"}:
            text = path.read_text(encoding="utf-8")
            if ABSOLUTE_WINDOWS_PATH.search(text):
                errors.append(f"Windows絶対パスが含まれています: {rel}")

    for path in sorted((ROOT / "src").glob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"Python構文エラー: {path.name}: {exc.msg}")

    if errors:
        print("公開前チェック: FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("公開前チェック: OK")
    print(f"Pythonファイル: {len(list((ROOT / 'src').glob('*.py')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
