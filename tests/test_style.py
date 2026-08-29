import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".txt", ".json", ".csv"}
TEXT_NAMES = {"LICENSE", ".gitignore", ".gitkeep"}
DOCSTRING_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
ENTITY_PATTERN = re.compile(r"&[a-z]+;")
WINDOWS_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/]Users[\\/]")
POSIX_HOME_PATTERN = re.compile(r"/home/[a-z]")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail("git ls-files failed, so this check verified nothing")
    paths = [PROJECT_ROOT / line for line in result.stdout.splitlines() if line]
    if not paths:
        pytest.fail("git ls-files listed no files, so this check verified nothing")
    return [path for path in paths if path.exists()]


def python_files() -> list[Path]:
    return [path for path in tracked_files() if path.suffix == ".py"]


def text_files() -> list[Path]:
    return [
        path for path in tracked_files() if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES
    ]


def test_the_scan_actually_finds_the_project():
    assert len(python_files()) >= 15
    assert len(text_files()) >= 20


def test_python_files_carry_no_comments():
    offenders = []
    for path in python_files():
        source = path.read_text(encoding="utf-8")
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                offenders.append(f"{path.name}:{token.start[0]}")
    assert offenders == []


def test_python_files_carry_no_docstrings():
    offenders = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, DOCSTRING_NODES) and ast.get_docstring(node) is not None:
                offenders.append(path.name)
    assert offenders == []


def test_every_tracked_text_file_is_plain_ascii():
    offenders = []
    for path in text_files():
        content = path.read_bytes()
        for index, value in enumerate(content):
            if value > 127:
                offenders.append(f"{path.name} byte {index}")
                break
    assert offenders == []


def test_no_html_entities_that_render_as_typography():
    offenders = [
        path.name
        for path in text_files()
        if ENTITY_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_absolute_paths_and_no_personal_email():
    offenders = []
    for path in text_files():
        content = path.read_text(encoding="utf-8")
        if WINDOWS_PATH_PATTERN.search(content) or POSIX_HOME_PATTERN.search(content):
            offenders.append(f"{path.name} holds an absolute path")
        if EMAIL_PATTERN.search(content):
            offenders.append(f"{path.name} holds an email address")
    assert offenders == []
