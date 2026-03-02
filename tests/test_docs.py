"""Tests for documentation code examples.

This module extracts and tests code examples from markdown files in AGENTS.md
and docs/*.md.

Markdown format:
- Code blocks with ```python are extracted and executed
- HTML comments <!-- test:skip --> before a code block skip that block
- HTML comments <!-- test:setup ... --> contain hidden setup code
- HTML comments <!-- test:teardown ... --> contain hidden teardown code
"""

import re
from pathlib import Path
from typing import List, Tuple

import pytest


def extract_code_blocks(markdown_content: str) -> List[Tuple[str, bool]]:
    """Extract Python code blocks from markdown content.

    Returns:
        List of (code, should_skip) tuples.
    """
    blocks: List[Tuple[str, bool]] = []

    # Extract hidden setup code from HTML comments
    setup_pattern = r"<!--\s*test:setup\s*\n(.*?)\n\s*-->"
    setup_matches = re.findall(setup_pattern, markdown_content, re.DOTALL)
    setup_code = "\n".join(setup_matches)

    # Find all code blocks with skip markers
    # Pattern matches optional skip comment followed by python code block
    pattern = r"(?:<!--\s*test:skip\s*-->\s*)?(```python\n(.*?)\n```)"
    matches = list(re.finditer(pattern, markdown_content, re.DOTALL))

    for match in matches:
        full_match = match.group(0)
        code = match.group(2)
        should_skip = full_match.strip().startswith("<!-- test:skip")

        if setup_code and not should_skip:
            code = setup_code + "\n" + code

        blocks.append((code, should_skip))

    return blocks


def run_code_block(code: str, filename: str, block_index: int) -> None:
    """Execute a code block and report errors with context."""
    try:
        exec(code, {"__name__": "__main__"})
    except Exception as e:
        raise AssertionError(
            f"Code block {block_index + 1} in {filename} failed:\n"
            f"Code:\n{code}\n\nError: {e}"
        ) from e


def get_markdown_files() -> List[Path]:
    """Get all markdown files to test."""
    root = Path(__file__).parent.parent
    files = [
        root / "AGENTS.md",
        root / "docs" / "README.md",
        root / "docs" / "data.md",
        root / "docs" / "flattener.md",
        root / "docs" / "util.md",
        root / "docs" / "examples.md",
        # Japanese documentation
        root / "docs" / "data.ja.md",
        root / "docs" / "flattener.ja.md",
        root / "docs" / "util.ja.md",
        root / "docs" / "examples.ja.md",
    ]
    return [f for f in files if f.exists()]


@pytest.mark.parametrize(
    "md_file", get_markdown_files(), ids=lambda p: p.name
)
def test_markdown_code_blocks(md_file: Path) -> None:
    """Test all Python code blocks in a markdown file."""
    content = md_file.read_text(encoding="utf-8")
    blocks = extract_code_blocks(content)

    if not blocks:
        pytest.skip(f"No Python code blocks found in {md_file.name}")

    for i, (code, should_skip) in enumerate(blocks):
        if should_skip:
            continue
        run_code_block(code, md_file.name, i)
