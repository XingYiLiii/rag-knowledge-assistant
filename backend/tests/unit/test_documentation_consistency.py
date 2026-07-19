"""Lightweight checks that portfolio documentation remains present and discoverable."""

from pathlib import Path


def test_project_documentation_files_exist() -> None:
    """README and the architecture/RAG design guides are committed at documented paths."""
    repository_root = Path(__file__).resolve().parents[3]

    assert (repository_root / "README.md").is_file()
    assert (repository_root / "docs" / "architecture.md").is_file()
    assert (repository_root / "docs" / "rag-design.md").is_file()


def test_readme_contains_major_portfolio_sections() -> None:
    """README exposes the core setup, API, quality, safety, and planning sections."""
    repository_root = Path(__file__).resolve().parents[3]
    readme = (repository_root / "README.md").read_text(encoding="utf-8")

    for heading in (
        "## 项目介绍",
        "## 核心功能",
        "## 技术栈",
        "## 快速启动",
        "## API示例",
        "## 测试",
        "## RAG评测",
        "## 安全说明",
        "## 已知限制",
        "## Roadmap",
    ):
        assert heading in readme
