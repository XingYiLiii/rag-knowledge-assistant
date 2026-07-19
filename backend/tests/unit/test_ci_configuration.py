"""Static checks for the GitHub Actions continuous-integration workflow."""

from pathlib import Path


def test_ci_workflow_declares_required_runner_python_and_checks() -> None:
    """The workflow uses the supported Python version and local-equivalent test commands."""
    repository_root = Path(__file__).resolve().parents[3]
    workflow = (repository_root / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "push:" in workflow
    assert "pull_request:" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert 'python-version: "3.11"' in workflow
    assert "working-directory: backend" in workflow
    assert 'python -m pip install ".[dev]"' in workflow
    assert "python -m ruff check ." in workflow
    assert "python -m ruff format --check ." in workflow
    assert "python -m pytest" in workflow


def test_ci_workflow_does_not_require_secrets_or_local_environment_file() -> None:
    """The CI workflow remains runnable without API keys, secrets, or a local .env file."""
    repository_root = Path(__file__).resolve().parents[3]
    workflow = (repository_root / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")

    assert "secrets." not in workflow
    assert ".env" not in workflow
    assert "APP_ENV: test" in workflow
