"""Packaging contract tests for the finance package (mirror of core policy)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import fuzzymatch_finance

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _load_pyproject() -> dict:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


class TestVersionSingleSourcing:
    def test_version_is_dynamic_not_duplicated(self) -> None:
        project = _load_pyproject()["project"]
        assert "version" in project.get("dynamic", [])
        assert "version" not in project

    def test_runtime_version_matches_installed_distribution(self) -> None:
        from importlib.metadata import version

        assert fuzzymatch_finance.__version__ == version("fuzzymatch-finance")


class TestDistributionMetadata:
    def test_readme_wired_as_long_description(self) -> None:
        assert _load_pyproject()["project"].get("readme") == "README.md"

    def test_license_declared(self) -> None:
        lic = _load_pyproject()["project"]["license"]
        assert lic == "MIT" or (isinstance(lic, dict) and lic.get("text") == "MIT")


class TestCoverageGate:
    def test_fail_under_configured_at_or_above_95(self) -> None:
        tool = _load_pyproject().get("tool", {})
        fail_under = tool.get("coverage", {}).get("report", {}).get("fail_under")
        assert isinstance(fail_under, (int, float))
        assert fail_under >= 95


class TestCiWorkflow:
    def _workflow(self) -> dict:
        import pytest

        yaml = pytest.importorskip("yaml")

        wf_path = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        assert wf_path.exists(), "ci.yml must exist"
        wf = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        assert isinstance(wf, dict)
        return wf

    def test_defines_lint_typecheck_test_jobs(self) -> None:
        jobs = self._workflow().get("jobs", {})
        assert {"lint", "typecheck", "test"} <= set(jobs)

    def test_test_job_runs_on_windows_and_ubuntu(self) -> None:
        test_job = self._workflow()["jobs"]["test"]
        os_list = test_job["strategy"]["matrix"]["os"]
        joined = ";".join(str(os_name) for os_name in os_list).lower()
        assert "windows" in joined
        assert "ubuntu" in joined


class TestCoreDependencyContract:
    """Finance must consume the core through its public typed surface."""

    def test_declares_core_dependency_with_version_floor(self) -> None:
        deps = _load_pyproject()["project"]["dependencies"]
        assert any(d.startswith("fuzzymatch>=") for d in deps)


class TestLocalResearchIgnored:
    """Finance repository must ignore local research and prune it from distributions."""

    def test_local_research_in_gitignore(self) -> None:
        gitignore_path = REPO_ROOT / ".gitignore"
        assert gitignore_path.exists(), ".gitignore must exist"
        content = gitignore_path.read_text(encoding="utf-8")
        assert "docs/" in content or "docs" in content, "docs must be gitignored"
        assert "experiments/" in content or "experiments" in content, (
            "experiments must be gitignored"
        )

    def test_local_research_pruned_in_manifest(self) -> None:
        manifest_path = REPO_ROOT / "MANIFEST.in"
        assert manifest_path.exists(), "MANIFEST.in must exist"
        content = manifest_path.read_text(encoding="utf-8")
        assert "prune docs" in content, "MANIFEST.in must prune docs"
        assert "prune experiments" in content, "MANIFEST.in must prune experiments"
