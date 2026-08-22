from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def load_repo_hygiene_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "repo_hygiene_check.py"
    spec = importlib.util.spec_from_file_location("repo_hygiene_check", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repo_hygiene_check = load_repo_hygiene_module()


def write_file(root: Path, relative_path: str, content: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_load_toml_module_falls_back_to_tomli(monkeypatch) -> None:
    calls: list[str] = []
    tomli_module = SimpleNamespace(loads=lambda text: {"ok": text})

    def fake_import_module(name: str):
        calls.append(name)
        if name == "tomllib":
            raise ModuleNotFoundError(name)
        if name == "tomli":
            return tomli_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(repo_hygiene_check.importlib, "import_module", fake_import_module)

    loaded = repo_hygiene_check._load_toml_module()

    assert loaded is tomli_module
    assert calls == ["tomllib", "tomli"]


def test_workflow_check_flags_missing_top_level_on(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/bad.yml",
        "name: Broken\non:\npermissions:\n  contents: read\n  pull_request:\n    paths:\n      - 'victor/**'\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("non-empty top-level trigger" in finding.message for finding in findings)


def _nightly_workflow(*, install_run: str, notify_run: str) -> str:
    install = "\n".join(f"          {line}" for line in install_run.splitlines())
    notify = "\n".join(f"          {line}" for line in notify_run.splitlines())
    return f"""
name: Nightly Develop Suite
on: workflow_dispatch
jobs:
  full-suite:
    steps:
      - name: Install dependencies
        run: |
{install}
  report:
    steps:
      - name: Notify on failure (tracking issue)
        run: |
{notify}
""".lstrip()


def test_nightly_workflow_requires_local_codegraph_install(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/ci-develop-nightly.yml",
        _nightly_workflow(
            install_run='pip install -e ".[dev,ci,api]"',
            notify_run='gh issue list --repo "$GITHUB_REPOSITORY"',
        ),
    )

    findings = repo_hygiene_check.check_nightly_workflow_contract(tmp_path)

    assert any("must install the in-repo victor-codegraph" in item.message for item in findings)


def test_nightly_workflow_notification_requires_explicit_repo(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/ci-develop-nightly.yml",
        _nightly_workflow(
            install_run="pip install --no-cache-dir -e ./victor-codegraph",
            notify_run="\n".join(
                (
                    "gh issue list --state open",
                    'gh issue comment "$num" --body-file /tmp/body.md',
                    'gh issue create --title "$title"',
                )
            ),
        ),
    )

    findings = repo_hygiene_check.check_nightly_workflow_contract(tmp_path)

    assert len(findings) == 3
    assert all("must pass an explicit repository" in item.message for item in findings)


def test_nightly_workflow_contract_accepts_production_shape(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/ci-develop-nightly.yml",
        _nightly_workflow(
            install_run="pip install --no-cache-dir -e ./victor-codegraph",
            notify_run="\n".join(
                (
                    'num=$(gh issue list --repo "$GITHUB_REPOSITORY" --state open)',
                    'gh issue comment "$num" --repo "$GITHUB_REPOSITORY"',
                    'gh issue create --repo "$GITHUB_REPOSITORY" --title "$title"',
                )
            ),
        ),
    )

    assert repo_hygiene_check.check_nightly_workflow_contract(tmp_path) == []


def test_vertical_extras_require_real_package_dependencies(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = []
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("extracted vertical extra 'coding'" in finding.message for finding in findings)


def test_legacy_noop_vertical_extras_are_rejected(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
rag = []
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("legacy no-op vertical extra 'rag'" in finding.message for finding in findings)


def test_self_referential_ci_extra_is_rejected(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
ci = ["victor-ai[dev]", "pytest-split>=0.8"]
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("must not self-reference victor-ai" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_legacy_entry_point_guidance(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]

[project.entry-points."victor.plugins"]
benchmark = "victor.benchmark:plugin"

# Requirements for external verticals:
#   1. Must inherit from victor.core.verticals.VerticalBase
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any(
        "victor_contracts.VerticalBase" in finding.message or "victor.plugins" in finding.message
        for finding in findings
    )


def test_primary_vertical_contract_docs_reject_legacy_victor_verticals_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "victor-contracts/README.md",
        """
[project.entry-points."victor.verticals"]
security = "victor_security:SecurityVertical"
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor.plugins" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_nested_victor_plugins_groups(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/verticals/best_practices.md",
        """
[project.entry-points."victor.plugins.my_company"]
tool = "my_package.plugin:plugin"
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("nested victor.plugins.* groups" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_legacy_sdk_extension_groups(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "victor-contracts/VERTICAL_DEVELOPMENT.md",
        """
[project.entry-points."victor.sdk.protocols"]
tools = "my_vertical.protocols:ToolProvider"

[project.entry-points."victor.sdk.capabilities"]
search = "my_vertical.capabilities:SearchCapability"

[project.entry-points."victor.sdk.validators"]
path = "my_vertical.validators:validate_path"
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor.extension.protocols" in finding.message for finding in findings)
    assert any("victor.extension.capabilities" in finding.message for finding in findings)
    assert any("victor.extension.validators" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_sdk_first_wording(
    tmp_path: Path,
) -> None:
    legacy_first = "SDK" "-first"
    legacy_only = "SDK" "-only"
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/demos/10-minute-vertical-demo.md",
        f"Create an {legacy_first} assistant with an {legacy_only} definition layer.\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("contract-first wording" in finding.message for finding in findings)
    assert any("contract-only wording" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_compat_sdk_import_namespace(
    tmp_path: Path,
) -> None:
    legacy_import = "from victor" "_sdk import VerticalBase, register_vertical"
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/guides/vertical-quickstart.md",
        f"""
{legacy_import}

Must inherit from victor.core.verticals.VerticalBase in every external vertical.
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_contracts import namespace" in finding.message for finding in findings)
    assert any("victor_contracts.VerticalBase" in finding.message for finding in findings)


def test_extended_vertical_authoring_guide_is_a_primary_contract_surface(
    tmp_path: Path,
) -> None:
    legacy_import = "from victor" "_sdk import VerticalBase"
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/development/extending/verticals.md",
        f"{legacy_import}\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_contracts import namespace" in finding.message for finding in findings)


def test_edge_model_guide_is_a_primary_contract_surface(tmp_path: Path) -> None:
    legacy_import = "from victor" "_sdk import VerticalBase"
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/guides/EDGE_MODEL.md",
        f"{legacy_import}\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_contracts import namespace" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_framework_shim_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "pyproject.toml",
        """
[project]
name = "victor-ai"

[project.optional-dependencies]
coding = ["victor-coding>=0.6.0"]
research = ["victor-research>=0.6.0"]
devops = ["victor-devops>=0.6.0"]
verticals = ["victor-coding>=0.6.0", "victor-research>=0.6.0", "victor-devops>=0.6.0"]
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "victor-contracts/VERTICAL_DEVELOPMENT.md",
        """
from victor.framework.vertical_base import VerticalBase
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor.framework.vertical_base" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_core_register_vertical_imports(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/verticals/api_reference.md",
        """
from victor.core.verticals.registration import register_vertical

# Public API reference
victor.core.verticals.registration.register_vertical
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any(
        "import register_vertical from victor_contracts" in finding.message for finding in findings
    )
    assert any("victor_contracts.register_vertical" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_framework_extensions_definition_imports(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "victor-contracts/VERTICAL_DEVELOPMENT.md",
        """
from victor.framework.extensions import VerticalBase
from victor.framework.extensions import StageDefinition
from victor.framework.extensions import VerticalConfig
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_contracts.VerticalBase" in finding.message for finding in findings)
    assert any("victor_contracts.StageDefinition" in finding.message for finding in findings)
    assert any("victor_contracts.VerticalConfig" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_legacy_entry_point_lookup_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/verticals/api_reference.md",
        """
from victor.framework.entry_point_registry import get_entry_point

coding = get_entry_point("victor.verticals", "coding")
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("entry-point lookups" in finding.message for finding in findings)


def test_primary_vertical_contract_docs_reject_runtime_vertical_authoring_examples(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "docs/reference/verticals/index.md",
        """
from victor.verticals import VerticalBase
from victor.verticals.base import StageDefinition
from victor.verticals.base import VerticalConfig
from victor.verticals import VerticalRegistry

VerticalRegistry.register(MyVertical)
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("victor_contracts.VerticalBase" in finding.message for finding in findings)
    assert any("victor_contracts.StageDefinition" in finding.message for finding in findings)
    assert any("victor_contracts.VerticalConfig" in finding.message for finding in findings)
    assert any("VictorPlugin/context.register_vertical" in finding.message for finding in findings)


def test_banned_repo_url_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "victor/verticals/contrib/coding/victor-vertical.toml",
        'repository = "https://github.com/vijay-singh/codingagent"\n',
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("banned repo URL" in finding.message for finding in findings)


def test_uppercase_roadmap_markdown_link_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(tmp_path, "README.md", "[Roadmap](ROADMAP.md)\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("non-canonical roadmap link target" in finding.message for finding in findings)


def test_archived_doc_requires_banner(tmp_path: Path, monkeypatch) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(tmp_path, "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md", "# Missing banner\n")

    # ARCHIVED_DOC_BANNERS is empty after #518's doc consolidation; patch to
    # test the banner-check logic still fires when a listed doc lacks its banner.
    monkeypatch.setattr(
        repo_hygiene_check,
        "ARCHIVED_DOC_BANNERS",
        {"docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md": "> **Archived**"},
    )
    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("required banner text" in finding.message for finding in findings)


def test_legacy_monolithic_protocol_module_is_flagged(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(tmp_path, "victor/agent/protocols.py", "class Legacy: ...\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("legacy monolithic protocol module" in finding.message for finding in findings)


def test_makefile_lint_gate_rejects_advisory_mypy(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs: {}\n",
    )
    write_file(
        tmp_path,
        "Makefile",
        "lint:\n\truff check victor tests\n\tmypy victor --ignore-missing-imports || true\n",
    )
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("must not suppress mypy failure" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_trivy_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs: {}\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Trivy step" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_pip_audit_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n| Dependency audit | Blocking |\n| Bandit (SAST) | Blocking |\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking pip-audit step" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_bandit_high_path(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n      - uses: pypa/gh-action-pip-audit@v1.0.8\n",
    )
    write_file(tmp_path, ".github/workflows/ci-fast.yml", "name: CI\non: push\njobs: {}\n")
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n| Dependency audit | Blocking |\n| Bandit (SAST) | Blocking |\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Bandit step" in finding.message for finding in findings)


def test_security_baseline_accepts_shell_based_blocking_pip_audit(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@0.33.1\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@0.33.1\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n      - name: Run pip-audit\n        run: |\n          pip install pip-audit\n          pip-audit --format json --output audit-report.json\n      - name: Run Bandit\n        run: |\n          bandit -r victor/ --severity-level high --confidence-level high\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n| Dependency audit | Blocking |\n| Bandit (SAST) | Blocking |\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert not any("blocking pip-audit step" in finding.message for finding in findings)


def test_security_baseline_requires_ignore_unfixed_for_blocking_trivy_path(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n",
    )
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs: {}\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n**Blocking today**\n**Advisory today**\n### Current Thresholds\nTrivy filesystem scan\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("blocking Trivy step" in finding.message for finding in findings)


def test_security_baseline_requires_threshold_docs(tmp_path: Path) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs: {}\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(tmp_path, "SECURITY.md", "## Current CI Enforcement Baseline\n")

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("Current Thresholds section" in finding.message for finding in findings)


def test_security_baseline_requires_blocking_and_advisory_summary_docs(
    tmp_path: Path,
) -> None:
    write_file(tmp_path, ".github/workflows/test.yml", "name: OK\non: push\n")
    write_file(
        tmp_path,
        ".github/workflows/ci-fast.yml",
        "name: CI\non: push\njobs:\n  sec:\n    steps:\n      - uses: aquasecurity/trivy-action@master\n        with:\n          severity: CRITICAL\n          exit-code: '1'\n          ignore-unfixed: 'true'\n",
    )
    write_file(
        tmp_path,
        ".github/workflows/security.yml",
        "name: Security\non: push\njobs: {}\n",
    )
    write_file(tmp_path, "Makefile", "lint:\n\tmypy victor\n")
    write_file(
        tmp_path,
        "docs/COMPREHENSIVE_IMPROVEMENT_ROADMAP.md",
        "Archived planning document\n",
    )
    write_file(
        tmp_path,
        "SECURITY.md",
        "## Current CI Enforcement Baseline\n### Current Thresholds\n",
    )

    findings = repo_hygiene_check.run_checks(tmp_path)

    assert any("Blocking today" in finding.message for finding in findings)
    assert any("Advisory today" in finding.message for finding in findings)


def test_deprecation_contract_requires_warning_and_docs(
    tmp_path: Path,
) -> None:
    write_file(
        tmp_path,
        "victor/framework/workflows/nodes.py",
        """
_DEPRECATED_ALIAS_MAP = {"TeamNode": object}

def __getattr__(name):
    warnings.warn(
        "TeamNode is deprecated; use TeamStep instead. "
        "It will be removed in a future release.",
        DeprecationWarning,
    )
        """.strip() + "\n",
    )
    write_file(tmp_path, "CHANGELOG.md", "## [Unreleased] (develop)\n")
    write_file(tmp_path, "docs/development/deprecation-inventory-2026-03-03.md", "")
    write_file(tmp_path, "docs/architecture/migration.md", "")

    findings = repo_hygiene_check.check_deprecation_contract(
        tmp_path, repo_hygiene_check.TEAM_NODE_DEPRECATION_CONTRACT
    )

    assert any("target removal version" in finding.message for finding in findings)
    assert any("target removal date" in finding.message for finding in findings)
    assert any("migration guide" in finding.message for finding in findings)
    assert any("inventory entry is missing" in finding.message for finding in findings)
    assert any("changelog entry is missing" in finding.message for finding in findings)


def test_deprecation_contract_accepts_complete_teamnode_contract(
    tmp_path: Path,
) -> None:
    contract = repo_hygiene_check.TEAM_NODE_DEPRECATION_CONTRACT
    warning_text = (
        "TeamNode is deprecated; use TeamStep instead. "
        "This compatibility alias remains supported through v0.9.0 (2027-03-31) "
        "and will be removed after that milestone. "
        "See docs/architecture/migration.md for migration guidance."
    )
    for rel_path in contract.runtime_files:
        write_file(
            tmp_path,
            rel_path.as_posix(),
            warning_text + "\n",
        )
    write_file(
        tmp_path,
        "docs/development/deprecation-inventory-2026-03-03.md",
        """
| `TeamNode*` workflow compatibility aliases | source | `TeamStep*` workflow names | Architecture Lead | `v0.9.0` | `2027-03-31` |
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "CHANGELOG.md",
        """
## [Unreleased] (develop)

### Deprecated
- **`TeamNode*` workflow compatibility aliases**
  - To be removed in: `v0.9.0`
  - Target removal date: `2027-03-31`
  - Replacement: `TeamStep*` workflow names
  - Compatibility shim status: warning-backed aliases remain supported through `v0.9.0`
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "docs/architecture/migration.md",
        """
Legacy `TeamNode*` workflow names are deprecated.
Use `TeamStep*` names during the migration window.
Removal target: `v0.9.0` (`2027-03-31`).
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.check_deprecation_contract(tmp_path, contract)

    assert findings == []


def test_public_shim_contracts_register_workflowgraph_alias() -> None:
    labels = {contract.label for contract in repo_hygiene_check.PUBLIC_SHIM_DEPRECATION_CONTRACTS}

    assert "FrameworkShim compatibility surface" in labels
    assert "TeamNode*" in labels
    assert "WorkflowGraph alias from victor.workflows.graph" in labels


def test_deprecation_contract_accepts_complete_workflowgraph_contract(
    tmp_path: Path,
) -> None:
    contract = repo_hygiene_check.WORKFLOW_GRAPH_ALIAS_DEPRECATION_CONTRACT
    warning_text = (
        "WorkflowGraph from victor.workflows.graph is deprecated. "
        "Use BasicWorkflowGraph for the simple graph container or "
        "victor.workflows.graph_dsl.WorkflowGraph for the typed workflow DSL. "
        "This warning-backed alias remains supported through v0.8.0 (2026-12-31). "
        "See docs/architecture/migration.md for migration guidance."
    )
    for rel_path in contract.runtime_files:
        write_file(
            tmp_path,
            rel_path.as_posix(),
            warning_text + "\n",
        )
    write_file(
        tmp_path,
        "docs/development/deprecation-inventory-2026-03-03.md",
        """
| `WorkflowGraph` alias from `victor.workflows.graph` | source | `BasicWorkflowGraph` or `victor.workflows.graph_dsl.WorkflowGraph` | Architecture Lead | `v0.8.0` | `2026-12-31` |
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "CHANGELOG.md",
        """
## [Unreleased] (develop)

### Deprecated
- **`WorkflowGraph` alias from `victor.workflows.graph`**
  - To be removed in: `v0.8.0`
  - Target removal date: `2026-12-31`
  - Replacement: `BasicWorkflowGraph` for the simple container or `victor.workflows.graph_dsl.WorkflowGraph` for the typed DSL
  - Compatibility shim status: warning-backed alias remains supported through `v0.8.0`
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "docs/architecture/migration.md",
        """
Legacy `WorkflowGraph` import from `victor.workflows.graph` is deprecated.
Use `BasicWorkflowGraph` for the simple container or `victor.workflows.graph_dsl.WorkflowGraph` for the typed DSL.
Removal target: `v0.8.0` (`2026-12-31`).
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.check_deprecation_contract(tmp_path, contract)

    assert findings == []


def test_deprecation_contract_accepts_complete_frameworkshim_contract(
    tmp_path: Path,
) -> None:
    contract = repo_hygiene_check.FRAMEWORK_SHIM_DEPRECATION_CONTRACT
    warning_text = (
        "FrameworkShim is deprecated. Use Agent.create() from the Framework API instead. "
        "Internal surface layers should compose AgentFactory / AgentCreationFactory. "
        "This warning-backed compatibility shim remains supported through v1.0.0 "
        "(2027-06-30). See docs/architecture/migration.md for migration guidance."
    )
    compat_export_text = (
        "victor.framework.FrameworkShim is deprecated. "
        "Use Agent.create() from the Framework API instead. "
        "This warning-backed compatibility export remains supported through v1.0.0 "
        "(2027-06-30). See docs/architecture/migration.md for migration guidance."
    )
    for rel_path in contract.runtime_files:
        text = compat_export_text if rel_path.name == "__init__.py" else warning_text
        write_file(tmp_path, rel_path.as_posix(), text + "\n")
    write_file(
        tmp_path,
        "docs/development/deprecation-inventory-2026-03-03.md",
        """
| `FrameworkShim` compatibility surface | `victor/framework/shim.py`, `victor/framework/__init__.py` | `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition | Architecture Lead | `v1.0.0` | `2027-06-30` |
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "CHANGELOG.md",
        """
## [Unreleased] (develop)

### Deprecated
- **`FrameworkShim` compatibility surface**
  - To be removed in: `v1.0.0`
  - Target removal date: `2027-06-30`
  - Replacement: `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition
  - Compatibility shim status: warning-backed shim remains supported through `v1.0.0`
        """.strip() + "\n",
    )
    write_file(
        tmp_path,
        "docs/architecture/migration.md",
        """
Legacy `FrameworkShim` usage is deprecated.
Use `Agent.create()` for public callers or `AgentFactory` / `AgentCreationFactory` for internal composition.
Removal target: `v1.0.0` (`2027-06-30`).
        """.strip() + "\n",
    )

    findings = repo_hygiene_check.check_deprecation_contract(tmp_path, contract)

    assert findings == []


def test_deprecation_contract_supports_non_teamnode_shims(tmp_path: Path) -> None:
    contract = repo_hygiene_check.DeprecationContract(
        label="LegacyFoo",
        runtime_files=(Path("victor/foo/shim.py"),),
        activation_needles=("LegacyFoo",),
        runtime_requirements=(
            repo_hygiene_check.TextRequirement(
                needle="v1.2.0",
                missing_message="deprecation warning must publish the target removal version",
            ),
            repo_hygiene_check.TextRequirement(
                needle="2027-06-30",
                missing_message="deprecation warning must publish the target removal date",
            ),
            repo_hygiene_check.TextRequirement(
                needle="docs/migration/foo.md",
                missing_message="deprecation warning must point at the migration guide",
            ),
        ),
        inventory_path=Path("docs/development/foo-inventory.md"),
        inventory_requirements=(
            repo_hygiene_check.TextRequirement(
                needle="LegacyFoo",
                missing_message="deprecation inventory entry is missing the shim family name",
            ),
        ),
        changelog_path=Path("CHANGELOG.md"),
        changelog_requirements=(
            repo_hygiene_check.TextRequirement(
                needle="## [Unreleased]",
                missing_message="deprecation changelog entry is missing the Unreleased section",
            ),
            repo_hygiene_check.TextRequirement(
                needle="LegacyFoo",
                missing_message="deprecation changelog entry is missing the shim family name",
            ),
        ),
        migration_path=Path("docs/migration/foo.md"),
        migration_requirements=(
            repo_hygiene_check.TextRequirement(
                needle="LegacyFoo",
                missing_message="migration guidance is missing the shim family name",
            ),
        ),
    )
    write_file(
        tmp_path,
        "victor/foo/shim.py",
        """
warnings.warn(
    "LegacyFoo is deprecated. Supported through v1.2.0 (2027-06-30). "
    "See docs/migration/foo.md for migration guidance.",
    DeprecationWarning,
)
        """.strip() + "\n",
    )
    write_file(tmp_path, "docs/development/foo-inventory.md", "LegacyFoo\n")
    write_file(tmp_path, "CHANGELOG.md", "## [Unreleased] (develop)\nLegacyFoo\n")
    write_file(tmp_path, "docs/migration/foo.md", "LegacyFoo\n")

    findings = repo_hygiene_check.check_deprecation_contract(tmp_path, contract)

    assert findings == []


def _write_canonical_pointer_docs(root: Path) -> None:
    write_file(root, "roadmap.md", "See [roadmap](docs/roadmap.md).\n")
    write_file(root, "ARCHITECTURE.md", "See [arch](docs/architecture.md).\n")
    write_file(root, "docs/roadmap.md", "See [debt](tech-stack.md#technical-debt-register).\n")
    write_file(root, "docs/architecture/adr/README.md", "See [template](000-template.md).\n")
    write_file(root, "docs/architecture.md", "arch\n")
    write_file(root, "docs/tech-stack.md", "debt\n")
    write_file(root, "docs/architecture/adr/000-template.md", "template\n")


def test_canonical_doc_pointers_pass_when_targets_resolve(tmp_path: Path) -> None:
    _write_canonical_pointer_docs(tmp_path)

    findings = repo_hygiene_check.check_canonical_doc_pointers(tmp_path)

    assert findings == []


def test_canonical_doc_pointers_flag_missing_doc_and_broken_link(
    tmp_path: Path,
) -> None:
    _write_canonical_pointer_docs(tmp_path)
    (tmp_path / "docs" / "roadmap.md").unlink()
    # Break a link INSIDE a canonical pointer doc (not ARCHITECTURE.md, which
    # isn't a canonical pointer doc after #518's doc consolidation).
    write_file(
        tmp_path,
        "docs/architecture/adr/README.md",
        "See [gone](does-not-exist.md).\n",
    )

    findings = repo_hygiene_check.check_canonical_doc_pointers(tmp_path)

    messages = {(str(f.path), f.message) for f in findings}
    assert ("docs/roadmap.md", "canonical doc is missing") in messages
    assert (
        "docs/architecture/adr/README.md",
        "broken canonical link target: does-not-exist.md",
    ) in messages


def test_canonical_doc_pointers_ignore_external_links(tmp_path: Path) -> None:
    _write_canonical_pointer_docs(tmp_path)
    write_file(
        tmp_path,
        "docs/architecture/adr/README.md",
        "See [adr](https://adr.github.io/x.md) and [template](000-template.md).\n",
    )

    findings = repo_hygiene_check.check_canonical_doc_pointers(tmp_path)

    assert findings == []


def _release_wf(tags: list[str]) -> str:
    body = "\n".join(f'      - "{t}"' for t in tags)
    return f"on:\n  push:\n    tags:\n{body}\n"


def test_release_tag_routing_accepts_excluded_siblings(tmp_path: Path) -> None:
    write_file(
        tmp_path,
        ".github/workflows/release.yml",
        _release_wf(["v*", "!sdk-v*", "!victor-codegraph-v*", "!verticals-v*"]),
    )
    write_file(tmp_path, ".github/workflows/release-contracts.yml", _release_wf(["sdk-v*"]))
    write_file(
        tmp_path,
        ".github/workflows/release-codegraph.yml",
        _release_wf(["victor-codegraph-v*"]),
    )
    write_file(
        tmp_path,
        ".github/workflows/release-verticals.yml",
        _release_wf(["verticals-v*"]),
    )

    assert repo_hygiene_check.check_release_tag_routing(tmp_path) == []


def test_release_tag_routing_flags_unexcluded_sibling(tmp_path: Path) -> None:
    # release.yml forgets to exclude verticals-v* → verticals-v0.0.0 also fires release.yml.
    write_file(tmp_path, ".github/workflows/release.yml", _release_wf(["v*", "!sdk-v*"]))
    write_file(
        tmp_path,
        ".github/workflows/release-verticals.yml",
        _release_wf(["verticals-v*"]),
    )

    findings = repo_hygiene_check.check_release_tag_routing(tmp_path)
    assert findings, "collision between release.yml (v*) and verticals-v* was not caught"
    assert "!verticals-v*" in findings[0].message


def test_current_repo_passes_hygiene_checks() -> None:
    findings = repo_hygiene_check.run_checks(Path.cwd())

    assert findings == []
