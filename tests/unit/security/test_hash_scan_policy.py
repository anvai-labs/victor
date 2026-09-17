"""B324 stays enabled globally; individual non-security uses declare intent."""

import configparser
from pathlib import Path
import re
import shlex

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def globally_skips_weak_hash_check(config_text: str, workflows: list[dict]) -> bool:
    config = configparser.ConfigParser()
    config.read_string(config_text)
    if "B324" in re.split(r"[\s,]+", config.get("bandit", "skips", fallback="")):
        return True
    for workflow in workflows:
        for job in workflow.get("jobs", {}).values():
            for step in job.get("steps", []):
                script = step.get("run", "")
                if not re.search(r"\bbandit\b", script):
                    continue
                tokens = shlex.split(script)
                for index, token in enumerate(tokens):
                    if token in {"-s", "--skip"}:
                        skips = tokens[index + 1] if index + 1 < len(tokens) else ""
                    elif token.startswith("--skip="):
                        skips = token.partition("=")[2]
                    elif token.startswith("-s"):
                        skips = token[2:]
                    else:
                        continue
                    if "B324" in re.split(r"[\s,]+", skips):
                        return True
    return False


def test_bandit_hash_check_is_not_globally_suppressed():
    workflows = [
        yaml.safe_load(path.read_text()) for path in (ROOT / ".github/workflows").glob("*.yml")
    ]
    assert not globally_skips_weak_hash_check((ROOT / ".bandit").read_text(), workflows)


def test_policy_guard_catches_configuration_suppression():
    assert globally_skips_weak_hash_check("[bandit]\nskips = B101,B324\n", [])


@pytest.mark.parametrize("option", ["-s B101,B324", "--skip B324", "--skip=B101,B324", "-sB324"])
def test_policy_guard_catches_command_suppression(option):
    workflow = {"jobs": {"scan": {"steps": [{"run": f"bandit -r victor {option}"}]}}}
    assert globally_skips_weak_hash_check("[bandit]\nskips = B101\n", [workflow])
