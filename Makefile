# Victor Makefile
# Common development and distribution commands
#
# Usage:
#   make install      # Install for development
#   make test         # Run tests
#   make build        # Build distribution packages
#   make release      # Create a release (requires version)

.PHONY: help install install-dev install-standalone install-verticals test-verticals check-vertical-test-floor check-vertical-boundaries lint-verticals lint-verticals-ruff lint-verticals-fmt-types test test-definition-boundaries lint check-repo-hygiene check-extracted-vertical-boundaries format clean build build-binary docker release sync-version check-version

PYTEST_TIMEOUT_ARG := $(shell pytest --help 2>/dev/null | grep -q -- "--timeout" && echo --timeout=120)

# Parallel test execution via pytest-xdist (no-op if the plugin is absent).
# --dist=loadfile keeps each test file on one worker in definition order: parts
# of the suite (e.g. test_orchestrator_core.py) depend on in-file execution
# order, which the default --dist=load scrambles.
# Override with e.g. `make test PYTEST_JOBS=8` or PYTEST_JOBS=0 to disable.
PYTEST_JOBS ?= 4
PYTEST_XDIST_ARG := $(shell pytest --help 2>/dev/null | grep -q -- "--numprocesses" && test "$(PYTEST_JOBS)" != "0" && echo --numprocesses=$(PYTEST_JOBS) --dist=loadfile)

# Default target
help:
	@echo "Victor Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install       Install for development"
	@echo "  make install-dev   Install with all dev dependencies"
	@echo "  make install-standalone PY=...  Bootstrap a venv from a standalone Python (+ contracts + verticals)"
	@echo "  make test          Run unit tests"
	@echo "  make test-all      Run all tests including integration"
	@echo "  make test-definition-boundaries  Run SDK-definition import guardrails"
	@echo "  make lint          Run linters"
	@echo "  make check-repo-hygiene  Validate workflow/link/metadata drift guards"
	@echo "  make check-extracted-vertical-boundaries  Audit extracted plugin repos when present"
	@echo "  make format        Format code"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Distribution:"
	@echo "  make build         Build Python packages (sdist + wheel)"
	@echo "  make build-binary  Build standalone binary"
	@echo "  make docker        Build Docker image"
	@echo "  make release       Create a release (VERSION=x.y.z required)"
	@echo ""
	@echo "Versioning:"
	@echo "  make check-version Verify victor-ai/victor-contracts versions are in sync"
	@echo "  make sync-version  Sync all package versions from VERSION file"
	@echo ""
	@echo "Utilities:"
	@echo "  make docs          Generate documentation"
	@echo "  make serve         Start API server"
	@echo "  make tui           Start interactive CLI (rich terminal chat; Textual dashboard: victor dashboard)"

# =============================================================================
# Development
# =============================================================================

install:
	# Install the in-repo contracts editable FIRST so victor-ai resolves the local
	# victor-contracts (which may carry unreleased modules, e.g. victor_contracts.tools
	# from FEP-0009) instead of pulling the last PyPI release.
	pip install -e ./victor-contracts
	pip install -e .

install-dev:
	# Contracts first (see `install`) — keeps dev/CI on the in-repo SDK, not PyPI.
	pip install -e ./victor-contracts
	pip install -e ".[dev,docs,build]"
	pip install pre-commit pytest-split
	pre-commit install || true

# First-party verticals folded into the monorepo under verticals/. Installed as
# their own editable packages (they register via the victor.plugins entry point).
VERTICALS := coding devops research rag dataanalysis

install-verticals: install-dev
	@for v in $(VERTICALS); do \
		echo "== install verticals/victor-$$v =="; \
		pip install -e ./verticals/victor-$$v || exit 1; \
	done

# Bootstrap a complete dev env into a fresh venv built FROM an arbitrary Python
# (e.g. a standalone / relocatable build). Drives every install through the venv
# interpreter so it never depends on which pip is on PATH. Usage:
#   make install-standalone PY=/path/to/standalone/bin/python3
# Override the venv location with VENV=... (default: .venv).
VENV ?= .venv
install-standalone:
	@test -n "$(PY)" || { echo "ERROR: set PY=/path/to/standalone/bin/python3"; exit 1; }
	@test -x "$(PY)" || { echo "ERROR: PY=$(PY) is not an executable"; exit 1; }
	"$(PY)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install --upgrade pip setuptools wheel
	# Contracts FIRST (see `install`) so victor-ai resolves the local
	# victor-contracts, not the last PyPI release.
	"$(VENV)/bin/python" -m pip install -e ./victor-contracts
	"$(VENV)/bin/python" -m pip install -e ".[dev,docs,build]"
	"$(VENV)/bin/python" -m pip install pre-commit pytest-split
	@for v in $(VERTICALS); do \
		echo "== install verticals/victor-$$v =="; \
		"$(VENV)/bin/python" -m pip install -e ./verticals/victor-$$v || exit 1; \
	done
	"$(VENV)/bin/pre-commit" install || true
	@echo ""
	@echo "✓ Victor dev env ready in $(VENV)/ (built from $(PY))"
	@echo "  activate: source $(VENV)/bin/activate"
	@echo "  verify:   $(VENV)/bin/victor --version"

test-verticals:
	@for v in $(VERTICALS); do \
		echo "== test verticals/victor-$$v =="; \
		pytest verticals/victor-$$v/tests -q || exit 1; \
	done
	@$(MAKE) check-vertical-test-floor

# Vertical test parity floor: minimum collected-test count per vertical.
# Guards the test baseline (victor-devops/research/dataanalysis were brought to
# parity with victor-coding's structure) against silent regression — a change
# that deletes or stops collecting a vertical's tests fails here even if the
# surviving tests pass. Floors sit ~5-10% below the current collected counts;
# raise them as suites grow, lower them only deliberately in the same PR that
# removes tests.
check-vertical-test-floor:
	@fail=0; for v in $(VERTICALS); do \
		case $$v in \
			coding) floor=1200 ;; \
			devops) floor=260 ;; \
			research) floor=240 ;; \
			rag) floor=75 ;; \
			dataanalysis) floor=260 ;; \
			*) floor=1 ;; \
		esac; \
		count=$$(pytest verticals/victor-$$v/tests --collect-only -q 2>/dev/null \
			| grep -oE "[0-9]+ tests? collected" | grep -oE "[0-9]+" | tail -1); \
		count=$${count:-0}; \
		echo "verticals/victor-$$v: $$count tests collected (floor: $$floor)"; \
		if [ "$$count" -lt "$$floor" ]; then \
			echo "FAIL: victor-$$v collected $$count tests, below floor $$floor"; \
			fail=1; \
		fi; \
	done; exit $$fail

# Vertical lint, each package using its OWN tooling config. Ruff is aligned with
# the framework's ruff config and clean across all verticals → BLOCKING. Black
# (formatting) and mypy (types) still carry folded-in debt → ADVISORY for now.
lint-verticals-ruff:
	@for v in $(VERTICALS); do \
		echo "== ruff verticals/victor-$$v =="; \
		ruff check verticals/victor-$$v || exit 1; \
	done

lint-verticals-fmt-types:
	@fail=0; for v in $(VERTICALS); do \
		echo "== black+mypy verticals/victor-$$v =="; \
		black --check verticals/victor-$$v || fail=1; \
		( cd verticals/victor-$$v && mypy victor_$$v ) || fail=1; \
	done; exit $$fail

lint-verticals: lint-verticals-ruff lint-verticals-fmt-types

# Contract-boundary audit for the in-repo verticals (verticals import only
# victor_contracts, never victor framework internals — the monorepo discipline).
check-vertical-boundaries:
	python scripts/ci/check_extracted_vertical_boundaries.py

test:
	pytest tests/unit -v --tb=short $(PYTEST_TIMEOUT_ARG) $(PYTEST_XDIST_ARG)

test-definition-boundaries:
	@echo "Definition import boundaries enforced by contract boundary tests"
	pytest tests/unit/contracts -q

test-all:
	pytest -v --tb=short $(PYTEST_XDIST_ARG)

test-cov:
	pytest tests/unit --cov=victor --cov-report=html --cov-report=term-missing $(PYTEST_XDIST_ARG)

test-quick:
	pytest tests/unit -v --tb=short -m "not slow" $(PYTEST_XDIST_ARG)

# EVR-5 / ADR-012: the FEP-0007 run≡stream + characterization batteries that the
# harness-acceptance oracle gates on. The characterization transcripts ARE the
# regression gate — any loop/prompt/completion edit must keep them byte-stable-or-justified.
test-battery:
	pytest tests/integration/streaming -v --tb=short

# EVR-5 / ADR-012: run the harness-acceptance oracle promotion gate — drives the scripted battery
# through the real loop, scores trajectories, and gates against acceptance_baseline.json at
# (model, harness-config) granularity. Re-record after an *intended* behavior change with:
#   EVR5_RECORD_BASELINE=1 make eval-accept   # then review + commit acceptance_baseline.json
eval-accept:
	pytest tests/integration/streaming/test_acceptance_oracle_gate.py -v --tb=short

# Flag graduation: decide whether an opt-in flag is safe to flip default-on by A/B-ing its
# battery (flag OFF vs ON) through the acceptance oracle. Produce a BatteryResult.to_dict() JSON
# for each arm (flag off/on over a task set), then:
#   make eval-graduate FLAG=effect_gated_completion BASELINE=off.json CANDIDATE=on.json
eval-graduate:
	python -m victor.evaluation.flag_graduation --flag "$(FLAG)" --baseline "$(BASELINE)" --candidate "$(CANDIDATE)"

# Flag A/B: run an opt-in flag OFF vs ON over the real agent, emit two battery snapshots + a
# GRADUATE/HOLD verdict (the run that produces the snapshots eval-graduate consumes). Point
# BASE_URL at a local Ollama for a token-free run. For a fair effect-gate test use the effect-gate
# corpus + verify scoring (task pass rate). Example:
#   make eval-flag-ab FLAG=effect_gated_completion BASE_URL=http://172.31.160.1:11434 \
#     MODEL=qwen3-coder-tools:30b VARIANTS=8 CORPUS=effect-gate SCORE=verify
eval-flag-ab:
	python -m victor.evaluation.flag_ab --flag "$(FLAG)" \
		$(if $(BASE_URL),--base-url "$(BASE_URL)") \
		$(if $(MODEL),--model "$(MODEL)") \
		$(if $(VARIANTS),--variants "$(VARIANTS)") \
		$(if $(CORPUS),--corpus "$(CORPUS)") \
		$(if $(SCORE),--score "$(SCORE)")

test-split:
	pytest tests/unit --splits=4 --group=1 -v --tb=short

lint:
	ruff check victor tests
	black --check victor tests
	mypy victor
	python scripts/ci/repo_hygiene_check.py

check-repo-hygiene:
	python scripts/ci/repo_hygiene_check.py

check-extracted-vertical-boundaries:
	python scripts/ci/check_extracted_vertical_boundaries.py

format:
	black victor tests
	ruff check --fix victor tests

format-check:
	black --check victor tests
	ruff check victor tests

pre-commit:
	pre-commit run --all-files

pre-commit-install:
	pre-commit install

hooks:  ## Install shared git hooks (.githooks/* -> .git/hooks/). Preserves pre-commit's hooks.
	@mkdir -p .git/hooks
	@cp -p .githooks/* .git/hooks/ 2>/dev/null || true
	@chmod +x .git/hooks/* 2>/dev/null || true
	@echo "shared hooks installed (.githooks -> .git/hooks); pre-commit hooks preserved"

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# =============================================================================
# Distribution
# =============================================================================

build: clean
	pip install build
	python -m build
	@echo ""
	@echo "Built packages:"
	@ls -la dist/

build-binary: clean
	pip install -e ".[build]"
	python scripts/build_binary.py --onefile
	@echo ""
	@echo "Built binary:"
	@ls -la dist/

docker:
	docker build -t victor:latest .
	@echo ""
	@echo "Built Docker image: victor:latest"

docker-push: docker
	docker tag victor:latest vijayksingh/victor:latest
	docker push vijayksingh/victor:latest

# Version management
sync-version:  ## Sync all package versions from VERSION files
	python scripts/sync_version.py

sync-version-ai:  ## Sync victor-ai version only
	python scripts/sync_version.py --ai

sync-version-sdk:  ## Sync victor-contracts version only
	python scripts/sync_version.py --sdk

check-version:  ## Verify all package versions are consistent
	python scripts/check_version_sync.py

# Release victor-ai (requires VERSION)
release:
ifndef VERSION
	$(error VERSION is required. Usage: make release VERSION=0.1.0)
endif
	@echo "Creating victor-ai release v$(VERSION)..."
	echo "$(VERSION)" > VERSION
	python scripts/sync_version.py --ai
	python scripts/check_version_sync.py
	git add VERSION pyproject.toml
	git commit -m "release: victor-ai v$(VERSION)"
	git tag -a "v$(VERSION)" -m "Release victor-ai v$(VERSION)"
	@echo ""
	@echo "Release v$(VERSION) created!"
	@echo "Run 'git push && git push --tags' to trigger the release workflow"

# Release victor-contracts independently (requires VERSION)
release-sdk:
ifndef VERSION
	$(error VERSION is required. Usage: make release-sdk VERSION=0.7.0)
endif
	@echo "Creating victor-contracts release sdk-v$(VERSION)..."
	echo "$(VERSION)" > victor-contracts/VERSION
	python scripts/sync_version.py --sdk
	python scripts/check_version_sync.py
	git add victor-contracts/VERSION victor-contracts/pyproject.toml pyproject.toml
	git commit -m "release: victor-contracts v$(VERSION)"
	git tag -a "sdk-v$(VERSION)" -m "Release victor-contracts v$(VERSION)"
	@echo ""
	@echo "SDK release sdk-v$(VERSION) created!"
	@echo "Run 'git push && git push --tags' to trigger the SDK release workflow"

# =============================================================================
# Utilities
# =============================================================================

docs:
	pip install -e ".[docs]"
	mkdocs build

docs-serve:
	pip install -e ".[docs]"
	mkdocs serve

serve:
	victor serve

# "tui" is a historical alias: bare `victor` IS the interactive rich CLI
# (same surface as `victor chat`). The Textual app is `victor dashboard`.
tui:
	victor

# Check if all distribution files are ready
check-dist:
	@echo "Checking distribution readiness..."
	@test -f pyproject.toml && echo "✓ pyproject.toml" || echo "✗ pyproject.toml missing"
	@test -f MANIFEST.in && echo "✓ MANIFEST.in" || echo "✗ MANIFEST.in missing"
	@test -f requirements.txt && echo "✓ requirements.txt" || echo "✗ requirements.txt missing"
	@test -f README.md && echo "✓ README.md" || echo "✗ README.md missing"
	@test -f LICENSE && echo "✓ LICENSE" || echo "✗ LICENSE missing"
	@test -f Dockerfile && echo "✓ Dockerfile" || echo "✗ Dockerfile missing"
	@test -f .github/workflows/release.yml && echo "✓ release.yml" || echo "✗ release.yml missing"
	@test -f scripts/install/install.sh && echo "✓ install.sh" || echo "✗ install.sh missing"
	@test -f scripts/install/install.ps1 && echo "✓ install.ps1" || echo "✗ install.ps1 missing"
	@test -f Formula/victor.rb && echo "✓ Homebrew formula" || echo "✗ Homebrew formula missing"
	@echo ""
	@echo "Run 'make build' to test the build process"
