#!/usr/bin/env python3
# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI script for running SWE-bench evaluations.

This script provides a command-line interface for running end-to-end
SWE-bench evaluations using Victor's agentic benchmark infrastructure.

Usage:
    python scripts/run_swe_bench_evaluation.py \
        --dataset swe-bench-lite.jsonl \
        --profile default \
        --output results/ \
        --parallel 4

    # Run specific instances
    python scripts/run_swe_bench_evaluation.py \
        --instance-ids django__django-12345 flask__flask-6789 \
        --profile default

    # Filter by repository
    python scripts/run_swe_bench_evaluation.py \
        --repos django/django pallets/flask \
        --max-tasks 10
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from victor.evaluation import (
    EvaluationOrchestrator,
    EvaluationStage,
    OrchestratorConfig,
    TaskProgress,
)
from victor.evaluation.agent_adapter import AdapterConfig


def setup_logging(verbose: bool, log_file: Optional[Path] = None) -> None:
    """Configure logging for the evaluation script."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def progress_callback(progress: TaskProgress, stage: EvaluationStage) -> None:
    """Print progress updates during evaluation."""
    status_icons = {
        EvaluationStage.LOADING: "📥",
        EvaluationStage.ENVIRONMENT_SETUP: "🔧",
        EvaluationStage.BASELINE_ESTABLISHMENT: "📊",
        EvaluationStage.AGENT_EXECUTION: "🤖",
        EvaluationStage.VALIDATION: "✅",
        EvaluationStage.CORRELATION: "🔗",
        EvaluationStage.REPORTING: "📝",
        EvaluationStage.COMPLETED: "✨",
        EvaluationStage.FAILED: "❌",
    }

    icon = status_icons.get(stage, "⏳")
    duration = f" ({progress.duration_seconds:.1f}s)" if progress.duration_seconds > 0 else ""

    print(f"{icon} [{progress.instance_id}] {stage.value}{duration}")

    if stage == EvaluationStage.FAILED and progress.error_message:
        print(f"   Error: {progress.error_message}")

    if stage == EvaluationStage.COMPLETED and progress.score:
        print(
            f"   Score: {progress.score.overall_score:.3f} "
            f"(F2P: {progress.score.f2p_score:.3f}, P2P: {progress.score.p2p_score:.3f})"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using Victor agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on SWE-bench Lite from HuggingFace
  python scripts/run_swe_bench_evaluation.py --profile default --max-tasks 10

  # Run on local JSONL dataset
  python scripts/run_swe_bench_evaluation.py --dataset swe-bench-lite.jsonl --parallel 4

  # Run specific instances
  python scripts/run_swe_bench_evaluation.py --instance-ids django__django-12345

  # Filter by repository
  python scripts/run_swe_bench_evaluation.py --repos django/django --max-tasks 5
        """,
    )

    # Dataset options
    dataset_group = parser.add_argument_group("Dataset Options")
    dataset_group.add_argument(
        "--dataset",
        "-d",
        type=Path,
        help="Path to JSONL dataset file (default: load from HuggingFace)",
    )
    dataset_group.add_argument(
        "--dataset-name",
        default="princeton-nlp/SWE-bench_Lite",
        help="HuggingFace dataset name (default: princeton-nlp/SWE-bench_Lite)",
    )
    dataset_group.add_argument(
        "--max-tasks",
        "-n",
        type=int,
        default=0,
        help="Maximum number of tasks to run (0 = all)",
    )
    dataset_group.add_argument(
        "--instance-ids",
        "-i",
        nargs="+",
        default=[],
        help="Specific instance IDs to run",
    )
    dataset_group.add_argument(
        "--repos",
        "-r",
        nargs="+",
        default=[],
        help="Filter by repository names (e.g., django/django)",
    )

    # Agent options
    agent_group = parser.add_argument_group("Agent Options")
    agent_group.add_argument(
        "--profile",
        "-p",
        default="default",
        help="Victor profile name (default: default)",
    )
    agent_group.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum agent turns per task (default: 20)",
    )
    agent_group.add_argument(
        "--max-tool-calls",
        type=int,
        default=50,
        help="Maximum tool calls per task (default: 50)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./swe_bench_results"),
        help="Output directory for results (default: ./swe_bench_results)",
    )
    output_group.add_argument(
        "--save-traces",
        action="store_true",
        default=True,
        help="Save execution traces (default: True)",
    )
    output_group.add_argument(
        "--no-save-traces",
        action="store_false",
        dest="save_traces",
        help="Don't save execution traces",
    )
    output_group.add_argument(
        "--save-patches",
        action="store_true",
        default=True,
        help="Save generated patches (default: True)",
    )
    output_group.add_argument(
        "--no-save-patches",
        action="store_false",
        dest="save_patches",
        help="Don't save generated patches",
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution Options")
    exec_group.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=1,
        help="Number of parallel tasks (default: 1)",
    )
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per task in seconds (default: 1800 = 30 min)",
    )
    exec_group.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue running after task failures (default: True)",
    )
    exec_group.add_argument(
        "--stop-on-error",
        action="store_false",
        dest="continue_on_error",
        help="Stop on first task failure",
    )

    # Cache options
    cache_group = parser.add_argument_group("Cache Options")
    cache_group.add_argument("--workspace-cache", type=Path, help="Directory for workspace cache")
    cache_group.add_argument("--baseline-cache", type=Path, help="Directory for baseline cache")
    cache_group.add_argument("--no-cache", action="store_true", help="Disable all caching")

    # Logging options
    log_group = parser.add_argument_group("Logging Options")
    log_group.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    log_group.add_argument("--log-file", type=Path, help="Write logs to file")
    log_group.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    return parser.parse_args()


async def main() -> int:
    """Main entry point for the evaluation script."""
    args = parse_args()

    # Setup logging
    log_file = args.log_file
    if log_file is None and args.output:
        log_file = args.output / "evaluation.log"
    setup_logging(args.verbose, log_file)

    logger = logging.getLogger(__name__)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save run configuration
    config_file = args.output / "run_config.json"
    run_config = {
        "started_at": datetime.now().isoformat(),
        "args": vars(args).copy(),
    }
    # Convert Path objects to strings for JSON serialization
    for key, value in run_config["args"].items():
        if isinstance(value, Path):
            run_config["args"][key] = str(value)
    config_file.write_text(json.dumps(run_config, indent=2))

    # Build orchestrator config
    adapter_config = AdapterConfig(
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
    )

    orchestrator_config = OrchestratorConfig(
        dataset_path=args.dataset,
        dataset_name=args.dataset_name,
        max_tasks=args.max_tasks,
        instance_ids=args.instance_ids,
        repos=args.repos,
        agent_profile=args.profile,
        adapter_config=adapter_config,
        output_dir=args.output,
        save_traces=args.save_traces,
        save_patches=args.save_patches,
        max_parallel=args.parallel,
        task_timeout=args.timeout,
        continue_on_error=args.continue_on_error,
        workspace_cache_dir=args.workspace_cache,
        baseline_cache_dir=args.baseline_cache,
        use_baseline_cache=not args.no_cache,
    )

    # Create orchestrator with progress callback
    callback = None if args.quiet else progress_callback
    orchestrator = EvaluationOrchestrator(orchestrator_config, callback)

    print("=" * 60)
    print("SWE-bench Evaluation")
    print("=" * 60)
    print(f"Profile: {args.profile}")
    print(f"Output: {args.output}")
    if args.dataset:
        print(f"Dataset: {args.dataset}")
    else:
        print(f"Dataset: {args.dataset_name} (HuggingFace)")
    if args.instance_ids:
        print(f"Instance IDs: {', '.join(args.instance_ids)}")
    if args.repos:
        print(f"Repos: {', '.join(args.repos)}")
    print(f"Parallel: {args.parallel}")
    print(f"Timeout: {args.timeout}s")
    print("=" * 60)
    print()

    try:
        # Run evaluation
        logger.info("Starting evaluation...")
        report = await orchestrator.run_evaluation()

        # Print summary
        summary = orchestrator.get_summary()
        print()
        print(summary.to_text())

        # Save report
        report_file = args.output / "correlation_report.json"
        report_file.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info(f"Report saved to: {report_file}")

        # Return exit code based on pass rate
        if summary.pass_rate >= 0.5:
            return 0  # Success
        elif summary.completed_tasks > 0:
            return 1  # Partial success
        else:
            return 2  # No tasks completed

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cli_main() -> None:
    """CLI entry point."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_main()
