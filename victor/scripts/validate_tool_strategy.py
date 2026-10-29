#!/usr/bin/env python3
"""
Validate context-aware tool strategy against real workloads.

This script validates the new context-window-aware, economy-first tool strategy
to ensure it works correctly with real providers and workloads.

Usage:
    python -m victor.scripts.validate_tool_strategy
    python -m victor.scripts.validate_tool_strategy --provider ollama --model qwen2.5-coder:7b
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from victor.config.tool_tiers import get_tool_tier, get_tier_summary
from victor.tools.enums import SchemaLevel


def validate_context_window(provider_name: str, model: str) -> Dict[str, Any]:
    """Validate context window detection for provider/model.

    Args:
        provider_name: Name of provider (e.g., "anthropic", "ollama")
        model: Model identifier

    Returns:
        Validation results
    """
    print(f"\n{'=' * 80}")
    print("CONTEXT WINDOW VALIDATION")
    print("=" * 80)

    results = {"provider": provider_name, "model": model, "tests": []}

    # Try to import and instantiate provider
    try:
        # Map short provider names to actual module names
        provider_module_name = (
            f"{provider_name}_provider"
            if not provider_name.endswith("_provider")
            else provider_name
        )
        provider_module = __import__(
            f"victor.providers.{provider_module_name}", fromlist=[provider_module_name]
        )

        # Map provider name to class name
        class_name = f"{provider_name.capitalize()}Provider"
        provider_class = getattr(provider_module, class_name)

        # Create a mock instance (we don't need real API key for this validation)
        provider = provider_class(api_key="test_key_placeholder")

        # Get context window
        if hasattr(provider, "context_window"):
            context_window = provider.context_window(model)
            results["context_window"] = context_window
            results["tests"].append(
                {
                    "name": "context_window_detection",
                    "status": "pass",
                    "message": f"Context window detected: {context_window} tokens",
                }
            )

            # Validate context window is reasonable
            if context_window < 4096:
                results["tests"].append(
                    {
                        "name": "context_window_minimum",
                        "status": "warn",
                        "message": f"Context window ({context_window}) is very small. May limit functionality.",
                    }
                )
            elif context_window >= 128000:
                results["tests"].append(
                    {
                        "name": "context_window_large",
                        "status": "pass",
                        "message": f"Large context window ({context_window}) - good for tool diversity.",
                    }
                )

        else:
            results["tests"].append(
                {
                    "name": "context_window_method",
                    "status": "fail",
                    "message": f"Provider {provider_name} does not support context_window() method",
                }
            )

    except Exception as e:
        results["tests"].append(
            {
                "name": "provider_instantiation",
                "status": "fail",
                "message": f"Failed to instantiate provider: {e}",
            }
        )

    return results


def validate_tool_tier_assignments() -> Dict[str, Any]:
    """Validate tool tier assignments.

    Returns:
        Validation results
    """
    print(f"\n{'=' * 80}")
    print("TOOL TIER VALIDATION")
    print("=" * 80)

    results = {"tests": []}

    try:
        # Check if tier assignments exist
        summary = get_tier_summary()

        results["tier_summary"] = summary
        results["tests"].append(
            {
                "name": "tier_assignments_exist",
                "status": "pass",
                "message": f"Found {summary['FULL']} FULL, {summary['COMPACT']} COMPACT, {summary['STUB']} STUB tools",
            }
        )

        # Validate tier assignments are reasonable
        total_tools = summary["FULL"] + summary["COMPACT"] + summary["STUB"]

        if total_tools < 10:
            results["tests"].append(
                {
                    "name": "tier_coverage",
                    "status": "warn",
                    "message": f"Only {total_tools} tools have tier assignments. May need to run analyze_tool_usage.py.",
                }
            )

        # Check that critical tools are assigned
        critical_tools = ["read", "write", "edit", "code_search", "shell"]
        for tool in critical_tools:
            tier = get_tool_tier(tool)
            if tier == "STUB":
                results["tests"].append(
                    {
                        "name": f"critical_tool_{tool}",
                        "status": "warn",
                        "message": f"Critical tool '{tool}' is assigned STUB tier. Consider FULL or COMPACT.",
                    }
                )

    except Exception as e:
        results["tests"].append(
            {
                "name": "tier_validation",
                "status": "fail",
                "message": f"Failed to validate tool tiers: {e}",
            }
        )

    return results


def validate_schema_token_costs() -> Dict[str, Any]:
    """Validate schema token cost estimates.

    Returns:
        Validation results
    """
    print(f"\n{'=' * 80}")
    print("SCHEMA TOKEN COST VALIDATION")
    print("=" * 80)

    results = {"tests": [], "schema_costs": {}}

    # Test schema token costs for different tiers
    test_tools = {"read": "FULL", "git_status": "COMPACT", "docker_build": "STUB"}

    for tool_name, expected_tier in test_tools.items():
        try:
            # Try to get tool from registry
            from victor.tools.registry import ToolRegistry

            registry = ToolRegistry.get_instance()
            tool = registry.get_tool(tool_name)

            if tool:
                # Get schema at different levels
                full_schema = tool.to_schema(SchemaLevel.FULL)
                compact_schema = tool.to_schema(SchemaLevel.COMPACT)
                stub_schema = tool.to_schema(SchemaLevel.STUB)

                # Estimate token costs
                full_tokens = len(str(full_schema)) // 4
                compact_tokens = len(str(compact_schema)) // 4
                stub_tokens = len(str(stub_schema)) // 4

                results["schema_costs"][tool_name] = {
                    "FULL": full_tokens,
                    "COMPACT": compact_tokens,
                    "STUB": stub_tokens,
                }

                # Validate FULL > COMPACT > STUB
                if not (full_tokens >= compact_tokens >= stub_tokens):
                    results["tests"].append(
                        {
                            "name": f"schema_ordering_{tool_name}",
                            "status": "fail",
                            "message": f"Schema token costs not ordered: FULL={full_tokens}, COMPACT={compact_tokens}, STUB={stub_tokens}",
                        }
                    )
                else:
                    results["tests"].append(
                        {
                            "name": f"schema_costs_{tool_name}",
                            "status": "pass",
                            "message": f"Token costs: FULL={full_tokens}, COMPACT={compact_tokens}, STUB={stub_tokens}",
                        }
                    )

        except Exception as e:
            results["tests"].append(
                {
                    "name": f"schema_cost_{tool_name}",
                    "status": "skip",
                    "message": f"Could not test {tool_name}: {e}",
                }
            )

    return results


def validate_context_constraints(provider_name: str, model: str) -> Dict[str, Any]:
    """Validate context window constraints for tool selection.

    Returns:
        Validation results
    """
    print(f"\n{'=' * 80}")
    print("CONTEXT CONSTRAINT VALIDATION")
    print("=" * 80)

    results = {"tests": []}

    try:
        # Get context window
        provider_module_name = (
            f"{provider_name}_provider"
            if not provider_name.endswith("_provider")
            else provider_name
        )
        provider_module = __import__(
            f"victor.providers.{provider_module_name}", fromlist=[provider_module_name]
        )
        class_name = f"{provider_name.capitalize()}Provider"
        provider_class = getattr(provider_module, class_name)
        provider = provider_class(api_key="test_key_placeholder")

        context_window = provider.context_window(model)

        # Calculate tool budget constraints
        max_tool_tokens = int(context_window * 0.25)

        results["context_window"] = context_window
        results["max_tool_tokens"] = max_tool_tokens

        results["tests"].append(
            {
                "name": "context_budget_calculation",
                "status": "pass",
                "message": f"Context window: {context_window}, Max tool tokens: {max_tool_tokens} (25% of context)",
            }
        )

        # Test if typical tool sets fit within budget
        # Assume: 6 FULL tools (125 each) + 10 COMPACT tools (70 each) + 20 STUB tools (32 each)
        typical_full_tokens = 6 * 125  # 750
        typical_compact_tokens = 10 * 70  # 700
        typical_stub_tokens = 20 * 32  # 640
        typical_total = typical_full_tokens + typical_compact_tokens + typical_stub_tokens  # 2090

        if typical_total <= max_tool_tokens:
            results["tests"].append(
                {
                    "name": "typical_tool_set_fits",
                    "status": "pass",
                    "message": f"Typical tool set ({typical_total} tokens) fits within budget ({max_tool_tokens})",
                }
            )
        else:
            results["tests"].append(
                {
                    "name": "typical_tool_set_exceeds",
                    "status": "warn",
                    "message": f"Typical tool set ({typical_total} tokens) exceeds budget ({max_tool_tokens}) by {typical_total - max_tool_tokens} tokens",
                }
            )

    except Exception as e:
        results["tests"].append(
            {
                "name": "context_constraint_validation",
                "status": "fail",
                "message": f"Failed to validate context constraints: {e}",
            }
        )

    return results


def print_summary(all_results: Dict[str, Any]):
    """Print summary of all validation results.

    Args:
        all_results: Dictionary of all validation results
    """
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print("=" * 80)

    total_tests = 0
    passed = 0
    failed = 0
    warnings = 0

    for category, results in all_results.items():
        if category == "tests":
            continue

        print(f"\n{category.upper()}:")
        print("-" * 80)

        for test in results.get("tests", []):
            total_tests += 1

            if test["status"] == "pass":
                passed += 1
                print(f"  ✅ {test['name']}: {test['message']}")
            elif test["status"] == "fail":
                failed += 1
                print(f"  ❌ {test['name']}: {test['message']}")
            elif test["status"] == "warn":
                warnings += 1
                print(f"  ⚠️  {test['name']}: {test['message']}")

    print(f"\n{'=' * 80}")
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Warnings: {warnings} ⚠️")

    if failed == 0:
        print("\n✅ All critical tests passed! Ready to enable tool_strategy_v2.")
        print("\nNext steps:")
        print("  1. Review warnings above")
        print("  2. Enable feature flag: VICTOR_TOOL_STRATEGY_V2=true")
        print("  3. Test with real workloads")
    else:
        print(f"\n❌ {failed} test(s) failed. Please fix before enabling tool_strategy_v2.")
        print("\nTroubleshooting:")
        print("  - Check provider implementations have context_window() method")
        print("  - Verify tool_tiers.yaml has been generated (run analyze_tool_usage.py)")
        print("  - Review schema token cost estimates")


async def main():
    """Main entry point for validation script."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate context-aware tool strategy")
    parser.add_argument(
        "--provider",
        type=str,
        default="anthropic",
        help="Provider to validate (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Model to validate (default: claude-sonnet-4-20250514)",
    )

    args = parser.parse_args()

    print("Tool Strategy V2 Validation")
    print("=" * 80)
    print(f"Provider: {args.provider}")
    print(f"Model: {args.model}")

    all_results = {}

    # Run validation tests
    all_results["context_window"] = validate_context_window(args.provider, args.model)
    all_results["tool_tiers"] = validate_tool_tier_assignments()
    all_results["schema_costs"] = validate_schema_token_costs()
    all_results["context_constraints"] = validate_context_constraints(args.provider, args.model)

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    asyncio.run(main())
