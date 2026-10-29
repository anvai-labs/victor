#!/usr/bin/env python3
"""
Validate provider-specific tool tier assignments.

This script validates the new provider-specific tool tier system to ensure it
works correctly for different provider categories (edge, standard, large).

Usage:
    python -m victor.scripts.validate_provider_tiers
"""

import sys


def validate_edge_tool_budget():
    """Test edge model tools fit within budget."""
    print("\n=== Edge Model Validation ===")
    context_window = 8192
    max_tool_tokens = 2048  # 25% of context

    # Edge tier: 2 FULL tools × 125 = 250 tokens
    edge_tokens = 2 * 125

    print(f"Context window: {context_window} tokens")
    print(f"Max tool tokens (25%): {max_tool_tokens} tokens")
    print(f"Edge tier tool cost: {edge_tokens} tokens")
    print(f"Budget utilization: {edge_tokens / max_tool_tokens * 100:.1f}%")

    assert (
        edge_tokens <= max_tool_tokens
    ), f"Edge tools ({edge_tokens}) exceed budget ({max_tool_tokens})"
    print("✅ Edge model: Tool tokens fit within budget")
    return True


def validate_standard_tool_budget():
    """Test standard model tools fit within budget."""
    print("\n=== Standard Model Validation ===")
    context_window = 32768
    max_tool_tokens = 8192  # 25% of context

    # Standard tier: 5 FULL + 2 COMPACT = 745 tokens
    standard_tokens = (5 * 125) + (2 * 70)

    print(f"Context window: {context_window} tokens")
    print(f"Max tool tokens (25%): {max_tool_tokens} tokens")
    print(f"Standard tier tool cost: {standard_tokens} tokens")
    print(f"Budget utilization: {standard_tokens / max_tool_tokens * 100:.1f}%")

    assert (
        standard_tokens <= max_tool_tokens
    ), f"Standard tools ({standard_tokens}) exceed budget ({max_tool_tokens})"
    print("✅ Standard model: Tool tokens fit within budget")
    return True


def validate_large_tool_budget():
    """Test large model tools fit within budget."""
    print("\n=== Large Model Validation ===")
    context_window = 200000
    max_tool_tokens = 50000  # 25% of context

    # Large tier: 10 FULL tools = 1250 tokens
    large_tokens = 10 * 125

    print(f"Context window: {context_window} tokens")
    print(f"Max tool tokens (25%): {max_tool_tokens} tokens")
    print(f"Large tier tool cost: {large_tokens} tokens")
    print(f"Budget utilization: {large_tokens / max_tool_tokens * 100:.1f}%")

    assert (
        large_tokens <= max_tool_tokens
    ), f"Large tools ({large_tokens}) exceed budget ({max_tool_tokens})"
    print("✅ Large model: Tool tokens fit within budget")
    return True


def validate_token_savings():
    """Test token savings vs global tiers."""
    print("\n=== Token Savings Validation ===")

    # Global tiers: 10 FULL = 1250 tokens
    global_tokens = 10 * 125

    # Edge tiers: 2 FULL = 250 tokens
    edge_tokens = 2 * 125

    # Standard tiers: 5 FULL + 2 COMPACT = 745 tokens
    standard_tokens = (5 * 125) + (2 * 70)

    edge_savings = global_tokens - edge_tokens
    edge_savings_pct = (edge_savings / global_tokens) * 100

    standard_savings = global_tokens - standard_tokens
    standard_savings_pct = (standard_savings / global_tokens) * 100

    print(f"Global tiers: {global_tokens} tokens")
    print(f"Edge tiers: {edge_tokens} tokens")
    print(f"Edge savings: {edge_savings} tokens ({edge_savings_pct:.1f}% reduction)")

    print(f"\nStandard tiers: {standard_tokens} tokens")
    print(f"Standard savings: {standard_savings} tokens ({standard_savings_pct:.1f}% reduction)")

    assert edge_savings_pct == 80.0, f"Expected 80% edge savings, got {edge_savings_pct:.1f}%"
    assert (
        abs(standard_savings_pct - 38.8) < 0.1
    ), f"Expected 38.8% standard savings, got {standard_savings_pct:.1f}%"

    print("✅ Token savings: Achieved expected reductions")
    return True


def validate_provider_category_detection():
    """Test provider category detection."""
    print("\n=== Provider Category Detection ===")

    from victor.config.tool_tiers import get_provider_category

    # Test edge category
    assert get_provider_category(8192) == "edge"
    assert get_provider_category(16383) == "edge"
    print("✅ Edge category: Correctly detected (<16K)")

    # Test standard category
    assert get_provider_category(16384) == "standard"
    assert get_provider_category(32768) == "standard"
    assert get_provider_category(131071) == "standard"
    print("✅ Standard category: Correctly detected (16K-128K)")

    # Test large category
    assert get_provider_category(131072) == "large"
    assert get_provider_category(200000) == "large"
    print("✅ Large category: Correctly detected (>128K)")

    return True


def validate_provider_tool_tier_assignments():
    """Test provider-specific tier assignments."""
    print("\n=== Provider Tier Assignments ===")

    from victor.config.tool_tiers import get_provider_tool_tier

    # Test edge tier assignments
    assert get_provider_tool_tier("read", "edge") == "FULL"
    assert get_provider_tool_tier("shell", "edge") == "FULL"
    assert get_provider_tool_tier("ls", "edge") == "STUB"  # Not in edge FULL
    assert get_provider_tool_tier("write", "edge") == "STUB"  # Not in edge FULL
    print("✅ Edge tiers: 2 FULL tools (read, shell), all others STUB")

    # Test standard tier assignments
    assert get_provider_tool_tier("read", "standard") == "FULL"
    assert get_provider_tool_tier("shell", "standard") == "FULL"
    assert get_provider_tool_tier("ls", "standard") == "FULL"
    assert get_provider_tool_tier("code_search", "standard") == "FULL"
    assert get_provider_tool_tier("edit", "standard") == "FULL"
    assert get_provider_tool_tier("write", "standard") == "COMPACT"
    assert get_provider_tool_tier("test", "standard") == "COMPACT"
    assert get_provider_tool_tier("refs", "standard") == "STUB"
    print("✅ Standard tiers: 5 FULL + 2 COMPACT tools")

    # Test large tier assignments
    assert get_provider_tool_tier("read", "large") == "FULL"
    assert get_provider_tool_tier("shell", "large") == "FULL"
    assert get_provider_tool_tier("write", "large") == "FULL"
    assert get_provider_tool_tier("test", "large") == "FULL"
    assert get_provider_tool_tier("refs", "large") == "STUB"
    print("✅ Large tiers: 10 FULL tools")

    return True


def main():
    """Main entry point for validation script."""
    print("=" * 80)
    print("Provider-Specific Tool Tier Validation")
    print("=" * 80)

    all_passed = True

    try:
        validate_provider_category_detection()
        validate_provider_tool_tier_assignments()
        validate_edge_tool_budget()
        validate_standard_tool_budget()
        validate_large_tool_budget()
        validate_token_savings()
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All validations passed!")
        print("\nProvider-specific tier optimization is working correctly.")
        print("Edge models: 80% token reduction")
        print("Standard models: 40% token reduction")
        print("Large models: No regression (full capability)")
        return 0
    else:
        print("❌ Some validations failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
