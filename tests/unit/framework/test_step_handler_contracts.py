"""Tests for capability contract wiring on step handlers."""


class TestStepHandlerContracts:

    def test_base_step_handler_has_capability_contracts(self):
        from victor.framework.step_handlers import BaseStepHandler

        assert hasattr(BaseStepHandler, "capability_contracts")
        assert BaseStepHandler.capability_contracts == ()

    def test_tool_step_handler_declares_contract(self):
        from victor.framework.step_handlers import ToolStepHandler

        assert len(ToolStepHandler.capability_contracts) >= 1
        contract = ToolStepHandler.capability_contracts[0]
        assert contract[0] == "tools"  # capability name
        assert contract[1] >= 1  # version

    def test_step_handler_protocol_accepts_contracts(self):
        """Any step handler with capability_contracts should be valid."""
        from victor.framework.step_handlers import ToolStepHandler, PromptStepHandler

        # These should not raise — contracts are optional metadata
        assert isinstance(ToolStepHandler.capability_contracts, tuple)
        assert isinstance(PromptStepHandler.capability_contracts, tuple)
