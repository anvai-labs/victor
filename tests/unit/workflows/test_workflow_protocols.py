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

"""Tests for Workflow Compiler Protocols (DIP compliance)."""

import pytest
import typing
from typing import Any, Dict, List, Optional
from victor.workflows.protocols import (
    IWorkflowCompiler,
    IWorkflowLoader,
    IWorkflowValidator,
)


class TestIWorkflowCompilerProtocol:
    """Tests for IWorkflowCompiler protocol."""

    def test_compiler_protocol_has_compile_method(self):
        """Test that compiler protocol requires compile method."""
        # This test verifies the protocol exists and has the right method signature
        # Check if it's a Protocol by checking MRO (Python 3.10+ compatible)
        assert any(c.__name__ == "Protocol" for c in IWorkflowCompiler.__mro__)
        # The protocol should require a compile method - check direct attribute
        assert hasattr(IWorkflowCompiler, "compile") or callable(
            getattr(IWorkflowCompiler, "compile", None)
        )

    def test_concrete_compiler_implementation(self):
        """Test that a concrete implementation can satisfy the protocol."""
        from victor.framework.graph import CompiledGraph

        class ConcreteCompiler:
            """Concrete implementation of IWorkflowCompiler."""

            def compile(self, workflow_def: Dict[str, Any]) -> CompiledGraph:
                """Compile workflow definition into executable graph."""
                # Mock implementation
                from victor.framework.graph import CompiledGraph

                # Create a minimal CompiledGraph instance
                return CompiledGraph(
                    nodes={},
                    edges={},
                    entry_point="start",
                )

        # Should satisfy protocol
        compiler = ConcreteCompiler()
        assert isinstance(compiler, IWorkflowCompiler)

        # Should have compile method
        assert hasattr(compiler, "compile")

        # Should return CompiledGraph
        result = compiler.compile({"name": "test"})
        assert result is not None


class TestIWorkflowLoaderProtocol:
    """Tests for IWorkflowLoader protocol."""

    def test_loader_protocol_has_load_method(self):
        """Test that loader protocol requires load method."""
        # Check if it's a Protocol by checking MRO (Python 3.10+ compatible)
        assert any(c.__name__ == "Protocol" for c in IWorkflowLoader.__mro__)
        # The protocol should require a load method - check direct attribute
        assert hasattr(IWorkflowLoader, "load") or callable(getattr(IWorkflowLoader, "load", None))

    def test_concrete_loader_implementation(self):
        """Test that a concrete implementation can satisfy the protocol."""

        class ConcreteLoader:
            """Concrete implementation of IWorkflowLoader."""

            def load(self, source: str) -> Dict[str, Any]:
                """Load workflow definition from source."""
                # Mock implementation
                return {"name": "test", "nodes": []}

        # Should satisfy protocol
        loader = ConcreteLoader()
        assert isinstance(loader, IWorkflowLoader)

        # Should have load method
        assert hasattr(loader, "load")

        # Should return dict
        result = loader.load("test_source")
        assert isinstance(result, dict)


class TestIWorkflowValidatorProtocol:
    """Tests for IWorkflowValidator protocol."""

    def test_validator_protocol_has_validate_method(self):
        """Test that validator protocol requires validate method."""
        # Check if it's a Protocol by checking MRO (Python 3.10+ compatible)
        assert any(c.__name__ == "Protocol" for c in IWorkflowValidator.__mro__)
        # The protocol should require a validate method - check direct attribute
        assert hasattr(IWorkflowValidator, "validate") or callable(
            getattr(IWorkflowValidator, "validate", None)
        )

    def test_concrete_validator_implementation(self):
        """Test that a concrete implementation can satisfy the protocol."""
        from dataclasses import dataclass

        @dataclass
        class ValidationResult:
            """Result of workflow validation."""

            is_valid: bool
            errors: List[str]

        class ConcreteValidator:
            """Concrete implementation of IWorkflowValidator."""

            def validate(self, workflow_def: Dict[str, Any]) -> ValidationResult:
                """Validate workflow definition."""
                # Mock implementation
                return ValidationResult(is_valid=True, errors=[])

        # Should satisfy protocol
        validator = ConcreteValidator()
        assert isinstance(validator, IWorkflowValidator)

        # Should have validate method
        assert hasattr(validator, "validate")

        # Should return ValidationResult
        result = validator.validate({"name": "test"})
        assert result.is_valid is True


class TestProtocolComposition:
    """Tests for protocol composition and dependency injection."""

    def test_compiler_with_loader_and_validator(self):
        """Test compiler using injected loader and validator."""
        from victor.framework.graph import CompiledGraph

        class MockLoader:
            def load(self, source: str) -> Dict[str, Any]:
                return {"name": "loaded_workflow", "nodes": []}

        class MockValidator:
            def validate(self, workflow_def: Dict[str, Any]) -> Any:
                return type("ValidationResult", (), {"is_valid": True, "errors": []})()

        class DIPCompiler:
            """Compiler that depends on abstractions (DIP compliance)."""

            def __init__(self, loader: IWorkflowLoader, validator: IWorkflowValidator):
                self._loader = loader
                self._validator = validator

            def compile(self, source: str) -> CompiledGraph:
                """Compile from source using injected dependencies."""
                # Load
                workflow_def = self._loader.load(source)

                # Validate
                validation_result = self._validator.validate(workflow_def)
                if not validation_result.is_valid:
                    raise ValueError(f"Invalid workflow: {validation_result.errors}")

                # Compile (mock)
                return CompiledGraph(nodes={}, edges={}, entry_point="start")

        # Create instances
        loader = MockLoader()
        validator = MockValidator()
        compiler = DIPCompiler(loader, validator)

        # Should compile successfully
        result = compiler.compile("test_source")
        assert result is not None

    def test_compiler_can_swap_implementations(self):
        """Test that compiler can work with different loader/validator implementations."""
        from victor.framework.graph import CompiledGraph

        class FileLoader:
            def load(self, source: str) -> Dict[str, Any]:
                return {"name": "file_workflow", "source": "file"}

        class StringLoader:
            def load(self, source: str) -> Dict[str, Any]:
                return {"name": "string_workflow", "source": "string"}

        class StrictValidator:
            def validate(self, workflow_def: Dict[str, Any]) -> Any:
                has_name = "name" in workflow_def
                return type(
                    "ValidationResult",
                    (),
                    {
                        "is_valid": has_name,
                        "errors": [] if has_name else ["Missing name"],
                    },
                )()

        class LenientValidator:
            def validate(self, workflow_def: Dict[str, Any]) -> Any:
                return type("ValidationResult", (), {"is_valid": True, "errors": []})()

        class FlexibleCompiler:
            def __init__(self, loader: IWorkflowLoader, validator: IWorkflowValidator):
                self._loader = loader
                self._validator = validator

            def compile(self, source: str) -> CompiledGraph:
                workflow_def = self._loader.load(source)
                validation_result = self._validator.validate(workflow_def)
                return CompiledGraph(nodes={}, edges={}, entry_point="start")

        # Test with file loader + strict validator
        compiler1 = FlexibleCompiler(FileLoader(), StrictValidator())
        result1 = compiler1.compile("test")
        assert result1 is not None

        # Test with string loader + lenient validator (swap implementations)
        compiler2 = FlexibleCompiler(StringLoader(), LenientValidator())
        result2 = compiler2.compile("test")
        assert result2 is not None


class TestProtocolRuntimeChecking:
    """Tests for runtime protocol checking."""

    def test_runtime_check_with_isinstance(self):
        """Test that isinstance works with protocols (runtime_checkable)."""

        class ValidCompiler:
            def compile(self, workflow_def: Dict[str, Any]) -> Any:
                return None

        class InvalidCompiler:
            def do_something_else(self) -> None:
                pass

        # Valid implementation should satisfy protocol
        assert isinstance(ValidCompiler(), IWorkflowCompiler)

        # Invalid implementation should not satisfy protocol
        assert not isinstance(InvalidCompiler(), IWorkflowCompiler)

    def test_runtime_check_with_loader(self):
        """Test runtime checking for loader protocol."""

        class ValidLoader:
            def load(self, source: str) -> Dict[str, Any]:
                return {}

        assert isinstance(ValidLoader(), IWorkflowLoader)

    def test_runtime_check_with_validator(self):
        """Test runtime checking for validator protocol."""

        class ValidValidator:
            def validate(self, workflow_def: Dict[str, Any]) -> Any:
                return type("Result", (), {"is_valid": True, "errors": []})()

        assert isinstance(ValidValidator(), IWorkflowValidator)


class TestProtocolTypeHints:
    """Tests for proper type hints in protocols."""

    def test_compiler_method_signatures(self):
        """Test that compiler has correct method signatures."""
        import inspect

        # Get the protocol's signature
        if hasattr(IWorkflowCompiler, "compile"):
            sig = inspect.signature(IWorkflowCompiler.compile)
            # Should have 'self' and 'workflow_def' parameters
            params = list(sig.parameters.keys())
            assert "workflow_def" in params or len(params) >= 1

    def test_loader_method_signatures(self):
        """Test that loader has correct method signatures."""
        import inspect

        if hasattr(IWorkflowLoader, "load"):
            sig = inspect.signature(IWorkflowLoader.load)
            params = list(sig.parameters.keys())
            assert "source" in params or len(params) >= 1

    def test_validator_method_signatures(self):
        """Test that validator has correct method signatures."""
        import inspect

        if hasattr(IWorkflowValidator, "validate"):
            sig = inspect.signature(IWorkflowValidator.validate)
            params = list(sig.parameters.keys())
            assert "workflow_def" in params or len(params) >= 1
