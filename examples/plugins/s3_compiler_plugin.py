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

"""Example S3 workflow compiler plugin.

This demonstrates how to create a plugin that loads workflows
from AWS S3 buckets.
"""

from typing import Any, Dict, List, Optional

from victor.workflows.plugins.compiler_plugin import WorkflowCompilerPlugin


class S3CompilerPlugin(WorkflowCompilerPlugin):
    """S3 workflow compiler plugin.

    Loads YAML workflows from AWS S3 buckets.

    Prerequisites:
        - boto3 installed: pip install boto3
        - AWS credentials configured
        - S3 bucket access

    Example:
        from victor.workflows.compiler_registry import WorkflowCompilerRegistry
        from examples.plugins.s3_compiler_plugin import S3CompilerPlugin

        registry = WorkflowCompilerRegistry()
        registry.register('s3', S3CompilerPlugin)

        compiler = create_compiler(
            's3://my-bucket/workflows/deep_research.yaml',
            bucket='my-bucket',
            region='us-west-2'
        )
        compiled = compiler.compile('workflows/deep_research.yaml')
        result = await compiled.invoke({'query': 'AI trends'})
    """

    supported_schemes = ["s3", "s3+https"]

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        **options,
    ):
        """Initialize S3 compiler plugin.

        Args:
            bucket: S3 bucket name
            region: AWS region (default: us-east-1)
            **options: Additional options (credentials, endpoint_url, etc.)
        """
        self.bucket = bucket
        self.region = region
        self._options = options

        # Lazy S3 client initialization
        self._s3_client = None

    def compile(
        self,
        source: str,
        *,
        workflow_name: Optional[str] = None,
        validate: bool = True,
    ) -> Any:
        """Compile workflow from S3.

        Args:
            source: S3 object key (e.g., 'workflows/my_workflow.yaml')
            workflow_name: Name of workflow to compile
            validate: Whether to validate before compilation

        Returns:
            CompiledGraphProtocol instance

        Raises:
            ValueError: If S3 key is invalid or validation fails
            FileNotFoundError: If S3 object doesn't exist
        """
        # Get YAML content from S3
        yaml_content = self._load_from_s3(source)

        # Delegate to YAML compiler
        from victor.workflows.create import create_compiler

        yaml_compiler = create_compiler("yaml://")

        return yaml_compiler.compile(
            yaml_content,
            workflow_name=workflow_name,
            validate=validate,
        )

    @property
    def supported_schemes(self) -> List[str]:
        """Return supported URI schemes."""
        return self.supported_schemes

    def _load_from_s3(self, key: str) -> str:
        """Load YAML content from S3.

        Args:
            key: S3 object key

        Returns:
            YAML content as string

        Raises:
            FileNotFoundError: If S3 object doesn't exist
        """
        # Lazy S3 client initialization
        if self._s3_client is None:
            import boto3

            # Initialize S3 client with options
            self._s3_client = boto3.client("s3", region_name=self.region, **self._options)

        try:
            # Get object from S3
            response = self._s3_client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read().decode("utf-8")
        except self._s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"S3 object not found: s3://{self.bucket}/{key}")
        except Exception as e:
            raise RuntimeError(f"Failed to load from S3: {e}") from e

    def validate_source(self, source: str) -> bool:
        """Validate S3 source.

        Args:
            source: S3 key to validate

        Returns:
            True if key appears valid, False otherwise
        """
        # Basic validation: check if key looks like a path
        if not source or source.startswith("/"):
            return False

        # Check for suspicious patterns
        if ".." in source:
            return False

        return True

    def get_cache_key(self, source: str) -> str:
        """Generate cache key for S3 source.

        Args:
            source: S3 key

        Returns:
            Cache key string
        """
        import hashlib

        # Include bucket and key in cache key
        cache_string = f"s3://{self.bucket}/{source}"
        return hashlib.md5(cache_string.encode()).hexdigest()


__all__ = [
    "S3CompilerPlugin",
]
