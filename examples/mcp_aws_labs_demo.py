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

"""Demo: Using AWS Labs MCP servers with Victor via Docker.

This example shows how to connect Victor to AWS Labs MCP servers
running in Docker containers for AWS documentation and CDK assistance.

AWS Labs provides official MCP servers for:
- AWS Documentation: Search and retrieve AWS documentation
- AWS CDK: Generate and validate CDK code
- AWS Cost Analysis: Analyze AWS costs and usage
- AWS Diagram: Generate AWS architecture diagrams

Prerequisites:
-------------
1. Docker installed and running

2. Pull the AWS MCP images:
   docker pull public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest
   docker pull public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest

3. AWS credentials configured (for CDK server):
   export AWS_ACCESS_KEY_ID=your-access-key
   export AWS_SECRET_ACCESS_KEY=your-secret-key
   export AWS_REGION=us-east-1

Usage:
------
    # Run the demo
    python examples/mcp_aws_labs_demo.py

    # Run with specific server
    python examples/mcp_aws_labs_demo.py --server documentation
    python examples/mcp_aws_labs_demo.py --server cdk

Docker Commands (for manual testing):
------------------------------------
    # Run AWS Documentation MCP server
    docker run -i --rm \\
        public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest

    # Run AWS CDK MCP server (requires AWS credentials)
    docker run -i --rm \\
        -e AWS_ACCESS_KEY_ID \\
        -e AWS_SECRET_ACCESS_KEY \\
        -e AWS_REGION \\
        public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest

References:
-----------
    - AWS MCP Servers: https://github.com/awslabs/mcp
    - MCP Protocol: https://modelcontextprotocol.io
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_docker_available() -> bool:
    """Check if Docker is available and running."""
    if not shutil.which("docker"):
        return False
    try:
        import subprocess

        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def get_aws_server_configs() -> Dict[str, dict]:
    """Get AWS MCP server configurations.

    Returns:
        Dictionary mapping server name to configuration.
    """
    return {
        "documentation": {
            "name": "aws-documentation",
            "description": "AWS Documentation MCP Server - Search and retrieve AWS docs",
            "image": "public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest",
            "command": [
                "docker",
                "run",
                "-i",
                "--rm",
                "public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest",
            ],
            "env": {},
            "tags": ["aws", "documentation", "docker"],
        },
        "cdk": {
            "name": "aws-cdk",
            "description": "AWS CDK MCP Server - Generate and validate CDK code",
            "image": "public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest",
            "command": [
                "docker",
                "run",
                "-i",
                "--rm",
                "-e",
                "AWS_ACCESS_KEY_ID",
                "-e",
                "AWS_SECRET_ACCESS_KEY",
                "-e",
                "AWS_REGION",
                "public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest",
            ],
            "env": {
                "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
                "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
            },
            "tags": ["aws", "cdk", "infrastructure", "docker"],
            "requires_credentials": True,
        },
    }


async def demo_aws_mcp_direct():
    """Demo using AWS MCP servers directly with MCPClient."""
    print("=" * 70)
    print("AWS Labs MCP Server Demo - Direct Connection")
    print("=" * 70)

    if not check_docker_available():
        print("\nError: Docker is not available or not running.")
        print("Please install Docker and ensure the Docker daemon is running.")
        print("\nInstallation:")
        print("  - macOS: https://docs.docker.com/desktop/mac/install/")
        print("  - Linux: https://docs.docker.com/engine/install/")
        print("  - Windows: https://docs.docker.com/desktop/windows/install/")
        return

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        print("Please install Victor: pip install -e '.[dev]'")
        return

    configs = get_aws_server_configs()
    doc_config = configs["documentation"]

    print(f"\n1. Connecting to {doc_config['name']}...")
    print("-" * 70)
    print(f"   Image: {doc_config['image']}")
    print(f"   Description: {doc_config['description']}")

    # Create MCP client
    client = MCPClient(
        name="Victor-AWS-Demo",
        version="1.0.0",
        health_check_interval=0,  # Disable health checks for demo
    )

    try:
        # Connect to AWS Documentation MCP server
        print("\n   Starting Docker container...")
        success = await client.connect(doc_config["command"])

        if not success:
            print("\n   Failed to connect to AWS Documentation MCP server.")
            print("\n   Troubleshooting:")
            print("   1. Ensure Docker is running: docker info")
            print(f"   2. Pull the image: docker pull {doc_config['image']}")
            print("   3. Check for errors: docker logs <container-id>")
            return

        print("   Connected successfully!")

        # Show server info
        if client.server_info:
            print(f"\n   Server: {client.server_info.name} v{client.server_info.version}")

        # List available tools
        print("\n2. Discovering available tools...")
        print("-" * 70)
        tools = await client.refresh_tools()
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            print(f"\n   - {tool.name}")
            print(
                f"     {tool.description[:80]}..."
                if len(tool.description) > 80
                else f"     {tool.description}"
            )

        # List available resources
        print("\n3. Discovering available resources...")
        print("-" * 70)
        resources = await client.refresh_resources()
        if resources:
            print(f"   Found {len(resources)} resources:")
            for resource in resources[:5]:  # Show first 5
                print(f"   - {resource.name}: {resource.uri}")
        else:
            print("   No resources exposed by this server.")

        # Demo: Search AWS documentation
        print("\n4. Demo: Searching AWS documentation...")
        print("-" * 70)

        # Find a search tool
        search_tools = [t for t in tools if "search" in t.name.lower() or "query" in t.name.lower()]
        if search_tools:
            search_tool = search_tools[0]
            print(f"   Using tool: {search_tool.name}")

            # Call the search tool
            result = await client.call_tool(search_tool.name, query="Lambda function timeout")

            if result.success:
                print("   Search completed successfully!")
                output = str(result.result)[:500]
                print(f"   Result preview:\n   {output}...")
            else:
                print(f"   Search failed: {result.error}")
        else:
            print("   No search tool found. Available tools:")
            for tool in tools:
                print(f"   - {tool.name}")

        # Test connection health
        print("\n5. Testing connection health...")
        print("-" * 70)
        ping_ok = await client.ping()
        print(f"   Ping: {'Success' if ping_ok else 'Failed'}")

    except Exception as e:
        print(f"\nError during demo: {e}")
        logger.exception("Demo error")
    finally:
        # Clean up
        print("\n6. Cleaning up...")
        print("-" * 70)
        client.disconnect()
        print("   Disconnected from server.")


async def demo_aws_mcp_registry():
    """Demo using AWS MCP servers with MCPRegistry for multi-server management."""
    print("\n" + "=" * 70)
    print("AWS Labs MCP Server Demo - Registry Mode")
    print("=" * 70)

    if not check_docker_available():
        print("\nError: Docker is not available or not running.")
        return

    try:
        from victor.integrations.mcp import MCPRegistry, MCPServerConfig
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        return

    configs = get_aws_server_configs()

    print("\n1. Setting up MCP Registry...")
    print("-" * 70)

    # Create registry
    registry = MCPRegistry(
        health_check_enabled=False,  # Disable for demo
    )

    # Register AWS MCP servers
    for server_key, config in configs.items():
        # Skip servers requiring credentials if not configured
        if config.get("requires_credentials"):
            if not os.environ.get("AWS_ACCESS_KEY_ID"):
                print(f"   Skipping {config['name']} (AWS credentials not configured)")
                continue

        server_config = MCPServerConfig(
            name=config["name"],
            command=config["command"],
            description=config["description"],
            tags=config["tags"],
            env=config.get("env", {}),
            auto_connect=False,  # Manual connect for demo
        )
        registry.register_server(server_config)
        print(f"   Registered: {config['name']}")

    print(f"\n   Total servers registered: {len(registry.list_servers())}")

    # Connect to documentation server
    print("\n2. Connecting to AWS Documentation server...")
    print("-" * 70)

    doc_server = "aws-documentation"
    if doc_server in registry.list_servers():
        success = await registry.connect(doc_server)
        if success:
            print(f"   Connected to {doc_server}")
            status = registry.get_server_status(doc_server)
            print(f"   Tools available: {status['tools_count']}")
            print(f"   Resources available: {status['resources_count']}")
        else:
            print(f"   Failed to connect to {doc_server}")

    # Show registry status
    print("\n3. Registry status...")
    print("-" * 70)
    status = registry.get_registry_status()
    print(f"   Total servers: {status['total_servers']}")
    print(f"   Connected servers: {status['connected_servers']}")
    print(f"   Total tools: {status['total_tools']}")
    print(f"   Total resources: {status['total_resources']}")

    # List all tools across servers
    print("\n4. All available tools...")
    print("-" * 70)
    all_tools = registry.get_all_tools()
    for tool in all_tools[:10]:  # Show first 10
        print(f"   - {tool.name}: {tool.description[:50]}...")

    # Clean up
    print("\n5. Cleaning up...")
    print("-" * 70)
    await registry.disconnect_all()
    print("   All servers disconnected.")


async def demo_aws_cdk_workflow():
    """Demo an AWS CDK workflow using MCP servers."""
    print("\n" + "=" * 70)
    print("AWS CDK Workflow Demo")
    print("=" * 70)

    if not check_docker_available():
        print("\nError: Docker is not available or not running.")
        return

    # Check for AWS credentials
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("\nNote: AWS credentials not configured.")
        print("To use the CDK server, set:")
        print("  export AWS_ACCESS_KEY_ID=your-key")
        print("  export AWS_SECRET_ACCESS_KEY=your-secret")
        print("  export AWS_REGION=us-east-1")
        print("\nSkipping CDK workflow demo.")
        return

    try:
        from victor.integrations.mcp import MCPClient
    except ImportError:
        print("\nError: Victor MCP integration not available.")
        return

    configs = get_aws_server_configs()
    cdk_config = configs["cdk"]

    print(f"\n1. Connecting to {cdk_config['name']}...")
    print("-" * 70)

    client = MCPClient(
        name="Victor-CDK-Demo",
        version="1.0.0",
        health_check_interval=0,
    )

    try:
        success = await client.connect(cdk_config["command"])

        if not success:
            print("   Failed to connect to AWS CDK MCP server.")
            return

        print("   Connected successfully!")

        # List CDK tools
        print("\n2. Available CDK tools...")
        print("-" * 70)
        tools = await client.refresh_tools()
        for tool in tools:
            print(f"   - {tool.name}")
            print(f"     {tool.description}")

        # Demo: Generate CDK code
        print("\n3. Demo: Generate CDK code for S3 bucket...")
        print("-" * 70)

        generate_tools = [t for t in tools if "generate" in t.name.lower()]
        if generate_tools:
            gen_tool = generate_tools[0]
            print(f"   Using tool: {gen_tool.name}")

            result = await client.call_tool(
                gen_tool.name,
                description="Create an S3 bucket with versioning enabled",
                language="python",
            )

            if result.success:
                print("   CDK code generated successfully!")
                print(f"\n   Generated code:\n{result.result[:800]}...")
            else:
                print(f"   Generation failed: {result.error}")
        else:
            print("   No code generation tool found.")

    except Exception as e:
        print(f"\nError during CDK demo: {e}")
    finally:
        client.disconnect()
        print("\n   Disconnected from CDK server.")


def show_integration_examples():
    """Show example code for integrating AWS MCP with Victor."""
    print("\n" + "=" * 70)
    print("Integration Examples")
    print("=" * 70)

    print("""
1. Claude Desktop Configuration
-------------------------------
Add to ~/Library/Application Support/Claude/claude_desktop_config.json:

{
  "mcpServers": {
    "aws-docs": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest"
      ]
    },
    "aws-cdk": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "AWS_ACCESS_KEY_ID",
        "-e", "AWS_SECRET_ACCESS_KEY",
        "-e", "AWS_REGION",
        "public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest"
      ],
      "env": {
        "AWS_ACCESS_KEY_ID": "your-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-secret-key",
        "AWS_REGION": "us-east-1"
      }
    }
  }
}

2. Victor MCP Configuration
---------------------------
Add to ~/.victor/mcp.yaml:

servers:
  - name: aws-documentation
    command:
      - docker
      - run
      - -i
      - --rm
      - public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest
    description: AWS Documentation MCP Server
    tags:
      - aws
      - documentation
    auto_connect: true

  - name: aws-cdk
    command:
      - docker
      - run
      - -i
      - --rm
      - -e
      - AWS_ACCESS_KEY_ID
      - -e
      - AWS_SECRET_ACCESS_KEY
      - -e
      - AWS_REGION
      - public.ecr.aws/aws-mcp/aws-cdk-mcp-server:latest
    description: AWS CDK MCP Server
    tags:
      - aws
      - cdk
      - infrastructure
    auto_connect: false

3. Programmatic Usage
---------------------
from victor.integrations.mcp import MCPRegistry, MCPServerConfig

# Create registry with AWS servers
registry = MCPRegistry()

registry.register_server(MCPServerConfig(
    name="aws-docs",
    command=[
        "docker", "run", "-i", "--rm",
        "public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest"
    ],
    tags=["aws", "documentation"],
))

# Connect and use
await registry.connect("aws-docs")
result = await registry.call_tool("search_docs", query="Lambda")
""")


async def main():
    """Run AWS Labs MCP demo."""
    parser = argparse.ArgumentParser(
        description="AWS Labs MCP Server Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/mcp_aws_labs_demo.py               # Run all demos
  python examples/mcp_aws_labs_demo.py --server documentation
  python examples/mcp_aws_labs_demo.py --server cdk
  python examples/mcp_aws_labs_demo.py --examples    # Show integration examples
        """,
    )
    parser.add_argument(
        "--server",
        choices=["documentation", "cdk", "all"],
        default="all",
        help="Which AWS MCP server to demo",
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show integration examples and exit",
    )

    args = parser.parse_args()

    print("AWS Labs MCP Servers Demo")
    print("=" * 70)
    print("\nThis demo shows how to use AWS Labs MCP servers with Victor.")
    print("AWS MCP servers provide tools for AWS documentation, CDK, and more.")

    if args.examples:
        show_integration_examples()
        return

    if args.server in ("documentation", "all"):
        await demo_aws_mcp_direct()

    if args.server == "all":
        await demo_aws_mcp_registry()

    if args.server in ("cdk", "all"):
        await demo_aws_cdk_workflow()

    show_integration_examples()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print(
        "  1. Pull AWS MCP images: docker pull public.ecr.aws/aws-mcp/aws-documentation-mcp-server:latest"
    )
    print("  2. Configure AWS credentials for CDK server")
    print("  3. Add servers to Victor config (~/.victor/mcp.yaml)")
    print("  4. Use AWS tools in your Victor workflows")


if __name__ == "__main__":
    asyncio.run(main())
