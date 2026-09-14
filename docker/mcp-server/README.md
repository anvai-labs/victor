# Victor MCP container

Build the MCP target from the repository root:

```bash
docker build --target mcp -t victor-mcp:local .
docker run --rm -i -v "$PWD:/workspace" victor-mcp:local
```

The server reads MCP JSON-RPC from standard input and writes responses to standard
output. Use `-i` without a pseudo-terminal so terminal processing does not alter
the protocol. Logs go to standard error. Override the logging argument with
`--log-level DEBUG` when diagnosing a client connection.

The image installs the core Victor package and exposes the tools discovered by
`victor mcp`. It does not implement tool/vertical allowlists through
`VICTOR_MCP_TOOLS` or `VICTOR_MCP_VERTICALS`; those formerly documented environment
variables did not enforce restrictions and have been removed from the examples.
Mount only the workspace the server should access. Optional tool backends need
their corresponding packages and services.

The Compose example uses the same build target. For an MCP client's stdio
transport, invoke `docker compose -f docker/mcp-server/docker-compose.yaml run
--rm -T victor-mcp` from the repository root, rather than connecting to a daemon
port. See [dependency and container maintenance](../../docs/development/dependencies.md)
for the shared image pipeline and release checks.
