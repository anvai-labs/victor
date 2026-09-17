# HTTP API Reference

REST API for Victor's agentic AI framework capabilities.

## Overview

The HTTP API provides language-agnostic access to Victor's features through a REST interface.

**Start server**:
```bash
victor serve --port 8080 --profile default --mode build
```

**Base URL**: `http://localhost:8080/api/v1`

**Authentication**: API key in `X-API-Key` header (optional, configurable)

---

## Quick Start

### Basic Chat Request

```bash
curl http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, Victor!",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514"
  }'
```

**Response**:
```json
{
  "success": true,
  "response": "Hello! How can I help you today?",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "tokens_used": 25,
  "duration_ms": 1250
}
```

### Streaming Chat

```bash
curl http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a REST API",
    "provider": "anthropic"
  }'
```

**Response** (Server-Sent Events):
```
data: {"type":"start","request_id":"req-123"}
data: {"type":"token","content":"Write"}
data: {"type":"token","content":" a"}
data: {"type":"token","content":" REST"}
data: {"type":"end","request_id":"req-123"}
```

### Capability Recommendations

```bash
curl "http://localhost:8080/api/v1/capabilities/recommend?task_type=feature&complexity=high&mode=build&vertical=coding"
```

**Response**:
```json
{
  "task_type": "feature",
  "complexity": "high",
  "mode": "build",
  "vertical": "coding",
  "count": 1,
  "recommendations": [
    {
      "vertical": "coding",
      "action": "auto_spawn",
      "primary_team": {
        "team_name": "feature_team",
        "confidence": 0.8,
        "reason": "Task type 'feature' matches pattern 'feature'",
        "formation": "parallel",
        "suggested_budget": null,
        "role_distribution": {},
        "source": "hybrid-rule"
      },
      "primary_workflow": {
        "workflow_name": "feature_implementation",
        "confidence": 0.8,
        "reason": "Task type 'feature' matches workflow pattern",
        "trigger_condition": null,
        "estimated_steps": null
      }
    }
  ]
}
```

---

## Endpoints

### POST /api/v1/chat

Send a single message to Victor.

**Request Body**:
```json
{
  "message": "string (required)",
  "provider": "string (optional)",
  "model": "string (optional)",
  "temperature": "float (optional, 0.0-1.0)",
  "max_tokens": "int (optional)",
  "conversation_id": "string (optional, for multi-turn)",
  "tools": ["array of tool names (optional)"]
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a function to sort an array",
    "provider": "anthropic",
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

**Response** (200 OK):
```json
{
  "success": true,
  "response": "Here's a function to sort an array...",
  "provider": "anthropic",
  "model": "claude-sonnet-4-20250514",
  "conversation_id": "conv-abc123",
  "tokens_used": {
    "input": 50,
    "output": 250,
    "total": 300
  },
  "duration_ms": 3200,
  "tools_used": []
}
```

**Error Response** (400 Bad Request):
```json
{
  "success": false,
  "error": "Missing required field: message",
  "error_code": "MISSING_FIELD"
}
```

### POST /api/v1/chat/stream

Stream a chat response in real-time.

**Request Body**: Same as `/api/v1/chat`

**Response**: Server-Sent Events (SSE) stream

```bash
curl -X POST http://localhost:8080/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Write code"}'
```

**SSE Events**:
```
event: start
data: {"type":"start","request_id":"req-123","timestamp":"2025-01-07T10:30:00Z"}

event: token
data: {"type":"token","content":"def","index":0}

event: token
data: {"type":"token","content":" sort","index":1}

event: tool_call
data: {"type":"tool_call","tool":"write_file","arguments":{"path":"sort.py"}}

event: end
data: {"type":"end","request_id":"req-123","timestamp":"2025-01-07T10:30:05Z"}
```

**Event Types**:
- `start`: Stream started
- `token`: Response token
- `tool_call`: Tool execution started
- `tool_result`: Tool execution result
- `end`: Stream ended

### Runtime Configuration

These endpoints let IDEs and external clients discover and switch the same
runtime state used by the CLI instead of relying on static preset lists.

#### GET /api/v1/config/effective

Return the effective profile, provider/model override, mode, and available
profiles/modes for the current server session.

```bash
curl http://localhost:8080/api/v1/config/effective
```

#### GET /api/v1/profiles

List profiles from `~/.victor/profiles.yaml`.

```bash
curl http://localhost:8080/api/v1/profiles
```

#### GET /api/v1/modes

List supported workflow modes: `build`, `plan`, `review`, `delegate`, and
`explore`.

```bash
curl http://localhost:8080/api/v1/modes
```

#### POST /api/v1/profile/switch

Switch the active profile for subsequent chat requests and clear any explicit
provider/model override.

```bash
curl -X POST http://localhost:8080/api/v1/profile/switch \
  -H "Content-Type: application/json" \
  -d '{"profile":"local"}'
```

#### POST /api/v1/model/switch

Override the active provider/model without changing the selected profile.

```bash
curl -X POST http://localhost:8080/api/v1/model/switch \
  -H "Content-Type: application/json" \
  -d '{"provider":"anthropic","model":"claude-sonnet-4-20250514"}'
```

#### POST /api/v1/mode/switch

Switch workflow mode for subsequent requests.

```bash
curl -X POST http://localhost:8080/api/v1/mode/switch \
  -H "Content-Type: application/json" \
  -d '{"mode":"review"}'
```

### GET /api/v1/capabilities/recommend

Recommend teams and workflows from the shared framework coordination catalogs.

**Query Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `task_type` | string | Yes | Task type or short task label |
| `complexity` | string | Yes | Complexity level such as `low`, `medium`, or `high` |
| `mode` | string | No | Coordination mode. Defaults to `build` |
| `vertical` | string | No | Restrict recommendations to one vertical |

**Example**:
```bash
curl "http://localhost:8080/api/v1/capabilities/recommend?task_type=feature&complexity=high&mode=build"
```

**Response** (200 OK):
```json
{
  "task_type": "feature",
  "complexity": "high",
  "mode": "build",
  "vertical": null,
  "count": 2,
  "recommendations": [
    {
      "vertical": "coding",
      "action": "auto_spawn",
      "primary_team": {"team_name": "feature_team"},
      "primary_workflow": {"workflow_name": "feature_implementation"}
    }
  ]
}
```

### POST /api/v1/conversations

Create or continue a multi-turn conversation.

**Request Body**:
```json
{
  "conversation_id": "string (optional, creates new if not provided)",
  "message": "string (required)",
  "provider": "string (optional)",
  "model": "string (optional)"
}
```

**Example**:
```bash
# First message
curl -X POST http://localhost:8080/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{"message": "Create a REST API"}'

# Response
{
  "conversation_id": "conv-abc123",
  "response": "I'll create a REST API for you...",
  "messages": [
    {"role": "user", "content": "Create a REST API"},
    {"role": "assistant", "content": "I'll create..."}
  ]
}

# Second message (continue conversation)
curl -X POST http://localhost:8080/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv-abc123",
    "message": "Add authentication"
  }'
```

### GET /api/v1/conversations/{id}

Get conversation history.

**Example**:
```bash
curl http://localhost:8080/api/v1/conversations/conv-abc123
```

**Response**:
```json
{
  "conversation_id": "conv-abc123",
  "created_at": "2025-01-07T10:00:00Z",
  "updated_at": "2025-01-07T10:05:00Z",
  "message_count": 5,
  "messages": [
    {"role": "user", "content": "Create a REST API"},
    {"role": "assistant", "content": "I'll create..."},
    {"role": "user", "content": "Add authentication"},
    {"role": "assistant", "content": "Here's the auth..."},
    {"role": "user", "content": "Add tests"}
  ]
}
```

### DELETE /api/v1/conversations/{id}

Delete a conversation.

**Example**:
```bash
curl -X DELETE http://localhost:8080/api/v1/conversations/conv-abc123
```

**Response** (204 No Content)

### POST /api/v1/workflows/run

Execute a workflow.

**Request Body**:
```json
{
  "workflow": "string (workflow name or file path)",
  "input": "object (workflow input)",
  "parameters": {
    "param1": "value1"
  }
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/api/v1/workflows/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "code-review",
    "input": {
      "pr_number": 123,
      "repo": "example/repo"
    }
  }'
```

**Response**:
```json
{
  "success": true,
  "workflow_id": "wf-abc123",
  "status": "completed",
  "result": {
    "review": "Code review complete...",
    "issues_found": 3,
    "suggestions": ["..."]
  },
  "execution_time_ms": 5000
}

# Or for async execution
{
  "success": true,
  "workflow_id": "wf-abc123",
  "status": "running",
  "message": "Workflow started",
  "status_url": "/api/v1/workflows/wf-abc123/status"
}
```

### GET /api/v1/workflows/{id}/status

Get workflow execution status.

**Example**:
```bash
curl http://localhost:8080/api/v1/workflows/wf-abc123/status
```

**Response**:
```json
{
  "workflow_id": "wf-abc123",
  "status": "running",  // running, completed, failed
  "progress": 0.6,
  "current_node": "test",
  "completed_nodes": ["analyze", "fix"],
  "result": null,  // Populated when completed
  "error": null  // Populated if failed
}
```

### GET /api/v1/providers

List available providers.

**Example**:
```bash
curl http://localhost:8080/api/v1/providers
```

**Response**:
```json
{
  "providers": [
    {
      "name": "anthropic",
      "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
      "features": {
        "tool_calling": true,
        "streaming": true,
        "vision": true
      }
    },
    {
      "name": "openai",
      "models": ["gpt-4o", "gpt-4o-mini", "o1-preview"],
      "features": {
        "tool_calling": true,
        "streaming": true,
        "vision": true
      }
    }
  ]
}
```

### GET /api/v1/tools

List available tools.

**Example**:
```bash
curl http://localhost:8080/api/v1/tools
```

**Response**:
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file from the file system",
      "category": "file_operations",
      "parameters": {
        "file_path": {
          "type": "string",
          "description": "Path to the file",
          "required": true
        }
      }
    },
    {
      "name": "write_file",
      "description": "Write content to a file",
      "category": "file_operations",
      "parameters": {
        "file_path": {"type": "string", "required": true},
        "content": {"type": "string", "required": true}
      }
    }
  ],
  "categories": [
    "file_operations",
    "git",
    "testing",
    "search"
  ]
}
```

### GET /health

Health check endpoint.

**Example**:
```bash
curl http://localhost:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": "healthy",
    "providers": "healthy",
    "cache": "healthy"
  }
}
```

### GET /api/v1/status

Server status and statistics.

**Example**:
```bash
curl http://localhost:8080/api/v1/status
```

**Response**:
```json
{
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "requests_total": 1234,
  "requests_active": 5,
  "providers": {
    "anthropic": {"healthy": true, "requests": 500},
    "openai": {"healthy": true, "requests": 300},
    "ollama": {"healthy": false, "error": "Not running"}
  },
  "cache": {
    "enabled": true,
    "size": 100,
    "hit_rate": 0.85
  }
}
```

---

## Authentication

### API Key Authentication

**Enable authentication**:
```yaml
# ~/.victor/config.yaml
server:
  auth:
    enabled: true
    api_key_env: VICTOR_API_KEY
```

**Set API key**:
```bash
export VICTOR_API_KEY=your-api-key
```

**Use in requests**:
```bash
curl http://localhost:8080/api/v1/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### Token-Based Authentication

**Generate token**:
```bash
victor tokens create
# Output: token-abc123...
```

**Use token**:
```bash
curl http://localhost:8080/api/v1/chat \
  -H "Authorization: Bearer token-abc123..." \
  -d '{"message": "Hello!"}'
```

---

## Configuration

### Server Configuration

**~/.victor/config.yaml**:
```yaml
server:
  host: 0.0.0.0
  port: 8080
  workers: 4

  # SSL/TLS
  ssl:
    enabled: false
    cert: /path/to/cert.pem
    key: /path/to/key.pem

  # CORS
  cors:
    enabled: true
    origins: ["*"]
    methods: ["GET", "POST", "PUT", "DELETE"]

  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60

  # Authentication
  auth:
    enabled: false
    api_key_env: VICTOR_API_KEY
```

### Start Options

```bash
# Basic
victor serve

# Custom port
victor serve --port 9000

# SSL/TLS
victor serve --ssl-cert cert.pem --ssl-key key.pem

# Multiple workers
victor serve --workers 4

# Debug mode
victor serve --debug
```

---

## Error Handling

### Error Response Format

All errors follow this format:

```json
{
  "success": false,
  "error": "Error message",
  "error_code": "ERROR_CODE",
  "details": {}
}
```

### HTTP Status Codes

| Code | Meaning | Example |
|------|---------|---------|
| **200** | Success | Chat request successful |
| **400** | Bad Request | Missing required field |
| **401** | Unauthorized | Invalid API key |
| **404** | Not Found | Conversation not found |
| **429** | Rate Limited | Too many requests |
| **500** | Server Error | Internal error |
| **503** | Service Unavailable | Provider not available |

### Error Codes

| Error Code | Description |
|------------|-------------|
| `MISSING_FIELD` | Required field missing |
| `INVALID_VALUE` | Invalid field value |
| `PROVIDER_NOT_FOUND` | Provider not available |
| `MODEL_NOT_FOUND` | Model not found |
| `AUTHENTICATION_FAILED` | Invalid API key |
| `RATE_LIMITED` | Rate limit exceeded |
| `PROVIDER_ERROR` | Provider API error |
| `TOOL_EXECUTION_FAILED` | Tool execution failed |

### Error Handling Example

**Request**:
```bash
curl http://localhost:8080/api/v1/chat \
  -d '{"invalid": "data"}'
```

**Response** (400 Bad Request):
```json
{
  "success": false,
  "error": "Missing required field: message",
  "error_code": "MISSING_FIELD",
  "details": {
    "required_fields": ["message"],
    "provided_fields": ["invalid"]
  }
}
```

---

## Rate Limiting

### Default Limits

- **60 requests/minute** per IP
- **1000 requests/day** per IP

### Rate Limit Headers

```bash
curl -I http://localhost:8080/api/v1/chat
```

**Response Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1704602400
```

### Handling Rate Limits

**Response** (429 Too Many Requests):
```json
{
  "success": false,
  "error": "Rate limit exceeded",
  "error_code": "RATE_LIMITED",
  "details": {
    "limit": 60,
    "window": 60,
    "retry_after": 30
  }
}
```

**Best practice**: Implement exponential backoff

```python
import time
import requests

def make_request_with_retry(url, data, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(url, json=data)

        if response.status_code == 429:
            retry_after = response.json().get('details', {}).get('retry_after', 5)
            time.sleep(retry_after)
            continue

        response.raise_for_status()
        return response.json()
```

---

## Examples

### Example 1: Chat Application

**JavaScript (Fetch)**:
```javascript
async function chat(message) {
  const response = await fetch('http://localhost:8080/api/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': apiKey
    },
    body: JSON.stringify({
      message: message,
      provider: 'anthropic'
    })
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();
  return data.response;
}

// Use
chat("Write a REST API").then(console.log);
```

### Example 2: Streaming Response

**Python (requests)**:
```python
import requests
import json

def stream_chat(message):
    response = requests.post(
        'http://localhost:8080/api/v1/chat/stream',
        json={'message': message},
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if data['type'] == 'token':
                    print(data['content'], end='')
                elif data['type'] == 'end':
                    print('\nDone!')

stream_chat("Write a function")
```

### Example 3: Multi-Turn Conversation

**Python**:
```python
import requests

def conversation():
    # First message
    response = requests.post(
        'http://localhost:8080/api/v1/conversations',
        json={'message': 'Create a REST API'}
    )
    data = response.json()
    conversation_id = data['conversation_id']

    # Continue conversation
    response = requests.post(
        'http://localhost:8080/api/v1/conversations',
        json={
            'conversation_id': conversation_id,
            'message': 'Add authentication'
        }
    )
    return response.json()['response']
```

### Example 4: Execute Workflow

**Bash**:
```bash
# Start workflow
curl -X POST http://localhost:8080/api/v1/workflows/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": "code-review",
    "input": {"pr_number": 123}
  }'

# Response (async)
{
  "workflow_id": "wf-abc123",
  "status": "running",
  "status_url": "/api/v1/workflows/wf-abc123/status"
}

# Check status
curl http://localhost:8080/api/v1/workflows/wf-abc123/status
```

---

## Testing

### Test with curl

```bash
# Health check
curl http://localhost:8080/health

# List providers
curl http://localhost:8080/api/v1/providers

# Simple chat
curl http://localhost:8080/api/v1/chat \
  -d '{"message": "Hello!"}'

# Streaming chat
curl http://localhost:8080/api/v1/chat/stream \
  -d '{"message": "Write code"}'
```

### Test with Python

```python
import requests

def test_chat():
    response = requests.post(
        'http://localhost:8080/api/v1/chat',
        json={'message': 'Hello!'}
    )
    assert response.status_code == 200
    data = response.json()
    assert data['success'] == True
    print("Test passed!")

if __name__ == '__main__':
    test_chat()
```

---

## Deployment

### Docker Deployment

```dockerfile
FROM ghcr.io/vjsingh1984/victor:latest

EXPOSE 8080

CMD ["victor", "serve", "--port", "8080"]
```

**docker-compose.yml**:
```yaml
version: '3'
services:
  victor:
    image: ghcr.io/vjsingh1984/victor:latest
    ports:
      - "8080:8080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - VICTOR_API_KEY=${VICTOR_API_KEY}
    volumes:
      - ~/.victor:/root/.victor
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: victor-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: victor
  template:
    metadata:
      labels:
        app: victor
    spec:
      containers:
      - name: victor
        image: ghcr.io/vjsingh1984/victor:latest
        ports:
        - containerPort: 8080
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: victor-secrets
              key: anthropic-api-key
        - name: VICTOR_API_KEY
          valueFrom:
            secretKeyRef:
              name: victor-secrets
              key: victor-api-key
```

---

## Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8080/metrics
```

**Prometheus format**:
```
# HELP victor_requests_total Total number of requests
victor_requests_total 1234

# HELP victor_request_duration_seconds Request duration
victor_request_duration_seconds_bucket{le="0.1"} 100
victor_request_duration_seconds_bucket{le="1.0"} 900
victor_request_duration_seconds_bucket{le="+Inf"} 1000
```

### Logging

Logs are written to:
- **Console**: stdout/stderr
- **File**: `~/.victor/logs/` (configurable)

**Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

---

## Troubleshooting

### Common Issues

**Server not starting**:
```bash
# Check port availability
lsof -i :8080

# Use different port
victor serve --port 8081
```

**Connection refused**:
```bash
# Check server is running
curl http://localhost:8080/health

# Check firewall
sudo ufw status
```

**Authentication failures**:
```bash
# Verify API key
echo $VICTOR_API_KEY

# Check auth is enabled
victor config show
```

[Full Troubleshooting →](../../user-guide/troubleshooting.md)

---

## Additional Resources

- **API Overview**: [API Reference →](index.md)
- **MCP Server**: [MCP Server →](mcp-server.md)
- **Python API**: [Python API →](python-api.md)
- **Configuration**: [Configuration →](../configuration/index.md)

---

**Next**: [MCP Server →](mcp-server.md) | [Python API →](python-api.md) | [Configuration →](../configuration/index.md)
