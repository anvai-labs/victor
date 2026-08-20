// Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Victor Protocol Types
//!
//! Portable message and protocol types for the Victor AI framework.
//! These types are designed for use in edge deployments and any context
//! where PyO3 is not available. They mirror the Python Pydantic models
//! used in `victor/providers/` and `victor/framework/events.py`.
//!
//! # Design
//!
//! - Pure Rust with serde (no PyO3 dependency)
//! - All types derive `Serialize`, `Deserialize`, `Clone`, `Debug`
//! - Optional fields use `skip_serializing_if` to produce clean JSON
//!
//! # Example
//!
//! ```rust
//! use victor_protocol::{Message, Role, ToolCall, CompletionResponse, Usage};
//!
//! let msg = Message {
//!     role: Role::User,
//!     content: Some("Hello, world!".to_string()),
//!     tool_calls: None,
//!     tool_call_id: None,
//! };
//!
//! let json = serde_json::to_string(&msg).unwrap();
//! assert!(json.contains("\"role\":\"user\""));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// The sender role for a chat message.
///
/// Serialized values intentionally remain lowercase strings to match provider
/// APIs and the Python protocol models.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System-level instructions.
    System,
    /// End-user input.
    User,
    /// Model-generated responses.
    Assistant,
    /// Tool result messages.
    Tool,
}

impl Role {
    /// Return the provider/API string representation for this role.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A chat message in the LLM conversation protocol.
///
/// Mirrors `victor.providers.base.Message` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Message {
    /// The role of the message sender.
    pub role: Role,

    /// The text content of the message. May be `None` for tool-call-only assistant messages.
    pub content: Option<String>,

    /// Tool calls requested by the assistant. Present only in assistant messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// The ID of the tool call this message is responding to. Present only in tool messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool invocation requested by the model.
///
/// Mirrors the tool_call structure used across all Victor providers.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,

    /// Name of the tool to invoke.
    pub name: String,

    /// Arguments to pass to the tool, as a JSON value.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Create a tool call from a JSON-serializable argument payload.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Serialize,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::to_value(arguments)?,
        })
    }

    /// Deserialize the argument payload into a typed structure.
    pub fn arguments_as<T>(&self) -> Result<T, serde_json::Error>
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_value(self.arguments.clone())
    }
}

/// A completion response from an LLM provider.
///
/// Mirrors `victor.providers.base.CompletionResponse` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CompletionResponse {
    /// The text content of the response.
    pub content: Option<String>,

    /// Tool calls requested by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Token usage statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,

    /// The model identifier that generated this response.
    pub model: String,

    /// The reason the model stopped generating (e.g., "stop", "tool_use", "length").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

impl CompletionResponse {
    /// Create an empty completion response for the given model.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            content: None,
            tool_calls: None,
            usage: None,
            model: model.into(),
            stop_reason: None,
        }
    }

    /// Create a text completion response for the given model.
    pub fn with_content(model: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            content: Some(content.into()),
            ..Self::new(model)
        }
    }

    /// Attach tool calls to this response.
    pub fn tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };
        self
    }

    /// Attach usage statistics to this response.
    pub fn usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Attach a stop reason to this response.
    pub fn stop_reason(mut self, stop_reason: impl Into<String>) -> Self {
        self.stop_reason = Some(stop_reason.into());
        self
    }
}

/// Token usage statistics for a completion request.
///
/// Mirrors `victor.providers.base.Usage` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Usage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,

    /// Number of tokens in the completion.
    pub completion_tokens: usize,

    /// Total tokens used (prompt + completion).
    pub total_tokens: usize,
}

/// A tool definition for the LLM to use.
///
/// Mirrors `victor.framework.tools.ToolDefinition` in Python.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToolDefinition {
    /// The name of the tool.
    pub name: String,

    /// A description of what the tool does.
    pub description: String,

    /// JSON Schema defining the tool's parameters.
    pub parameters: serde_json::Value,
}

/// A chunk from a streaming LLM response.
///
/// Mirrors the streaming event structure used in `victor.framework.events`.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StreamChunk {
    /// Incremental text content in this chunk.
    pub content: Option<String>,

    /// Incremental tool call data in this chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Whether this is the final chunk in the stream.
    pub is_final: bool,

    /// Token usage statistics (typically only present in the final chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message {
            role: Role::User,
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
        // Optional fields with None should be omitted
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_message_with_tool_calls() {
        let msg = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "/tmp/test.txt"}),
            }]),
            tool_call_id: None,
        };

        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);

        let tool_calls = deserialized.tool_calls.unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "read_file");
    }

    #[test]
    fn test_tool_message() {
        let msg = Message {
            role: Role::Tool,
            content: Some("File contents here".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_123".to_string()),
        };

        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"tool_call_id\":\"call_123\""));

        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, deserialized);
    }

    #[test]
    fn test_completion_response() {
        let resp = CompletionResponse::with_content("gpt-4", "Hello!")
            .usage(Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            })
            .stop_reason("stop");

        let json = serde_json::to_string(&resp).unwrap();
        let deserialized: CompletionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp, deserialized);
        assert_eq!(deserialized.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn test_tool_definition() {
        let tool = ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file from disk".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read"
                    }
                },
                "required": ["path"]
            }),
        };

        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(tool, deserialized);
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct ReadFileArgs {
        path: String,
    }

    #[test]
    fn test_tool_call_typed_arguments() {
        let tool_call = ToolCall::new(
            "call_123",
            "read_file",
            ReadFileArgs {
                path: "/tmp/test.txt".to_string(),
            },
        )
        .unwrap();

        let args: ReadFileArgs = tool_call.arguments_as().unwrap();
        assert_eq!(
            args,
            ReadFileArgs {
                path: "/tmp/test.txt".to_string()
            }
        );
    }

    #[test]
    fn test_stream_chunk_intermediate() {
        let chunk = StreamChunk {
            content: Some("Hello".to_string()),
            tool_calls: None,
            is_final: false,
            usage: None,
        };

        let json = serde_json::to_string(&chunk).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("usage"));

        let deserialized: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
        assert!(!deserialized.is_final);
    }

    #[test]
    fn test_stream_chunk_final() {
        let chunk = StreamChunk {
            content: None,
            tool_calls: None,
            is_final: true,
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
        };

        let json = serde_json::to_string(&chunk).unwrap();
        let deserialized: StreamChunk = serde_json::from_str(&json).unwrap();
        assert_eq!(chunk, deserialized);
        assert!(deserialized.is_final);
        assert_eq!(deserialized.usage.unwrap().total_tokens, 150);
    }

    #[test]
    fn test_usage() {
        let usage = Usage {
            prompt_tokens: 42,
            completion_tokens: 13,
            total_tokens: 55,
        };

        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(usage, deserialized);
    }

    #[test]
    fn test_role_rejects_unknown_values() {
        let json = r#"{"role":"developer","content":"nope"}"#;
        let result: Result<Message, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
