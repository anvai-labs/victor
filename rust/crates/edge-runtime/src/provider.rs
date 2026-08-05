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

//! HTTP-based LLM provider for edge deployment.
//!
//! Supports two API formats:
//! - **Ollama** (`/api/chat`) — auto-detected when base_url contains `:11434`
//! - **OpenAI-compatible** (`/v1/chat/completions`) — default for everything else
//!
//! # Example
//!
//! ```rust,no_run
//! use victor_edge::provider::{HttpProvider, ProviderConfig};
//! use victor_protocol::{Message, Role};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = HttpProvider::new(ProviderConfig::default());
//!
//! let messages = vec![Message {
//!     role: Role::User,
//!     content: Some("Hello!".to_string()),
//!     tool_calls: None,
//!     tool_call_id: None,
//! }];
//!
//! let response = provider.chat(&messages, None).await?;
//! println!("{:?}", response.content);
//! # Ok(())
//! # }
//! ```

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;
use tracing::debug;
use victor_protocol::{CompletionResponse, Message, ToolCall, ToolDefinition, Usage};

/// Provider configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Base URL for the LLM API (e.g., "http://localhost:11434" for Ollama).
    pub base_url: String,

    /// Model identifier (e.g., "qwen2.5-coder:1.5b", "gpt-4o").
    pub model: String,

    /// Optional API key for authenticated endpoints.
    pub api_key: Option<String>,

    /// Request timeout in seconds.
    pub timeout_secs: u64,

    /// Maximum tokens in the completion response.
    pub max_tokens: usize,

    /// Sampling temperature (0.0 = deterministic, 1.0 = creative).
    pub temperature: f64,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderConfig")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("timeout_secs", &self.timeout_secs)
            .field("max_tokens", &self.max_tokens)
            .field("temperature", &self.temperature)
            .finish()
    }
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: "qwen2.5-coder:1.5b".into(),
            api_key: None,
            timeout_secs: 120,
            max_tokens: 4096,
            temperature: 0.7,
        }
    }
}

impl ProviderConfig {
    /// Returns `true` if the base_url looks like an Ollama endpoint.
    fn is_ollama(&self) -> bool {
        self.base_url.contains(":11434")
    }
}

/// HTTP-based LLM provider -- works with Ollama and OpenAI-compatible APIs.
pub struct HttpProvider {
    client: Client,
    config: ProviderConfig,
}

impl HttpProvider {
    /// Create a new `HttpProvider` with the given configuration.
    pub fn new(config: ProviderConfig) -> Self {
        Self::try_new(config).expect("Failed to build HTTP client")
    }

    /// Try to create a new `HttpProvider` with the given configuration.
    pub fn try_new(config: ProviderConfig) -> Result<Self, ProviderError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()?;
        Ok(Self { client, config })
    }

    /// Send a chat completion request.
    ///
    /// Automatically selects the correct API format (Ollama vs OpenAI-compatible)
    /// based on the configured base URL.
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<CompletionResponse, ProviderError> {
        if self.config.is_ollama() {
            self.chat_ollama(messages, tools).await
        } else {
            self.chat_openai(messages, tools).await
        }
    }

    /// Send a chat request using the Ollama API format (`/api/chat`).
    async fn chat_ollama(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<CompletionResponse, ProviderError> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));
        debug!(url = %url, model = %self.config.model, "Ollama chat request");

        let ollama_messages: Vec<OllamaMessage> =
            messages.iter().map(OllamaMessage::from_message).collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": false,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            }
        });

        // Add tools if provided
        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                let ollama_tools: Vec<serde_json::Value> = tool_defs
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters,
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::json!(ollama_tools);
            }
        }

        let response = self.client.post(&url).json(&body).send().await?;
        let status = response.status();

        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let ollama_resp: OllamaResponse = response.json().await?;

        // Convert tool calls from Ollama format
        let tool_calls = ollama_resp.message.tool_calls.and_then(|calls| {
            let converted: Vec<ToolCall> = calls
                .into_iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{i}"),
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                })
                .collect();
            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        });

        let usage = ollama_resp.prompt_eval_count.map(|prompt_tokens| {
            let completion_tokens = ollama_resp.eval_count.unwrap_or(0);
            Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }
        });

        Ok(CompletionResponse {
            content: ollama_resp.message.content,
            tool_calls,
            usage,
            model: ollama_resp.model,
            stop_reason: Some("stop".to_string()),
        })
    }

    /// Send a chat request using the OpenAI-compatible API format (`/v1/chat/completions`).
    async fn chat_openai(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
    ) -> Result<CompletionResponse, ProviderError> {
        let url = format!(
            "{}/v1/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        debug!(url = %url, model = %self.config.model, "OpenAI chat request");

        let openai_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut msg = serde_json::json!({
                    "role": m.role.as_str(),
                });
                if let Some(ref content) = m.content {
                    msg["content"] = serde_json::json!(content);
                }
                if let Some(ref tool_calls) = m.tool_calls {
                    let tc: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            serde_json::json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                                }
                            })
                        })
                        .collect();
                    msg["tool_calls"] = serde_json::json!(tc);
                }
                if let Some(ref tool_call_id) = m.tool_call_id {
                    msg["tool_call_id"] = serde_json::json!(tool_call_id);
                }
                msg
            })
            .collect();

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": openai_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        });

        // Add tools if provided
        if let Some(tool_defs) = tools {
            if !tool_defs.is_empty() {
                let openai_tools: Vec<serde_json::Value> = tool_defs
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description,
                                "parameters": t.parameters,
                            }
                        })
                    })
                    .collect();
                body["tools"] = serde_json::json!(openai_tools);
            }
        }

        let mut request = self.client.post(&url).json(&body);
        if let Some(ref api_key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

        let response = request.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                body: body_text,
            });
        }

        let openai_resp: OpenAIResponse = response.json().await?;

        let choice = openai_resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::Api {
                status: 200,
                body: "No choices in response".to_string(),
            })?;

        let tool_calls = match choice.message.tool_calls {
            Some(calls) => {
                let converted = calls
                    .into_iter()
                    .map(|tc| {
                        let arguments =
                            serde_json::from_str(&tc.function.arguments).map_err(|error| {
                                ProviderError::InvalidToolArguments {
                                    tool: tc.function.name.clone(),
                                    arguments: tc.function.arguments.clone(),
                                    error,
                                }
                            })?;
                        Ok(ToolCall {
                            id: tc.id,
                            name: tc.function.name,
                            arguments,
                        })
                    })
                    .collect::<Result<Vec<_>, ProviderError>>()?;
                if converted.is_empty() {
                    None
                } else {
                    Some(converted)
                }
            }
            None => None,
        };

        let usage = openai_resp.usage.map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        });

        Ok(CompletionResponse {
            content: choice.message.content,
            tool_calls,
            usage,
            model: openai_resp.model,
            stop_reason: choice.finish_reason,
        })
    }
}

// ---------------------------------------------------------------------------
// Provider errors
// ---------------------------------------------------------------------------

/// Errors from the HTTP provider.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// HTTP transport error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// API returned a non-success status code.
    #[error("API error: {status} {body}")]
    Api {
        /// HTTP status code.
        status: u16,
        /// Response body text.
        body: String,
    },

    /// Failed to parse the response body.
    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),

    /// Provider returned tool-call arguments that were not valid JSON.
    #[error("Invalid JSON arguments for tool '{tool}': {error}; raw arguments: {arguments}")]
    InvalidToolArguments {
        /// Tool/function name.
        tool: String,
        /// Raw arguments payload returned by the provider.
        arguments: String,
        /// JSON parse failure.
        error: serde_json::Error,
    },

    /// Request timed out.
    #[error("Timeout after {0}s")]
    Timeout(u64),
}

// ---------------------------------------------------------------------------
// Ollama API types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
struct OllamaMessage {
    role: String,
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

impl OllamaMessage {
    fn from_message(msg: &Message) -> Self {
        Self {
            role: msg.role.as_str().to_string(),
            content: msg.content.clone(),
            tool_calls: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaToolCall {
    function: OllamaFunction,
}

#[derive(Serialize, Deserialize, Debug)]
struct OllamaFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    model: String,
    message: OllamaMessage,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

// ---------------------------------------------------------------------------
// OpenAI API types
// ---------------------------------------------------------------------------

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAIMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIToolCall {
    id: String,
    function: OpenAIFunction,
}

#[derive(Deserialize, Debug)]
struct OpenAIFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use victor_protocol::Role;

    #[test]
    fn test_provider_config_default() {
        let config = ProviderConfig::default();
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "qwen2.5-coder:1.5b");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.max_tokens, 4096);
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_provider_config_is_ollama() {
        let ollama_config = ProviderConfig {
            base_url: "http://localhost:11434".into(),
            ..Default::default()
        };
        assert!(ollama_config.is_ollama());

        let openai_config = ProviderConfig {
            base_url: "https://api.openai.com".into(),
            ..Default::default()
        };
        assert!(!openai_config.is_ollama());

        let custom_ollama = ProviderConfig {
            base_url: "http://192.168.1.100:11434".into(),
            ..Default::default()
        };
        assert!(custom_ollama.is_ollama());
    }

    #[test]
    fn test_provider_config_serialization() {
        let config = ProviderConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: ProviderConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.base_url, config.base_url);
        assert_eq!(restored.model, config.model);
        assert_eq!(restored.timeout_secs, config.timeout_secs);
    }

    #[test]
    fn test_provider_config_debug_redacts_api_key() {
        let config = ProviderConfig {
            api_key: Some("sk-secret".to_string()),
            ..Default::default()
        };

        let debug = format!("{config:?}");
        assert!(debug.contains("<redacted>"));
        assert!(!debug.contains("sk-secret"));
    }

    #[test]
    fn test_http_provider_creation() {
        let provider = HttpProvider::new(ProviderConfig::default());
        assert!(provider.config.is_ollama());
    }

    #[test]
    fn test_http_provider_try_new() {
        let provider = HttpProvider::try_new(ProviderConfig::default()).unwrap();
        assert!(provider.config.is_ollama());
    }

    #[test]
    fn test_ollama_message_from_message() {
        let msg = Message {
            role: Role::User,
            content: Some("Hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let ollama_msg = OllamaMessage::from_message(&msg);
        assert_eq!(ollama_msg.role, "user");
        assert_eq!(ollama_msg.content, Some("Hello".to_string()));
        assert!(ollama_msg.tool_calls.is_none());
    }

    #[test]
    fn test_ollama_response_deserialization() {
        let json = r#"{
            "model": "qwen2.5-coder:1.5b",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "prompt_eval_count": 10,
            "eval_count": 8
        }"#;
        let resp: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "qwen2.5-coder:1.5b");
        assert_eq!(
            resp.message.content,
            Some("Hello! How can I help?".to_string())
        );
        assert_eq!(resp.prompt_eval_count, Some(10));
        assert_eq!(resp.eval_count, Some(8));
    }

    #[test]
    fn test_openai_response_deserialization() {
        let json = r#"{
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": "Hi there!",
                    "tool_calls": null
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 5,
                "total_tokens": 20
            }
        }"#;
        let resp: OpenAIResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.model, "gpt-4o");
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(
            resp.choices[0].message.content,
            Some("Hi there!".to_string())
        );
        assert_eq!(resp.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 20);
    }

    #[test]
    fn test_openai_response_with_tool_calls() {
        let json = r#"{
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": "{\"path\": \"/tmp/test.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": null
        }"#;
        let resp: OpenAIResponse = serde_json::from_str(json).unwrap();
        let tool_calls = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, "call_abc123");
        assert_eq!(tool_calls[0].function.name, "read_file");
    }

    #[test]
    fn test_invalid_tool_arguments_error_display() {
        let error = ProviderError::InvalidToolArguments {
            tool: "read_file".to_string(),
            arguments: "{bad json".to_string(),
            error: serde_json::from_str::<serde_json::Value>("{bad json").unwrap_err(),
        };

        let message = format!("{error}");
        assert!(message.contains("read_file"));
        assert!(message.contains("{bad json"));
    }

    #[test]
    fn test_provider_error_display() {
        let http_err = ProviderError::Timeout(30);
        assert_eq!(format!("{http_err}"), "Timeout after 30s");

        let api_err = ProviderError::Api {
            status: 429,
            body: "Rate limited".to_string(),
        };
        assert!(format!("{api_err}").contains("429"));
        assert!(format!("{api_err}").contains("Rate limited"));
    }
}
