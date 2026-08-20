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

//! Edge agent — lightweight agent loop for resource-constrained devices.
//!
//! The agent orchestrates conversations by maintaining message history,
//! sending messages to the LLM provider, and managing conversation state.
//!
//! # Example
//!
//! ```rust,no_run
//! use victor_edge::agent::{EdgeAgent, AgentConfig};
//! use victor_edge::provider::ProviderConfig;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AgentConfig {
//!     provider: ProviderConfig::default(),
//!     system_prompt: "You are a helpful assistant.".to_string(),
//!     max_turns: 10,
//! };
//!
//! let mut agent = EdgeAgent::new(config);
//!
//! // Single turn
//! let response = agent.chat("What is 2 + 2?").await?;
//! println!("{response}");
//!
//! // Multi-turn task execution
//! let result = agent.run("Explain quicksort step by step").await?;
//! println!("{result}");
//! # Ok(())
//! # }
//! ```

use tracing::{debug, info, warn};
use victor_protocol::{Message, Role};
use victor_state::ConversationState;
use victor_tools::ToolRegistry;

use crate::provider::{HttpProvider, ProviderConfig};

/// Edge agent configuration.
#[derive(Clone, Debug)]
pub struct AgentConfig {
    /// LLM provider configuration.
    pub provider: ProviderConfig,

    /// System prompt prepended to every conversation.
    pub system_prompt: String,

    /// Maximum number of turns before the agent stops (safety limit).
    pub max_turns: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider: ProviderConfig::default(),
            system_prompt: "You are a helpful AI assistant.".to_string(),
            max_turns: 20,
        }
    }
}

/// Lightweight agent for edge devices.
///
/// Manages a conversation with an LLM provider, maintaining message history
/// and conversation state. Tool calls are logged but not yet executed
/// (tool execution is planned for a future phase).
pub struct EdgeAgent {
    provider: HttpProvider,
    state: ConversationState,
    tools: ToolRegistry,
    config: AgentConfig,
    messages: Vec<Message>,
}

impl EdgeAgent {
    /// Create a new `EdgeAgent` with the given configuration.
    ///
    /// Initializes the provider, shared state, and an empty tool registry.
    /// The system prompt is stored but not added to messages until the first
    /// call to `chat()` or `run()`.
    pub fn new(config: AgentConfig) -> Self {
        let provider = HttpProvider::new(config.provider.clone());
        let state = ConversationState::new();
        let tools = ToolRegistry::new();

        Self {
            provider,
            state,
            tools,
            config,
            messages: Vec::new(),
        }
    }

    /// Run a single turn: send user message, get response.
    ///
    /// Steps:
    /// 1. If this is the first message, prepend the system prompt.
    /// 2. Add the user message to history.
    /// 3. Call the provider with full message history + tools.
    /// 4. If the response contains tool calls, log them (execution is future work).
    /// 5. Add the assistant response to history.
    /// 6. Update conversation state.
    /// 7. Return the content string.
    pub async fn chat(&mut self, user_message: &str) -> Result<String, AgentError> {
        // Prepend system prompt on first message
        if self.messages.is_empty() && !self.config.system_prompt.is_empty() {
            self.messages.push(Message {
                role: Role::System,
                content: Some(self.config.system_prompt.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Add user message
        self.messages.push(Message {
            role: Role::User,
            content: Some(user_message.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        debug!(message_count = self.messages.len(), "Sending chat request");

        // Get tool definitions for the provider
        let tool_defs = if self.tools.is_empty() {
            None
        } else {
            Some(self.tools.get_schemas())
        };

        // Call the provider
        let response = self.provider.chat(&self.messages, tool_defs).await?;

        // Log tool calls if present
        if let Some(ref tool_calls) = response.tool_calls {
            for tc in tool_calls {
                info!(
                    tool = %tc.name,
                    id = %tc.id,
                    "Tool call requested (execution not yet implemented)"
                );
            }
        }

        // Extract content
        let content = response
            .content
            .clone()
            .unwrap_or_else(|| "[no content]".to_string());

        // Add assistant response to history
        self.messages.push(Message {
            role: Role::Assistant,
            content: response.content,
            tool_calls: response.tool_calls,
            tool_call_id: None,
        });

        // Update conversation state
        self.state.message_count += 2; // user + assistant
        if self.state.stage == "initial" {
            self.state.stage = "active".to_string();
        }

        debug!(response_len = content.len(), "Chat turn complete");

        Ok(content)
    }

    /// Run a multi-turn conversation until completion or max turns.
    ///
    /// Sends the initial task as a user message, then continues the conversation
    /// if the model requests tool calls (up to `max_turns` rounds). Currently,
    /// since tool execution is not yet implemented, this effectively runs a
    /// single turn.
    pub async fn run(&mut self, task: &str) -> Result<String, AgentError> {
        info!(task = %task, max_turns = self.config.max_turns, "Starting agent run");

        let last_response = self.chat(task).await?;
        let turns = 1;

        // Continue if the model wants to use tools (future: execute them)
        // Note: Tool execution not yet implemented, so we stop after first turn
        // When implemented, increment turns in the loop
        if turns < self.config.max_turns {
            let has_tool_calls = self
                .messages
                .last()
                .and_then(|m| m.tool_calls.as_ref())
                .map(|tc| !tc.is_empty())
                .unwrap_or(false);

            if !has_tool_calls {
                return Ok(last_response);
            }

            warn!(
                turn = turns,
                "Model requested tool calls but execution is not yet implemented. Stopping."
            );
        }

        if turns >= self.config.max_turns {
            return Err(AgentError::MaxTurns(self.config.max_turns));
        }

        // Update final state
        self.state.stage = "complete".to_string();
        self.state.stage_confidence = 1.0;

        info!(turns = turns, "Agent run complete");
        Ok(last_response)
    }

    /// Get a snapshot of the current conversation state.
    pub fn state(&self) -> ConversationState {
        self.state.clone()
    }

    /// Get the full conversation message history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Clear the conversation history and reset state.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.state = ConversationState::new();
        debug!("Conversation cleared");
    }

    /// Get a mutable reference to the tool registry for registration.
    pub fn tools_mut(&mut self) -> &mut ToolRegistry {
        &mut self.tools
    }

    /// Get a reference to the tool registry.
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }
}

/// Errors from the edge agent.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Error from the LLM provider.
    #[error("Provider error: {0}")]
    Provider(#[from] crate::provider::ProviderError),

    /// The maximum number of turns was exceeded.
    #[error("Max turns ({0}) exceeded")]
    MaxTurns(usize),
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_config_default() {
        let config = AgentConfig::default();
        assert_eq!(config.system_prompt, "You are a helpful AI assistant.");
        assert_eq!(config.max_turns, 20);
        assert_eq!(config.provider.model, "qwen2.5-coder:1.5b");
    }

    #[test]
    fn test_agent_creation() {
        let agent = EdgeAgent::new(AgentConfig::default());
        assert!(agent.messages().is_empty());
        assert!(agent.tools().is_empty());

        let state = agent.state();
        assert_eq!(state.stage, "initial");
        assert_eq!(state.message_count, 0);
    }

    #[test]
    fn test_agent_clear() {
        let mut agent = EdgeAgent::new(AgentConfig::default());

        // Simulate some state changes
        agent.state.stage = "active".to_string();
        agent.state.message_count = 5;
        agent.messages.push(Message {
            role: Role::User,
            content: Some("test".to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        // Clear
        agent.clear();

        assert!(agent.messages().is_empty());
        let state = agent.state();
        assert_eq!(state.stage, "initial");
        assert_eq!(state.message_count, 0);
    }

    #[test]
    fn test_agent_tools_registration() {
        let mut agent = EdgeAgent::new(AgentConfig::default());

        agent.tools_mut().register(victor_protocol::ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                },
                "required": ["path"]
            }),
        });

        assert_eq!(agent.tools().len(), 1);
        assert!(agent.tools().contains("read_file"));
    }

    #[test]
    fn test_agent_error_display() {
        let err = AgentError::MaxTurns(10);
        assert_eq!(format!("{err}"), "Max turns (10) exceeded");
    }
}
