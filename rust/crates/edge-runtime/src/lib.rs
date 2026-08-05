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

//! Victor Edge Runtime — lightweight agent for resource-constrained devices.
//!
//! Provides:
//! - HTTP-based LLM provider (connects to cloud APIs or local Ollama)
//! - Conversation state management (via victor-state)
//! - Tool registry and dispatch (via victor-tools)
//! - Agent loop orchestration
//!
//! # Architecture
//!
//! The edge runtime is designed to run standalone without Python. It connects
//! to any OpenAI-compatible API (including local Ollama) and orchestrates
//! tool-augmented conversations.
//!
//! ```text
//! User Input -> EdgeAgent -> HttpProvider -> LLM API
//!                  |              |
//!                  v              v
//!            SharedState    ToolRegistry
//! ```
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
//! let response = agent.chat("Hello!").await?;
//! println!("{response}");
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod config;
pub mod provider;
