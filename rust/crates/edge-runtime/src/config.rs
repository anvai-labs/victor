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

//! Edge runtime configuration — loadable from JSON files.
//!
//! # Example
//!
//! ```rust
//! use victor_edge::config::EdgeConfig;
//!
//! let config = EdgeConfig::from_json(r#"{
//!     "provider": {
//!         "base_url": "http://localhost:11434",
//!         "model": "llama3.2:3b",
//!         "timeout_secs": 60,
//!         "max_tokens": 2048,
//!         "temperature": 0.5
//!     },
//!     "system_prompt": "You are a coding assistant.",
//!     "max_turns": 10
//! }"#).unwrap();
//!
//! let agent_config = config.to_agent_config();
//! assert_eq!(agent_config.provider.model, "llama3.2:3b");
//! assert_eq!(agent_config.max_turns, 10);
//! ```

use serde::{Deserialize, Serialize};

use crate::agent::AgentConfig;
use crate::provider::ProviderConfig;

/// Default system prompt used when none is specified in configuration.
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful AI assistant.";

/// Default maximum turns if not specified.
const DEFAULT_MAX_TURNS: usize = 20;

/// Edge runtime configuration, loadable from JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeConfig {
    /// LLM provider configuration.
    pub provider: ProviderConfig,

    /// Optional system prompt (uses a sensible default if not set).
    pub system_prompt: Option<String>,

    /// Optional maximum turns (defaults to 20).
    pub max_turns: Option<usize>,
}

impl EdgeConfig {
    /// Parse an `EdgeConfig` from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Load an `EdgeConfig` from a JSON file on disk.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config = Self::from_json(&contents)?;
        Ok(config)
    }

    /// Convert to an `AgentConfig`, filling in defaults for optional fields.
    pub fn to_agent_config(self) -> AgentConfig {
        AgentConfig {
            provider: self.provider,
            system_prompt: self
                .system_prompt
                .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string()),
            max_turns: self.max_turns.unwrap_or(DEFAULT_MAX_TURNS),
        }
    }
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            provider: ProviderConfig::default(),
            system_prompt: None,
            max_turns: Some(DEFAULT_MAX_TURNS),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_config_default() {
        let config = EdgeConfig::default();
        assert!(config.system_prompt.is_none());
        assert_eq!(config.max_turns, Some(20));
        assert_eq!(config.provider.model, "qwen2.5-coder:1.5b");
    }

    #[test]
    fn test_edge_config_from_json() {
        let json = r#"{
            "provider": {
                "base_url": "https://api.openai.com",
                "model": "gpt-4o",
                "api_key": "sk-test",
                "timeout_secs": 30,
                "max_tokens": 8192,
                "temperature": 0.3
            },
            "system_prompt": "You are a coding expert.",
            "max_turns": 5
        }"#;

        let config = EdgeConfig::from_json(json).unwrap();
        assert_eq!(config.provider.base_url, "https://api.openai.com");
        assert_eq!(config.provider.model, "gpt-4o");
        assert_eq!(config.provider.api_key, Some("sk-test".to_string()));
        assert_eq!(config.provider.timeout_secs, 30);
        assert_eq!(config.provider.max_tokens, 8192);
        assert!((config.provider.temperature - 0.3).abs() < f64::EPSILON);
        assert_eq!(
            config.system_prompt,
            Some("You are a coding expert.".to_string())
        );
        assert_eq!(config.max_turns, Some(5));
    }

    #[test]
    fn test_edge_config_from_json_minimal() {
        // Only provider is required; system_prompt and max_turns are optional
        let json = r#"{
            "provider": {
                "base_url": "http://localhost:11434",
                "model": "qwen2.5-coder:1.5b",
                "timeout_secs": 120,
                "max_tokens": 4096,
                "temperature": 0.7
            }
        }"#;

        let config = EdgeConfig::from_json(json).unwrap();
        assert!(config.system_prompt.is_none());
        assert!(config.max_turns.is_none());
    }

    #[test]
    fn test_edge_config_to_agent_config_with_defaults() {
        let config = EdgeConfig {
            provider: ProviderConfig::default(),
            system_prompt: None,
            max_turns: None,
        };

        let agent_config = config.to_agent_config();
        assert_eq!(agent_config.system_prompt, DEFAULT_SYSTEM_PROMPT);
        assert_eq!(agent_config.max_turns, DEFAULT_MAX_TURNS);
    }

    #[test]
    fn test_edge_config_to_agent_config_with_overrides() {
        let config = EdgeConfig {
            provider: ProviderConfig {
                model: "llama3.2:3b".into(),
                ..Default::default()
            },
            system_prompt: Some("Custom prompt".to_string()),
            max_turns: Some(5),
        };

        let agent_config = config.to_agent_config();
        assert_eq!(agent_config.system_prompt, "Custom prompt");
        assert_eq!(agent_config.max_turns, 5);
        assert_eq!(agent_config.provider.model, "llama3.2:3b");
    }

    #[test]
    fn test_edge_config_serialization_roundtrip() {
        let config = EdgeConfig {
            provider: ProviderConfig {
                base_url: "https://api.example.com".into(),
                model: "test-model".into(),
                api_key: Some("key123".into()),
                timeout_secs: 60,
                max_tokens: 2048,
                temperature: 0.5,
            },
            system_prompt: Some("Test prompt".into()),
            max_turns: Some(15),
        };

        let json = serde_json::to_string(&config).unwrap();
        let restored = EdgeConfig::from_json(&json).unwrap();

        assert_eq!(restored.provider.base_url, "https://api.example.com");
        assert_eq!(restored.provider.model, "test-model");
        assert_eq!(restored.system_prompt, Some("Test prompt".to_string()));
        assert_eq!(restored.max_turns, Some(15));
    }

    #[test]
    fn test_edge_config_from_file_nonexistent() {
        let result = EdgeConfig::from_file("/nonexistent/path/config.json");
        assert!(result.is_err());
    }
}
