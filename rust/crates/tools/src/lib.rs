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

//! Tool Schema Registry
//!
//! Pure Rust implementation of Victor's tool registry for edge deployment.
//! Mirrors the Python `ToolRegistry` from `victor/tools/registry.py` and
//! uses `ToolDefinition` from `victor-protocol`.
//!
//! # Design
//!
//! - [`ToolRegistry`]: Pre-compiled registry storing tool definitions for fast lookup.
//!   Built once at startup, queried per tool-call validation. Uses `AHashMap` for
//!   faster hashing than the standard `HashMap`.
//! - [`validate_tool_call`]: Validates a tool call against registered schemas.
//! - [`ToolCallError`]: Error type for validation failures.
//!
//! # Example
//!
//! ```rust
//! use victor_tools::{ToolRegistry, validate_tool_call};
//! use victor_protocol::ToolDefinition;
//!
//! let mut registry = ToolRegistry::new();
//! registry.register(ToolDefinition {
//!     name: "read_file".to_string(),
//!     description: "Read a file".to_string(),
//!     parameters: serde_json::json!({
//!         "type": "object",
//!         "properties": {
//!             "path": { "type": "string" }
//!         },
//!         "required": ["path"]
//!     }),
//! });
//!
//! assert!(registry.contains("read_file"));
//! assert_eq!(registry.len(), 1);
//!
//! let result = validate_tool_call(
//!     &registry,
//!     "read_file",
//!     &serde_json::json!({"path": "/tmp/test.txt"}),
//! );
//! assert!(result.is_ok());
//! ```

use ahash::AHashMap;
use victor_protocol::{ToolCall, ToolDefinition};

/// Pre-compiled tool registry -- stores tool definitions for fast lookup.
///
/// Built once at startup, queried per tool-call validation. Uses `AHashMap`
/// (from the `ahash` crate) for faster hashing than `std::collections::HashMap`.
///
/// The `schemas_cache` field maintains a pre-built `Vec<ToolDefinition>` that
/// is returned by `get_schemas()` without allocation. It is rebuilt whenever
/// a tool is registered, following the same invalidation pattern as the Python
/// `ToolRegistry._schema_cache`.
#[derive(Clone, Debug)]
pub struct ToolRegistry {
    tools: AHashMap<String, ToolDefinition>,
    schemas_cache: Vec<ToolDefinition>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: AHashMap::new(),
            schemas_cache: Vec::new(),
        }
    }

    /// Register a tool definition.
    ///
    /// If a tool with the same name already exists, it is replaced.
    /// The schemas cache is rebuilt after each registration.
    pub fn register(&mut self, tool: ToolDefinition) {
        self.tools.insert(tool.name.clone(), tool);
        self.rebuild_cache();
    }

    /// Register several tool definitions and rebuild the schemas cache once.
    ///
    /// This avoids repeated full-cache clones when tools are loaded in batches
    /// during startup.
    pub fn register_many<I>(&mut self, tools: I)
    where
        I: IntoIterator<Item = ToolDefinition>,
    {
        for tool in tools {
            self.tools.insert(tool.name.clone(), tool);
        }
        self.rebuild_cache();
    }

    /// Get a tool definition by name.
    pub fn get(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    /// Get all registered tool schemas as a slice.
    ///
    /// Returns a pre-built slice -- no allocation on each call.
    /// Mirrors `ToolRegistry.get_tool_schemas()` in Python.
    pub fn get_schemas(&self) -> &[ToolDefinition] {
        &self.schemas_cache
    }

    /// Check whether a tool with the given name is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Return all registered tool names.
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Unregister a tool by name. Returns `true` if the tool existed.
    pub fn unregister(&mut self, name: &str) -> bool {
        let removed = self.tools.remove(name).is_some();
        if removed {
            self.rebuild_cache();
        }
        removed
    }

    /// Build a registry from a JSON array of tool definitions.
    ///
    /// The JSON must be a top-level array where each element is a
    /// `ToolDefinition` object with `name`, `description`, and `parameters`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let tools: Vec<ToolDefinition> = serde_json::from_str(json)?;
        let mut registry = Self::new();
        for tool in tools {
            registry.tools.insert(tool.name.clone(), tool);
        }
        registry.rebuild_cache();
        Ok(registry)
    }

    /// Serialize all tool definitions to a JSON array string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.schemas_cache)
    }

    /// Rebuild the schemas cache from the tools map.
    fn rebuild_cache(&mut self) {
        self.schemas_cache = self.tools.values().cloned().collect();
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for tool call validation failures.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ToolCallError {
    /// The tool name is not registered.
    #[error("Unknown tool: '{0}'")]
    UnknownTool(String),
    /// The arguments do not match the tool's parameter schema.
    #[error("Invalid arguments for tool '{tool}': {reason}")]
    InvalidArguments { tool: String, reason: String },
}

/// Validate a tool call against registered schemas.
///
/// Checks:
/// 1. The tool exists in the registry.
/// 2. The arguments are a JSON object (not null, array, etc.).
/// 3. All `required` parameters (from the tool's JSON Schema) are present.
///
/// Returns `Ok(())` if valid, `Err(ToolCallError)` with the reason if invalid.
///
/// Note: This performs structural validation (required fields, type of arguments
/// value). Full JSON Schema validation (type checking individual properties,
/// pattern matching, etc.) is left to a dedicated JSON Schema validator if needed.
pub fn validate_tool_call(
    registry: &ToolRegistry,
    tool_name: &str,
    arguments: &serde_json::Value,
) -> Result<(), ToolCallError> {
    // 1. Check tool exists
    let tool = registry
        .get(tool_name)
        .ok_or_else(|| ToolCallError::UnknownTool(tool_name.to_string()))?;

    // 2. Arguments must be an object
    let args_obj = arguments
        .as_object()
        .ok_or_else(|| ToolCallError::InvalidArguments {
            tool: tool_name.to_string(),
            reason: "arguments must be a JSON object".to_string(),
        })?;

    // 3. Check required parameters are present
    if let Some(required) = tool.parameters.get("required") {
        if let Some(required_arr) = required.as_array() {
            let missing: Vec<&str> = required_arr
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|key| !args_obj.contains_key(*key))
                .collect();

            if !missing.is_empty() {
                return Err(ToolCallError::InvalidArguments {
                    tool: tool_name.to_string(),
                    reason: format!("missing required parameters: {}", missing.join(", ")),
                });
            }
        }
    }

    Ok(())
}

/// Validate a protocol-level tool call against registered schemas.
pub fn validate_protocol_tool_call(
    registry: &ToolRegistry,
    tool_call: &ToolCall,
) -> Result<(), ToolCallError> {
    validate_tool_call(registry, &tool_call.name, &tool_call.arguments)
}

/// Convenience type alias for tool validation results.
pub type ToolCallResult = Result<(), ToolCallError>;

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tool(name: &str, required: &[&str]) -> ToolDefinition {
        let mut props = serde_json::Map::new();
        for &r in required {
            props.insert(r.to_string(), serde_json::json!({ "type": "string" }));
        }

        ToolDefinition {
            name: name.to_string(),
            description: format!("Test tool: {name}"),
            parameters: serde_json::json!({
                "type": "object",
                "properties": props,
                "required": required,
            }),
        }
    }

    // ---- ToolRegistry basic operations ----

    #[test]
    fn test_registry_new_is_empty() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.get_schemas().is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        let tool = make_tool("read_file", &["path"]);
        registry.register(tool.clone());

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert!(registry.contains("read_file"));
        assert!(!registry.contains("write_file"));

        let retrieved = registry.get("read_file").unwrap();
        assert_eq!(retrieved.name, "read_file");
        assert_eq!(retrieved.description, "Test tool: read_file");
    }

    #[test]
    fn test_registry_register_replaces_existing() {
        let mut registry = ToolRegistry::new();
        let tool1 = ToolDefinition {
            name: "read".to_string(),
            description: "v1".to_string(),
            parameters: serde_json::json!({}),
        };
        let tool2 = ToolDefinition {
            name: "read".to_string(),
            description: "v2".to_string(),
            parameters: serde_json::json!({}),
        };

        registry.register(tool1);
        registry.register(tool2);

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get("read").unwrap().description, "v2");
    }

    #[test]
    fn test_registry_register_many() {
        let mut registry = ToolRegistry::new();
        registry.register_many(vec![
            make_tool("read", &["path"]),
            make_tool("write", &["path", "content"]),
        ]);

        assert_eq!(registry.len(), 2);
        assert!(registry.contains("read"));
        assert!(registry.contains("write"));
        assert_eq!(registry.get_schemas().len(), 2);
    }

    #[test]
    fn test_registry_unregister() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("a", &[]));
        registry.register(make_tool("b", &[]));

        assert_eq!(registry.len(), 2);
        assert!(registry.unregister("a"));
        assert_eq!(registry.len(), 1);
        assert!(!registry.contains("a"));
        assert!(registry.contains("b"));

        // Unregister non-existent
        assert!(!registry.unregister("nonexistent"));
    }

    #[test]
    fn test_registry_names() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("alpha", &[]));
        registry.register(make_tool("beta", &[]));
        registry.register(make_tool("gamma", &[]));

        let mut names = registry.names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_registry_get_schemas() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read", &["path"]));
        registry.register(make_tool("write", &["path", "content"]));

        let schemas = registry.get_schemas();
        assert_eq!(schemas.len(), 2);

        let schema_names: Vec<&str> = schemas.iter().map(|s| s.name.as_str()).collect();
        assert!(schema_names.contains(&"read"));
        assert!(schema_names.contains(&"write"));
    }

    // ---- JSON serialization ----

    #[test]
    fn test_registry_json_roundtrip() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));
        registry.register(make_tool("write_file", &["path", "content"]));

        let json = registry.to_json().unwrap();
        let restored = ToolRegistry::from_json(&json).unwrap();

        assert_eq!(restored.len(), 2);
        assert!(restored.contains("read_file"));
        assert!(restored.contains("write_file"));

        // Verify tool details survived roundtrip
        let tool = restored.get("read_file").unwrap();
        assert_eq!(tool.description, "Test tool: read_file");
    }

    #[test]
    fn test_registry_from_json_empty_array() {
        let registry = ToolRegistry::from_json("[]").unwrap();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_from_json_invalid() {
        let result = ToolRegistry::from_json("not valid json");
        assert!(result.is_err());
    }

    // ---- validate_tool_call ----

    #[test]
    fn test_validate_tool_call_success() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));

        let result = validate_tool_call(
            &registry,
            "read_file",
            &serde_json::json!({"path": "/tmp/test.txt"}),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_protocol_tool_call_success() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));
        let tool_call = ToolCall::new(
            "call_123",
            "read_file",
            serde_json::json!({"path": "/tmp/test.txt"}),
        )
        .unwrap();

        assert!(validate_protocol_tool_call(&registry, &tool_call).is_ok());
    }

    #[test]
    fn test_validate_tool_call_with_extra_args() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));

        // Extra arguments beyond required are allowed
        let result = validate_tool_call(
            &registry,
            "read_file",
            &serde_json::json!({"path": "/tmp/test.txt", "encoding": "utf-8"}),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_tool_call_unknown_tool() {
        let registry = ToolRegistry::new();
        let result = validate_tool_call(&registry, "nonexistent", &serde_json::json!({}));
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolCallError::UnknownTool(name) => assert_eq!(name, "nonexistent"),
            other => panic!("Expected UnknownTool, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_tool_call_non_object_arguments() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));

        let result =
            validate_tool_call(&registry, "read_file", &serde_json::json!("not an object"));
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolCallError::InvalidArguments { tool, reason } => {
                assert_eq!(tool, "read_file");
                assert!(reason.contains("JSON object"));
            }
            other => panic!("Expected InvalidArguments, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_tool_call_missing_required() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("write_file", &["path", "content"]));

        // Missing "content"
        let result = validate_tool_call(
            &registry,
            "write_file",
            &serde_json::json!({"path": "/tmp/out.txt"}),
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolCallError::InvalidArguments { tool, reason } => {
                assert_eq!(tool, "write_file");
                assert!(reason.contains("content"));
            }
            other => panic!("Expected InvalidArguments, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_tool_call_missing_all_required() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("write_file", &["path", "content"]));

        let result = validate_tool_call(&registry, "write_file", &serde_json::json!({}));
        assert!(result.is_err());
        match result.unwrap_err() {
            ToolCallError::InvalidArguments { tool, reason } => {
                assert_eq!(tool, "write_file");
                assert!(reason.contains("path"));
                assert!(reason.contains("content"));
            }
            other => panic!("Expected InvalidArguments, got: {other:?}"),
        }
    }

    #[test]
    fn test_validate_tool_call_no_required_params() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("list_files", &[]));

        // No required params -- empty args is valid
        let result = validate_tool_call(&registry, "list_files", &serde_json::json!({}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_null_arguments() {
        let mut registry = ToolRegistry::new();
        registry.register(make_tool("read_file", &["path"]));

        let result = validate_tool_call(&registry, "read_file", &serde_json::Value::Null);
        assert!(result.is_err());
    }

    // ---- ToolCallError Display ----

    #[test]
    fn test_tool_call_error_display() {
        let err1 = ToolCallError::UnknownTool("foo".to_string());
        assert_eq!(format!("{err1}"), "Unknown tool: 'foo'");

        let err2 = ToolCallError::InvalidArguments {
            tool: "bar".to_string(),
            reason: "missing required parameters: path".to_string(),
        };
        assert!(format!("{err2}").contains("bar"));
        assert!(format!("{err2}").contains("path"));
    }
}
