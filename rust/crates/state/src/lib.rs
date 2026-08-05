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

//! Conversation State Management
//!
//! Pure Rust implementation of Victor's conversation state model for edge deployment.
//! Mirrors the Python `ConversationState` from `victor/agent/conversation_state.py`
//! and the `StateScope`/`IStateManager` protocols from `victor/state/protocols.py`.
//!
//! # Design
//!
//! - [`ConversationState`]: Core state that persists across turns (serializable).
//! - [`SharedState`]: Thread-safe wrapper using `Arc<RwLock<>>` — replaces Python's
//!   `copy.deepcopy()` with concurrent read access and exclusive write.
//! - [`StateScope`]: Four-level scope enum matching the Python `StateScope`.
//! - [`ScopedStateStore`]: Key-value store with scope isolation for multi-level state.
//!
//! # Example
//!
//! ```rust
//! use victor_state::{ConversationState, SharedState, StateScope, ScopedStateStore};
//!
//! // Create and share state across threads
//! let state = ConversationState::new();
//! let shared = SharedState::new(state);
//!
//! // Read access (non-blocking for concurrent readers)
//! {
//!     let guard = shared.read();
//!     assert_eq!(guard.stage, "initial");
//! }
//!
//! // Write access (exclusive)
//! {
//!     let mut guard = shared.write();
//!     guard.stage = "reading".to_string();
//!     guard.message_count += 1;
//! }
//!
//! // Scoped key-value store
//! let mut store = ScopedStateStore::new();
//! store.set(StateScope::Conversation, "tool_history".into(), serde_json::json!(["read"]));
//! assert!(store.get(&StateScope::Conversation, "tool_history").is_some());
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// Conversation state -- the core state that persists across turns.
///
/// Mirrors `ConversationState` from `victor/agent/conversation_state.py`.
/// Uses `serde(flatten)` for the metadata map so that additional fields
/// round-trip through JSON without loss.
///
/// Thread-safe shared access is provided by wrapping in [`SharedState`],
/// which replaces the Python pattern of `copy.deepcopy()` with
/// `Arc<RwLock<>>` for concurrent read access with exclusive write.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ConversationState {
    /// Current conversation stage (e.g., "initial", "reading", "execution").
    pub stage: String,

    /// Ordered list of all tools executed during the conversation.
    pub tool_history: Vec<String>,

    /// Files that have been read/observed during the conversation.
    pub observed_files: Vec<String>,

    /// Files that have been modified during the conversation.
    pub modified_files: Vec<String>,

    /// Number of messages exchanged in the conversation.
    pub message_count: usize,

    /// Last N tools executed (sliding window for pattern detection).
    pub last_tools: Vec<String>,

    /// Confidence in the current stage assignment (0.0 to 1.0).
    pub stage_confidence: f64,

    /// Arbitrary metadata for extensibility. Flattened into the JSON
    /// representation so that unknown fields are preserved on round-trip.
    #[serde(flatten)]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ConversationState {
    /// Create a new `ConversationState` with default values.
    ///
    /// Defaults mirror the Python dataclass defaults:
    /// - stage: "initial"
    /// - stage_confidence: 0.5
    /// - all collections empty
    pub fn new() -> Self {
        Self {
            stage: "initial".to_string(),
            tool_history: Vec::new(),
            observed_files: Vec::new(),
            modified_files: Vec::new(),
            message_count: 0,
            last_tools: Vec::new(),
            stage_confidence: 0.5,
            metadata: HashMap::new(),
        }
    }

    /// Set the current stage, preserving the fluent construction style.
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = stage.into();
        self
    }

    /// Set the stage confidence, preserving the fluent construction style.
    pub fn with_stage_confidence(mut self, confidence: f64) -> Self {
        self.stage_confidence = confidence;
        self
    }

    /// Set the message count, preserving the fluent construction style.
    pub fn with_message_count(mut self, message_count: usize) -> Self {
        self.message_count = message_count;
        self
    }

    /// Insert one metadata entry, preserving the fluent construction style.
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validate invariants that cannot be enforced while keeping public fields.
    pub fn validate(&self) -> Result<(), ConversationStateValidationError> {
        if self.stage.trim().is_empty() {
            return Err(ConversationStateValidationError::EmptyStage);
        }
        if !self.stage_confidence.is_finite() || !(0.0..=1.0).contains(&self.stage_confidence) {
            return Err(ConversationStateValidationError::InvalidStageConfidence(
                self.stage_confidence,
            ));
        }
        Ok(())
    }

    /// Serialize to a JSON string.
    ///
    /// Equivalent to Python's `to_dict()` followed by `json.dumps()`.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize from a JSON string.
    ///
    /// Equivalent to Python's `ConversationState.from_dict(json.loads(s))`.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Validation errors for [`ConversationState`].
#[derive(Clone, Debug, PartialEq)]
pub enum ConversationStateValidationError {
    /// The stage is empty or only whitespace.
    EmptyStage,
    /// Stage confidence must be finite and in the inclusive range 0.0..=1.0.
    InvalidStageConfidence(f64),
}

impl fmt::Display for ConversationStateValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyStage => f.write_str("conversation stage cannot be empty"),
            Self::InvalidStageConfidence(confidence) => write!(
                f,
                "stage confidence must be finite and between 0.0 and 1.0, got {confidence}"
            ),
        }
    }
}

impl Error for ConversationStateValidationError {}

impl Default for ConversationState {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe shared state container.
///
/// Wraps [`ConversationState`] in `Arc<RwLock<>>` for concurrent read access
/// with exclusive write. This eliminates the Python pattern of `copy.deepcopy()`
/// for state snapshots -- readers can access state without blocking each other,
/// and only writes acquire an exclusive lock.
///
/// Clone is cheap (Arc clone), so `SharedState` can be passed to multiple threads.
#[derive(Clone)]
pub struct SharedState {
    inner: Arc<RwLock<ConversationState>>,
}

impl SharedState {
    /// Create a new `SharedState` wrapping the given state.
    pub fn new(state: ConversationState) -> Self {
        Self {
            inner: Arc::new(RwLock::new(state)),
        }
    }

    /// Acquire a read lock on the state.
    ///
    /// Multiple readers can hold the lock simultaneously.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, ConversationState> {
        self.inner.read()
    }

    /// Acquire a write lock on the state.
    ///
    /// Only one writer can hold the lock at a time; blocks readers.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, ConversationState> {
        self.inner.write()
    }

    /// Create a deep-copy snapshot of the current state for checkpointing.
    ///
    /// This replaces Python's `copy.deepcopy(state)` pattern.
    pub fn snapshot(&self) -> ConversationState {
        self.inner.read().clone()
    }

    /// Restore the state from a previously taken snapshot.
    ///
    /// Acquires a write lock and replaces the inner state entirely.
    pub fn restore(&self, state: ConversationState) {
        *self.inner.write() = state;
    }
}

/// State scope for multi-level state management.
///
/// Mirrors `StateScope` from `victor/state/protocols.py`.
/// Four scopes isolate state at different levels of the system:
/// - `Workflow`: Single workflow execution
/// - `Conversation`: Multi-turn conversation
/// - `Team`: Multi-agent team coordination
/// - `Global`: Cross-cutting application state
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum StateScope {
    Workflow,
    Conversation,
    Team,
    Global,
}

/// Simple key-value state store with scope isolation.
///
/// Each [`StateScope`] has its own independent namespace. Operations on one
/// scope never affect another. Supports CRUD, snapshot, and restore per scope.
pub struct ScopedStateStore {
    scopes: HashMap<StateScope, HashMap<String, serde_json::Value>>,
}

impl ScopedStateStore {
    /// Create a new empty store with all scopes initialized.
    pub fn new() -> Self {
        let mut scopes = HashMap::new();
        scopes.insert(StateScope::Workflow, HashMap::new());
        scopes.insert(StateScope::Conversation, HashMap::new());
        scopes.insert(StateScope::Team, HashMap::new());
        scopes.insert(StateScope::Global, HashMap::new());
        Self { scopes }
    }

    /// Get a value by key within a scope.
    pub fn get(&self, scope: &StateScope, key: &str) -> Option<&serde_json::Value> {
        self.scopes.get(scope).and_then(|m| m.get(key))
    }

    /// Set a value by key within a scope.
    pub fn set(&mut self, scope: StateScope, key: String, value: serde_json::Value) {
        self.scopes.entry(scope).or_default().insert(key, value);
    }

    /// Delete a value by key within a scope. Returns the removed value if present.
    pub fn delete(&mut self, scope: &StateScope, key: &str) -> Option<serde_json::Value> {
        self.scopes.get_mut(scope).and_then(|m| m.remove(key))
    }

    /// List all keys in a scope.
    pub fn keys(&self, scope: &StateScope) -> Vec<String> {
        self.scopes
            .get(scope)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }

    /// Create a snapshot (clone) of a single scope's data.
    pub fn snapshot(&self, scope: &StateScope) -> HashMap<String, serde_json::Value> {
        self.scopes.get(scope).cloned().unwrap_or_default()
    }

    /// Restore a scope from a previously taken snapshot.
    ///
    /// Replaces the scope's data entirely with the provided map.
    pub fn restore(&mut self, scope: StateScope, data: HashMap<String, serde_json::Value>) {
        self.scopes.insert(scope, data);
    }
}

impl Default for ScopedStateStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    // ---- ConversationState serialization tests ----

    #[test]
    fn test_conversation_state_defaults() {
        let state = ConversationState::new();
        assert_eq!(state.stage, "initial");
        assert_eq!(state.stage_confidence, 0.5);
        assert_eq!(state.message_count, 0);
        assert!(state.tool_history.is_empty());
        assert!(state.observed_files.is_empty());
        assert!(state.modified_files.is_empty());
        assert!(state.last_tools.is_empty());
        assert!(state.metadata.is_empty());
    }

    #[test]
    fn test_conversation_state_json_roundtrip() {
        let mut state = ConversationState::new();
        state.stage = "execution".to_string();
        state.tool_history = vec!["read".into(), "edit".into()];
        state.observed_files = vec!["main.py".into()];
        state.modified_files = vec!["test.py".into()];
        state.message_count = 5;
        state.last_tools = vec!["edit".into()];
        state.stage_confidence = 0.85;
        state
            .metadata
            .insert("custom_key".to_string(), serde_json::json!("custom_value"));

        let json = state.to_json().unwrap();
        let restored = ConversationState::from_json(&json).unwrap();

        assert_eq!(restored.stage, "execution");
        assert_eq!(restored.tool_history, vec!["read", "edit"]);
        assert_eq!(restored.observed_files, vec!["main.py"]);
        assert_eq!(restored.modified_files, vec!["test.py"]);
        assert_eq!(restored.message_count, 5);
        assert_eq!(restored.last_tools, vec!["edit"]);
        assert_eq!(restored.stage_confidence, 0.85);
        assert_eq!(
            restored.metadata.get("custom_key"),
            Some(&serde_json::json!("custom_value"))
        );
    }

    #[test]
    fn test_conversation_state_default_json_roundtrip() {
        let state = ConversationState::default();
        let json = state.to_json().unwrap();
        let restored = ConversationState::from_json(&json).unwrap();
        assert_eq!(restored.stage, state.stage);
        assert_eq!(restored.stage_confidence, state.stage_confidence);
    }

    #[test]
    fn test_conversation_state_builder_and_validation() {
        let state = ConversationState::new()
            .with_stage("execution")
            .with_stage_confidence(0.8)
            .with_message_count(3)
            .with_metadata("trace_id", serde_json::json!("abc"));

        assert_eq!(state.stage, "execution");
        assert_eq!(state.stage_confidence, 0.8);
        assert_eq!(state.message_count, 3);
        assert_eq!(
            state.metadata.get("trace_id"),
            Some(&serde_json::json!("abc"))
        );
        assert!(state.validate().is_ok());
    }

    #[test]
    fn test_conversation_state_validation_rejects_invalid_values() {
        let empty_stage = ConversationState::new().with_stage(" ");
        assert_eq!(
            empty_stage.validate(),
            Err(ConversationStateValidationError::EmptyStage)
        );

        let invalid_confidence = ConversationState::new().with_stage_confidence(1.5);
        assert_eq!(
            invalid_confidence.validate(),
            Err(ConversationStateValidationError::InvalidStageConfidence(
                1.5
            ))
        );
    }

    #[test]
    fn test_conversation_state_metadata_flattened() {
        // Verify that metadata is flattened into the JSON (not nested under "metadata" key)
        let mut state = ConversationState::new();
        state
            .metadata
            .insert("extra".to_string(), serde_json::json!(42));

        let json = state.to_json().unwrap();
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();

        // The "extra" key should appear at the top level due to #[serde(flatten)]
        assert_eq!(value["extra"], serde_json::json!(42));
        // There should NOT be a nested "metadata" key
        assert!(value.get("metadata").is_none());
    }

    // ---- SharedState tests ----

    #[test]
    fn test_shared_state_read_write() {
        let shared = SharedState::new(ConversationState::new());

        // Write
        {
            let mut guard = shared.write();
            guard.stage = "reading".to_string();
            guard.message_count = 3;
        }

        // Read
        {
            let guard = shared.read();
            assert_eq!(guard.stage, "reading");
            assert_eq!(guard.message_count, 3);
        }
    }

    #[test]
    fn test_shared_state_snapshot_and_restore() {
        let shared = SharedState::new(ConversationState::new());

        // Mutate
        {
            let mut guard = shared.write();
            guard.stage = "execution".to_string();
            guard.tool_history.push("edit".to_string());
        }

        // Snapshot
        let snap = shared.snapshot();
        assert_eq!(snap.stage, "execution");
        assert_eq!(snap.tool_history, vec!["edit"]);

        // Mutate further
        {
            let mut guard = shared.write();
            guard.stage = "verification".to_string();
        }

        // Restore from snapshot
        shared.restore(snap);
        {
            let guard = shared.read();
            assert_eq!(guard.stage, "execution");
        }
    }

    #[test]
    fn test_shared_state_concurrent_reads() {
        let shared = SharedState::new(ConversationState::new());

        {
            let mut guard = shared.write();
            guard.stage = "analysis".to_string();
        }

        // Spawn multiple reader threads
        let mut handles = Vec::new();
        for _ in 0..8 {
            let shared_clone = shared.clone();
            handles.push(thread::spawn(move || {
                let guard = shared_clone.read();
                assert_eq!(guard.stage, "analysis");
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_shared_state_concurrent_read_write() {
        let shared = SharedState::new(ConversationState::new());

        // Writer thread increments message_count 100 times
        let writer = {
            let shared_clone = shared.clone();
            thread::spawn(move || {
                for _ in 0..100 {
                    let mut guard = shared_clone.write();
                    guard.message_count += 1;
                }
            })
        };

        // Reader threads just read without panicking
        let mut readers = Vec::new();
        for _ in 0..4 {
            let shared_clone = shared.clone();
            readers.push(thread::spawn(move || {
                for _ in 0..100 {
                    let guard = shared_clone.read();
                    // message_count is always a valid value (0..=100)
                    assert!(guard.message_count <= 100);
                }
            }));
        }

        writer.join().unwrap();
        for reader in readers {
            reader.join().unwrap();
        }

        // After all threads complete, message_count should be exactly 100
        assert_eq!(shared.read().message_count, 100);
    }

    // ---- StateScope tests ----

    #[test]
    fn test_state_scope_serialization() {
        let scope = StateScope::Conversation;
        let json = serde_json::to_string(&scope).unwrap();
        assert_eq!(json, "\"conversation\"");

        let restored: StateScope = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, StateScope::Conversation);
    }

    #[test]
    fn test_state_scope_all_variants() {
        let variants = vec![
            (StateScope::Workflow, "\"workflow\""),
            (StateScope::Conversation, "\"conversation\""),
            (StateScope::Team, "\"team\""),
            (StateScope::Global, "\"global\""),
        ];

        for (scope, expected_json) in variants {
            let json = serde_json::to_string(&scope).unwrap();
            assert_eq!(json, expected_json);
            let restored: StateScope = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, scope);
        }
    }

    // ---- ScopedStateStore tests ----

    #[test]
    fn test_scoped_store_crud() {
        let mut store = ScopedStateStore::new();

        // Set
        store.set(
            StateScope::Conversation,
            "key1".into(),
            serde_json::json!("value1"),
        );
        store.set(StateScope::Workflow, "key2".into(), serde_json::json!(42));

        // Get
        assert_eq!(
            store.get(&StateScope::Conversation, "key1"),
            Some(&serde_json::json!("value1"))
        );
        assert_eq!(
            store.get(&StateScope::Workflow, "key2"),
            Some(&serde_json::json!(42))
        );

        // Cross-scope isolation
        assert!(store.get(&StateScope::Conversation, "key2").is_none());
        assert!(store.get(&StateScope::Workflow, "key1").is_none());

        // Delete
        let deleted = store.delete(&StateScope::Conversation, "key1");
        assert_eq!(deleted, Some(serde_json::json!("value1")));
        assert!(store.get(&StateScope::Conversation, "key1").is_none());

        // Delete non-existent
        let deleted = store.delete(&StateScope::Conversation, "nonexistent");
        assert!(deleted.is_none());
    }

    #[test]
    fn test_scoped_store_keys() {
        let mut store = ScopedStateStore::new();
        store.set(StateScope::Team, "a".into(), serde_json::json!(1));
        store.set(StateScope::Team, "b".into(), serde_json::json!(2));
        store.set(StateScope::Team, "c".into(), serde_json::json!(3));

        let mut keys = store.keys(&StateScope::Team);
        keys.sort();
        assert_eq!(keys, vec!["a", "b", "c"]);

        // Empty scope
        assert!(store.keys(&StateScope::Global).is_empty());
    }

    #[test]
    fn test_scoped_store_snapshot_and_restore() {
        let mut store = ScopedStateStore::new();
        store.set(StateScope::Workflow, "x".into(), serde_json::json!("hello"));
        store.set(StateScope::Workflow, "y".into(), serde_json::json!(99));

        // Snapshot
        let snap = store.snapshot(&StateScope::Workflow);
        assert_eq!(snap.len(), 2);
        assert_eq!(snap.get("x"), Some(&serde_json::json!("hello")));

        // Mutate
        store.set(StateScope::Workflow, "z".into(), serde_json::json!("new"));
        store.delete(&StateScope::Workflow, "x");
        assert!(store.get(&StateScope::Workflow, "x").is_none());
        assert!(store.get(&StateScope::Workflow, "z").is_some());

        // Restore
        store.restore(StateScope::Workflow, snap);
        assert_eq!(
            store.get(&StateScope::Workflow, "x"),
            Some(&serde_json::json!("hello"))
        );
        assert!(store.get(&StateScope::Workflow, "z").is_none());
    }

    #[test]
    fn test_scoped_store_snapshot_empty_scope() {
        let store = ScopedStateStore::new();
        let snap = store.snapshot(&StateScope::Global);
        assert!(snap.is_empty());
    }

    #[test]
    fn test_scoped_store_overwrite() {
        let mut store = ScopedStateStore::new();
        store.set(StateScope::Global, "key".into(), serde_json::json!("old"));
        store.set(StateScope::Global, "key".into(), serde_json::json!("new"));
        assert_eq!(
            store.get(&StateScope::Global, "key"),
            Some(&serde_json::json!("new"))
        );
    }
}
