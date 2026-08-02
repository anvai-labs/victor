// Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

//! PyO3 wrappers around `victor_protocol` types — FEP-0010 Phase 4.
//!
//! The `victor_protocol` crate is intentionally PyO3-free (it is built for edge
//! deployments where PyO3 is unavailable, per its own design). These wrappers
//! live here in `python-bindings` and convert to/from the canonical
//! `victor_protocol` types, so Python and the edge runtime share ONE shape at
//! the FFI boundary. Drift between the two is caught at the conversion boundary
//! (and by the round-trip tests).
//!
//! The Python orchestrator keeps its own dataclasses as the public API; these
//! `victor_native.*` types are the internal FFI agreement.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use victor_protocol::{Message, Role, StreamChunk, ToolCall, Usage};

// =============================================================================
// Role
// =============================================================================

/// Sender role for a chat message (mirrors ``victor_protocol::Role``).
#[pyclass(name = "Role", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRole {
    System,
    User,
    Assistant,
    Tool,
}

impl From<Role> for PyRole {
    fn from(r: Role) -> Self {
        match r {
            Role::System => PyRole::System,
            Role::User => PyRole::User,
            Role::Assistant => PyRole::Assistant,
            Role::Tool => PyRole::Tool,
        }
    }
}

impl From<PyRole> for Role {
    fn from(r: PyRole) -> Self {
        match r {
            PyRole::System => Role::System,
            PyRole::User => Role::User,
            PyRole::Assistant => Role::Assistant,
            PyRole::Tool => Role::Tool,
        }
    }
}

#[pymethods]
impl PyRole {
    /// Provider/API string representation ("system" | "user" | "assistant" | "tool").
    #[pyo3(name = "as_str")]
    fn as_str(&self) -> &'static str {
        Role::from(*self).as_str()
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyRole::System => "Role.System",
            PyRole::User => "Role.User",
            PyRole::Assistant => "Role.Assistant",
            PyRole::Tool => "Role.Tool",
        }
    }

    fn __str__(&self) -> &'static str {
        self.as_str()
    }
}

// =============================================================================
// ToolCall
// =============================================================================

/// A tool invocation requested by the model (mirrors ``victor_protocol::ToolCall``).
///
/// ``arguments`` is carried as a JSON string at the Python boundary.
#[pyclass(name = "ToolCall")]
#[derive(Clone)]
pub struct PyToolCall {
    inner: ToolCall,
}

impl From<ToolCall> for PyToolCall {
    fn from(tc: ToolCall) -> Self {
        Self { inner: tc }
    }
}

impl From<PyToolCall> for ToolCall {
    fn from(tc: PyToolCall) -> Self {
        tc.inner
    }
}

#[pymethods]
impl PyToolCall {
    /// Construct a tool call. ``arguments`` is a JSON string (defaults to "{}").
    #[new]
    #[pyo3(signature = (id, name, arguments="{}".to_string()))]
    fn new(id: String, name: String, arguments: String) -> PyResult<Self> {
        let value = serde_json::from_str(&arguments)
            .map_err(|e| PyValueError::new_err(format!("arguments is not valid JSON: {e}")))?;
        Ok(Self {
            inner: ToolCall {
                id,
                name,
                arguments: value,
            },
        })
    }

    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The tool arguments as a JSON string.
    #[getter]
    fn arguments(&self) -> String {
        self.inner.arguments.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ToolCall(id={:?}, name={:?})", self.inner.id, self.inner.name)
    }
}

// =============================================================================
// Usage
// =============================================================================

/// Token usage statistics (mirrors ``victor_protocol::Usage``).
#[pyclass(name = "Usage")]
#[derive(Clone)]
pub struct PyUsage {
    inner: Usage,
}

impl From<Usage> for PyUsage {
    fn from(u: Usage) -> Self {
        Self { inner: u }
    }
}

impl From<PyUsage> for Usage {
    fn from(u: PyUsage) -> Self {
        u.inner
    }
}

#[pymethods]
impl PyUsage {
    #[new]
    #[pyo3(signature = (prompt_tokens=0, completion_tokens=0, total_tokens=0))]
    fn new(prompt_tokens: usize, completion_tokens: usize, total_tokens: usize) -> Self {
        Self {
            inner: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            },
        }
    }

    #[getter]
    fn prompt_tokens(&self) -> usize {
        self.inner.prompt_tokens
    }

    #[getter]
    fn completion_tokens(&self) -> usize {
        self.inner.completion_tokens
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens
    }
}

// =============================================================================
// Message
// =============================================================================

/// A chat message (mirrors ``victor_protocol::Message``).
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    inner: Message,
}

impl From<Message> for PyMessage {
    fn from(m: Message) -> Self {
        Self { inner: m }
    }
}

impl From<PyMessage> for Message {
    fn from(m: PyMessage) -> Self {
        m.inner
    }
}

#[pymethods]
impl PyMessage {
    /// Construct a message. ``tool_calls`` is a list of ``ToolCall`` (or None).
    #[new]
    #[pyo3(signature = (role, content=None, tool_calls=None, tool_call_id=None))]
    fn new(
        role: PyRole,
        content: Option<String>,
        tool_calls: Option<Vec<Py<PyToolCall>>>,
        tool_call_id: Option<String>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let tool_calls = match tool_calls {
            Some(v) => {
                let mut out = Vec::with_capacity(v.len());
                for tc in v {
                    out.push(tc.borrow(py).inner.clone());
                }
                Some(out)
            }
            None => None,
        };
        Ok(Self {
            inner: Message {
                role: role.into(),
                content,
                tool_calls,
                tool_call_id,
            },
        })
    }

    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }

    #[getter]
    fn content(&self) -> Option<String> {
        self.inner.content.clone()
    }

    #[getter]
    fn tool_call_id(&self) -> Option<String> {
        self.inner.tool_call_id.clone()
    }

    #[getter]
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyToolCall>>>> {
        match &self.inner.tool_calls {
            Some(v) => {
                let mut out = Vec::with_capacity(v.len());
                for tc in v {
                    out.push(Py::new(py, PyToolCall::from(tc.clone()))?);
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Message(role={}, content={:?})",
            self.inner.role, self.inner.content
        )
    }
}

// =============================================================================
// StreamChunk
// =============================================================================

/// A chunk from a streaming response (mirrors ``victor_protocol::StreamChunk``).
#[pyclass(name = "StreamChunk")]
#[derive(Clone)]
pub struct PyStreamChunk {
    inner: StreamChunk,
}

impl From<StreamChunk> for PyStreamChunk {
    fn from(c: StreamChunk) -> Self {
        Self { inner: c }
    }
}

impl From<PyStreamChunk> for StreamChunk {
    fn from(c: PyStreamChunk) -> Self {
        c.inner
    }
}

#[pymethods]
impl PyStreamChunk {
    #[new]
    #[pyo3(signature = (content=None, tool_calls=None, is_final=false, usage=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        content: Option<String>,
        tool_calls: Option<Vec<Py<PyToolCall>>>,
        is_final: bool,
        usage: Option<Py<PyUsage>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let tool_calls = match tool_calls {
            Some(v) => {
                let mut out = Vec::with_capacity(v.len());
                for tc in v {
                    out.push(tc.borrow(py).inner.clone());
                }
                Some(out)
            }
            None => None,
        };
        let usage = usage.map(|u| u.borrow(py).inner.clone());
        Ok(Self {
            inner: StreamChunk {
                content,
                tool_calls,
                is_final,
                usage,
            },
        })
    }

    #[getter]
    fn content(&self) -> Option<String> {
        self.inner.content.clone()
    }

    #[getter]
    fn is_final(&self) -> bool {
        self.inner.is_final
    }

    #[getter]
    fn usage(&self, py: Python<'_>) -> PyResult<Option<Py<PyUsage>>> {
        match &self.inner.usage {
            Some(u) => Ok(Some(Py::new(py, PyUsage::from(u.clone()))?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn tool_calls(&self, py: Python<'_>) -> PyResult<Option<Vec<Py<PyToolCall>>>> {
        match &self.inner.tool_calls {
            Some(v) => {
                let mut out = Vec::with_capacity(v.len());
                for tc in v {
                    out.push(Py::new(py, PyToolCall::from(tc.clone()))?);
                }
                Ok(Some(out))
            }
            None => Ok(None),
        }
    }
}

// =============================================================================
// FFI round-trip helpers (prove the shape works both directions)
// =============================================================================

/// Echo a Message back unchanged — proves PyO3 can accept and return the
/// protocol wrapper at the FFI boundary (used by the Python round-trip test).
#[pyfunction]
pub fn echo_message(msg: PyMessage) -> PyMessage {
    msg
}

/// Echo a StreamChunk back unchanged.
#[pyfunction]
pub fn echo_stream_chunk(chunk: PyStreamChunk) -> PyStreamChunk {
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_roundtrip() {
        for role in [Role::System, Role::User, Role::Assistant, Role::Tool] {
            let py: PyRole = role.into();
            let back: Role = py.into();
            assert_eq!(role, back);
        }
    }

    #[test]
    fn test_message_roundtrip() {
        let msg = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: serde_json::json!({"path": "/tmp/x"}),
            }]),
            tool_call_id: None,
        };
        // protocol -> wrapper -> protocol preserves the canonical value.
        let py_msg: PyMessage = msg.clone().into();
        let back: Message = py_msg.into();
        assert_eq!(msg, back);
    }

    #[test]
    fn test_stream_chunk_roundtrip() {
        let chunk = StreamChunk {
            content: Some("hi".to_string()),
            tool_calls: None,
            is_final: true,
            usage: Some(Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            }),
        };
        let py_chunk: PyStreamChunk = chunk.clone().into();
        let back: StreamChunk = py_chunk.into();
        assert_eq!(chunk, back);
    }
}
