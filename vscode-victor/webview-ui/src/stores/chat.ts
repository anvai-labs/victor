/**
 * Svelte store for chat state management
 * Uses Svelte's reactive stores for efficient updates
 */

import { writable, derived, type Writable, type Readable } from 'svelte/store';
import type { ChatMessage, ToolCall, ChatState, VsCodeApi, ExtensionMessage } from './types';

// Generate unique ID for messages
function generateId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// Create the main chat state store
function createChatStore() {
  const initialState: ChatState = {
    messages: [],
    isThinking: false,
    isConnected: false,
    currentStreamContent: '',
    error: null,
  };

  const { subscribe, set, update }: Writable<ChatState> = writable(initialState);

  return {
    subscribe,

    // Initialize with messages from extension
    init(messages: ChatMessage[]) {
      update(state => ({
        ...state,
        messages: messages.map(m => ({
          ...m,
          id: m.id || generateId(),
          timestamp: m.timestamp || Date.now(),
        })),
        error: null,
      }));
    },

    // Add a new user message
    addUserMessage(content: string) {
      const message: ChatMessage = {
        id: generateId(),
        role: 'user',
        content,
        timestamp: Date.now(),
      };

      update(state => ({
        ...state,
        messages: [...state.messages, message],
        isThinking: true,
        error: null,
      }));

      return message;
    },

    // Start assistant response (streaming)
    startAssistantResponse() {
      update(state => ({
        ...state,
        currentStreamContent: '',
        isThinking: true,
      }));
    },

    // Update streaming content
    updateStreamContent(content: string) {
      update(state => ({
        ...state,
        currentStreamContent: content,
      }));
    },

    // Finalize assistant response
    finalizeAssistantResponse(content: string, toolCalls?: ToolCall[]) {
      const message: ChatMessage = {
        id: generateId(),
        role: 'assistant',
        content,
        timestamp: Date.now(),
        toolCalls,
      };

      update(state => ({
        ...state,
        messages: [...state.messages, message],
        currentStreamContent: '',
        isThinking: false,
      }));

      return message;
    },

    // Set thinking state
    setThinking(isThinking: boolean, thinkingContent?: string) {
      update(state => ({
        ...state,
        isThinking,
      }));
    },

    // Add or update a tool call
    addToolCall(toolCall: ToolCall) {
      update(state => {
        // Find if there's an ongoing assistant message to attach this to
        const lastMessage = state.messages[state.messages.length - 1];
        if (lastMessage?.role === 'assistant') {
          const toolCalls = lastMessage.toolCalls || [];
          const existingIndex = toolCalls.findIndex(tc => tc.id === toolCall.id);

          if (existingIndex >= 0) {
            toolCalls[existingIndex] = toolCall;
          } else {
            toolCalls.push(toolCall);
          }

          return {
            ...state,
            messages: [
              ...state.messages.slice(0, -1),
              { ...lastMessage, toolCalls },
            ],
          };
        }

        return state;
      });
    },

    // Update tool call result
    updateToolCall(
      id: string,
      status: ToolCall['status'],
      result?: string,
      elapsedMs?: number,
      truncated?: boolean
    ) {
      update(state => {
        const messages = state.messages.map(msg => {
          if (msg.toolCalls) {
            return {
              ...msg,
              toolCalls: msg.toolCalls.map(tc =>
                tc.id === id
                  ? { ...tc, status, result, elapsedMs, truncated, endTime: Date.now() }
                  : tc
              ),
            };
          }
          return msg;
        });

        return { ...state, messages };
      });
    },

    // Set connection status
    setConnected(connected: boolean) {
      update(state => ({
        ...state,
        isConnected: connected,
      }));
    },

    // Set error
    setError(error: string | null) {
      update(state => ({
        ...state,
        error,
        isThinking: false,
      }));
    },

    // Clear all messages
    clear() {
      set(initialState);
    },

    // Replace all messages
    setMessages(messages: ChatMessage[]) {
      update(state => ({
        ...state,
        messages: messages.map(m => ({
          ...m,
          id: m.id || generateId(),
          timestamp: m.timestamp || Date.now(),
        })),
      }));
    },
  };
}

// Create the store instance
export const chatStore = createChatStore();

// Derived store: check if there are any messages
export const hasMessages: Readable<boolean> = derived(
  chatStore,
  $chat => $chat.messages.length > 0
);

// Derived store: get the last message
export const lastMessage: Readable<ChatMessage | null> = derived(
  chatStore,
  $chat => $chat.messages[$chat.messages.length - 1] || null
);

// Derived store: count of pending tool calls
export const pendingToolCalls: Readable<number> = derived(
  chatStore,
  $chat => {
    let count = 0;
    for (const msg of $chat.messages) {
      if (msg.toolCalls) {
        count += msg.toolCalls.filter(tc => tc.status === 'pending' || tc.status === 'running').length;
      }
    }
    return count;
  }
);

// VS Code API singleton
let vscodeApi: VsCodeApi | null = null;

export function getVsCodeApi(): VsCodeApi {
  if (!vscodeApi) {
    // @ts-expect-error - acquireVsCodeApi is injected by VS Code
    vscodeApi = acquireVsCodeApi();
  }
  return vscodeApi!;
}

// Send message to extension
export function sendToExtension(message: unknown): void {
  getVsCodeApi().postMessage(message);
}

// Handle messages from extension
export function handleExtensionMessage(data: ExtensionMessage): void {
  switch (data.type) {
    case 'init':
    case 'messages':
      chatStore.setMessages(data.messages);
      break;

    case 'stream':
      chatStore.updateStreamContent(data.content);
      break;

    case 'thinking':
      chatStore.setThinking(data.thinking, data.content);
      break;

    case 'toolCall':
      chatStore.addToolCall(data.toolCall);
      break;

    case 'toolCallResult':
      chatStore.updateToolCall(data.id, data.status, data.result, data.elapsedMs, data.truncated);
      break;

    case 'error':
      chatStore.setError(data.message);
      break;

    case 'connected':
      chatStore.setConnected(data.connected);
      break;
  }
}
