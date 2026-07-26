/**
 * Type definitions for Victor chat webview
 */

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  toolCalls?: ToolCall[];
  thinking?: string;
  isStreaming?: boolean;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  status: 'pending' | 'running' | 'success' | 'error';
  result?: string;
  error?: string;
  startTime?: number;
  endTime?: number;
  /** Measured tool duration from the v1 wire contract (tool_result.elapsed_ms). */
  elapsedMs?: number;
  /** Result was bounded upstream or on the wire (tool_result.truncated). */
  truncated?: boolean;
  category?: string;
  costTier?: 'free' | 'low' | 'medium' | 'high';
  isDangerous?: boolean;
}

export interface ChatState {
  messages: ChatMessage[];
  isThinking: boolean;
  isConnected: boolean;
  currentStreamContent: string;
  error: string | null;
}

export interface Settings {
  provider: string;
  model: string;
  mode: 'build' | 'plan' | 'explore';
  theme: 'dark' | 'light';
}

// VS Code API type for webview communication
export interface VsCodeApi {
  postMessage(message: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
}

// Message types from extension to webview
export type ExtensionMessage =
  | { type: 'init'; messages: ChatMessage[] }
  | { type: 'messages'; messages: ChatMessage[] }
  | { type: 'stream'; content: string }
  | { type: 'thinking'; thinking: boolean; content?: string }
  | { type: 'toolCall'; toolCall: ToolCall }
  | {
      type: 'toolCallResult';
      id: string;
      status: ToolCall['status'];
      result?: string;
      elapsedMs?: number;
      truncated?: boolean;
    }
  | { type: 'error'; message: string }
  | { type: 'connected'; connected: boolean }
  | { type: 'settings'; settings: Settings };

// Message types from webview to extension
export type WebviewMessage =
  | { type: 'webviewReady' }
  | { type: 'sendMessage'; message: string }
  | { type: 'clearHistory' }
  | { type: 'applyCode'; code: string; file?: string }
  | { type: 'cancelRequest' }
  | { type: 'retryToolCall'; id: string };
