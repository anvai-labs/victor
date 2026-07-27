/**
 * Chat View Provider
 *
 * Provides the webview-based chat panel for interacting with Victor.
 * Supports both legacy HTML mode and modern Svelte-built UI.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import { VictorClient, ChatMessage, ToolCall, StreamEvent } from './victorClient';
import { EventBridgeClient, VictorEvent, getEventBridgeClient } from './eventBridgeClient';

export class ChatViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
    public static readonly viewType = 'victor.chatView';
    private _view?: vscode.WebviewView;
    private _messages: ChatMessage[] = [];
    private _disposables: vscode.Disposable[] = [];
    private _webviewReady = false;
    private _useSvelteUI = true; // Use Svelte-built UI by default
    private _activeRequestId: string | null = null;
    private _activeToolCalls: ToolCall[] = [];
    private _trackedToolCalls: Map<string, ToolCall> = new Map();
    private readonly _eventBridge: EventBridgeClient;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _client: VictorClient,
        private readonly _log?: vscode.OutputChannel
    ) {
        this._eventBridge = getEventBridgeClient();
        this._disposables.push(this._eventBridge.onAny((event) => {
            this._handleEventBridgeEvent(event);
        }));

        // Check if Svelte build exists
        const webviewPath = vscode.Uri.joinPath(this._extensionUri, 'out', 'webview', 'index.html');
        try {
            fs.accessSync(webviewPath.fsPath);
            this._useSvelteUI = true;
            this._log?.appendLine('[Chat] Using Svelte-built UI');
        } catch {
            this._useSvelteUI = false;
            this._log?.appendLine('[Chat] Svelte UI not found, using legacy HTML');
        }
    }

    public dispose(): void {
        // Clean up all registered event listeners
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ) {
        this._view = webviewView;
        this._log?.appendLine('[Chat] Webview resolved');

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._useSvelteUI
            ? this._getSvelteHtml(webviewView.webview)
            : this._getHtmlForWebview(webviewView.webview);

        // Handle messages from the webview - store disposable for cleanup
        const messageListener = webviewView.webview.onDidReceiveMessage(async (data) => {
            this._log?.appendLine(`[Chat] Received message from webview: ${data?.type ?? 'unknown'}`);
            try {
                switch (data.type) {
                    case 'webviewReady':
                        this._webviewReady = true;
                        this._log?.appendLine('[Chat] Webview script reported ready');
                        break;
                    case 'sendClick':
                        this._log?.appendLine(`[Chat] Webview send clicked (${data.length ?? 0} chars)`);
                        break;
                    case 'sendMessage':
                        this._log?.appendLine(`[Chat] Webview requested send (${(data.message ?? '').length} chars)`);
                        await this.sendMessage(data.message);
                        break;
                    case 'clearHistory':
                        this._messages = [];
                        this._client.resetChatSession();
                        this._updateMessages();
                        break;
                    case 'applyCode':
                        await this._applyCode(data.code, data.file);
                        break;
                }
            } catch (error) {
                const errorMessage = error instanceof Error ? error.message : String(error);
                console.error('[Chat] Error handling webview message:', errorMessage);
                this._log?.appendLine(`[Chat] Error: ${errorMessage}`);
                // Notify user of error
                vscode.window.showErrorMessage(`Victor chat error: ${errorMessage}`);
                // Also update webview to show error state
                this._postMessage({ type: 'error', message: errorMessage });
            }
        });
        this._disposables.push(messageListener);

        // Clean up when webview is disposed
        const viewDisposedListener = webviewView.onDidDispose(() => {
            this._view = undefined;
            this._webviewReady = false;
        });
        this._disposables.push(viewDisposedListener);

        // Initialize with welcome message
        this._postMessage({
            type: 'init',
            messages: this._messages,
        });
    }

    public async sendMessage(content: string): Promise<void> {
        if (!this._view) {
            this._log?.appendLine('[Chat] Ignoring send (webview not ready)');
            return;
        }

        if (!content || !content.trim()) {
            this._log?.appendLine('[Chat] Ignoring send (empty content)');
            return;
        }

        if (!this._webviewReady) {
            this._log?.appendLine('[Chat] Warning: webviewReady not observed yet');
        }

        this._log?.appendLine(`[Chat] Sending user message (${content.length} chars)`);

        // Add user message
        const userMessage: ChatMessage = { role: 'user', content };
        this._messages.push(userMessage);
        this._updateMessages();

        // Show thinking indicator
        this._postMessage({ type: 'thinking', thinking: true });

        try {
            // Stream response
            let assistantContent = '';
            this._activeRequestId = null;
            this._activeToolCalls = [];
            this._trackedToolCalls.clear();
            this._ensureEventBridgeConnected();

            await this._client.streamChat(
                this._messages,
                (chunk: string) => {
                    assistantContent += chunk;
                    this._postMessage({
                        type: 'stream',
                        content: assistantContent,
                    });
                },
                (toolCall: ToolCall) => {
                    const trackedToolCall = this._upsertTrackedToolCall(toolCall);
                    this._postMessage({
                        type: 'toolCall',
                        toolCall: trackedToolCall,
                    });
                },
                (event: StreamEvent) => {
                    this._handleStreamEvent(event);
                }
            );

            // Add assistant message
            const assistantMessage: ChatMessage = {
                role: 'assistant',
                content: assistantContent,
                toolCalls: this._activeToolCalls.length > 0 ? [...this._activeToolCalls] : undefined,
            };
            this._messages.push(assistantMessage);
            this._updateMessages();

        } catch (error) {
            this._log?.appendLine(`[Chat] Error: ${error}`);

            // Format user-friendly error message
            let errorMessage = 'An unexpected error occurred.';
            let errorDetails = '';

            if (error instanceof Error) {
                const errMsg = error.message.toLowerCase();

                if (errMsg.includes('econnrefused') || errMsg.includes('network')) {
                    errorMessage = 'Unable to connect to Victor server.';
                    errorDetails = 'Make sure the server is running (victor serve).';
                } else if (errMsg.includes('timeout') || errMsg.includes('timed out')) {
                    errorMessage = 'Request timed out.';
                    errorDetails = 'The server took too long to respond. Try a simpler query.';
                } else if (errMsg.includes('rate limit') || errMsg.includes('429')) {
                    errorMessage = 'Rate limit exceeded.';
                    errorDetails = 'Please wait a moment before trying again.';
                } else if (errMsg.includes('unauthorized') || errMsg.includes('401')) {
                    errorMessage = 'Authentication failed.';
                    errorDetails = 'Check your API key configuration.';
                } else if (errMsg.includes('500') || errMsg.includes('internal server')) {
                    errorMessage = 'Server error occurred.';
                    errorDetails = 'Check the Victor Server output for details.';
                } else {
                    errorMessage = error.message;
                }

                this._log?.appendLine(`[Chat] Error type: ${error.name}, message: ${error.message}`);
            }

            this._postMessage({
                type: 'error',
                message: errorMessage,
                details: errorDetails,
            });

            // Show VS Code notification for connection errors
            if (errorMessage.includes('connect')) {
                vscode.window.showWarningMessage(
                    `Victor: ${errorMessage}`,
                    'Show Server Output'
                ).then(selection => {
                    if (selection === 'Show Server Output') {
                        vscode.commands.executeCommand('victor.showServerOutput');
                    }
                });
            }
        } finally {
            this._activeRequestId = null;
            this._activeToolCalls = [];
            this._trackedToolCalls.clear();
            this._log?.appendLine('[Chat] Done');
            this._postMessage({ type: 'thinking', thinking: false });
        }
    }

    private _ensureEventBridgeConnected(): void {
        this._eventBridge.connect(this._client.getServerUrl(), {
            categories: ['tool.start', 'tool.progress', 'tool.complete', 'tool.error'],
        });
    }

    private _handleStreamEvent(event: StreamEvent): void {
        // v1 wire contract: reasoning text streams as `thinking` events.
        if (event.type === 'thinking') {
            this._postMessage({ type: 'thinking', thinking: true, content: event.content });
            return;
        }

        // v1 wire contract: tool results arrive in-stream (with call_id
        // correlation and measured duration), no event bridge required.
        if (event.type === 'tool_result') {
            const status = event.success === false ? 'error' : 'success';
            let tracked: ToolCall | undefined =
                event.callId ? this._trackedToolCalls.get(event.callId) : undefined;
            if (!tracked) {
                tracked = this._upsertTrackedToolCall({
                    id: event.callId,
                    name: event.toolName || 'tool',
                    arguments: {},
                });
            }
            tracked.status = status;
            tracked.result = event.content;
            this._postMessage({
                type: 'toolCallResult',
                id: tracked.id,
                status,
                result: event.content,
                elapsedMs: event.elapsedMs,
                truncated: event.truncated,
            });
            return;
        }

        // Legacy pre-v1 correlation event: wire up the /events bridge.
        if (event.type !== 'request' || !event.requestId) {
            return;
        }

        this._activeRequestId = event.requestId;
        this._log?.appendLine(`[Chat] Request correlation id: ${event.requestId}`);
        this._eventBridge.subscribe(
            ['tool.start', 'tool.progress', 'tool.complete', 'tool.error'],
            event.requestId
        );
        void this._hydrateRecentToolEvents(event.requestId);
    }

    private async _hydrateRecentToolEvents(requestId: string): Promise<void> {
        try {
            const events = await this._client.getRecentEvents({
                limit: 12,
                categories: ['tool.start', 'tool.progress', 'tool.complete', 'tool.error'],
                correlationId: requestId,
            });
            for (const event of events.reverse()) {
                this._handleEventBridgeEvent(event);
            }
        } catch (error) {
            this._log?.appendLine(`[Chat] Failed to hydrate recent tool events: ${error}`);
        }
    }

    private _handleEventBridgeEvent(event: VictorEvent): void {
        if (!this._activeRequestId) {
            return;
        }

        const requestId = typeof event.data.request_id === 'string'
            ? event.data.request_id
            : typeof event.data.correlation_id === 'string'
                ? event.data.correlation_id
                : undefined;
        if (requestId && requestId !== this._activeRequestId) {
            return;
        }

        switch (event.type) {
            case 'tool.start': {
                const toolCall = this._buildToolCallFromBridgeEvent(event);
                if (!toolCall) {
                    return;
                }
                const tracked = this._upsertTrackedToolCall(toolCall);
                this._postMessage({ type: 'toolCall', toolCall: tracked });
                break;
            }
            case 'tool.complete':
            case 'tool.error': {
                const trackedId = this._findTrackedToolCallId(event);
                const status: ToolCall['status'] = event.type === 'tool.complete' ? 'success' : 'error';
                const result = this._formatBridgeResult(event);

                if (trackedId) {
                    const tracked = this._trackedToolCalls.get(trackedId);
                    if (tracked) {
                        tracked.status = status;
                        tracked.result = result;
                    }
                    this._postMessage({
                        type: 'toolCallResult',
                        id: trackedId,
                        status,
                        result,
                    });
                    return;
                }

                const fallbackToolCall = this._buildToolCallFromBridgeEvent(event, status, result);
                if (fallbackToolCall) {
                    const tracked = this._upsertTrackedToolCall(fallbackToolCall);
                    this._postMessage({ type: 'toolCall', toolCall: tracked });
                }
                break;
            }
            default:
                break;
        }
    }

    private _upsertTrackedToolCall(toolCall: ToolCall): ToolCall {
        const id = toolCall.id || `tool-${Date.now()}`;
        const existing = this._trackedToolCalls.get(id);
        if (existing) {
            Object.assign(existing, toolCall, { id });
            return existing;
        }

        const normalized: ToolCall = {
            ...toolCall,
            id,
            status: toolCall.status || 'running',
        };
        this._trackedToolCalls.set(id, normalized);
        this._activeToolCalls.push(normalized);
        return normalized;
    }

    private _findTrackedToolCallId(event: VictorEvent): string | null {
        const toolId = typeof event.data.tool_id === 'string' ? event.data.tool_id : undefined;
        if (toolId && this._trackedToolCalls.has(toolId)) {
            return toolId;
        }

        const toolName = typeof event.data.tool_name === 'string' ? event.data.tool_name : undefined;
        if (!toolName) {
            return null;
        }

        const eventArgs = this._stableStringify(event.data.arguments);
        for (let i = this._activeToolCalls.length - 1; i >= 0; i--) {
            const candidate = this._activeToolCalls[i];
            if (candidate.name !== toolName) {
                continue;
            }
            if (candidate.status === 'success' || candidate.status === 'error') {
                continue;
            }
            if (eventArgs && this._stableStringify(candidate.arguments) !== eventArgs) {
                continue;
            }
            return candidate.id || null;
        }

        return null;
    }

    private _buildToolCallFromBridgeEvent(
        event: VictorEvent,
        status: ToolCall['status'] = 'running',
        result?: unknown
    ): ToolCall | null {
        const toolName = typeof event.data.tool_name === 'string' ? event.data.tool_name : undefined;
        if (!toolName) {
            return null;
        }

        const rawArgs = event.data.arguments;
        const argumentsValue = typeof rawArgs === 'object' && rawArgs !== null
            ? rawArgs as Record<string, unknown>
            : {};
        const toolId = typeof event.data.tool_id === 'string'
            ? event.data.tool_id
            : `bridge-${event.id}`;

        return {
            id: toolId,
            name: toolName,
            arguments: argumentsValue,
            status,
            result,
        };
    }

    private _formatBridgeResult(event: VictorEvent): unknown {
        const preview = event.data.preview;
        if (typeof preview === 'string' && preview.trim()) {
            return preview;
        }
        if (preview !== undefined) {
            return this._stableStringify(preview);
        }

        const excerpt = event.data.result_excerpt;
        if (typeof excerpt === 'string' && excerpt.trim()) {
            return excerpt;
        }

        const error = event.data.error;
        if (typeof error === 'string' && error.trim()) {
            return error;
        }

        return undefined;
    }

    private _stableStringify(value: unknown): string | undefined {
        if (value === undefined) {
            return undefined;
        }
        try {
            return JSON.stringify(value, Object.keys(value as Record<string, unknown>).sort());
        } catch {
            return String(value);
        }
    }

    private async _applyCode(code: string, _file?: string): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor to apply code');
            return;
        }

        const edit = new vscode.WorkspaceEdit();
        const selection = editor.selection;

        if (selection.isEmpty) {
            // Insert at cursor
            edit.insert(editor.document.uri, selection.start, code);
        } else {
            // Replace selection
            edit.replace(editor.document.uri, selection, code);
        }

        await vscode.workspace.applyEdit(edit);
        vscode.window.showInformationMessage('Code applied');
    }

    private _updateMessages(): void {
        this._postMessage({
            type: 'messages',
            messages: this._messages,
        });
    }

    private _postMessage(message: unknown): void {
        if (this._view) {
            this._view.webview.postMessage(message);
        }
    }

    /**
     * Generate HTML for Svelte-built webview UI
     * Loads the pre-built Svelte application from out/webview/
     */
    private _getSvelteHtml(webview: vscode.Webview): string {
        const nonce = this._getNonce();

        // Get URIs for the built Svelte assets
        const scriptUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'out', 'webview', 'assets', 'main.js')
        );
        const styleUri = webview.asWebviewUri(
            vscode.Uri.joinPath(this._extensionUri, 'out', 'webview', 'assets', 'main.css')
        );

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; font-src ${webview.cspSource}; img-src ${webview.cspSource} https: data:;">
    <title>Victor Chat</title>
    <link rel="stylesheet" href="${styleUri}">
</head>
<body>
    <div id="app"></div>
    <script nonce="${nonce}" type="module" src="${scriptUri}"></script>
</body>
</html>`;
    }

    /**
     * Generate legacy HTML (fallback when Svelte build is not available)
     */
    private _getHtmlForWebview(webview: vscode.Webview): string {
        const nonce = this._getNonce();

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
    <title>Victor Chat</title>
    <style>
        :root {
            --bg-primary: var(--vscode-editor-background);
            --bg-secondary: var(--vscode-sideBar-background);
            --text-primary: var(--vscode-editor-foreground);
            --text-secondary: var(--vscode-descriptionForeground);
            --border-color: var(--vscode-panel-border);
            --user-bg: var(--vscode-button-background);
            --assistant-bg: var(--vscode-editor-inactiveSelectionBackground);
            --accent: var(--vscode-button-background);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--text-primary);
            background: var(--bg-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .message {
            max-width: 90%;
            padding: 10px 14px;
            border-radius: 12px;
            line-height: 1.5;
        }

        .message.user {
            align-self: flex-end;
            background: var(--user-bg);
            color: var(--vscode-button-foreground);
        }

        .message.assistant {
            align-self: flex-start;
            background: var(--assistant-bg);
        }

        .message pre {
            background: var(--vscode-textCodeBlock-background);
            padding: 10px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 8px 0;
            font-family: var(--vscode-editor-font-family);
            font-size: 12px;
            position: relative;
        }

        .message pre .copy-btn {
            position: absolute;
            top: 4px;
            right: 4px;
            background: var(--accent);
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .message pre:hover .copy-btn {
            opacity: 1;
        }

        .message code {
            background: var(--vscode-textCodeBlock-background);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
        }

        /* Tool Call - Enhanced Styling */
        .tool-call {
            background: var(--vscode-editorWidget-background);
            border: 1px solid var(--vscode-editorWidget-border);
            border-radius: 8px;
            margin: 8px 0;
            font-size: 12px;
            overflow: hidden;
            transition: border-color 0.2s;
        }

        .tool-call.dangerous {
            border-left: 3px solid var(--vscode-editorWarning-foreground);
        }

        .tool-call-header {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 12px;
            cursor: pointer;
            user-select: none;
        }

        .tool-call-header:hover {
            background: var(--vscode-list-hoverBackground);
        }

        .tool-call-icon {
            font-size: 14px;
            width: 20px;
            text-align: center;
        }

        .tool-call-name {
            font-weight: 600;
            color: var(--vscode-symbolIcon-functionForeground, #dcdcaa);
            flex: 1;
        }

        .tool-call-status {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .tool-call-badge {
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .tool-call-badge.free { background: var(--vscode-testing-iconPassed); color: #000; }
        .tool-call-badge.low { background: var(--vscode-charts-blue); color: #fff; }
        .tool-call-badge.medium { background: var(--vscode-editorWarning-foreground); color: #000; }
        .tool-call-badge.high { background: var(--vscode-errorForeground); color: #fff; }
        .tool-call-badge.danger { background: var(--vscode-errorForeground); color: #fff; }

        .tool-call-status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .tool-call-status-indicator.pending { background: var(--vscode-editorInfo-foreground); }
        .tool-call-status-indicator.running {
            background: var(--vscode-progressBar-background);
            animation: pulse 1s infinite;
        }
        .tool-call-status-indicator.success { background: var(--vscode-testing-iconPassed); }
        .tool-call-status-indicator.error { background: var(--vscode-errorForeground); }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }

        .tool-call-chevron {
            transition: transform 0.2s;
            color: var(--text-secondary);
        }

        .tool-call.expanded .tool-call-chevron {
            transform: rotate(90deg);
        }

        .tool-call-body {
            display: none;
            border-top: 1px solid var(--vscode-editorWidget-border);
            padding: 10px 12px;
        }

        .tool-call.expanded .tool-call-body {
            display: block;
        }

        .tool-call-section {
            margin-bottom: 10px;
        }

        .tool-call-section:last-child {
            margin-bottom: 0;
        }

        .tool-call-section-title {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .tool-call-args {
            background: var(--vscode-textCodeBlock-background);
            padding: 8px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            overflow-x: auto;
            max-height: 150px;
            overflow-y: auto;
        }

        .tool-call-args-key {
            color: var(--vscode-symbolIcon-propertyForeground, #9cdcfe);
        }

        .tool-call-args-value {
            color: var(--vscode-symbolIcon-stringForeground, #ce9178);
        }

        .tool-call-result {
            background: var(--vscode-textCodeBlock-background);
            padding: 8px;
            border-radius: 4px;
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            max-height: 200px;
            overflow: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .tool-call-result.success {
            border-left: 3px solid var(--vscode-testing-iconPassed);
        }

        .tool-call-result.error {
            border-left: 3px solid var(--vscode-errorForeground);
            color: var(--vscode-errorForeground);
        }

        .tool-call-spinner {
            display: inline-flex;
            gap: 3px;
            align-items: center;
        }

        .tool-call-spinner span {
            width: 4px;
            height: 4px;
            background: var(--accent);
            border-radius: 50%;
            animation: spinnerBounce 1.4s infinite ease-in-out;
        }

        .tool-call-spinner span:nth-child(1) { animation-delay: -0.32s; }
        .tool-call-spinner span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes spinnerBounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .tool-call-duration {
            font-size: 10px;
            color: var(--text-secondary);
            margin-left: auto;
        }

        .thinking {
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-secondary);
            padding: 10px;
        }

        .thinking-dots {
            display: flex;
            gap: 4px;
        }

        .thinking-dots span {
            width: 8px;
            height: 8px;
            background: var(--accent);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .thinking-dots span:nth-child(1) { animation-delay: -0.32s; }
        .thinking-dots span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        .input-container {
            padding: 12px;
            border-top: 1px solid var(--border-color);
            background: var(--bg-secondary);
        }

        .input-wrapper {
            display: flex;
            gap: 8px;
        }

        #messageInput {
            flex: 1;
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border);
            color: var(--vscode-input-foreground);
            padding: 10px 12px;
            border-radius: 8px;
            font-family: inherit;
            font-size: inherit;
            resize: none;
            min-height: 40px;
            max-height: 150px;
        }

        #messageInput:focus {
            outline: none;
            border-color: var(--accent);
        }

        #sendBtn {
            background: var(--accent);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 10px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
        }

        #sendBtn:hover {
            opacity: 0.9;
        }

        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .welcome {
            text-align: center;
            color: var(--text-secondary);
            padding: 40px 20px;
        }

        .welcome h2 {
            margin-bottom: 12px;
            color: var(--text-primary);
        }

        .shortcuts {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin-top: 16px;
        }

        .shortcut {
            background: var(--bg-secondary);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            border: 1px solid var(--border-color);
        }

        .shortcut:hover {
            border-color: var(--accent);
        }

        /* Syntax highlighting using VS Code theme colors */
        .hljs-keyword { color: var(--vscode-symbolIcon-keywordForeground, #569cd6); font-weight: 500; }
        .hljs-string { color: var(--vscode-symbolIcon-stringForeground, #ce9178); }
        .hljs-number { color: var(--vscode-symbolIcon-numberForeground, #b5cea8); }
        .hljs-function { color: var(--vscode-symbolIcon-functionForeground, #dcdcaa); }
        .hljs-class { color: var(--vscode-symbolIcon-classForeground, #4ec9b0); }
        .hljs-comment { color: var(--vscode-symbolIcon-colorForeground, #6a9955); font-style: italic; }
        .hljs-operator { color: var(--vscode-symbolIcon-operatorForeground, #d4d4d4); }
        .hljs-variable { color: var(--vscode-symbolIcon-variableForeground, #9cdcfe); }
        .hljs-type { color: var(--vscode-symbolIcon-typeParameterForeground, #4ec9b0); }
        .hljs-property { color: var(--vscode-symbolIcon-propertyForeground, #9cdcfe); }
        .hljs-decorator { color: var(--vscode-symbolIcon-referenceForeground, #dcdcaa); }
        .hljs-punctuation { color: var(--vscode-editor-foreground, #d4d4d4); }
        .hljs-builtin { color: var(--vscode-symbolIcon-methodForeground, #dcdcaa); }
        .hljs-constant { color: var(--vscode-symbolIcon-constantForeground, #4fc1ff); }
        .hljs-tag { color: var(--vscode-symbolIcon-keywordForeground, #569cd6); }
        .hljs-attr { color: var(--vscode-symbolIcon-propertyForeground, #9cdcfe); }

        /* Code block with language label */
        .code-block-wrapper {
            position: relative;
            margin: 8px 0;
        }

        .code-lang-label {
            position: absolute;
            top: 0;
            left: 0;
            background: var(--vscode-badge-background);
            color: var(--vscode-badge-foreground);
            padding: 2px 8px;
            font-size: 10px;
            border-radius: 4px 0 4px 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .code-block-wrapper pre {
            margin-top: 0;
            padding-top: 24px !important;
        }

        /* Apply code button */
        .apply-btn {
            position: absolute;
            top: 4px;
            right: 60px;
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
            border: none;
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .code-block-wrapper:hover .apply-btn,
        .code-block-wrapper:hover .copy-btn {
            opacity: 1;
        }

        /* Markdown enhancements */
        .message h1, .message h2, .message h3, .message h4 {
            margin: 12px 0 8px 0;
            color: var(--text-primary);
        }
        .message h1 { font-size: 1.4em; border-bottom: 1px solid var(--border-color); padding-bottom: 4px; }
        .message h2 { font-size: 1.2em; }
        .message h3 { font-size: 1.1em; }
        .message h4 { font-size: 1em; }

        .message ul, .message ol {
            margin: 8px 0;
            padding-left: 24px;
        }

        .message li { margin: 4px 0; }

        .message blockquote {
            border-left: 3px solid var(--accent);
            margin: 8px 0;
            padding: 4px 12px;
            color: var(--text-secondary);
            background: var(--vscode-textBlockQuote-background);
        }

        .message a {
            color: var(--vscode-textLink-foreground);
            text-decoration: none;
        }
        .message a:hover { text-decoration: underline; }

        .message table {
            border-collapse: collapse;
            margin: 8px 0;
            width: 100%;
        }
        .message th, .message td {
            border: 1px solid var(--border-color);
            padding: 6px 10px;
            text-align: left;
        }
        .message th {
            background: var(--vscode-editorWidget-background);
            font-weight: 600;
        }

        .message hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 12px 0;
        }

        /* Keyboard shortcuts help */
        .kbd {
            background: var(--vscode-keybindingLabel-background);
            border: 1px solid var(--vscode-keybindingLabel-border);
            border-radius: 3px;
            padding: 1px 5px;
            font-family: var(--vscode-editor-font-family);
            font-size: 11px;
            box-shadow: 0 1px 1px rgba(0,0,0,0.1);
        }

        .help-text {
            font-size: 11px;
            color: var(--text-secondary);
            text-align: center;
            padding: 4px;
        }
    </style>
</head>
<body>
    <div class="chat-container" id="chatContainer">
        <div class="welcome" id="welcome">
            <h2>Victor AI Assistant</h2>
            <p>Ask me anything about your code!</p>
            <div class="shortcuts">
                <span class="shortcut" onclick="useShortcut('Explain the selected code')">Explain</span>
                <span class="shortcut" onclick="useShortcut('Refactor this code')">Refactor</span>
                <span class="shortcut" onclick="useShortcut('Write tests for')">Test</span>
                <span class="shortcut" onclick="useShortcut('Fix the bugs in')">Fix</span>
                <span class="shortcut" onclick="useShortcut('Add documentation to')">Document</span>
                <span class="shortcut" onclick="useShortcut('Optimize this code')">Optimize</span>
            </div>
            <div class="help-text" style="margin-top: 20px;">
                <span class="kbd">Enter</span> to send &nbsp;|&nbsp;
                <span class="kbd">Shift+Enter</span> for new line &nbsp;|&nbsp;
                <span class="kbd">Esc</span> to clear
            </div>
        </div>
    </div>

    <div class="thinking" id="thinking" style="display: none;">
        <div class="thinking-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
        <span>Victor is thinking...</span>
    </div>

    <div class="input-container">
        <div class="input-wrapper">
            <textarea
                id="messageInput"
                placeholder="Ask Victor anything..."
                rows="1"
            ></textarea>
            <button id="sendBtn">Send</button>
        </div>
    </div>

    <script nonce="${nonce}">
        const vscode = acquireVsCodeApi();
        // Handshake so the extension knows the webview script loaded
        vscode.postMessage({ type: 'webviewReady' });
        const chatContainer = document.getElementById('chatContainer');
        const messageInput = document.getElementById('messageInput');
        const sendBtn = document.getElementById('sendBtn');
        const thinking = document.getElementById('thinking');
        const welcome = document.getElementById('welcome');

        let isStreaming = false;
        let streamingElement = null;

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });

        // Keyboard shortcuts
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            } else if (e.key === 'Escape') {
                messageInput.value = '';
                messageInput.style.height = 'auto';
                messageInput.blur();
            }
        });

        sendBtn.addEventListener('click', sendMessage);

        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isStreaming) return;

            vscode.postMessage({ type: 'sendClick', length: message.length });
            vscode.postMessage({ type: 'sendMessage', message });
            messageInput.value = '';
            messageInput.style.height = 'auto';
        }

        function useShortcut(prefix) {
            messageInput.value = prefix + ' ';
            messageInput.focus();
        }

        // Handle messages from extension
        window.addEventListener('message', (event) => {
            const data = event.data;

            switch (data.type) {
                case 'init':
                case 'messages':
                    renderMessages(data.messages);
                    break;

                case 'stream':
                    if (!streamingElement) {
                        streamingElement = createMessageElement('assistant', '');
                        chatContainer.appendChild(streamingElement);
                    }
                    streamingElement.innerHTML = formatContent(data.content);
                    scrollToBottom();
                    break;

                case 'thinking':
                    isStreaming = data.thinking;
                    thinking.style.display = data.thinking ? 'flex' : 'none';
                    sendBtn.disabled = data.thinking;
                    if (!data.thinking) {
                        streamingElement = null;
                    }
                    break;

                case 'toolCall':
                    appendToolCall(data.toolCall);
                    break;

                case 'toolCallResult':
                    updateToolCallStatus(data.id, data.status, data.result);
                    break;

                case 'error':
                    appendError(data.message);
                    break;
            }
        });

        function renderMessages(messages) {
            // Hide welcome if messages exist
            welcome.style.display = messages.length > 0 ? 'none' : 'block';

            // Clear existing messages (except welcome)
            const existingMessages = chatContainer.querySelectorAll('.message');
            existingMessages.forEach(m => m.remove());

            messages.forEach(msg => {
                const el = createMessageElement(msg.role, msg.content);
                chatContainer.appendChild(el);

                if (msg.toolCalls) {
                    msg.toolCalls.forEach(tc => {
                        const tcEl = createToolCallElement(tc);
                        chatContainer.appendChild(tcEl);
                    });
                }
            });

            scrollToBottom();
        }

        function createMessageElement(role, content) {
            const el = document.createElement('div');
            el.className = 'message ' + role;
            el.innerHTML = formatContent(content);
            return el;
        }

        // Tool call tracking
        const toolCallElements = new Map();

        function createToolCallElement(toolCall) {
            const el = document.createElement('div');
            const id = toolCall.id || 'tc-' + Date.now();
            el.id = id;
            el.className = 'tool-call' + (toolCall.is_dangerous ? ' dangerous' : '');
            el.dataset.status = toolCall.status || 'running';

            // Get icon based on tool category
            const icon = getToolIcon(toolCall.category || toolCall.name);
            const statusClass = toolCall.status || 'running';
            const costBadge = toolCall.cost_tier ? '<span class="tool-call-badge ' + toolCall.cost_tier.toLowerCase() + '">' + toolCall.cost_tier + '</span>' : '';
            const dangerBadge = toolCall.is_dangerous ? '<span class="tool-call-badge danger">⚠</span>' : '';

            // Format arguments for display
            const argsHtml = formatToolArgs(toolCall.arguments || toolCall.args || {});

            el.innerHTML =
                '<div class="tool-call-header" onclick="toggleToolCall(\\'' + id + '\\')">' +
                    '<span class="tool-call-icon">' + icon + '</span>' +
                    '<span class="tool-call-name">' + escapeHtml(toolCall.name) + '</span>' +
                    '<span class="tool-call-status">' +
                        dangerBadge +
                        costBadge +
                        '<span class="tool-call-status-indicator ' + statusClass + '"></span>' +
                    '</span>' +
                    '<span class="tool-call-chevron">▶</span>' +
                '</div>' +
                '<div class="tool-call-body">' +
                    (argsHtml ? '<div class="tool-call-section"><div class="tool-call-section-title">Arguments</div><div class="tool-call-args">' + argsHtml + '</div></div>' : '') +
                    '<div class="tool-call-section tool-call-result-section" style="display:none;"><div class="tool-call-section-title">Result</div><div class="tool-call-result"></div></div>' +
                '</div>';

            toolCallElements.set(id, el);
            return el;
        }

        function getToolIcon(categoryOrName) {
            const icons = {
                'filesystem': '📁', 'file': '📁',
                'search': '🔍', 'code_search': '🔍',
                'git': '🔀', 'git_tool': '🔀',
                'shell': '💻', 'bash': '💻',
                'analysis': '📊', 'code_review': '📊',
                'web': '🌐', 'web_search': '🌐',
                'docker': '🐳',
                'testing': '🧪', 'test': '🧪',
                'refactor': '🔧',
                'code intelligence': '🧠', 'lsp': '🧠',
                'database': '🗃️',
                'documentation': '📖',
                'infrastructure': '🏗️',
                'batch': '📦',
                'cache': '💾',
                'mcp': '🔌',
                'workflow': '⚙️',
                'patch': '📝', 'edit': '📝',
                'read': '👁️',
                'write': '✏️',
                'default': '🔧'
            };
            const key = (categoryOrName || '').toLowerCase();
            return icons[key] || icons['default'];
        }

        function formatToolArgs(args) {
            if (!args || Object.keys(args).length === 0) return '';

            let html = '';
            for (const [key, value] of Object.entries(args)) {
                const valueStr = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
                const truncated = valueStr.length > 100 ? valueStr.substring(0, 100) + '...' : valueStr;
                html += '<div><span class="tool-call-args-key">' + escapeHtml(key) + ':</span> <span class="tool-call-args-value">' + escapeHtml(truncated) + '</span></div>';
            }
            return html;
        }

        function toggleToolCall(id) {
            const el = document.getElementById(id);
            if (el) {
                el.classList.toggle('expanded');
            }
        }

        function updateToolCallStatus(id, status, result) {
            const el = toolCallElements.get(id) || document.getElementById(id);
            if (!el) return;

            el.dataset.status = status;
            const indicator = el.querySelector('.tool-call-status-indicator');
            if (indicator) {
                indicator.className = 'tool-call-status-indicator ' + status;
            }

            // Update result section if provided
            if (result !== undefined) {
                const resultSection = el.querySelector('.tool-call-result-section');
                const resultDiv = el.querySelector('.tool-call-result');
                if (resultSection && resultDiv) {
                    resultSection.style.display = 'block';
                    resultDiv.className = 'tool-call-result ' + (status === 'error' ? 'error' : 'success');
                    const resultText = typeof result === 'object' ? JSON.stringify(result, null, 2) : String(result);
                    resultDiv.textContent = resultText.length > 500 ? resultText.substring(0, 500) + '...' : resultText;
                }
            }
        }

        function appendToolCall(toolCall) {
            const el = createToolCallElement(toolCall);
            chatContainer.appendChild(el);
            scrollToBottom();
        }

        function appendError(message) {
            const el = document.createElement('div');
            el.className = 'message assistant';
            el.style.color = 'var(--vscode-errorForeground)';
            el.textContent = message;
            chatContainer.appendChild(el);
            scrollToBottom();
        }

        function formatContent(content) {
            if (!content) return '';

            // First handle code blocks (before escaping)
            const codeBlocks = [];
            let processedContent = content.replace(/\`\`\`(\\w*)\\n([\\s\\S]*?)\`\`\`/g, (match, lang, code) => {
                const placeholder = '___CODEBLOCK_' + codeBlocks.length + '___';
                codeBlocks.push({ lang: lang || 'text', code });
                return placeholder;
            });

            // Escape HTML
            let html = escapeHtml(processedContent);

            // Restore code blocks with syntax highlighting
            codeBlocks.forEach((block, i) => {
                const highlighted = highlightCode(block.code, block.lang);
                const langLabel = block.lang ? '<span class="code-lang-label">' + block.lang + '</span>' : '';
                const codeId = 'code-' + Date.now() + '-' + i;
                html = html.replace('___CODEBLOCK_' + i + '___',
                    '<div class="code-block-wrapper">' + langLabel +
                    '<pre><code id="' + codeId + '" class="language-' + block.lang + '">' + highlighted + '</code>' +
                    '<button class="apply-btn" onclick="applyCode(\\'' + codeId + '\\')">Apply</button>' +
                    '<button class="copy-btn" onclick="copyCode(this)">Copy</button></pre></div>'
                );
            });

            // Headers (process before other formatting)
            html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
            html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

            // Blockquotes
            html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

            // Horizontal rules
            html = html.replace(/^---$/gm, '<hr>');

            // Unordered lists
            html = html.replace(/^[\\*\\-] (.+)$/gm, '<li>$1</li>');
            html = html.replace(/(<li>.*<\\/li>\\n?)+/g, '<ul>$&</ul>');

            // Ordered lists
            html = html.replace(/^\\d+\\. (.+)$/gm, '<li>$1</li>');

            // Links [text](url)
            html = html.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');

            // Inline code
            html = html.replace(/\`([^\`]+)\`/g, '<code>$1</code>');

            // Bold
            html = html.replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>');

            // Italic
            html = html.replace(/\\*([^*]+)\\*/g, '<em>$1</em>');

            // Strikethrough
            html = html.replace(/~~([^~]+)~~/g, '<del>$1</del>');

            // Line breaks (but not inside block elements)
            html = html.replace(/\\n/g, '<br>');

            // Clean up extra breaks after block elements
            html = html.replace(/<\\/(h[1-4]|ul|ol|blockquote|pre|hr)><br>/g, '</$1>');
            html = html.replace(/<br><(h[1-4]|ul|ol|blockquote|pre)/g, '<$1');

            return html;
        }

        // Simple syntax highlighting for common languages
        function highlightCode(code, lang) {
            const escaped = escapeHtml(code);

            // Language-specific keyword sets
            const keywords = {
                js: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'extends', 'new', 'this', 'import', 'export', 'from', 'async', 'await', 'try', 'catch', 'throw', 'switch', 'case', 'break', 'default', 'typeof', 'instanceof'],
                ts: ['const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'class', 'extends', 'new', 'this', 'import', 'export', 'from', 'async', 'await', 'try', 'catch', 'throw', 'interface', 'type', 'enum', 'implements', 'private', 'public', 'protected', 'readonly', 'abstract', 'static'],
                python: ['def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'import', 'from', 'as', 'try', 'except', 'finally', 'raise', 'with', 'yield', 'lambda', 'pass', 'break', 'continue', 'and', 'or', 'not', 'in', 'is', 'None', 'True', 'False', 'async', 'await', 'self'],
                rust: ['fn', 'let', 'mut', 'const', 'struct', 'enum', 'impl', 'trait', 'pub', 'use', 'mod', 'if', 'else', 'match', 'for', 'while', 'loop', 'return', 'Self', 'self', 'async', 'await', 'where', 'type', 'dyn', 'move'],
                go: ['func', 'var', 'const', 'type', 'struct', 'interface', 'package', 'import', 'return', 'if', 'else', 'for', 'range', 'switch', 'case', 'default', 'break', 'continue', 'go', 'defer', 'chan', 'select', 'map', 'make', 'new'],
                java: ['public', 'private', 'protected', 'class', 'interface', 'extends', 'implements', 'static', 'final', 'void', 'return', 'if', 'else', 'for', 'while', 'switch', 'case', 'break', 'new', 'this', 'super', 'try', 'catch', 'throw', 'throws', 'import', 'package'],
                cpp: ['int', 'float', 'double', 'char', 'void', 'bool', 'class', 'struct', 'public', 'private', 'protected', 'virtual', 'override', 'const', 'static', 'return', 'if', 'else', 'for', 'while', 'switch', 'case', 'break', 'new', 'delete', 'nullptr', 'template', 'typename', 'namespace', 'using', 'include'],
                bash: ['if', 'then', 'else', 'fi', 'for', 'do', 'done', 'while', 'case', 'esac', 'function', 'return', 'export', 'local', 'echo', 'exit', 'cd', 'pwd', 'ls', 'cat', 'grep', 'sed', 'awk', 'find', 'xargs', 'pip', 'npm', 'git'],
                sql: ['SELECT', 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'ON', 'AND', 'OR', 'NOT', 'IN', 'NULL', 'IS', 'AS', 'ORDER', 'BY', 'GROUP', 'HAVING', 'LIMIT', 'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE', 'CREATE', 'TABLE', 'INDEX', 'DROP', 'ALTER'],
                html: ['html', 'head', 'body', 'div', 'span', 'p', 'a', 'img', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th', 'form', 'input', 'button', 'script', 'style', 'link', 'meta', 'title'],
                css: ['color', 'background', 'margin', 'padding', 'border', 'width', 'height', 'display', 'position', 'top', 'left', 'right', 'bottom', 'flex', 'grid', 'font', 'text', 'align', 'justify', 'transform', 'transition', 'animation']
            };

            // Normalize language name
            const langMap = { javascript: 'js', typescript: 'ts', py: 'python', sh: 'bash', shell: 'bash', c: 'cpp', 'c++': 'cpp' };
            const normalizedLang = langMap[lang?.toLowerCase()] || lang?.toLowerCase() || 'text';
            const langKeywords = keywords[normalizedLang] || [];

            let result = escaped;

            // Comments (single line // and #, multi-line /* */)
            result = result.replace(/(\/\/[^\\n]*)/g, '<span class="hljs-comment">$1</span>');
            result = result.replace(/(#[^\\n]*)/g, '<span class="hljs-comment">$1</span>');
            result = result.replace(/(\\/\\*[\\s\\S]*?\\*\\/)/g, '<span class="hljs-comment">$1</span>');

            // Strings (double and single quotes)
            result = result.replace(/(&quot;[^&]*&quot;)/g, '<span class="hljs-string">$1</span>');
            result = result.replace(/('[^']*')/g, '<span class="hljs-string">$1</span>');
            result = result.replace(/(\`[^\`]*\`)/g, '<span class="hljs-string">$1</span>');

            // Numbers
            result = result.replace(/\\b(\\d+\\.?\\d*)\\b/g, '<span class="hljs-number">$1</span>');

            // Keywords
            if (langKeywords.length > 0) {
                const keywordPattern = new RegExp('\\\\b(' + langKeywords.join('|') + ')\\\\b', 'g');
                result = result.replace(keywordPattern, '<span class="hljs-keyword">$1</span>');
            }

            // Decorators (Python @decorator, TypeScript @Decorator)
            result = result.replace(/(@\\w+)/g, '<span class="hljs-decorator">$1</span>');

            // Function calls
            result = result.replace(/(\\w+)\\(/g, '<span class="hljs-function">$1</span>(');

            // Types/Classes (PascalCase)
            result = result.replace(/\\b([A-Z][a-zA-Z0-9_]*)\\b/g, '<span class="hljs-type">$1</span>');

            return result;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function copyCode(btn) {
            const pre = btn.closest('pre');
            const code = pre.querySelector('code').textContent;
            navigator.clipboard.writeText(code);
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 1500);
        }

        function applyCode(codeId) {
            const codeEl = document.getElementById(codeId);
            if (codeEl) {
                const code = codeEl.textContent;
                vscode.postMessage({ type: 'applyCode', code });
            }
        }

        function scrollToBottom() {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
</body>
</html>`;
    }

    private _getNonce(): string {
        let text = '';
        const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        for (let i = 0; i < 32; i++) {
            text += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        return text;
    }
}
