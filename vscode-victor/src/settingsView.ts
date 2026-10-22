/**
 * Settings View Provider
 *
 * Provides a webview panel for configuring Victor AI settings:
 * - Provider and model selection
 * - API key management
 * - Tool preferences
 * - Appearance settings
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

interface VictorSettings {
    provider: string;
    model: string;
    mode: string;
    serverUrl: string;
    serverPort: number;
    autoStart: boolean;
    showInlineCompletions: boolean;
    semanticSearchEnabled: boolean;
    semanticSearchMaxResults: number;
}

interface ProviderInfo {
    name: string;
    display_name: string;
    is_local: boolean;
    configured: boolean;
    supports_tools: boolean;
    supports_streaming: boolean;
}

/**
 * Settings View Provider
 */
export class SettingsViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
    public static readonly viewType = 'victor.settingsView';

    private _view?: vscode.WebviewView;
    private _disposables: vscode.Disposable[] = [];

    constructor(private readonly _extensionUri: vscode.Uri) {}

    public dispose(): void {
        // Clean up all registered event listeners
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtmlContent(webviewView.webview);

        // Handle messages from webview - store disposable for cleanup
        const messageListener = webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'saveSettings':
                    await this._saveSettings(data.settings);
                    break;
                case 'loadSettings':
                    this._sendSettings();
                    break;
                case 'loadProviders':
                    await this._loadProviders();
                    break;
                case 'loadModels':
                    await this._loadModels();
                    break;
                case 'testConnection':
                    await this._testConnection(data.settings);
                    break;
            }
        });
        this._disposables.push(messageListener);

        // Clean up when webview is disposed
        const viewDisposedListener = webviewView.onDidDispose(() => {
            this._view = undefined;
        });
        this._disposables.push(viewDisposedListener);

        // Send initial settings
        this._sendSettings();
    }

    private _getSettings(): VictorSettings {
        const config = vscode.workspace.getConfiguration('victor');
        return {
            provider: config.get('provider', 'anthropic'),
            model: config.get('model', 'claude-sonnet-4-20250514'),
            mode: config.get('mode', 'build'),
            serverUrl: this._normalizeServerUrl(config.get('serverUrl', '')),
            serverPort: config.get('serverPort', 8765),
            autoStart: config.get('autoStart', false),
            showInlineCompletions: config.get('showInlineCompletions', true),
            semanticSearchEnabled: config.get('semanticSearch.enabled', true),
            semanticSearchMaxResults: config.get('semanticSearch.maxResults', 10),
        };
    }

    private _normalizeServerUrl(serverUrl: string | undefined): string {
        return (serverUrl || '').trim().replace(/\/+$/, '');
    }

    private _resolveBaseUrl(settings?: Partial<VictorSettings>): string {
        const config = vscode.workspace.getConfiguration('victor');
        const serverUrl = this._normalizeServerUrl(settings?.serverUrl ?? config.get<string>('serverUrl', ''));

        if (serverUrl) {
            let parsed: URL;
            try {
                parsed = new URL(serverUrl);
            } catch {
                throw new Error('Server URL must be a valid http:// or https:// URL');
            }

            if (!['http:', 'https:'].includes(parsed.protocol)) {
                throw new Error('Server URL must use http:// or https://');
            }

            return serverUrl;
        }

        const rawPort = settings?.serverPort ?? config.get<number>('serverPort', 8765);
        const serverPort = Number.isFinite(rawPort) ? rawPort : 8765;
        return `http://localhost:${serverPort}`;
    }

    private async _saveSettings(settings: VictorSettings): Promise<void> {
        const config = vscode.workspace.getConfiguration('victor');
        const serverUrl = this._normalizeServerUrl(settings.serverUrl);

        try {
            if (serverUrl) {
                this._resolveBaseUrl({ serverUrl });
            }
        } catch (error) {
            vscode.window.showErrorMessage(
                error instanceof Error ? error.message : 'Invalid Victor server URL'
            );
            return;
        }

        await config.update('provider', settings.provider, true);
        await config.update('model', settings.model, true);
        await config.update('mode', settings.mode, true);
        await config.update('serverUrl', serverUrl || undefined, true);
        await config.update('serverPort', settings.serverPort, true);
        await config.update('autoStart', settings.autoStart, true);
        await config.update('showInlineCompletions', settings.showInlineCompletions, true);
        await config.update('semanticSearch.enabled', settings.semanticSearchEnabled, true);
        await config.update('semanticSearch.maxResults', settings.semanticSearchMaxResults, true);

        vscode.window.showInformationMessage('Victor settings saved');
    }

    private _sendSettings(): void {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'settings',
                settings: this._getSettings(),
            });
        }
    }

    private async _testConnection(settings?: Partial<VictorSettings>): Promise<void> {
        try {
            const baseUrl = this._resolveBaseUrl(settings);
            const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/health`);
            if (response.ok) {
                this._view?.webview.postMessage({
                    type: 'connectionStatus',
                    status: 'connected',
                    message: 'Server is running',
                });
            } else {
                throw new Error('Server not healthy');
            }
        } catch (error) {
            this._view?.webview.postMessage({
                type: 'connectionStatus',
                status: 'disconnected',
                message: error instanceof Error ? error.message : 'Server not running',
            });
        }
    }

    private async _loadProviders(): Promise<void> {
        const defaultProviders: ProviderInfo[] = [
            { name: 'anthropic', display_name: 'Anthropic (Claude)', is_local: false, configured: false, supports_tools: true, supports_streaming: true },
            { name: 'openai', display_name: 'OpenAI (GPT-4)', is_local: false, configured: false, supports_tools: true, supports_streaming: true },
            { name: 'google', display_name: 'Google (Gemini)', is_local: false, configured: false, supports_tools: true, supports_streaming: true },
            { name: 'ollama', display_name: 'Ollama (Local)', is_local: true, configured: true, supports_tools: true, supports_streaming: true },
            { name: 'lmstudio', display_name: 'LM Studio (Local)', is_local: true, configured: true, supports_tools: false, supports_streaming: true },
            { name: 'xai', display_name: 'xAI (Grok)', is_local: false, configured: false, supports_tools: true, supports_streaming: true },
        ];

        try {
            const providers = getProviders();
            if (providers?.victorClient) {
                const serverProviders = await providers.victorClient.getProviders();
                this._view?.webview.postMessage({
                    type: 'providers',
                    providers: serverProviders,
                });
                return;
            }
        } catch (error) {
            console.error('Failed to load providers from server:', error);
        }

        // Fallback to defaults
        this._view?.webview.postMessage({
            type: 'providers',
            providers: defaultProviders,
        });
    }

    private async _loadModels(): Promise<void> {
        try {
            const providers = getProviders();
            if (providers?.victorClient) {
                const models = await providers.victorClient.getModels();
                this._view?.webview.postMessage({
                    type: 'models',
                    models,
                });
                return;
            }
        } catch (error) {
            console.error('Failed to load models from server:', error);
        }

        // Fallback to empty - the HTML has hardcoded defaults
        this._view?.webview.postMessage({
            type: 'models',
            models: [],
        });
    }

    private _getHtmlContent(_webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Victor Settings</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background-color: var(--vscode-editor-background);
            padding: 10px;
            margin: 0;
        }

        .section {
            margin-bottom: 20px;
        }

        .section-title {
            font-weight: bold;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid var(--vscode-widget-border);
        }

        .setting {
            margin-bottom: 12px;
        }

        .setting-hint {
            margin-top: 4px;
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
            line-height: 1.4;
        }

        label {
            display: block;
            margin-bottom: 4px;
            color: var(--vscode-descriptionForeground);
        }

        select, input[type="text"], input[type="number"] {
            width: 100%;
            padding: 6px 8px;
            background-color: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            border-radius: 2px;
            box-sizing: border-box;
        }

        select:focus, input:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }

        .checkbox-setting {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .checkbox-setting input {
            width: auto;
        }

        .checkbox-setting label {
            margin-bottom: 0;
            color: var(--vscode-foreground);
        }

        button {
            background-color: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 8px 16px;
            border-radius: 2px;
            cursor: pointer;
            margin-right: 8px;
        }

        button:hover {
            background-color: var(--vscode-button-hoverBackground);
        }

        button.secondary {
            background-color: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        .status {
            padding: 8px;
            border-radius: 4px;
            margin-top: 10px;
        }

        .status.connected {
            background-color: var(--vscode-testing-iconPassed);
            color: white;
        }

        .status.disconnected {
            background-color: var(--vscode-testing-iconFailed);
            color: white;
        }

        .button-row {
            margin-top: 20px;
        }

        /* Provider Card Styles */
        .provider-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 8px;
            margin-bottom: 12px;
        }

        .provider-card {
            background: var(--vscode-input-background);
            border: 1px solid var(--vscode-input-border);
            border-radius: 4px;
            padding: 8px 10px;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        .provider-card:hover {
            border-color: var(--vscode-focusBorder);
        }

        .provider-card.selected {
            border-color: var(--vscode-button-background);
            background: var(--vscode-list-activeSelectionBackground);
        }

        .provider-card.not-configured {
            opacity: 0.6;
        }

        .provider-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }

        .provider-name {
            font-weight: 500;
            flex: 1;
        }

        .provider-badges {
            display: flex;
            gap: 4px;
            flex-wrap: wrap;
        }

        .provider-badge {
            font-size: 9px;
            padding: 1px 5px;
            border-radius: 8px;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }

        .provider-badge.local {
            background: var(--vscode-testing-iconPassed);
            color: #000;
        }

        .provider-badge.cloud {
            background: var(--vscode-charts-blue);
            color: #fff;
        }

        .provider-badge.configured {
            background: var(--vscode-testing-iconPassed);
            color: #000;
        }

        .provider-badge.not-configured {
            background: var(--vscode-errorForeground);
            color: #fff;
        }

        .provider-badge.tools {
            background: var(--vscode-editorInfo-foreground);
            color: #fff;
        }

        .provider-badge.streaming {
            background: var(--vscode-symbolIcon-functionForeground);
            color: #000;
        }

        .provider-capabilities {
            font-size: 10px;
            color: var(--vscode-descriptionForeground);
            margin-top: 4px;
        }

        .loading-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
            color: var(--vscode-descriptionForeground);
        }

        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid var(--vscode-button-background);
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .refresh-btn {
            font-size: 11px;
            padding: 4px 8px;
            background: transparent;
            border: 1px solid var(--vscode-input-border);
            color: var(--vscode-foreground);
            cursor: pointer;
            border-radius: 2px;
        }

        .refresh-btn:hover {
            background: var(--vscode-list-hoverBackground);
        }
    </style>
</head>
<body>
    <div class="section">
        <div class="section-title">
            Provider Settings
            <button class="refresh-btn" onclick="refreshProviders()" title="Refresh providers">Refresh</button>
        </div>

        <div class="setting">
            <label>AI Provider</label>
            <div id="providerList" class="provider-grid">
                <div class="loading-indicator">
                    <div class="spinner"></div>
                    <span>Loading providers...</span>
                </div>
            </div>
            <input type="hidden" id="provider" value="anthropic">
        </div>

        <div class="setting">
            <label for="model">Model</label>
            <select id="model">
                <option value="">Loading models...</option>
            </select>
        </div>

        <div class="setting">
            <label for="mode">Agent Mode</label>
            <select id="mode">
                <option value="build">Build - Full implementation mode</option>
                <option value="plan">Plan - Read-only analysis</option>
                <option value="explore">Explore - Codebase exploration</option>
            </select>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Server Settings</div>

        <div class="setting">
            <label for="serverUrl">Server URL (optional)</label>
            <input type="text" id="serverUrl" placeholder="http://127.0.0.1:8765">
            <div class="setting-hint">
                Use a full URL for a remote or shared Victor server. When set, it overrides Server Port.
            </div>
        </div>

        <div class="setting">
            <label for="serverPort">Server Port</label>
            <input type="number" id="serverPort" min="1024" max="65535" value="8765">
            <div class="setting-hint" id="serverConnectionHint">
                Leave Server URL empty to connect to a local Victor server on this port.
            </div>
        </div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="autoStart" checked>
            <label for="autoStart">Auto-start server on extension activation</label>
        </div>

        <button class="secondary" onclick="testConnection()">Test Connection</button>
        <div id="connectionStatus"></div>
    </div>

    <div class="section">
        <div class="section-title">Features</div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="showInlineCompletions" checked>
            <label for="showInlineCompletions">Show inline code completions</label>
        </div>

        <div class="setting checkbox-setting">
            <input type="checkbox" id="semanticSearchEnabled" checked>
            <label for="semanticSearchEnabled">Enable semantic code search</label>
        </div>

        <div class="setting">
            <label for="semanticSearchMaxResults">Semantic search max results</label>
            <input type="number" id="semanticSearchMaxResults" min="1" max="50" value="10">
        </div>
    </div>

    <div class="button-row">
        <button onclick="saveSettings()">Save Settings</button>
        <button class="secondary" onclick="loadSettings()">Reset</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();
        let currentProviders = [];
        let currentModels = [];
        let currentSettings = {};

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;

            switch (message.type) {
                case 'settings':
                    currentSettings = message.settings;
                    applySettings(message.settings);
                    break;
                case 'providers':
                    currentProviders = message.providers;
                    renderProviders(message.providers);
                    break;
                case 'models':
                    currentModels = message.models;
                    renderModels(message.models);
                    break;
                case 'connectionStatus':
                    showConnectionStatus(message.status, message.message);
                    break;
            }
        });

        function applySettings(settings) {
            document.getElementById('provider').value = settings.provider;
            document.getElementById('mode').value = settings.mode;
            document.getElementById('serverUrl').value = settings.serverUrl || '';
            document.getElementById('serverPort').value = settings.serverPort;
            document.getElementById('autoStart').checked = settings.autoStart;
            document.getElementById('showInlineCompletions').checked = settings.showInlineCompletions;
            document.getElementById('semanticSearchEnabled').checked = settings.semanticSearchEnabled;
            document.getElementById('semanticSearchMaxResults').value = settings.semanticSearchMaxResults;
            updateServerConnectionMode();

            // Update provider selection in UI
            updateProviderSelection(settings.provider);

            // Set model after models are loaded
            if (currentModels.length > 0) {
                renderModels(currentModels);
            }
            document.getElementById('model').value = settings.model;
        }

        function renderProviders(providers) {
            const container = document.getElementById('providerList');
            const selectedProvider = document.getElementById('provider').value || 'anthropic';

            container.innerHTML = providers.map(p => {
                const isSelected = p.name === selectedProvider;
                const badges = [];

                badges.push(p.is_local
                    ? '<span class="provider-badge local">Local</span>'
                    : '<span class="provider-badge cloud">Cloud</span>');

                badges.push(p.configured
                    ? '<span class="provider-badge configured">Ready</span>'
                    : '<span class="provider-badge not-configured">Not Configured</span>');

                if (p.supports_tools) {
                    badges.push('<span class="provider-badge tools">Tools</span>');
                }

                return '<div class="provider-card' +
                    (isSelected ? ' selected' : '') +
                    (!p.configured ? ' not-configured' : '') +
                    '" data-provider="' + p.name + '" onclick="selectProvider(\\'' + p.name + '\\')">' +
                    '<div class="provider-header">' +
                    '<span class="provider-name">' + p.display_name + '</span>' +
                    '<div class="provider-badges">' + badges.join('') + '</div>' +
                    '</div>' +
                    '</div>';
            }).join('');
        }

        function selectProvider(providerName) {
            document.getElementById('provider').value = providerName;
            updateProviderSelection(providerName);
            loadModelsForProvider(providerName);
        }

        function updateProviderSelection(providerName) {
            document.querySelectorAll('.provider-card').forEach(card => {
                card.classList.toggle('selected', card.dataset.provider === providerName);
            });
        }

        function loadModelsForProvider(providerName) {
            // Filter models for the selected provider
            const filteredModels = currentModels.filter(m => m.provider === providerName);
            renderModels(filteredModels.length > 0 ? filteredModels : currentModels);
        }

        function renderModels(models) {
            const select = document.getElementById('model');
            const currentValue = select.value || currentSettings.model;
            const selectedProvider = document.getElementById('provider').value;

            if (models.length === 0) {
                // Fallback to hardcoded models
                select.innerHTML = getDefaultModelOptions(selectedProvider);
            } else {
                // Group models by provider
                const grouped = {};
                models.forEach(m => {
                    if (!grouped[m.provider]) grouped[m.provider] = [];
                    grouped[m.provider].push(m);
                });

                let html = '';
                for (const [provider, providerModels] of Object.entries(grouped)) {
                    const isSelectedProvider = provider === selectedProvider;
                    html += '<optgroup label="' + provider + '">';
                    providerModels.forEach(m => {
                        const style = isSelectedProvider ? '' : 'display:none;';
                        html += '<option value="' + m.model_id + '" style="' + style + '">' +
                            (m.is_local ? '[Local] ' : '') + m.display_name + '</option>';
                    });
                    html += '</optgroup>';
                }
                select.innerHTML = html;
            }

            // Try to restore previous value
            if (currentValue && select.querySelector('option[value="' + currentValue + '"]')) {
                select.value = currentValue;
            }
        }

        function getDefaultModelOptions(provider) {
            const defaults = {
                anthropic: [
                    { id: 'claude-sonnet-4-20250514', name: 'Claude Sonnet 4' },
                    { id: 'claude-opus-4-5-20251101', name: 'Claude Opus 4.5' },
                ],
                openai: [
                    { id: 'gpt-4-turbo', name: 'GPT-4 Turbo' },
                    { id: 'gpt-4o', name: 'GPT-4o' },
                ],
                google: [
                    { id: 'gemini-2.0-flash', name: 'Gemini 2.0 Flash' },
                ],
                ollama: [
                    { id: 'qwen2.5-coder:14b', name: 'Qwen 2.5 Coder 14B' },
                    { id: 'llama3.1:8b', name: 'Llama 3.1 8B' },
                ],
            };

            const models = defaults[provider] || defaults.anthropic;
            return models.map(m => '<option value="' + m.id + '">' + m.name + '</option>').join('');
        }

        function getSettings() {
            return {
                provider: document.getElementById('provider').value,
                model: document.getElementById('model').value,
                mode: document.getElementById('mode').value,
                serverUrl: document.getElementById('serverUrl').value.trim(),
                serverPort: parseInt(document.getElementById('serverPort').value),
                autoStart: document.getElementById('autoStart').checked,
                showInlineCompletions: document.getElementById('showInlineCompletions').checked,
                semanticSearchEnabled: document.getElementById('semanticSearchEnabled').checked,
                semanticSearchMaxResults: parseInt(document.getElementById('semanticSearchMaxResults').value),
            };
        }

        function updateServerConnectionMode() {
            const serverUrl = document.getElementById('serverUrl').value.trim();
            const serverPortInput = document.getElementById('serverPort');
            const hint = document.getElementById('serverConnectionHint');
            const usingRemoteServer = serverUrl.length > 0;

            serverPortInput.disabled = usingRemoteServer;
            serverPortInput.title = usingRemoteServer
                ? 'Server Port is ignored while Server URL is set.'
                : 'Port used when connecting to a local Victor server.';
            hint.textContent = usingRemoteServer
                ? 'Using Server URL. Clear it to reconnect to localhost with Server Port.'
                : 'Leave Server URL empty to connect to a local Victor server on this port.';
        }

        function saveSettings() {
            vscode.postMessage({
                type: 'saveSettings',
                settings: getSettings(),
            });
        }

        function loadSettings() {
            vscode.postMessage({ type: 'loadSettings' });
        }

        function refreshProviders() {
            document.getElementById('providerList').innerHTML =
                '<div class="loading-indicator"><div class="spinner"></div><span>Loading providers...</span></div>';
            vscode.postMessage({ type: 'loadProviders' });
            vscode.postMessage({ type: 'loadModels' });
        }

        function testConnection() {
            vscode.postMessage({
                type: 'testConnection',
                settings: getSettings(),
            });
        }

        function showConnectionStatus(status, message) {
            const statusDiv = document.getElementById('connectionStatus');
            statusDiv.className = 'status ' + status;
            statusDiv.textContent = message;
        }

        document.getElementById('serverUrl').addEventListener('input', updateServerConnectionMode);

        // Initial load
        loadSettings();
        vscode.postMessage({ type: 'loadProviders' });
        vscode.postMessage({ type: 'loadModels' });
    </script>
</body>
</html>`;
    }
}

/**
 * Register settings commands
 */
export function registerSettingsCommands(context: vscode.ExtensionContext): void {
    // Open settings
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.openSettings', () => {
            vscode.commands.executeCommand('workbench.action.openSettings', 'victor');
        })
    );

    // Open settings JSON
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.openSettingsJson', () => {
            vscode.commands.executeCommand('workbench.action.openSettingsJson');
        })
    );
}
