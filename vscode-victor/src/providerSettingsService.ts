/**
 * Provider Settings Service
 *
 * Manages secure storage and configuration of LLM provider credentials.
 * Uses platform-native credential storage via Victor backend:
 * - macOS Keychain (via Python keyring)
 * - Windows Credential Manager (via Python keyring)
 * - Linux Secret Service/libsecret (via Python keyring)
 *
 * Falls back to VS Code SecretStorage when backend is unavailable.
 *
 * Features:
 * - Platform-native secure API key storage (preferred)
 * - VS Code SecretStorage fallback
 * - Provider-specific configuration (temperature, max_tokens, etc.)
 * - Connection testing with health checks
 * - Status tracking and health monitoring
 * - Event-driven architecture for UI updates
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

/**
 * Credential storage backend type
 */
export type CredentialStorageBackend = 'system' | 'vscode';

/**
 * Credential storage info
 */
interface CredentialStorageInfo {
    backend: CredentialStorageBackend;
    description: string;
    available: boolean;
}

/**
 * Provider configuration metadata
 */
export interface ProviderConfig {
    name: string;
    displayName: string;
    isLocal: boolean;
    apiKeyRequired: boolean;
    apiKeyEnvVar?: string;
    endpointConfigurable: boolean;
    defaultEndpoint?: string;
    supportedModels?: string[];
    docsUrl?: string;
}

/**
 * Provider status information
 */
export interface ProviderStatus {
    name: string;
    configured: boolean;
    connected: boolean;
    lastChecked?: number;
    error?: string;
    latencyMs?: number;
}

/**
 * Provider-specific parameters
 */
export interface ProviderParameters {
    temperature?: number;
    maxTokens?: number;
    topP?: number;
    topK?: number;
    endpoint?: string;
}

/**
 * Connection test result
 */
export interface ConnectionTestResult {
    success: boolean;
    latencyMs?: number;
    modelCount?: number;
    error?: string;
    details?: Record<string, unknown>;
}

/**
 * Built-in provider configurations
 */
export const PROVIDER_CONFIGS: ProviderConfig[] = [
    {
        name: 'anthropic',
        displayName: 'Anthropic (Claude)',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'ANTHROPIC_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://docs.anthropic.com/',
    },
    {
        name: 'openai',
        displayName: 'OpenAI (GPT-4)',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'OPENAI_API_KEY',
        endpointConfigurable: true,
        defaultEndpoint: 'https://api.openai.com/v1',
        docsUrl: 'https://platform.openai.com/docs/',
    },
    {
        name: 'google',
        displayName: 'Google (Gemini)',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'GOOGLE_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://ai.google.dev/docs',
    },
    {
        name: 'xai',
        displayName: 'xAI (Grok)',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'XAI_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://docs.x.ai/',
    },
    {
        name: 'deepseek',
        displayName: 'DeepSeek',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'DEEPSEEK_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://api-docs.deepseek.com/',
    },
    {
        name: 'groq',
        displayName: 'Groq',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'GROQ_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://console.groq.com/docs',
    },
    {
        name: 'mistral',
        displayName: 'Mistral AI',
        isLocal: false,
        apiKeyRequired: true,
        apiKeyEnvVar: 'MISTRAL_API_KEY',
        endpointConfigurable: false,
        docsUrl: 'https://docs.mistral.ai/',
    },
    {
        name: 'ollama',
        displayName: 'Ollama (Local)',
        isLocal: true,
        apiKeyRequired: false,
        endpointConfigurable: true,
        defaultEndpoint: 'http://localhost:11434',
        docsUrl: 'https://ollama.ai/docs',
    },
    {
        name: 'lmstudio',
        displayName: 'LM Studio (Local)',
        isLocal: true,
        apiKeyRequired: false,
        endpointConfigurable: true,
        defaultEndpoint: 'http://localhost:1234/v1',
        docsUrl: 'https://lmstudio.ai/docs',
    },
    {
        name: 'vllm',
        displayName: 'vLLM (Local)',
        isLocal: true,
        apiKeyRequired: false,
        endpointConfigurable: true,
        defaultEndpoint: 'http://localhost:8000/v1',
        docsUrl: 'https://docs.vllm.ai/',
    },
];

/**
 * Secret storage key prefix for VS Code fallback
 */
const SECRET_KEY_PREFIX = 'victor.provider.apiKey.';
const PARAMS_KEY_PREFIX = 'victor.provider.params.';

/**
 * Service for managing provider settings and credentials
 */
export class ProviderSettingsService implements vscode.Disposable {
    private readonly _onProviderStatusChanged = new vscode.EventEmitter<ProviderStatus>();
    readonly onProviderStatusChanged = this._onProviderStatusChanged.event;

    private readonly _onConfigurationChanged = new vscode.EventEmitter<string>();
    readonly onConfigurationChanged = this._onConfigurationChanged.event;

    private providerStatuses: Map<string, ProviderStatus> = new Map();
    private readonly disposables: vscode.Disposable[] = [];
    private healthCheckInterval?: NodeJS.Timeout;
    private preferredBackend: CredentialStorageBackend = 'system';
    private systemKeyringAvailable = false;

    constructor(
        private readonly context: vscode.ExtensionContext
    ) {
        // Initialize statuses for all providers
        for (const config of PROVIDER_CONFIGS) {
            this.providerStatuses.set(config.name, {
                name: config.name,
                configured: false,
                connected: false,
            });
        }

        // Load preferred backend from settings
        this.loadPreferredBackend();

        // Start periodic health checks
        this.startHealthChecks();
    }

    /**
     * Load preferred credential storage backend from settings
     */
    private loadPreferredBackend(): void {
        const config = vscode.workspace.getConfiguration('victor');
        this.preferredBackend = config.get('credentialStorage', 'system') as CredentialStorageBackend;
    }

    /**
     * Get available credential storage backends
     */
    async getAvailableBackends(): Promise<CredentialStorageInfo[]> {
        const backends: CredentialStorageInfo[] = [];

        // Check system keyring availability via backend
        const providers = getProviders();
        if (providers?.victorClient) {
            try {
                // Try to call backend to check keyring availability
                const keyringStatus = await providers.victorClient.checkKeyringAvailable();
                this.systemKeyringAvailable = keyringStatus.available;

                backends.push({
                    backend: 'system',
                    description: this.getSystemKeychainDescription(),
                    available: this.systemKeyringAvailable,
                });
            } catch {
                this.systemKeyringAvailable = false;
                backends.push({
                    backend: 'system',
                    description: this.getSystemKeychainDescription(),
                    available: false,
                });
            }
        }

        // VS Code SecretStorage is always available
        backends.push({
            backend: 'vscode',
            description: 'VS Code Secure Storage (uses system keychain under the hood)',
            available: true,
        });

        return backends;
    }

    /**
     * Get description for system keychain based on platform
     */
    private getSystemKeychainDescription(): string {
        switch (process.platform) {
            case 'darwin':
                return 'macOS Keychain (via Victor backend)';
            case 'win32':
                return 'Windows Credential Manager (via Victor backend)';
            case 'linux':
                return 'Secret Service/GNOME Keyring (via Victor backend)';
            default:
                return 'System Keyring (via Victor backend)';
        }
    }

    /**
     * Set preferred credential storage backend
     */
    async setPreferredBackend(backend: CredentialStorageBackend): Promise<void> {
        this.preferredBackend = backend;
        const config = vscode.workspace.getConfiguration('victor');
        await config.update('credentialStorage', backend, true);
        this._onConfigurationChanged.fire('_backend');
    }

    /**
     * Get current preferred backend
     */
    getPreferredBackend(): CredentialStorageBackend {
        return this.preferredBackend;
    }

    /**
     * Get all provider configurations
     */
    getProviderConfigs(): ProviderConfig[] {
        return [...PROVIDER_CONFIGS];
    }

    /**
     * Get configuration for a specific provider
     */
    getProviderConfig(name: string): ProviderConfig | undefined {
        return PROVIDER_CONFIGS.find(p => p.name === name);
    }

    /**
     * Store API key securely using preferred backend
     * Primary: System keyring via Victor backend
     * Fallback: VS Code SecretStorage
     */
    async setApiKey(providerName: string, apiKey: string): Promise<{ success: boolean; backend: CredentialStorageBackend }> {
        let success = false;
        let usedBackend: CredentialStorageBackend = 'vscode';

        // Try system keyring first if preferred and available
        if (this.preferredBackend === 'system') {
            const providers = getProviders();
            if (providers?.victorClient) {
                try {
                    await providers.victorClient.setApiKey(providerName, apiKey);
                    success = true;
                    usedBackend = 'system';
                } catch (error) {
                    console.warn('System keyring storage failed, falling back to VS Code:', error);
                }
            }
        }

        // Fallback to VS Code SecretStorage
        if (!success) {
            const key = `${SECRET_KEY_PREFIX}${providerName}`;
            await this.context.secrets.store(key, apiKey);
            success = true;
            usedBackend = 'vscode';
        }

        // Update status
        const status = this.providerStatuses.get(providerName);
        if (status) {
            status.configured = true;
            this._onProviderStatusChanged.fire(status);
        }

        this._onConfigurationChanged.fire(providerName);
        return { success, backend: usedBackend };
    }

    /**
     * Retrieve API key from secure storage
     * Checks system keyring first, then VS Code fallback
     */
    async getApiKey(providerName: string): Promise<string | undefined> {
        // Try system keyring first
        const providers = getProviders();
        if (providers?.victorClient) {
            try {
                const result = await providers.victorClient.getApiKey(providerName);
                if (result.apiKey) {
                    return result.apiKey;
                }
            } catch {
                // Fall through to VS Code storage
            }
        }

        // Fallback to VS Code SecretStorage
        const key = `${SECRET_KEY_PREFIX}${providerName}`;
        return await this.context.secrets.get(key);
    }

    /**
     * Delete API key from all storage backends
     */
    async deleteApiKey(providerName: string): Promise<void> {
        // Delete from system keyring
        const providers = getProviders();
        if (providers?.victorClient) {
            try {
                await providers.victorClient.deleteApiKey(providerName);
            } catch {
                // Ignore errors, continue with VS Code deletion
            }
        }

        // Delete from VS Code SecretStorage
        const key = `${SECRET_KEY_PREFIX}${providerName}`;
        await this.context.secrets.delete(key);

        // Update status
        const status = this.providerStatuses.get(providerName);
        if (status) {
            status.configured = false;
            status.connected = false;
            this._onProviderStatusChanged.fire(status);
        }

        this._onConfigurationChanged.fire(providerName);
    }

    /**
     * Check if provider has API key configured
     */
    async hasApiKey(providerName: string): Promise<boolean> {
        const apiKey = await this.getApiKey(providerName);
        return !!apiKey;
    }

    /**
     * Get where the API key is stored
     */
    async getApiKeyLocation(providerName: string): Promise<CredentialStorageBackend | null> {
        // Check system keyring first
        const providers = getProviders();
        if (providers?.victorClient) {
            try {
                const result = await providers.victorClient.getApiKey(providerName);
                if (result.apiKey) {
                    return 'system';
                }
            } catch {
                // Not in system keyring
            }
        }

        // Check VS Code SecretStorage
        const key = `${SECRET_KEY_PREFIX}${providerName}`;
        const vscodeKey = await this.context.secrets.get(key);
        if (vscodeKey) {
            return 'vscode';
        }

        return null;
    }

    /**
     * Store provider-specific parameters
     */
    async setProviderParameters(providerName: string, params: ProviderParameters): Promise<void> {
        const key = `${PARAMS_KEY_PREFIX}${providerName}`;
        await this.context.globalState.update(key, params);
        this._onConfigurationChanged.fire(providerName);
    }

    /**
     * Get provider-specific parameters
     */
    getProviderParameters(providerName: string): ProviderParameters {
        const key = `${PARAMS_KEY_PREFIX}${providerName}`;
        return this.context.globalState.get<ProviderParameters>(key) || {};
    }

    /**
     * Get status for a provider
     */
    getProviderStatus(providerName: string): ProviderStatus | undefined {
        return this.providerStatuses.get(providerName);
    }

    /**
     * Get statuses for all providers
     */
    getAllProviderStatuses(): ProviderStatus[] {
        return Array.from(this.providerStatuses.values());
    }

    /**
     * Test connection to a provider
     */
    async testConnection(providerName: string): Promise<ConnectionTestResult> {
        const config = this.getProviderConfig(providerName);
        if (!config) {
            return { success: false, error: 'Unknown provider' };
        }

        const startTime = Date.now();
        const status = this.providerStatuses.get(providerName)!;

        try {
            // Check if API key is required and configured
            if (config.apiKeyRequired) {
                const hasKey = await this.hasApiKey(providerName);
                if (!hasKey) {
                    status.configured = false;
                    status.connected = false;
                    status.error = 'API key not configured';
                    status.lastChecked = Date.now();
                    this._onProviderStatusChanged.fire(status);
                    return { success: false, error: 'API key not configured' };
                }
                status.configured = true;
            } else {
                // Local providers are always "configured"
                status.configured = true;
            }

            // Try to get models from the provider via backend
            const providers = getProviders();
            if (!providers?.victorClient) {
                status.connected = false;
                status.error = 'Victor server not available';
                status.lastChecked = Date.now();
                this._onProviderStatusChanged.fire(status);
                return { success: false, error: 'Victor server not available' };
            }

            // Test by fetching models for this provider
            const models = await providers.victorClient.getModels();
            const providerModels = models.filter(m => m.provider === providerName);

            const latencyMs = Date.now() - startTime;

            status.connected = true;
            status.error = undefined;
            status.latencyMs = latencyMs;
            status.lastChecked = Date.now();
            this._onProviderStatusChanged.fire(status);

            return {
                success: true,
                latencyMs,
                modelCount: providerModels.length,
                details: {
                    models: providerModels.map(m => m.model_id),
                },
            };
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Connection failed';
            status.connected = false;
            status.error = errorMessage;
            status.lastChecked = Date.now();
            this._onProviderStatusChanged.fire(status);

            return {
                success: false,
                error: errorMessage,
                latencyMs: Date.now() - startTime,
            };
        }
    }

    /**
     * Start periodic health checks for connected providers
     */
    private startHealthChecks(): void {
        // Check every 5 minutes
        const intervalMs = 5 * 60 * 1000;

        this.healthCheckInterval = setInterval(async () => {
            await this.checkAllProviders();
        }, intervalMs);

        // Run initial check after a delay to allow server to start
        setTimeout(() => {
            this.checkAllProviders().catch(console.error);
        }, 10000);
    }

    /**
     * Check all providers' connectivity
     */
    async checkAllProviders(): Promise<void> {
        for (const config of PROVIDER_CONFIGS) {
            // Only check providers that are configured
            if (config.apiKeyRequired) {
                const hasKey = await this.hasApiKey(config.name);
                if (!hasKey) {
                    continue;
                }
            }

            // Run checks in background, don't await all
            this.testConnection(config.name).catch(console.error);
        }
    }

    /**
     * Initialize provider status based on stored credentials
     */
    async initialize(): Promise<void> {
        for (const config of PROVIDER_CONFIGS) {
            const status = this.providerStatuses.get(config.name)!;

            if (config.apiKeyRequired) {
                const hasKey = await this.hasApiKey(config.name);
                status.configured = hasKey;
            } else {
                // Local providers are always configured
                status.configured = true;
            }

            this._onProviderStatusChanged.fire(status);
        }
    }

    dispose(): void {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        this._onProviderStatusChanged.dispose();
        this._onConfigurationChanged.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}

/**
 * Register provider settings commands
 */
export function registerProviderSettingsCommands(
    context: vscode.ExtensionContext,
    service: ProviderSettingsService
): void {
    // Configure provider API key
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.configureProviderKey', async () => {
            // Show provider selection
            const configs = service.getProviderConfigs().filter(c => c.apiKeyRequired);
            const items = configs.map(c => ({
                label: c.displayName,
                description: c.name,
                detail: `Environment variable: ${c.apiKeyEnvVar}`,
                config: c,
            }));

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select provider to configure',
            });

            if (!selected) {
                return;
            }

            // Check if already configured
            const hasKey = await service.hasApiKey(selected.config.name);
            if (hasKey) {
                const action = await vscode.window.showQuickPick([
                    { label: 'Update API Key', action: 'update' },
                    { label: 'Delete API Key', action: 'delete' },
                    { label: 'Test Connection', action: 'test' },
                    { label: 'Cancel', action: 'cancel' },
                ], {
                    placeHolder: `${selected.config.displayName} is already configured`,
                });

                if (!action || action.action === 'cancel') {
                    return;
                }

                if (action.action === 'delete') {
                    await service.deleteApiKey(selected.config.name);
                    vscode.window.showInformationMessage(`${selected.config.displayName} API key removed`);
                    return;
                }

                if (action.action === 'test') {
                    await testProviderConnection(service, selected.config.name);
                    return;
                }
            }

            // Prompt for API key
            const apiKey = await vscode.window.showInputBox({
                prompt: `Enter API key for ${selected.config.displayName}`,
                placeHolder: `Paste your ${selected.config.apiKeyEnvVar} here`,
                password: true,
                ignoreFocusOut: true,
            });

            if (!apiKey) {
                return;
            }

            // Store the API key
            await service.setApiKey(selected.config.name, apiKey);
            vscode.window.showInformationMessage(`${selected.config.displayName} API key saved securely`);

            // Offer to test connection
            const testNow = await vscode.window.showQuickPick(['Yes', 'No'], {
                placeHolder: 'Test connection now?',
            });

            if (testNow === 'Yes') {
                await testProviderConnection(service, selected.config.name);
            }
        })
    );

    // Test provider connection
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.testProviderConnection', async () => {
            const statuses = service.getAllProviderStatuses();
            const configuredProviders = statuses.filter(s => s.configured);

            if (configuredProviders.length === 0) {
                vscode.window.showWarningMessage('No providers configured. Use "Victor: Configure Provider API Key" first.');
                return;
            }

            const items = configuredProviders.map(s => {
                const config = service.getProviderConfig(s.name);
                return {
                    label: `${s.connected ? '$(check)' : '$(x)'} ${config?.displayName || s.name}`,
                    description: s.connected ? 'Connected' : s.error || 'Not connected',
                    provider: s.name,
                };
            });

            items.push({
                label: '$(sync) Test All Providers',
                description: 'Check connectivity for all configured providers',
                provider: '_all',
            });

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select provider to test',
            });

            if (!selected) {
                return;
            }

            if (selected.provider === '_all') {
                await vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: 'Testing all providers...',
                    cancellable: false,
                }, async () => {
                    await service.checkAllProviders();
                });
                vscode.window.showInformationMessage('Provider connectivity checks complete');
            } else {
                await testProviderConnection(service, selected.provider);
            }
        })
    );

    // Configure provider parameters
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.configureProviderParams', async () => {
            const configs = service.getProviderConfigs();
            const items = configs.map(c => {
                const params = service.getProviderParameters(c.name);
                return {
                    label: c.displayName,
                    description: c.name,
                    detail: formatParams(params),
                    config: c,
                };
            });

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Select provider to configure parameters',
            });

            if (!selected) {
                return;
            }

            const currentParams = service.getProviderParameters(selected.config.name);

            // Show parameter configuration
            const paramItems = [
                {
                    label: 'Temperature',
                    description: `Current: ${currentParams.temperature ?? 'default'}`,
                    param: 'temperature',
                },
                {
                    label: 'Max Tokens',
                    description: `Current: ${currentParams.maxTokens ?? 'default'}`,
                    param: 'maxTokens',
                },
                {
                    label: 'Top P',
                    description: `Current: ${currentParams.topP ?? 'default'}`,
                    param: 'topP',
                },
            ];

            // Add endpoint for configurable providers
            if (selected.config.endpointConfigurable) {
                paramItems.push({
                    label: 'Custom Endpoint',
                    description: `Current: ${currentParams.endpoint ?? selected.config.defaultEndpoint ?? 'default'}`,
                    param: 'endpoint',
                });
            }

            paramItems.push({
                label: '$(trash) Reset All Parameters',
                description: 'Reset to defaults',
                param: '_reset',
            });

            const paramSelected = await vscode.window.showQuickPick(paramItems, {
                placeHolder: `Configure ${selected.config.displayName} parameters`,
            });

            if (!paramSelected) {
                return;
            }

            if (paramSelected.param === '_reset') {
                await service.setProviderParameters(selected.config.name, {});
                vscode.window.showInformationMessage(`${selected.config.displayName} parameters reset to defaults`);
                return;
            }

            // Get new value
            const value = await vscode.window.showInputBox({
                prompt: `Enter ${paramSelected.label} for ${selected.config.displayName}`,
                placeHolder: paramSelected.param === 'endpoint' ? 'https://...' : 'Leave empty for default',
                value: String(currentParams[paramSelected.param as keyof ProviderParameters] ?? ''),
            });

            if (value === undefined) {
                return;
            }

            // Update parameter
            const newParams = { ...currentParams };
            if (value === '') {
                delete newParams[paramSelected.param as keyof ProviderParameters];
            } else {
                if (paramSelected.param === 'endpoint') {
                    newParams.endpoint = value;
                } else {
                    const numValue = parseFloat(value);
                    if (!isNaN(numValue)) {
                        (newParams as Record<string, number | string | undefined>)[paramSelected.param] = numValue;
                    }
                }
            }

            await service.setProviderParameters(selected.config.name, newParams);
            vscode.window.showInformationMessage(`${selected.config.displayName} ${paramSelected.label} updated`);
        })
    );

    // View provider status
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.viewProviderStatus', async () => {
            const statuses = service.getAllProviderStatuses();
            const configs = service.getProviderConfigs();

            const items = statuses.map(s => {
                const config = configs.find(c => c.name === s.name);
                const icon = s.connected ? '$(check)' : s.configured ? '$(warning)' : '$(x)';
                const status = s.connected ? 'Connected' : s.configured ? 'Configured (not connected)' : 'Not configured';

                return {
                    label: `${icon} ${config?.displayName || s.name}`,
                    description: status,
                    detail: [
                        s.latencyMs ? `Latency: ${s.latencyMs}ms` : '',
                        s.lastChecked ? `Last checked: ${new Date(s.lastChecked).toLocaleTimeString()}` : '',
                        s.error ? `Error: ${s.error}` : '',
                    ].filter(Boolean).join(' | '),
                    config,
                    status: s,
                };
            });

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Provider status overview',
            });

            if (selected && selected.config) {
                // Show actions for selected provider
                const actions = [];

                if (selected.config.apiKeyRequired) {
                    actions.push({
                        label: selected.status.configured ? '$(key) Update API Key' : '$(key) Configure API Key',
                        action: 'configure',
                    });
                }

                if (selected.status.configured) {
                    actions.push({ label: '$(sync) Test Connection', action: 'test' });
                    actions.push({ label: '$(settings-gear) Configure Parameters', action: 'params' });
                }

                if (selected.config.docsUrl) {
                    actions.push({ label: '$(book) Open Documentation', action: 'docs' });
                }

                const action = await vscode.window.showQuickPick(actions, {
                    placeHolder: `Actions for ${selected.config.displayName}`,
                });

                if (action) {
                    switch (action.action) {
                        case 'configure':
                            await vscode.commands.executeCommand('victor.configureProviderKey');
                            break;
                        case 'test':
                            await testProviderConnection(service, selected.config.name);
                            break;
                        case 'params':
                            await vscode.commands.executeCommand('victor.configureProviderParams');
                            break;
                        case 'docs':
                            if (selected.config.docsUrl) {
                                await vscode.env.openExternal(vscode.Uri.parse(selected.config.docsUrl));
                            }
                            break;
                    }
                }
            }
        })
    );
}

/**
 * Test connection with progress indicator
 */
async function testProviderConnection(service: ProviderSettingsService, providerName: string): Promise<void> {
    const config = service.getProviderConfig(providerName);

    await vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: `Testing ${config?.displayName || providerName}...`,
        cancellable: false,
    }, async () => {
        const result = await service.testConnection(providerName);

        if (result.success) {
            vscode.window.showInformationMessage(
                `${config?.displayName}: Connected (${result.latencyMs}ms, ${result.modelCount} models)`
            );
        } else {
            vscode.window.showErrorMessage(
                `${config?.displayName}: ${result.error}`
            );
        }
    });
}

/**
 * Format provider parameters for display
 */
function formatParams(params: ProviderParameters): string {
    const parts: string[] = [];
    if (params.temperature !== undefined) {
        parts.push(`temp: ${params.temperature}`);
    }
    if (params.maxTokens !== undefined) {
        parts.push(`max: ${params.maxTokens}`);
    }
    if (params.topP !== undefined) {
        parts.push(`topP: ${params.topP}`);
    }
    if (params.endpoint) {
        parts.push(`endpoint: ${params.endpoint.substring(0, 30)}...`);
    }
    return parts.length > 0 ? parts.join(', ') : 'Using defaults';
}
