/**
 * Victor AI VS Code Extension
 *
 * Provides AI-powered coding assistance through the Victor backend.
 * Features automatic server management for seamless experience.
 */

import * as vscode from 'vscode';
import { disposeEventBridgeClient } from './eventBridgeClient';
import { VictorClient } from './victorClient';
import { ChatViewProvider } from './chatViewProvider';
import { SemanticSearchProvider } from './semanticSearch';
import { InlineCompletionProvider } from './inlineCompletion';
import { InlineEditProvider } from './inlineEdit';
import { ServerManager, ServerStatus, createServerManager } from './serverManager';
import { HistoryViewProvider, registerHistoryCommands } from './historyView';
import { ToolsViewProvider, registerToolsCommands } from './toolsView';
import { DiffViewProvider, registerDiffCommands } from './diffView';
import { createFileWatcher, FileWatcher, ViewRefreshManager } from './fileWatcher';
import { ContextProvider, registerContextCommands } from './contextProvider';
import { TerminalProvider, TerminalHistoryProvider, registerTerminalCommands } from './terminalProvider';
import { registerCodeActionCommands } from './codeActions';
import { DiagnosticsViewProvider, registerDiagnosticsCommands } from './diagnosticsView';
import { SettingsViewProvider, registerSettingsCommands } from './settingsView';
import { WorkspaceInsightsViewProvider, registerWorkspaceInsightsCommands } from './workspaceInsightsView';
import { McpViewProvider, registerMcpCommands } from './mcpView';
import { AgentsViewProvider, registerAgentCommands } from './agentsView';
import { PlansViewProvider, registerPlanCommands } from './plansView';
import { ToolApprovalService, registerToolApprovalCommands } from './toolApprovalService';
import { ProviderSettingsService, registerProviderSettingsCommands } from './providerSettingsService';
import { VictorCodeLensProvider, registerCodeLensCommands } from './codeLensProvider';
import { VictorCodeActionProvider, registerCodeActionProviderCommands } from './codeActionProvider';
import { VictorHoverProvider, registerHoverCommands } from './hoverProvider';
import { ToolExecutionService, registerToolExecutionCommands } from './toolExecutionService';
import { RLService, registerRLCommands } from './rlService';
import { TerminalAgentProvider, registerTerminalAgentCommands } from './terminalAgent';
import { TerminalHistoryService } from './terminalHistory';
import { SmartPasteProvider, registerSmartPasteCommands } from './smartPaste';
import { ComposerViewProvider, registerComposerCommands } from './composer';
import { TeamsViewProvider, registerTeamCommands } from './teamsView';
import { WorkflowsViewProvider, registerWorkflowCommands } from './workflowsView';
import { getStore, selectors, AgentMode, ModelInfo } from './state';

// Provider instances (managed centrally, accessed via module)
interface ExtensionProviders {
    victorClient: VictorClient;
    chatViewProvider: ChatViewProvider;
    semanticSearchProvider: SemanticSearchProvider;
    inlineCompletionProvider: InlineCompletionProvider;
    inlineEditProvider: InlineEditProvider;
    serverManager: ServerManager;
    historyViewProvider: HistoryViewProvider;
    toolsViewProvider: ToolsViewProvider;
    agentsViewProvider: AgentsViewProvider;
    plansViewProvider: PlansViewProvider;
    diffViewProvider: DiffViewProvider;
    fileWatcher: FileWatcher;
    refreshManager: ViewRefreshManager;
    contextProvider: ContextProvider;
    terminalProvider: TerminalProvider;
    terminalHistoryProvider: TerminalHistoryProvider;
    diagnosticsViewProvider: DiagnosticsViewProvider;
    settingsViewProvider: SettingsViewProvider;
    workspaceInsightsViewProvider: WorkspaceInsightsViewProvider;
    mcpViewProvider: McpViewProvider;
    toolApprovalService: ToolApprovalService;
    providerSettingsService: ProviderSettingsService;
    codeLensProvider: VictorCodeLensProvider;
    codeActionProvider: VictorCodeActionProvider;
    hoverProvider: VictorHoverProvider;
    toolExecutionService: ToolExecutionService;
    rlService: RLService;
    terminalAgentProvider: TerminalAgentProvider;
    smartPasteProvider: SmartPasteProvider;
    composerViewProvider: ComposerViewProvider;
    teamsViewProvider: TeamsViewProvider;
    workflowsViewProvider: WorkflowsViewProvider;
    statusBarItem: vscode.StatusBarItem;
    serverStatusBarItem: vscode.StatusBarItem;
}

interface ProfileQuickPickItem extends vscode.QuickPickItem {
    value: string;
    provider?: string;
    model?: string;
}

let providers: ExtensionProviders | null = null;

/**
 * Validate extension configuration values
 */
function validateConfiguration(config: vscode.WorkspaceConfiguration): void {
    const warnings: string[] = [];
    const serverUrl = (config.get<string>('serverUrl', '') || '').trim();

    // Validate serverPort
    const serverPort = config.get<number>('serverPort', 8765);
    if (serverPort < 1 || serverPort > 65535) {
        warnings.push(`Invalid serverPort ${serverPort}. Using default 8765.`);
    }

    if (serverUrl) {
        try {
            const parsed = new URL(serverUrl);
            if (!['http:', 'https:'].includes(parsed.protocol)) {
                warnings.push(`Invalid serverUrl "${serverUrl}". Use http:// or https://.`);
            }
        } catch {
            warnings.push(`Invalid serverUrl "${serverUrl}". Use a full http:// or https:// URL.`);
        }
    }

    // Validate fallbackPorts
    const fallbackPorts = config.get<number[]>('fallbackPorts', []);
    const invalidPorts = fallbackPorts.filter(p => p < 1 || p > 65535);
    if (invalidPorts.length > 0) {
        warnings.push(`Invalid fallback ports: ${invalidPorts.join(', ')}. These will be ignored.`);
    }

    // Validate mode
    const mode = config.get<string>('mode', 'build');
    const validModes = ['build', 'plan', 'review', 'delegate', 'explore'];
    if (!validModes.includes(mode)) {
        warnings.push(`Invalid mode "${mode}". Using default "build".`);
    }

    // Validate autonomy level
    const autonomyLevel = config.get<string>('autonomy.level', 'semi-auto');
    const validLevels = ['manual', 'semi-auto', 'auto'];
    if (!validLevels.includes(autonomyLevel)) {
        warnings.push(`Invalid autonomy level "${autonomyLevel}". Using default "semi-auto".`);
    }

    // Show warnings to user
    if (warnings.length > 0) {
        const message = `Victor configuration issues:\n${warnings.join('\n')}`;
        console.warn('[Victor]', message);
        vscode.window.showWarningMessage(`Victor: ${warnings[0]}${warnings.length > 1 ? ` (+${warnings.length - 1} more)` : ''}`);
    }
}

export async function activate(context: vscode.ExtensionContext) {
    console.log('Victor AI extension activating...');

    // Initialize centralized state store
    const store = getStore();
    await store.initialize(context);

    const config = vscode.workspace.getConfiguration('victor');

    // Validate configuration
    validateConfiguration(config);

    // Initialize server manager
    const serverManager = createServerManager();

    // Create server status bar item
    const serverStatusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        99
    );
    serverStatusBarItem.command = 'victor.serverStatus';
    updateServerStatusBar(serverStatusBarItem, ServerStatus.Stopped);
    serverStatusBarItem.show();
    context.subscriptions.push(serverStatusBarItem);

    // Listen for server status changes and update state store
    serverManager.onStatusChange((status) => {
        store.updateState({
            server: {
                status,
                port: serverManager.getActivePort(),
                url: serverManager.getServerUrl(),
            },
        });
        const activePort = serverManager.getActivePort();
        const configPort = config.get<number>('serverPort', 8765);
        const hasExplicitServerUrl = !!(config.get<string>('serverUrl') || '').trim();
        const usingFallback = !hasExplicitServerUrl && activePort !== configPort;
        updateServerStatusBar(serverStatusBarItem, status, activePort, usingFallback);
    });

    // Initialize the Victor client with server URL from state
    const apiKey = config.get<string>('serverApiKey') || undefined;
    const victorClient = new VictorClient(store.select(selectors.serverUrl), undefined, apiKey);
    // Pre-fetch session token when API key is configured
    victorClient.prefetchSessionToken().catch((error) => {
        console.warn('[Victor] Failed to prefetch session token (server may be offline):', error.message || error);
    });
    // Keep Victor client auth in sync with configuration updates
    context.subscriptions.push(
        store.subscribe('server', (_state) => {
            victorClient.setServerUrl(store.select(selectors.serverUrl));
        })
    );
    context.subscriptions.push(
        store.subscribe('settings', (_state) => {
            const latestKey = store.select(selectors.serverApiKey) || undefined;
            victorClient.setApiToken(latestKey);
        })
    );

    // Create mode status bar item
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right,
        100
    );
    statusBarItem.command = 'victor.switchMode';
    updateStatusBar(statusBarItem, store.select(selectors.mode));
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Subscribe to mode changes from state store
    context.subscriptions.push(
        store.subscribeToSelector(selectors.mode, (mode) => {
            updateStatusBar(statusBarItem, mode);
        })
    );

    // Initialize providers
    const clientOutput = vscode.window.createOutputChannel('Victor Client');
    clientOutput.appendLine('[Victor] Extension activated');
    context.subscriptions.push(clientOutput);

    const chatViewProvider = new ChatViewProvider(context.extensionUri, victorClient, clientOutput);
    context.subscriptions.push(chatViewProvider);
    const semanticSearchProvider = new SemanticSearchProvider(victorClient);
    const inlineCompletionProvider = new InlineCompletionProvider(victorClient);
    const inlineEditProvider = new InlineEditProvider(victorClient, clientOutput);
    const historyViewProvider = new HistoryViewProvider(victorClient);
    const toolsViewProvider = new ToolsViewProvider();
    const agentsViewProvider = new AgentsViewProvider(victorClient, clientOutput);
    const plansViewProvider = new PlansViewProvider(victorClient, clientOutput);

    clientOutput.appendLine('[Victor] Providers initialized');

    // Register chat view (left sidebar)
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'victor.chatView',
            chatViewProvider,
            { webviewOptions: { retainContextWhenHidden: true } }
        )
    );

    // Register chat panel view (bottom panel / secondary sidebar)
    // Users can drag this to the right side for Copilot-like experience
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            'victor.chatPanelView',
            chatViewProvider,
            { webviewOptions: { retainContextWhenHidden: true } }
        )
    );

    // Register history view
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.historyView',
            historyViewProvider
        )
    );
    registerHistoryCommands(context, victorClient, historyViewProvider);

    // Register tools view
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.toolsView',
            toolsViewProvider
        )
    );
    registerToolsCommands(context, toolsViewProvider);

    // Register agents view (shows active agent tasks)
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.agentsView',
            agentsViewProvider
        )
    );
    context.subscriptions.push({ dispose: () => agentsViewProvider.dispose() });

    // Register plans view (shows execution plans)
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.plansView',
            plansViewProvider
        )
    );
    context.subscriptions.push({ dispose: () => plansViewProvider.dispose() });

    // Register agent and plan commands
    registerAgentCommands(context, agentsViewProvider);
    registerPlanCommands(context, plansViewProvider);

    // Initialize teams view
    const teamsViewProvider = new TeamsViewProvider(victorClient, clientOutput);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.teamsView',
            teamsViewProvider
        )
    );
    context.subscriptions.push({ dispose: () => teamsViewProvider.dispose() });
    registerTeamCommands(context, teamsViewProvider);

    // Initialize workflows view
    const workflowsViewProvider = new WorkflowsViewProvider(victorClient, clientOutput);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.workflowsView',
            workflowsViewProvider
        )
    );
    context.subscriptions.push({ dispose: () => workflowsViewProvider.dispose() });
    registerWorkflowCommands(context, workflowsViewProvider);

    // Initialize diff view provider
    const diffViewProvider = new DiffViewProvider();
    registerDiffCommands(context, diffViewProvider);
    context.subscriptions.push({ dispose: () => diffViewProvider.dispose() });

    // Initialize file watcher for auto-refresh
    const watcherComponents = createFileWatcher(context);
    const fileWatcher = watcherComponents.fileWatcher;
    const refreshManager = watcherComponents.refreshManager;

    // Register views for auto-refresh on file changes
    refreshManager.registerView('victor.historyView', () => historyViewProvider.refresh());
    refreshManager.registerView('victor.toolsView', () => toolsViewProvider.refresh());

    // Initialize context provider for @-mentions
    const contextProvider = new ContextProvider();
    registerContextCommands(context, contextProvider);

    // Initialize terminal provider
    const terminalProvider = new TerminalProvider();
    const terminalHistoryProvider = new TerminalHistoryProvider(terminalProvider);
    registerTerminalCommands(context, terminalProvider);
    context.subscriptions.push(terminalProvider);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.terminalHistoryView',
            terminalHistoryProvider
        )
    );

    // Initialize Terminal Agent provider (AI-assisted command execution)
    const terminalAgentProvider = new TerminalAgentProvider(victorClient, clientOutput);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.terminalAgentView',
            terminalAgentProvider
        )
    );
    registerTerminalAgentCommands(context, terminalAgentProvider);
    context.subscriptions.push({ dispose: () => terminalAgentProvider.dispose() });

    // Initialize shared Terminal History service (used by inline completions and terminal agent)
    const terminalHistoryService = TerminalHistoryService.getInstance();
    context.subscriptions.push({ dispose: () => terminalHistoryService.dispose() });

    // Initialize Smart Paste provider (adapts pasted code to context)
    const smartPasteProvider = new SmartPasteProvider(victorClient, clientOutput);
    registerSmartPasteCommands(context, smartPasteProvider);

    // Initialize Multi-file Composer (like Cursor's Composer)
    const composerViewProvider = new ComposerViewProvider(
        context.extensionUri,
        victorClient,
        diffViewProvider,
        clientOutput
    );
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            ComposerViewProvider.viewType,
            composerViewProvider
        )
    );
    registerComposerCommands(context, composerViewProvider);
    context.subscriptions.push({ dispose: () => composerViewProvider.dispose() });

    // Initialize diagnostics view
    const diagnosticsViewProvider = new DiagnosticsViewProvider();
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.diagnosticsView',
            diagnosticsViewProvider
        )
    );
    // Helper function to send messages to chat
    const sendToChat = async (message: string) => {
        clientOutput.appendLine(`[Chat] Queued send from command (${message.length} chars)`);
        await chatViewProvider.sendMessage(message);
        await vscode.commands.executeCommand('victor.chatView.focus');
    };
    registerDiagnosticsCommands(context, diagnosticsViewProvider, sendToChat);
    context.subscriptions.push(diagnosticsViewProvider);

    // Register code action commands
    registerCodeActionCommands(context, victorClient, sendToChat);

    // Initialize settings view
    const settingsViewProvider = new SettingsViewProvider(context.extensionUri);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider(
            SettingsViewProvider.viewType,
            settingsViewProvider
        )
    );
    registerSettingsCommands(context);

    // Initialize workspace insights view
    const workspaceInsightsViewProvider = new WorkspaceInsightsViewProvider();
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.workspaceInsightsView',
            workspaceInsightsViewProvider
        )
    );
    registerWorkspaceInsightsCommands(context, workspaceInsightsViewProvider);

    // Initialize MCP view
    const mcpViewProvider = new McpViewProvider();
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider(
            'victor.mcpView',
            mcpViewProvider
        )
    );
    registerMcpCommands(context, mcpViewProvider);

    // Initialize tool approval service
    const toolApprovalService = new ToolApprovalService(victorClient, context);
    context.subscriptions.push(toolApprovalService);
    registerToolApprovalCommands(context, toolApprovalService);

    // Initialize provider settings service
    const providerSettingsService = new ProviderSettingsService(context);
    context.subscriptions.push(providerSettingsService);
    registerProviderSettingsCommands(context, providerSettingsService);

    // Initialize provider settings
    await providerSettingsService.initialize();

    // Initialize CodeLens provider
    const codeLensProvider = new VictorCodeLensProvider();
    context.subscriptions.push(
        vscode.languages.registerCodeLensProvider(
            { pattern: '**/*.{ts,tsx,js,jsx,py,java,cs,go,rs,cpp,c,rb,php,swift,kt}' },
            codeLensProvider
        )
    );
    registerCodeLensCommands(context);
    context.subscriptions.push({ dispose: () => codeLensProvider.dispose() });

    // Initialize Code Action provider (lightbulb menu)
    const codeActionProvider = new VictorCodeActionProvider();
    context.subscriptions.push(
        vscode.languages.registerCodeActionsProvider(
            { pattern: '**/*.{ts,tsx,js,jsx,py,java,cs,go,rs,cpp,c,rb,php,swift,kt}' },
            codeActionProvider,
            {
                providedCodeActionKinds: VictorCodeActionProvider.providedCodeActionKinds,
            }
        )
    );
    registerCodeActionProviderCommands(context, sendToChat);

    // Initialize Hover provider for AI tooltips
    const hoverProvider = new VictorHoverProvider();
    hoverProvider.setVictorClient(victorClient);
    context.subscriptions.push(
        vscode.languages.registerHoverProvider(
            { pattern: '**/*.{ts,tsx,js,jsx,py,java,cs,go,rs,cpp,c,rb,php,swift,kt}' },
            hoverProvider
        )
    );
    registerHoverCommands(context);
    context.subscriptions.push({ dispose: () => hoverProvider.dispose() });

    // Initialize Tool Execution Service for progress visibility
    const toolExecutionService = new ToolExecutionService();
    toolExecutionService.setVictorClient(victorClient);
    registerToolExecutionCommands(context, toolExecutionService);
    context.subscriptions.push(toolExecutionService);

    // Initialize RL Model Selector Service
    const rlService = new RLService();
    registerRLCommands(context, rlService);
    context.subscriptions.push(rlService);

    // Store all providers in centralized object
    providers = {
        victorClient,
        chatViewProvider,
        semanticSearchProvider,
        inlineCompletionProvider,
        inlineEditProvider,
        serverManager,
        historyViewProvider,
        toolsViewProvider,
        agentsViewProvider,
        plansViewProvider,
        diffViewProvider,
        fileWatcher,
        refreshManager,
        contextProvider,
        terminalProvider,
        terminalHistoryProvider,
        diagnosticsViewProvider,
        settingsViewProvider,
        workspaceInsightsViewProvider,
        mcpViewProvider,
        toolApprovalService,
        providerSettingsService,
        codeLensProvider,
        codeActionProvider,
        hoverProvider,
        toolExecutionService,
        rlService,
        terminalAgentProvider,
        smartPasteProvider,
        composerViewProvider,
        teamsViewProvider,
        workflowsViewProvider,
        statusBarItem,
        serverStatusBarItem,
    };

    // Register inline completions if enabled
    if (config.get('showInlineCompletions', true)) {
        context.subscriptions.push(
            vscode.languages.registerInlineCompletionItemProvider(
                { pattern: '**' },
                inlineCompletionProvider
            )
        );
    }

    // Register inline edit provider (commands registered internally)
    context.subscriptions.push(inlineEditProvider);

    // Register commands with providers
    registerCommands(context, providers);

    // Auto-start server if configured
    if (config.get('autoStart', true)) {
        await autoStartServer(serverManager);
    }

    // Register dispose handler
    context.subscriptions.push({
        dispose: () => {
            serverManager.dispose();
        }
    });

    console.log('Victor AI extension activated');
}

/**
 * Auto-start the server with user feedback
 */
async function autoStartServer(serverManager: ServerManager): Promise<void> {
    // First check if server is already running (e.g., user started manually)
    if (await serverManager.checkHealth()) {
        vscode.window.showInformationMessage('Victor: Connected to existing server');
        return;
    }

    // Show progress notification while starting
    await vscode.window.withProgress(
        {
            location: vscode.ProgressLocation.Notification,
            title: 'Victor: Starting server...',
            cancellable: false
        },
        async (progress) => {
            progress.report({ increment: 0 });

            const started = await serverManager.start();

            if (started) {
                vscode.window.showInformationMessage('Victor: Server started');
            } else {
                const action = await vscode.window.showErrorMessage(
                    'Victor: Failed to start server. Make sure Victor is installed.',
                    'Show Logs',
                    'Manual Start',
                    'Install Help'
                );

                if (action === 'Show Logs') {
                    serverManager.showOutput();
                } else if (action === 'Manual Start') {
                    vscode.commands.executeCommand('victor.startServer');
                } else if (action === 'Install Help') {
                    vscode.env.openExternal(
                        vscode.Uri.parse('https://github.com/anvai-labs/victor#quick-start')
                    );
                }
            }
        }
    );
}

function registerCommands(context: vscode.ExtensionContext, p: ExtensionProviders) {
    const store = getStore();
    const { victorClient, chatViewProvider, semanticSearchProvider, serverManager } = p;

    // Chat command
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.chat', async () => {
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Explain selected code
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.explain', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return;}

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            await chatViewProvider.sendMessage(
                `Explain this code:\n\`\`\`\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Refactor selected code
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refactor', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return;}

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const suggestion = await vscode.window.showInputBox({
                prompt: 'What refactoring would you like?',
                placeHolder: 'e.g., Extract to function, simplify logic...'
            });

            if (suggestion) {
                await chatViewProvider.sendMessage(
                    `Refactor this code (${suggestion}):\n\`\`\`\n${text}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        })
    );

    // Fix issues in selection
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.fix', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return;}

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            // Get diagnostics for the selection
            const diagnostics = vscode.languages.getDiagnostics(editor.document.uri)
                .filter(d => selection.contains(d.range))
                .map(d => `- ${d.message}`)
                .join('\n');

            const prompt = diagnostics
                ? `Fix these issues:\n${diagnostics}\n\nIn this code:\n\`\`\`\n${text}\n\`\`\``
                : `Fix any issues in this code:\n\`\`\`\n${text}\n\`\`\``;

            await chatViewProvider.sendMessage(prompt);
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Generate tests
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.test', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return;}

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const language = editor.document.languageId;
            await chatViewProvider.sendMessage(
                `Generate comprehensive unit tests for this ${language} code:\n\`\`\`${language}\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Add documentation
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.document', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return;}

            const selection = editor.selection;
            const text = editor.document.getText(selection);
            if (!text) {
                vscode.window.showWarningMessage('Please select some code first');
                return;
            }

            const language = editor.document.languageId;
            await chatViewProvider.sendMessage(
                `Add documentation/docstrings to this ${language} code:\n\`\`\`${language}\n${text}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Semantic search
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.search', async () => {
            const query = await vscode.window.showInputBox({
                prompt: 'Semantic code search',
                placeHolder: 'e.g., function that handles user authentication'
            });

            if (query) {
                await semanticSearchProvider.search(query);
            }
        })
    );

    // Switch model - dynamically fetches available models from server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.switchModel', async () => {
            // Show loading indicator while fetching models
            const models = await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'Fetching available models...',
                    cancellable: false
                },
                async () => {
                    try {
                        const serverModels = await victorClient.getModels();
                        if (serverModels.length > 0) {
                            return serverModels.map(m => ({
                                label: `${m.is_local ? '$(server-environment) ' : '$(cloud) '}${m.display_name}`,
                                description: m.provider,
                                detail: m.model_id,
                                value: m.model_id,
                                provider: m.provider,
                                displayName: m.display_name,
                            }));
                        }
                        // Fallback to default models if server returns empty
                        return getDefaultModels();
                    } catch {
                        return getDefaultModels();
                    }
                }
            );

            const selected = await vscode.window.showQuickPick(models, {
                placeHolder: 'Select AI model',
                matchOnDescription: true,
                matchOnDetail: true,
            });

            if (selected) {
                try {
                    await victorClient.switchModel(selected.provider, selected.value);

                    // Update centralized state store (also updates VS Code config)
                    await store.setModel({
                        provider: selected.provider as ModelInfo['provider'],
                        modelId: selected.value,
                        displayName: selected.displayName,
                    });

                    vscode.window.showInformationMessage(`Switched to ${selected.displayName}`);
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to switch model: ${error}`);
                }
            }
        })
    );

    context.subscriptions.push(
        vscode.commands.registerCommand('victor.switchProfile', async () => {
            const profiles = await vscode.window.withProgress<ProfileQuickPickItem[]>(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'Fetching Victor profiles...',
                    cancellable: false
                },
                async () => {
                    try {
                        const serverProfiles = await victorClient.getProfiles();
                        if (serverProfiles.length > 0) {
                            return serverProfiles.map((profile): ProfileQuickPickItem => ({
                                label: `${profile.is_default ? '$(star-full) ' : ''}${profile.name}`,
                                description: `${profile.provider} / ${profile.model}`,
                                detail: profile.description || '',
                                value: profile.name,
                                provider: profile.provider,
                                model: profile.model,
                            }));
                        }
                    } catch {
                        // Fallback below.
                    }
                    const configuredProfile = vscode.workspace
                        .getConfiguration('victor')
                        .get<string>('profile', 'default');
                    return [{
                        label: configuredProfile,
                        description: 'Configured VS Code profile',
                        detail: 'Backend profile discovery unavailable',
                        value: configuredProfile,
                    } satisfies ProfileQuickPickItem];
                }
            );

            const selected = await vscode.window.showQuickPick<ProfileQuickPickItem>(profiles, {
                placeHolder: 'Select Victor profile',
                matchOnDescription: true,
                matchOnDetail: true,
            });

            if (selected) {
                try {
                    await victorClient.switchProfile(selected.value);
                    await store.setProfile(selected.value);
                    if (selected.provider && selected.model) {
                        await store.setModel({
                            provider: selected.provider as ModelInfo['provider'],
                            modelId: selected.model,
                            displayName: selected.model,
                        });
                    }
                    vscode.window.showInformationMessage(`Switched to profile ${selected.value}`);
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to switch profile: ${error}`);
                }
            }
        })
    );

    // Helper function for fallback models
    function getDefaultModels() {
        return [
            { label: '$(cloud) Claude Sonnet 4', description: 'anthropic', detail: 'claude-sonnet-4-20250514', value: 'claude-sonnet-4-20250514', provider: 'anthropic', displayName: 'Claude Sonnet 4' },
            { label: '$(cloud) Claude Opus 4.5', description: 'anthropic', detail: 'claude-opus-4-5-20251101', value: 'claude-opus-4-5-20251101', provider: 'anthropic', displayName: 'Claude Opus 4.5' },
            { label: '$(cloud) GPT-4 Turbo', description: 'openai', detail: 'gpt-4-turbo', value: 'gpt-4-turbo', provider: 'openai', displayName: 'GPT-4 Turbo' },
            { label: '$(cloud) GPT-4o', description: 'openai', detail: 'gpt-4o', value: 'gpt-4o', provider: 'openai', displayName: 'GPT-4o' },
            { label: '$(cloud) Gemini 2.0 Flash', description: 'google', detail: 'gemini-2.0-flash', value: 'gemini-2.0-flash', provider: 'google', displayName: 'Gemini 2.0 Flash' },
            { label: '$(server-environment) Qwen 2.5 Coder', description: 'ollama', detail: 'qwen2.5-coder:14b', value: 'qwen2.5-coder:14b', provider: 'ollama', displayName: 'Qwen 2.5 Coder' },
            { label: '$(server-environment) Llama 3.1', description: 'ollama', detail: 'llama3.1:8b', value: 'llama3.1:8b', provider: 'ollama', displayName: 'Llama 3.1' },
        ];
    }

    // Switch mode - now uses state store
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.switchMode', async () => {
            let modes = [
                { label: '$(tools) Build', description: 'Implementation mode - all tools available', value: 'build' as AgentMode },
                { label: '$(search) Plan', description: 'Planning mode - read-only analysis', value: 'plan' as AgentMode },
                { label: '$(eye) Review', description: 'Review mode - bugs, risks, and regressions', value: 'review' as AgentMode },
                { label: '$(organization) Delegate', description: 'Delegation mode - coordinate multi-agent work', value: 'delegate' as AgentMode },
                { label: '$(eye) Explore', description: 'Exploration mode - understand codebase', value: 'explore' as AgentMode },
            ];
            try {
                const serverModes = await victorClient.getModes();
                if (serverModes.length > 0) {
                    modes = serverModes.map((mode) => ({
                        label: mode.name.charAt(0).toUpperCase() + mode.name.slice(1),
                        description: mode.description,
                        value: mode.name as AgentMode,
                    }));
                }
            } catch {
                // Keep local fallback modes.
            }

            const selected = await vscode.window.showQuickPick(modes, {
                placeHolder: 'Select agent mode'
            });

            if (selected) {
                try {
                    await victorClient.switchMode(selected.value);

                    // Update centralized state store (also updates VS Code config)
                    await store.setMode(selected.value);

                    vscode.window.showInformationMessage(`Switched to ${selected.value} mode`);
                } catch (error) {
                    vscode.window.showErrorMessage(`Failed to switch mode: ${error}`);
                }
            }
        })
    );

    // Server status command
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.serverStatus', async () => {
            const status = serverManager.getStatus();
            const actions: string[] = [];

            switch (status) {
                case ServerStatus.Running:
                    actions.push('Restart Server', 'Stop Server', 'Show Logs');
                    break;
                case ServerStatus.Stopped:
                case ServerStatus.Error:
                    actions.push('Start Server', 'Show Logs');
                    break;
                case ServerStatus.Starting:
                    actions.push('Show Logs');
                    break;
            }

            const action = await vscode.window.showQuickPick(actions, {
                placeHolder: `Victor Server: ${status}`
            });

            if (action === 'Start Server') {
                await serverManager.start();
            } else if (action === 'Stop Server') {
                await serverManager.stop();
            } else if (action === 'Restart Server') {
                await serverManager.restart();
            } else if (action === 'Show Logs') {
                serverManager.showOutput();
            }
        })
    );

    // Undo
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.undo', async () => {
            try {
                const result = await victorClient.undo();
                if (result.success) {
                    vscode.window.showInformationMessage(result.message);
                } else {
                    vscode.window.showWarningMessage(result.message);
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Undo failed: ${error}`);
            }
        })
    );

    // Redo
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.redo', async () => {
            try {
                const result = await victorClient.redo();
                if (result.success) {
                    vscode.window.showInformationMessage(result.message);
                } else {
                    vscode.window.showWarningMessage(result.message);
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Redo failed: ${error}`);
            }
        })
    );

    // Start server (manual)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.startServer', async () => {
            const started = await serverManager.start();
            if (started) {
                vscode.window.showInformationMessage('Victor server started');
            }
        })
    );

    // Stop server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.stopServer', async () => {
            await serverManager.stop();
            vscode.window.showInformationMessage('Victor server stopped');
        })
    );

    // Restart server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.restartServer', async () => {
            await serverManager.restart();
        })
    );

    // Show server logs
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showServerLogs', async () => {
            serverManager.showOutput();
        })
    );

    // Export conversation
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.exportConversation', async () => {
            const exportOption = await vscode.window.showQuickPick([
                { label: 'Export as Markdown', value: 'markdown' },
                { label: 'Export as JSON', value: 'json' },
                { label: 'Copy to Clipboard', value: 'clipboard' },
            ], {
                placeHolder: 'Choose export format',
            });

            if (!exportOption) {
                return;
            }

            try {
                // For clipboard, use markdown format
                const format = exportOption.value === 'clipboard' ? 'markdown' : exportOption.value;
                const content = await victorClient.exportConversation(format as 'json' | 'markdown');

                if (exportOption.value === 'clipboard') {
                    await vscode.env.clipboard.writeText(content);
                    vscode.window.showInformationMessage('Conversation copied to clipboard');
                } else {
                    // Show save dialog
                    const extension = exportOption.value === 'markdown' ? 'md' : 'json';
                    const defaultUri = vscode.Uri.file(`conversation.${extension}`);

                    const saveUri = await vscode.window.showSaveDialog({
                        defaultUri,
                        filters: {
                            [exportOption.value === 'markdown' ? 'Markdown' : 'JSON']: [extension],
                        },
                    });

                    if (saveUri) {
                        await vscode.workspace.fs.writeFile(
                            saveUri,
                            Buffer.from(content, 'utf-8')
                        );
                        vscode.window.showInformationMessage(`Conversation exported to ${saveUri.fsPath}`);
                    }
                }
            } catch (error) {
                vscode.window.showErrorMessage(`Export failed: ${error}`);
            }
        })
    );
}

function updateStatusBar(statusBarItem: vscode.StatusBarItem, mode: string): void {
    const icons: Record<string, string> = {
        'build': '$(tools)',
        'plan': '$(search)',
        'explore': '$(eye)'
    };
    statusBarItem.text = `${icons[mode] || '$(hubot)'} Victor: ${mode}`;
    statusBarItem.tooltip = `Victor AI - ${mode} mode\nClick to switch modes`;
}

function updateServerStatusBar(
    serverStatusBarItem: vscode.StatusBarItem,
    status: ServerStatus,
    port?: number,
    usingFallback?: boolean
): void {
    const portInfo = port ? ` (port ${port})` : '';
    const fallbackNote = usingFallback ? ' [fallback]' : '';

    const statusConfig: Record<ServerStatus, { icon: string; color?: string; tooltip: string }> = {
        [ServerStatus.Running]: {
            icon: '$(circle-filled)',
            color: 'statusBarItem.prominentBackground',
            tooltip: `Victor Server: Running${portInfo}${fallbackNote}\nClick for options`
        },
        [ServerStatus.Starting]: {
            icon: '$(sync~spin)',
            tooltip: 'Victor Server: Starting...'
        },
        [ServerStatus.Stopped]: {
            icon: '$(circle-outline)',
            tooltip: 'Victor Server: Stopped\nClick to start'
        },
        [ServerStatus.Error]: {
            icon: '$(error)',
            color: 'statusBarItem.errorBackground',
            tooltip: 'Victor Server: Error\nClick for options'
        },
        [ServerStatus.Reconnecting]: {
            icon: '$(sync~spin)',
            color: 'statusBarItem.warningBackground',
            tooltip: 'Victor Server: Reconnecting...'
        }
    };

    const cfg = statusConfig[status];
    // Show port in text only when using fallback port
    const portText = usingFallback && port ? `:${port}` : '';
    serverStatusBarItem.text = `${cfg.icon} Server${portText}`;
    serverStatusBarItem.tooltip = cfg.tooltip;

    if (cfg.color) {
        serverStatusBarItem.backgroundColor = new vscode.ThemeColor(cfg.color);
    } else {
        serverStatusBarItem.backgroundColor = undefined;
    }
}

export function deactivate() {
    console.log('Victor AI extension deactivating...');
    disposeEventBridgeClient();

    // Dispose state store
    const store = getStore();
    store.dispose();

    // Dispose all providers
    if (providers) {
        if (providers.serverManager) {
            providers.serverManager.dispose();
        }
        if (providers.historyViewProvider) {
            providers.historyViewProvider.dispose();
        }
        if (providers.terminalProvider) {
            providers.terminalProvider.dispose();
        }
        if (providers.diagnosticsViewProvider) {
            providers.diagnosticsViewProvider.dispose();
        }
        if (providers.diffViewProvider) {
            providers.diffViewProvider.dispose();
        }
        if (providers.codeLensProvider) {
            providers.codeLensProvider.dispose();
        }
        if (providers.hoverProvider) {
            providers.hoverProvider.dispose();
        }
        if (providers.toolExecutionService) {
            providers.toolExecutionService.dispose();
        }
        providers = null;
    }

    console.log('Victor AI extension deactivated');
}

/**
 * Get the extension providers (for use by other modules)
 */
export function getProviders(): ExtensionProviders | null {
    return providers;
}
