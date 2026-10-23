/**
 * MCP (Model Context Protocol) View Provider
 *
 * Provides a tree view for managing MCP servers:
 * - List connected MCP servers
 * - Show available tools per server
 * - Connect/disconnect servers
 * - Configure new MCP endpoints
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

interface McpServer {
    name: string;
    status: 'connected' | 'disconnected' | 'error';
    connected?: boolean;
    endpoint?: string;
    tools?: string[];
    error?: string;
}

type McpItemType = 'server' | 'tool' | 'action' | 'loading' | 'error' | 'info';

export class McpViewProvider implements vscode.TreeDataProvider<McpItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<McpItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private isLoading = false;
    private servers: McpServer[] = [];
    private lastError: string | null = null;

    constructor() {
        this.refresh();
    }

    async refresh(): Promise<void> {
        if (this.isLoading) {
            return;
        }

        this.isLoading = true;
        this.lastError = null;
        this._onDidChangeTreeData.fire(undefined);

        try {
            const providers = getProviders();
            if (!providers?.victorClient) {
                this.lastError = 'Victor client not available';
                return;
            }

            const result = await providers.victorClient.getMcpServers();
            // Transform API response to include status field
            this.servers = (result.servers || []).map(s => ({
                ...s,
                status: s.connected ? 'connected' as const : 'disconnected' as const,
            }));

        } catch (error) {
            console.error('Failed to load MCP servers:', error);
            this.lastError = error instanceof Error ? error.message : 'Unknown error';
        } finally {
            this.isLoading = false;
            this._onDidChangeTreeData.fire(undefined);
        }
    }

    getTreeItem(element: McpItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: McpItem): McpItem[] {
        if (!element) {
            // Root level
            if (this.isLoading) {
                return [new McpItem('Loading MCP servers...', 'loading', vscode.TreeItemCollapsibleState.None)];
            }

            if (this.lastError) {
                return [new McpItem(`Error: ${this.lastError}`, 'error', vscode.TreeItemCollapsibleState.None)];
            }

            const items: McpItem[] = [];

            // Add "Add Server" action at top
            items.push(new McpItem(
                'Add MCP Server',
                'action',
                vscode.TreeItemCollapsibleState.None,
                { action: 'add_server' },
                'add'
            ));

            if (this.servers.length === 0) {
                items.push(new McpItem(
                    'No MCP servers configured',
                    'info',
                    vscode.TreeItemCollapsibleState.None,
                    undefined,
                    'info'
                ));
                return items;
            }

            // List servers
            for (const server of this.servers) {
                const statusIcon = server.status === 'connected' ? 'check'
                    : server.status === 'error' ? 'error'
                    : 'circle-outline';

                const hasTools = server.tools && server.tools.length > 0;
                const item = new McpItem(
                    server.name,
                    'server',
                    hasTools ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None,
                    { server },
                    statusIcon
                );

                item.description = server.status;
                if (server.endpoint) {
                    item.tooltip = `${server.name}\nEndpoint: ${server.endpoint}\nStatus: ${server.status}${server.error ? '\nError: ' + server.error : ''}`;
                }
                items.push(item);
            }

            return items;
        }

        // Children for a server - show its tools
        if (element.itemType === 'server' && element.data?.server) {
            const server = element.data.server as McpServer;
            if (server.tools && server.tools.length > 0) {
                return server.tools.map(tool =>
                    new McpItem(
                        tool,
                        'tool',
                        vscode.TreeItemCollapsibleState.None,
                        { tool, serverName: server.name },
                        'symbol-function'
                    )
                );
            }
        }

        return [];
    }
}

class McpItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly itemType: McpItemType,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly data?: Record<string, unknown>,
        iconName?: string
    ) {
        super(label, collapsibleState);

        if (iconName) {
            const color = itemType === 'error' ? new vscode.ThemeColor('errorForeground')
                : itemType === 'action' ? new vscode.ThemeColor('textLink.foreground')
                : undefined;
            this.iconPath = new vscode.ThemeIcon(iconName, color);
        }

        switch (itemType) {
            case 'loading':
                this.iconPath = new vscode.ThemeIcon('loading~spin');
                break;
            case 'error':
                this.iconPath = new vscode.ThemeIcon('error', new vscode.ThemeColor('errorForeground'));
                break;
            case 'server':
                this.contextValue = data?.server && (data.server as McpServer).status === 'connected'
                    ? 'connectedServer'
                    : 'disconnectedServer';
                break;
            case 'action':
                this.contextValue = 'action';
                if (data?.action === 'add_server') {
                    this.command = {
                        command: 'victor.addMcpServer',
                        title: 'Add MCP Server',
                    };
                }
                break;
        }
    }
}

/**
 * Register MCP view commands
 */
export function registerMcpCommands(
    context: vscode.ExtensionContext,
    provider: McpViewProvider
): void {
    // Refresh MCP view
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshMcp', () => {
            provider.refresh();
        })
    );

    // Add MCP server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.addMcpServer', async () => {
            const serverName = await vscode.window.showInputBox({
                prompt: 'Enter MCP server name',
                placeHolder: 'e.g., filesystem, github, sqlite',
            });

            if (!serverName) {
                return;
            }

            const endpoint = await vscode.window.showInputBox({
                prompt: 'Enter MCP server endpoint (optional for built-in servers)',
                placeHolder: 'e.g., http://localhost:3000 or leave empty',
            });

            const providers = getProviders();
            if (!providers?.victorClient) {
                vscode.window.showWarningMessage('Victor client not available');
                return;
            }

            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: `Connecting to ${serverName}...`,
                cancellable: false,
            }, async () => {
                try {
                    const result = await providers.victorClient.connectMcpServer(
                        serverName,
                        endpoint || undefined
                    );

                    if (result.success) {
                        vscode.window.showInformationMessage(`Connected to MCP server: ${serverName}`);
                        provider.refresh();
                    } else {
                        vscode.window.showErrorMessage('Failed to connect to MCP server');
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Connection error: ${error}`);
                }
            });
        })
    );

    // Connect to server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.connectMcpServer', async (item: McpItem) => {
            if (item.data?.server) {
                const server = item.data.server as McpServer;
                const providers = getProviders();
                if (!providers?.victorClient) {
                    vscode.window.showWarningMessage('Victor client not available');
                    return;
                }

                await vscode.window.withProgress({
                    location: vscode.ProgressLocation.Notification,
                    title: `Connecting to ${server.name}...`,
                    cancellable: false,
                }, async () => {
                    try {
                        const result = await providers.victorClient.connectMcpServer(
                            server.name,
                            server.endpoint
                        );

                        if (result.success) {
                            vscode.window.showInformationMessage(`Connected to ${server.name}`);
                            provider.refresh();
                        } else {
                            vscode.window.showErrorMessage('Failed to connect to MCP server');
                        }
                    } catch (error) {
                        vscode.window.showErrorMessage(`Connection error: ${error}`);
                    }
                });
            }
        })
    );

    // Disconnect from server
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.disconnectMcpServer', async (item: McpItem) => {
            if (item.data?.server) {
                const server = item.data.server as McpServer;
                const providers = getProviders();
                if (!providers?.victorClient) {
                    vscode.window.showWarningMessage('Victor client not available');
                    return;
                }

                try {
                    const result = await providers.victorClient.disconnectMcpServer(server.name);

                    if (result.success) {
                        vscode.window.showInformationMessage(`Disconnected from ${server.name}`);
                        provider.refresh();
                    } else {
                        vscode.window.showErrorMessage('Failed to disconnect from MCP server');
                    }
                } catch (error) {
                    vscode.window.showErrorMessage(`Disconnect error: ${error}`);
                }
            }
        })
    );

    // Show MCP server info
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showMcpServerInfo', async (item: McpItem) => {
            if (item.data?.server) {
                const server = item.data.server as McpServer;
                const info = [
                    `Name: ${server.name}`,
                    `Status: ${server.status}`,
                    server.endpoint ? `Endpoint: ${server.endpoint}` : '',
                    server.tools ? `Tools: ${server.tools.length}` : '',
                    server.error ? `Error: ${server.error}` : '',
                ].filter(Boolean).join('\n');

                await vscode.window.showInformationMessage(info, { modal: true });
            }
        })
    );
}
