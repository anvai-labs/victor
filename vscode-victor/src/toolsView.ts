/**
 * Tools View Provider
 *
 * Provides a tree view of available Victor tools organized by category.
 * Dynamically fetches tools from the backend server.
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

export interface ToolInfo {
    name: string;
    description: string;
    category: string;
    cost_tier: string;
    parameters: Record<string, unknown>;
    is_dangerous: boolean;
    requires_approval: boolean;
}

// Default tools to show when server is unavailable
const DEFAULT_TOOLS: ToolInfo[] = [
    { name: 'read', description: 'Read file contents', category: 'Filesystem', cost_tier: 'free', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'write', description: 'Write or create files', category: 'Filesystem', cost_tier: 'free', parameters: {}, is_dangerous: true, requires_approval: true },
    { name: 'edit', description: 'Edit files atomically', category: 'Filesystem', cost_tier: 'free', parameters: {}, is_dangerous: true, requires_approval: true },
    { name: 'ls', description: 'List directory contents', category: 'Filesystem', cost_tier: 'free', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'search', description: 'Search code (semantic + literal)', category: 'Search', cost_tier: 'free', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'shell', description: 'Run shell commands', category: 'Shell', cost_tier: 'free', parameters: {}, is_dangerous: true, requires_approval: true },
    { name: 'git', description: 'Git operations', category: 'Git', cost_tier: 'free', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'code_review', description: 'Review code quality', category: 'Analysis', cost_tier: 'low', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'test', description: 'Run tests with pytest', category: 'Testing', cost_tier: 'free', parameters: {}, is_dangerous: false, requires_approval: false },
    { name: 'web_search', description: 'Search the web', category: 'Web', cost_tier: 'medium', parameters: {}, is_dangerous: false, requires_approval: false },
];

export class ToolsViewProvider implements vscode.TreeDataProvider<ToolItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<ToolItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private tools: ToolInfo[] = [];
    private categories: string[] = [];
    private isLoading = false;
    private lastError: string | null = null;

    constructor() {
        // Initial load
        this.loadTools();
    }

    async loadTools(): Promise<void> {
        if (this.isLoading) {
            return;
        }

        this.isLoading = true;
        this.lastError = null;

        try {
            const providers = getProviders();
            if (providers?.victorClient) {
                const response = await providers.victorClient.getTools();
                if (response.tools.length > 0) {
                    this.tools = response.tools;
                    this.categories = response.categories;
                } else {
                    this.tools = DEFAULT_TOOLS;
                    this.categories = [...new Set(DEFAULT_TOOLS.map(t => t.category))];
                }
            } else {
                this.tools = DEFAULT_TOOLS;
                this.categories = [...new Set(DEFAULT_TOOLS.map(t => t.category))];
            }
        } catch (error) {
            console.error('Failed to load tools:', error);
            this.lastError = error instanceof Error ? error.message : 'Unknown error';
            this.tools = DEFAULT_TOOLS;
            this.categories = [...new Set(DEFAULT_TOOLS.map(t => t.category))];
        } finally {
            this.isLoading = false;
            this._onDidChangeTreeData.fire(undefined);
        }
    }

    getTreeItem(element: ToolItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: ToolItem): ToolItem[] {
        if (!element) {
            // Root level - show categories with tool counts
            const categoryGroups = this.groupToolsByCategory();

            // Add summary item at the top
            const items: ToolItem[] = [
                new ToolItem(
                    `${this.tools.length} tools available`,
                    undefined,
                    vscode.TreeItemCollapsibleState.None,
                    'summary'
                ),
            ];

            // Add error item if there was an error
            if (this.lastError) {
                items.push(new ToolItem(
                    `Error: ${this.lastError}`,
                    undefined,
                    vscode.TreeItemCollapsibleState.None,
                    'error'
                ));
            }

            // Add category items
            for (const category of Object.keys(categoryGroups).sort()) {
                const count = categoryGroups[category].length;
                items.push(new ToolItem(
                    category,
                    undefined,
                    vscode.TreeItemCollapsibleState.Collapsed,
                    'category',
                    count
                ));
            }

            return items;
        }

        if (element.itemType === 'category') {
            // Show tools in category
            const categoryTools = this.tools.filter(t => t.category === element.label);
            return categoryTools.map(tool => new ToolItem(
                tool.name,
                tool,
                vscode.TreeItemCollapsibleState.None,
                'tool'
            ));
        }

        return [];
    }

    private groupToolsByCategory(): Record<string, ToolInfo[]> {
        const groups: Record<string, ToolInfo[]> = {};
        for (const tool of this.tools) {
            const category = tool.category || 'Other';
            if (!groups[category]) {
                groups[category] = [];
            }
            groups[category].push(tool);
        }
        return groups;
    }

    refresh(): void {
        this.loadTools();
    }
}

class ToolItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly tool: ToolInfo | undefined,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly itemType: 'category' | 'tool' | 'summary' | 'error',
        public readonly toolCount?: number
    ) {
        super(label, collapsibleState);

        switch (itemType) {
            case 'summary':
                this.iconPath = new vscode.ThemeIcon('tools');
                this.contextValue = 'summary';
                break;

            case 'error':
                this.iconPath = new vscode.ThemeIcon('warning', new vscode.ThemeColor('errorForeground'));
                this.contextValue = 'error';
                break;

            case 'category':
                this.iconPath = new vscode.ThemeIcon('folder');
                this.contextValue = 'category';
                this.description = `${toolCount} tools`;
                break;

            case 'tool':
                if (tool) {
                    this.description = tool.description;
                    this.tooltip = this._buildTooltip(tool);
                    this.iconPath = this._getIcon(tool);
                    this.contextValue = tool.is_dangerous ? 'dangerousTool' : 'tool';

                    // Command to show tool details
                    this.command = {
                        command: 'victor.showToolInfo',
                        title: 'Show Tool Info',
                        arguments: [tool]
                    };
                }
                break;
        }
    }

    private _buildTooltip(tool: ToolInfo): vscode.MarkdownString {
        const md = new vscode.MarkdownString();
        md.appendMarkdown(`### ${tool.name}\n\n`);
        md.appendMarkdown(`${tool.description}\n\n`);
        md.appendMarkdown(`**Category:** ${tool.category}\n\n`);
        md.appendMarkdown(`**Cost Tier:** ${this._formatCostTier(tool.cost_tier)}\n\n`);

        if (tool.is_dangerous) {
            md.appendMarkdown(`⚠️ **Dangerous:** Requires approval\n\n`);
        }

        if (tool.parameters && Object.keys(tool.parameters).length > 0) {
            md.appendMarkdown(`**Parameters:**\n`);
            for (const [key, value] of Object.entries(tool.parameters)) {
                md.appendMarkdown(`- \`${key}\`: ${JSON.stringify(value)}\n`);
            }
        }

        return md;
    }

    private _formatCostTier(tier: string): string {
        const badges: Record<string, string> = {
            'free': '🟢 Free',
            'low': '🔵 Low',
            'medium': '🟡 Medium',
            'high': '🔴 High',
        };
        return badges[tier.toLowerCase()] || tier;
    }

    private _getIcon(tool: ToolInfo): vscode.ThemeIcon {
        // Icon based on danger level first
        if (tool.is_dangerous) {
            return new vscode.ThemeIcon('warning', new vscode.ThemeColor('editorWarning.foreground'));
        }

        // Icon based on category
        const categoryIcons: Record<string, string> = {
            'Filesystem': 'file',
            'Search': 'search',
            'Git': 'git-branch',
            'Shell': 'terminal',
            'Analysis': 'graph',
            'Web': 'globe',
            'Docker': 'package',
            'Testing': 'beaker',
            'Refactor': 'wand',
            'Code Intelligence': 'symbol-method',
            'Database': 'database',
            'Documentation': 'book',
            'Infrastructure': 'server',
            'Batch': 'layers',
            'Cache': 'archive',
            'Mcp': 'plug',
            'Workflow': 'workflow',
            'Patch': 'diff',
        };

        const iconName = categoryIcons[tool.category] || 'tools';
        return new vscode.ThemeIcon(iconName);
    }
}

/**
 * Register tools view commands
 */
export function registerToolsCommands(
    context: vscode.ExtensionContext,
    provider: ToolsViewProvider
): void {
    // Show tool info in modal dialog
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showToolInfo', (tool: ToolInfo) => {
            const dangerWarning = tool.is_dangerous
                ? '\n\n⚠️ **This tool can make changes and requires approval.**'
                : '';

            const params = Object.keys(tool.parameters || {}).length > 0
                ? '\n\nParameters:\n' + Object.entries(tool.parameters)
                    .map(([k, v]) => `• ${k}: ${JSON.stringify(v)}`)
                    .join('\n')
                : '';

            const message = `**${tool.name}**\n\n${tool.description}\n\nCategory: ${tool.category}\nCost Tier: ${tool.cost_tier}${dangerWarning}${params}`;

            vscode.window.showInformationMessage(message, { modal: true });
        })
    );

    // Refresh tools (fetches from server)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshTools', () => {
            provider.refresh();
            vscode.window.showInformationMessage('Refreshing tools from server...');
        })
    );

    // Filter tools by category
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.filterToolsByCategory', async () => {
            const providers = getProviders();
            if (!providers?.victorClient) {
                vscode.window.showWarningMessage('Victor client not available');
                return;
            }

            const response = await providers.victorClient.getTools();
            const categories = response.categories;

            const selected = await vscode.window.showQuickPick(
                ['All', ...categories],
                { placeHolder: 'Filter by category' }
            );

            if (selected) {
                // For now, just show a message - could implement actual filtering
                vscode.window.showInformationMessage(`Selected category: ${selected}`);
            }
        })
    );
}
