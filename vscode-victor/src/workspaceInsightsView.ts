/**
 * Workspace Insights View Provider
 *
 * Provides a tree view of workspace analysis including:
 * - Project structure overview
 * - Code metrics
 * - Security scan results
 * - Dependency information
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

interface WorkspaceMetrics {
    lines_of_code: number;
    files_by_type: Record<string, number>;
    largest_files: { path: string; lines: number; size: number }[];
}

interface SecurityFinding {
    file: string;
    line: number;
    type: string;
    severity: string;
    snippet: string;
}

interface DependencyInfo {
    file: string;
    count?: number;
    packages?: string[];
}

type InsightItemType = 'category' | 'metric' | 'file' | 'finding' | 'dependency' | 'loading' | 'error';

export class WorkspaceInsightsViewProvider implements vscode.TreeDataProvider<InsightItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<InsightItem | undefined>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

    private isLoading = false;
    private metrics: WorkspaceMetrics | null = null;
    private securityFindings: SecurityFinding[] = [];
    private dependencies: Record<string, DependencyInfo> = {};
    private lastError: string | null = null;

    constructor() {
        // Load data on initialization
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

            // Load all data in parallel
            const [metricsResult, securityResult, dependenciesResult] = await Promise.all([
                providers.victorClient.getWorkspaceMetrics(),
                providers.victorClient.getWorkspaceSecurity(),
                providers.victorClient.getWorkspaceDependencies(),
            ]);

            this.metrics = metricsResult;
            this.securityFindings = securityResult.findings || [];
            this.dependencies = dependenciesResult.dependencies || {};

        } catch (error) {
            console.error('Failed to load workspace insights:', error);
            this.lastError = error instanceof Error ? error.message : 'Unknown error';
        } finally {
            this.isLoading = false;
            this._onDidChangeTreeData.fire(undefined);
        }
    }

    getTreeItem(element: InsightItem): vscode.TreeItem {
        return element;
    }

    getChildren(element?: InsightItem): InsightItem[] {
        if (!element) {
            // Root level - show main categories
            if (this.isLoading) {
                return [new InsightItem('Loading workspace insights...', 'loading', vscode.TreeItemCollapsibleState.None)];
            }

            if (this.lastError) {
                return [new InsightItem(`Error: ${this.lastError}`, 'error', vscode.TreeItemCollapsibleState.None)];
            }

            const items: InsightItem[] = [];

            // Metrics category
            items.push(new InsightItem(
                'Code Metrics',
                'category',
                vscode.TreeItemCollapsibleState.Expanded,
                undefined,
                'graph'
            ));

            // Security category
            const securityLabel = this.securityFindings.length > 0
                ? `Security (${this.securityFindings.length} findings)`
                : 'Security (clean)';
            items.push(new InsightItem(
                securityLabel,
                'category',
                vscode.TreeItemCollapsibleState.Collapsed,
                undefined,
                this.securityFindings.length > 0 ? 'warning' : 'shield'
            ));

            // Dependencies category
            const depCount = Object.keys(this.dependencies).length;
            items.push(new InsightItem(
                `Dependencies (${depCount} ecosystems)`,
                'category',
                vscode.TreeItemCollapsibleState.Collapsed,
                undefined,
                'package'
            ));

            // Largest files category
            if (this.metrics?.largest_files?.length) {
                items.push(new InsightItem(
                    'Largest Files',
                    'category',
                    vscode.TreeItemCollapsibleState.Collapsed,
                    undefined,
                    'file-code'
                ));
            }

            return items;
        }

        // Children based on category
        if (element.label === 'Code Metrics') {
            return this.getMetricsChildren();
        }

        if (element.label?.toString().startsWith('Security')) {
            return this.getSecurityChildren();
        }

        if (element.label?.toString().startsWith('Dependencies')) {
            return this.getDependenciesChildren();
        }

        if (element.label === 'Largest Files') {
            return this.getLargestFilesChildren();
        }

        // Children for dependency ecosystem
        if (element.itemType === 'dependency' && element.data) {
            const info = element.data as DependencyInfo;
            if (info.packages) {
                return info.packages.slice(0, 10).map(pkg =>
                    new InsightItem(pkg, 'dependency', vscode.TreeItemCollapsibleState.None, undefined, 'symbol-package')
                );
            }
        }

        return [];
    }

    private getMetricsChildren(): InsightItem[] {
        if (!this.metrics) {
            return [new InsightItem('No metrics available', 'metric', vscode.TreeItemCollapsibleState.None)];
        }

        const items: InsightItem[] = [];

        // Total lines of code
        items.push(new InsightItem(
            `Lines of Code: ${this.formatNumber(this.metrics.lines_of_code)}`,
            'metric',
            vscode.TreeItemCollapsibleState.None,
            undefined,
            'symbol-number'
        ));

        // Files by type
        const fileTypes = Object.entries(this.metrics.files_by_type)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

        for (const [ext, count] of fileTypes) {
            items.push(new InsightItem(
                `${ext || 'no ext'}: ${count} files`,
                'metric',
                vscode.TreeItemCollapsibleState.None,
                undefined,
                'file'
            ));
        }

        return items;
    }

    private getSecurityChildren(): InsightItem[] {
        if (this.securityFindings.length === 0) {
            return [new InsightItem('No security issues found', 'finding', vscode.TreeItemCollapsibleState.None, undefined, 'check')];
        }

        return this.securityFindings.slice(0, 20).map(finding => {
            const item = new InsightItem(
                `${finding.type} in ${finding.file}:${finding.line}`,
                'finding',
                vscode.TreeItemCollapsibleState.None,
                finding,
                finding.severity === 'high' ? 'error' : 'warning'
            );
            item.description = finding.snippet;
            item.command = {
                command: 'vscode.open',
                title: 'Open File',
                arguments: [
                    vscode.Uri.file(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath + '/' + finding.file),
                    { selection: new vscode.Range(finding.line - 1, 0, finding.line - 1, 100) }
                ]
            };
            return item;
        });
    }

    private getDependenciesChildren(): InsightItem[] {
        const items: InsightItem[] = [];

        for (const [ecosystem, info] of Object.entries(this.dependencies)) {
            const label = info.count !== undefined
                ? `${ecosystem}: ${info.count} packages`
                : `${ecosystem}: ${info.file}`;

            const collapsible = info.packages && info.packages.length > 0
                ? vscode.TreeItemCollapsibleState.Collapsed
                : vscode.TreeItemCollapsibleState.None;

            items.push(new InsightItem(
                label,
                'dependency',
                collapsible,
                info,
                this.getEcosystemIcon(ecosystem)
            ));
        }

        return items;
    }

    private getLargestFilesChildren(): InsightItem[] {
        if (!this.metrics?.largest_files) {
            return [];
        }

        return this.metrics.largest_files.slice(0, 10).map(file => {
            const item = new InsightItem(
                file.path,
                'file',
                vscode.TreeItemCollapsibleState.None,
                file,
                'file-code'
            );
            item.description = `${this.formatNumber(file.lines)} lines`;
            item.command = {
                command: 'vscode.open',
                title: 'Open File',
                arguments: [vscode.Uri.file(vscode.workspace.workspaceFolders?.[0]?.uri.fsPath + '/' + file.path)]
            };
            return item;
        });
    }

    private formatNumber(num: number): string {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    private getEcosystemIcon(ecosystem: string): string {
        const icons: Record<string, string> = {
            python: 'symbol-method',
            node: 'symbol-namespace',
            rust: 'symbol-struct',
            go: 'symbol-interface',
        };
        return icons[ecosystem] || 'package';
    }
}

class InsightItem extends vscode.TreeItem {
    constructor(
        public readonly label: string,
        public readonly itemType: InsightItemType,
        public readonly collapsibleState: vscode.TreeItemCollapsibleState,
        public readonly data?: unknown,
        iconName?: string
    ) {
        super(label, collapsibleState);

        if (iconName) {
            const color = itemType === 'error' ? new vscode.ThemeColor('errorForeground')
                : itemType === 'finding' ? new vscode.ThemeColor('editorWarning.foreground')
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
            case 'category':
                this.contextValue = 'category';
                break;
        }
    }
}

/**
 * Register workspace insights commands
 */
export function registerWorkspaceInsightsCommands(
    context: vscode.ExtensionContext,
    provider: WorkspaceInsightsViewProvider
): void {
    // Refresh workspace insights
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refreshWorkspaceInsights', () => {
            provider.refresh();
            vscode.window.showInformationMessage('Refreshing workspace insights...');
        })
    );

    // Run security scan
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.runSecurityScan', async () => {
            const providers = getProviders();
            if (!providers?.victorClient) {
                vscode.window.showWarningMessage('Victor client not available');
                return;
            }

            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: 'Running security scan...',
                cancellable: false,
            }, async () => {
                await providers.victorClient.getWorkspaceSecurity();
                provider.refresh();
            });
        })
    );
}
