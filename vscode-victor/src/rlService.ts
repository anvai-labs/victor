/**
 * RL Model Selector Service
 *
 * Manages reinforcement learning-based model selection in VS Code.
 * Features:
 * - View RL statistics and Q-values
 * - Get model recommendations (with optional task type)
 * - Configure exploration rate and strategy
 * - Reset learning state
 */

import * as vscode from 'vscode';
import { getProviders } from './providerAccessor';

/**
 * RL Statistics interface
 */
export interface RLStats {
    total_selections: number;
    epsilon: number;
    strategy: string;
    q_table: Record<string, number>;
    selection_counts: Record<string, number>;
    q_table_by_task?: Record<string, Record<string, number>>;
}

/**
 * RL Recommendation interface
 */
export interface RLRecommendation {
    recommended: string;
    strategy: string;
    exploration_rate: number;
    q_values: Record<string, number>;
    available_providers: string[];
}

/**
 * Task types for RL recommendations
 */
export const TASK_TYPES = [
    { label: 'Simple', value: 'simple', description: 'Quick questions, lookups' },
    { label: 'Complex', value: 'complex', description: 'Multi-step reasoning' },
    { label: 'Action', value: 'action', description: 'File edits, code changes' },
    { label: 'Generation', value: 'generation', description: 'Code generation' },
    { label: 'Analysis', value: 'analysis', description: 'Code review, explanation' },
];

/**
 * Selection strategies
 */
export const STRATEGIES = [
    { label: 'Epsilon-Greedy', value: 'epsilon_greedy', description: 'Balance exploration/exploitation' },
    { label: 'UCB (Upper Confidence Bound)', value: 'ucb', description: 'Explore less-tried providers' },
    { label: 'Greedy', value: 'greedy', description: 'Always pick best known provider' },
    { label: 'Random', value: 'random', description: 'Pure exploration' },
];

/**
 * Service for managing RL model selection
 */
export class RLService implements vscode.Disposable {
    private readonly _onStatsChanged = new vscode.EventEmitter<RLStats>();
    readonly onStatsChanged = this._onStatsChanged.event;

    private readonly disposables: vscode.Disposable[] = [];
    private statusBarItem?: vscode.StatusBarItem;

    constructor() {
        // Create status bar item for RL status
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            98
        );
        this.statusBarItem.command = 'victor.rlStats';
        this.statusBarItem.tooltip = 'Victor RL Model Selector';
        this.disposables.push(this.statusBarItem);
    }

    /**
     * Update status bar with RL stats
     */
    async updateStatusBar(): Promise<void> {
        const providers = getProviders();
        if (!providers?.victorClient || !this.statusBarItem) {
            if (this.statusBarItem) {
                this.statusBarItem.hide();
            }
            return;
        }

        try {
            const stats = await providers.victorClient.getRLStats();
            const topProvider = this.getTopProvider(stats.q_table);

            this.statusBarItem.text = `$(lightbulb) RL: ${topProvider || 'learning'}`;
            this.statusBarItem.tooltip = `Victor RL - ${stats.total_selections} selections, ε=${stats.epsilon.toFixed(2)}`;
            this.statusBarItem.show();

            this._onStatsChanged.fire(stats);
        } catch {
            this.statusBarItem.hide();
        }
    }

    /**
     * Get the provider with highest Q-value
     */
    private getTopProvider(qTable: Record<string, number>): string | null {
        let maxQ = -Infinity;
        let topProvider: string | null = null;

        for (const [provider, q] of Object.entries(qTable)) {
            if (q > maxQ) {
                maxQ = q;
                topProvider = provider;
            }
        }

        return topProvider;
    }

    /**
     * Get RL statistics
     */
    async getStats(): Promise<RLStats | null> {
        const providers = getProviders();
        if (!providers?.victorClient) {
            return null;
        }
        return providers.victorClient.getRLStats();
    }

    /**
     * Get model recommendation
     */
    async getRecommendation(taskType?: string): Promise<RLRecommendation | null> {
        const providers = getProviders();
        if (!providers?.victorClient) {
            return null;
        }
        return providers.victorClient.getRLRecommendation(taskType);
    }

    /**
     * Set exploration rate
     */
    async setExplorationRate(rate: number): Promise<boolean> {
        const providers = getProviders();
        if (!providers?.victorClient) {
            return false;
        }

        try {
            await providers.victorClient.setRLExplorationRate(rate);
            await this.updateStatusBar();
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Set selection strategy
     */
    async setStrategy(strategy: string): Promise<boolean> {
        const providers = getProviders();
        if (!providers?.victorClient) {
            return false;
        }

        try {
            await providers.victorClient.setRLStrategy(strategy);
            await this.updateStatusBar();
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Reset Q-values
     */
    async resetQValues(): Promise<boolean> {
        const providers = getProviders();
        if (!providers?.victorClient) {
            return false;
        }

        try {
            await providers.victorClient.resetRLQValues();
            await this.updateStatusBar();
            return true;
        } catch {
            return false;
        }
    }

    dispose(): void {
        this._onStatsChanged.dispose();
        this.disposables.forEach(d => d.dispose());
    }
}

/**
 * Register RL-related commands
 */
export function registerRLCommands(
    context: vscode.ExtensionContext,
    service: RLService
): void {
    // View RL Statistics
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rlStats', async () => {
            const stats = await service.getStats();
            if (!stats) {
                vscode.window.showWarningMessage('Victor server not available');
                return;
            }

            // Format Q-table for display
            const qTableItems = Object.entries(stats.q_table)
                .sort(([, a], [, b]) => b - a)
                .map(([provider, q]) => {
                    const count = stats.selection_counts[provider] || 0;
                    return `${provider}: Q=${q.toFixed(3)} (${count} selections)`;
                });

            // Show stats in quick pick
            const items = [
                {
                    label: '$(graph) Overview',
                    description: `${stats.total_selections} total selections`,
                    detail: `Strategy: ${stats.strategy}, ε=${stats.epsilon.toFixed(2)}`,
                    action: 'overview',
                },
                {
                    label: '$(list-ordered) Q-Values by Provider',
                    description: `${Object.keys(stats.q_table).length} providers`,
                    detail: qTableItems.slice(0, 3).join(' | ') || 'No data yet',
                    action: 'q_table',
                },
                {
                    label: '$(lightbulb) Get Recommendation',
                    description: 'Get best provider for a task',
                    action: 'recommend',
                },
                {
                    label: '$(settings-gear) Configure',
                    description: 'Change exploration rate or strategy',
                    action: 'configure',
                },
                {
                    label: '$(trash) Reset Learning',
                    description: 'Clear all Q-values and start fresh',
                    action: 'reset',
                },
            ];

            const selected = await vscode.window.showQuickPick(items, {
                placeHolder: 'Victor RL Model Selector',
            });

            if (!selected) {
                return;
            }

            switch (selected.action) {
                case 'overview':
                    await showRLOverview(stats);
                    break;
                case 'q_table':
                    await showQTable(stats);
                    break;
                case 'recommend':
                    await vscode.commands.executeCommand('victor.rlRecommend');
                    break;
                case 'configure':
                    await vscode.commands.executeCommand('victor.rlConfigure');
                    break;
                case 'reset':
                    await vscode.commands.executeCommand('victor.rlReset');
                    break;
            }
        })
    );

    // Get RL Recommendation
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rlRecommend', async () => {
            // Ask for task type
            const taskItems = [
                { label: '$(question) Any Task', description: 'General recommendation', value: undefined },
                ...TASK_TYPES.map(t => ({
                    label: `$(tag) ${t.label}`,
                    description: t.description,
                    value: t.value,
                })),
            ];

            const taskSelected = await vscode.window.showQuickPick(taskItems, {
                placeHolder: 'What type of task?',
            });

            if (!taskSelected) {
                return;
            }

            const recommendation = await service.getRecommendation(taskSelected.value);
            if (!recommendation) {
                vscode.window.showWarningMessage('Victor server not available');
                return;
            }

            // Show recommendation
            const taskInfo = taskSelected.value ? ` for ${taskSelected.label.replace('$(tag) ', '')} tasks` : '';
            const message = `Recommended: **${recommendation.recommended}**${taskInfo}`;

            const action = await vscode.window.showInformationMessage(
                message,
                'Use This Provider',
                'View Q-Values',
                'Dismiss'
            );

            if (action === 'Use This Provider') {
                // Switch to recommended provider
                const providers = getProviders();
                if (providers?.victorClient) {
                    try {
                        await providers.victorClient.switchModel(recommendation.recommended, 'default');
                        vscode.window.showInformationMessage(`Switched to ${recommendation.recommended}`);
                    } catch (error) {
                        vscode.window.showErrorMessage(`Failed to switch: ${error}`);
                    }
                }
            } else if (action === 'View Q-Values') {
                const qItems = Object.entries(recommendation.q_values)
                    .sort(([, a], [, b]) => b - a)
                    .map(([provider, q]) => ({
                        label: provider === recommendation.recommended ? `$(star) ${provider}` : provider,
                        description: `Q=${q.toFixed(3)}`,
                    }));

                await vscode.window.showQuickPick(qItems, {
                    placeHolder: 'Q-Values (higher = better)',
                });
            }
        })
    );

    // Configure RL
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rlConfigure', async () => {
            const stats = await service.getStats();

            const configItems = [
                {
                    label: '$(symbol-number) Exploration Rate',
                    description: `Current: ${stats?.epsilon.toFixed(2) || '0.10'}`,
                    detail: 'Higher = more exploration, lower = more exploitation',
                    action: 'epsilon',
                },
                {
                    label: '$(symbol-method) Strategy',
                    description: `Current: ${stats?.strategy || 'epsilon_greedy'}`,
                    detail: 'How to balance exploration vs exploitation',
                    action: 'strategy',
                },
            ];

            const selected = await vscode.window.showQuickPick(configItems, {
                placeHolder: 'Configure RL Model Selector',
            });

            if (!selected) {
                return;
            }

            if (selected.action === 'epsilon') {
                const rateStr = await vscode.window.showInputBox({
                    prompt: 'Enter exploration rate (0.0 to 1.0)',
                    value: stats?.epsilon.toString() || '0.1',
                    validateInput: (value) => {
                        const num = parseFloat(value);
                        if (isNaN(num) || num < 0 || num > 1) {
                            return 'Must be a number between 0.0 and 1.0';
                        }
                        return null;
                    },
                });

                if (rateStr) {
                    const success = await service.setExplorationRate(parseFloat(rateStr));
                    if (success) {
                        vscode.window.showInformationMessage(`Exploration rate set to ${rateStr}`);
                    } else {
                        vscode.window.showErrorMessage('Failed to set exploration rate');
                    }
                }
            } else if (selected.action === 'strategy') {
                const strategyItems = STRATEGIES.map(s => ({
                    label: s.value === stats?.strategy ? `$(check) ${s.label}` : s.label,
                    description: s.description,
                    value: s.value,
                }));

                const strategySelected = await vscode.window.showQuickPick(strategyItems, {
                    placeHolder: 'Select strategy',
                });

                if (strategySelected) {
                    const success = await service.setStrategy(strategySelected.value);
                    if (success) {
                        vscode.window.showInformationMessage(`Strategy set to ${strategySelected.label}`);
                    } else {
                        vscode.window.showErrorMessage('Failed to set strategy');
                    }
                }
            }
        })
    );

    // Reset RL
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rlReset', async () => {
            const confirm = await vscode.window.showWarningMessage(
                'This will reset all learned Q-values. The model selector will start learning from scratch.',
                { modal: true },
                'Reset',
                'Cancel'
            );

            if (confirm === 'Reset') {
                const success = await service.resetQValues();
                if (success) {
                    vscode.window.showInformationMessage('RL Q-values have been reset');
                } else {
                    vscode.window.showErrorMessage('Failed to reset Q-values');
                }
            }
        })
    );

    // Initialize status bar
    service.updateStatusBar().catch(console.error);

    // Periodic status bar updates
    const updateInterval = setInterval(() => {
        service.updateStatusBar().catch(console.error);
    }, 60000); // Update every minute

    context.subscriptions.push({
        dispose: () => clearInterval(updateInterval),
    });
}

/**
 * Show RL overview in output channel
 */
async function showRLOverview(stats: RLStats): Promise<void> {
    const output = vscode.window.createOutputChannel('Victor RL Stats');
    output.clear();
    output.appendLine('═══════════════════════════════════════════════════════════');
    output.appendLine('                   Victor RL Model Selector');
    output.appendLine('═══════════════════════════════════════════════════════════');
    output.appendLine('');
    output.appendLine(`Strategy:          ${stats.strategy}`);
    output.appendLine(`Exploration Rate:  ${stats.epsilon.toFixed(3)}`);
    output.appendLine(`Total Selections:  ${stats.total_selections}`);
    output.appendLine('');
    output.appendLine('───────────────────────────────────────────────────────────');
    output.appendLine('                     Global Q-Values');
    output.appendLine('───────────────────────────────────────────────────────────');

    const sorted = Object.entries(stats.q_table).sort(([, a], [, b]) => b - a);
    for (const [provider, q] of sorted) {
        const count = stats.selection_counts[provider] || 0;
        const bar = '█'.repeat(Math.max(0, Math.round(q * 20)));
        output.appendLine(`${provider.padEnd(15)} ${q.toFixed(3).padStart(7)} ${bar} (${count})`);
    }

    if (stats.q_table_by_task && Object.keys(stats.q_table_by_task).length > 0) {
        output.appendLine('');
        output.appendLine('───────────────────────────────────────────────────────────');
        output.appendLine('                   Q-Values by Task Type');
        output.appendLine('───────────────────────────────────────────────────────────');

        for (const [taskType, qTable] of Object.entries(stats.q_table_by_task)) {
            output.appendLine(`\n  ${taskType.toUpperCase()}:`);
            const taskSorted = Object.entries(qTable).sort(([, a], [, b]) => b - a);
            for (const [provider, q] of taskSorted) {
                output.appendLine(`    ${provider.padEnd(15)} ${q.toFixed(3)}`);
            }
        }
    }

    output.appendLine('');
    output.appendLine('═══════════════════════════════════════════════════════════');
    output.show();
}

/**
 * Show Q-table in quick pick
 */
async function showQTable(stats: RLStats): Promise<void> {
    const items = Object.entries(stats.q_table)
        .sort(([, a], [, b]) => b - a)
        .map(([provider, q], index) => {
            const count = stats.selection_counts[provider] || 0;
            const icon = index === 0 ? '$(star)' : '$(circle-outline)';
            return {
                label: `${icon} ${provider}`,
                description: `Q=${q.toFixed(3)}`,
                detail: `${count} selections`,
            };
        });

    if (items.length === 0) {
        vscode.window.showInformationMessage('No Q-values yet. Use Victor more to build learning data.');
        return;
    }

    await vscode.window.showQuickPick(items, {
        placeHolder: 'Q-Values by Provider (higher = better performance)',
    });
}
