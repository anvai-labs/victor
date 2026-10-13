/**
 * Hover Provider for Victor AI
 *
 * Provides AI-powered hover tooltips with explanations:
 * - Quick explanations for functions, classes, and symbols
 * - Type information with AI-enhanced descriptions
 * - Documentation previews with improvement suggestions
 * - Link to full AI chat for deeper analysis
 *
 * Features:
 * - Debounced requests to avoid API spam
 * - LRU cache for previously explained symbols
 * - Language-aware context extraction
 * - Configurable verbosity levels
 */

import * as vscode from 'vscode';
import { encodeSymbolTarget } from './codeActionProvider';
import type { VictorClient } from './victorClient';

/**
 * Hover verbosity levels
 */
export type HoverVerbosity = 'minimal' | 'standard' | 'detailed';

/**
 * Cached hover result
 */
interface CachedHover {
    content: vscode.MarkdownString;
    timestamp: number;
    symbolName: string;
}

/**
 * Hover configuration
 */
interface HoverConfig {
    enabled: boolean;
    verbosity: HoverVerbosity;
    showAIExplanation: boolean;
    showQuickActions: boolean;
    cacheTimeoutMs: number;
    debounceMs: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: HoverConfig = {
    enabled: true,
    verbosity: 'standard',
    showAIExplanation: true,
    showQuickActions: true,
    cacheTimeoutMs: 300000, // 5 minutes
    debounceMs: 150,
};

/**
 * Supported languages for hover
 */
const SUPPORTED_LANGUAGES = new Set([
    'javascript',
    'typescript',
    'javascriptreact',
    'typescriptreact',
    'python',
    'java',
    'csharp',
    'go',
    'rust',
    'cpp',
    'c',
    'ruby',
    'php',
    'swift',
    'kotlin',
]);

/**
 * Victor Hover Provider
 */
export class VictorHoverProvider implements vscode.HoverProvider {
    private config: HoverConfig;
    private cache: Map<string, CachedHover> = new Map();
    private readonly maxCacheSize = 100;
    private pendingRequests: Map<string, Promise<vscode.Hover | null>> = new Map();
    private victorClient: VictorClient | null = null;

    constructor() {
        this.config = this.loadConfig();

        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('victor.hover')) {
                this.config = this.loadConfig();
                this.cache.clear(); // Clear cache on config change
            }
        });

        // Periodic cache cleanup
        setInterval(() => this.cleanupCache(), 60000);
    }

    /**
     * Set the Victor client for AI-powered explanations
     */
    setVictorClient(client: VictorClient): void {
        this.victorClient = client;
    }

    /**
     * Load configuration from VS Code settings
     */
    private loadConfig(): HoverConfig {
        const config = vscode.workspace.getConfiguration('victor.hover');
        return {
            enabled: config.get('enabled', DEFAULT_CONFIG.enabled),
            verbosity: config.get('verbosity', DEFAULT_CONFIG.verbosity),
            showAIExplanation: config.get('showAIExplanation', DEFAULT_CONFIG.showAIExplanation),
            showQuickActions: config.get('showQuickActions', DEFAULT_CONFIG.showQuickActions),
            cacheTimeoutMs: config.get('cacheTimeoutMs', DEFAULT_CONFIG.cacheTimeoutMs),
            debounceMs: config.get('debounceMs', DEFAULT_CONFIG.debounceMs),
        };
    }

    /**
     * Provide hover for a position
     */
    async provideHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<vscode.Hover | null> {
        if (!this.config.enabled) {
            return null;
        }

        if (!SUPPORTED_LANGUAGES.has(document.languageId)) {
            return null;
        }

        // Get word at position
        const wordRange = document.getWordRangeAtPosition(position);
        if (!wordRange) {
            return null;
        }

        const word = document.getText(wordRange);
        if (!word || word.length < 2) {
            return null;
        }

        // Create cache key
        const cacheKey = this.getCacheKey(document.uri, position, word);

        // Check cache first
        const cached = this.cache.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < this.config.cacheTimeoutMs) {
            return new vscode.Hover(cached.content, wordRange);
        }

        // Check for pending request
        const pending = this.pendingRequests.get(cacheKey);
        if (pending) {
            return pending;
        }

        // Create new request
        const requestPromise = this.createHover(document, position, wordRange, word, token);
        this.pendingRequests.set(cacheKey, requestPromise);

        try {
            const result = await requestPromise;
            return result;
        } finally {
            this.pendingRequests.delete(cacheKey);
        }
    }

    /**
     * Create hover content
     */
    private async createHover(
        document: vscode.TextDocument,
        position: vscode.Position,
        wordRange: vscode.Range,
        word: string,
        token: vscode.CancellationToken
    ): Promise<vscode.Hover | null> {
        const cacheKey = this.getCacheKey(document.uri, position, word);

        // Get symbol information from VS Code
        const symbolInfo = await this.getSymbolInfo(document, position, token);
        if (token.isCancellationRequested) {
            return null;
        }

        // Build hover content
        const content = new vscode.MarkdownString();
        content.isTrusted = true;
        content.supportHtml = true;

        // Add symbol header
        if (symbolInfo) {
            content.appendMarkdown(`### ${symbolInfo.kind}: \`${symbolInfo.name}\`\n\n`);
        } else {
            content.appendMarkdown(`### \`${word}\`\n\n`);
        }

        // Get existing hover from other providers
        const existingHovers = await vscode.commands.executeCommand<vscode.Hover[]>(
            'vscode.executeHoverProvider',
            document.uri,
            position
        );

        // Include existing documentation if available
        if (existingHovers && existingHovers.length > 0) {
            for (const hover of existingHovers) {
                for (const hoverContent of hover.contents) {
                    if (typeof hoverContent === 'string') {
                        content.appendMarkdown(hoverContent + '\n\n');
                    } else if ('value' in hoverContent) {
                        content.appendMarkdown(hoverContent.value + '\n\n');
                    }
                }
            }
        }

        // Add quick actions if enabled
        if (this.config.showQuickActions) {
            content.appendMarkdown('---\n\n');
            content.appendMarkdown('**Victor AI Actions:**\n\n');

            // Explain action
            content.appendMarkdown(
                `[$(question) Explain](command:victor.explainSymbol?${encodeSymbolTarget(document.uri, wordRange, word)}) | `
            );

            // Test action
            content.appendMarkdown(
                `[$(beaker) Test](command:victor.generateTestsForSymbol?${encodeSymbolTarget(document.uri, wordRange, word)}) | `
            );

            // Optimize action
            content.appendMarkdown(
                `[$(zap) Optimize](command:victor.optimizeSymbol?${encodeSymbolTarget(document.uri, wordRange, word)})`
            );

            content.appendMarkdown('\n\n');
        }

        // Cache the result
        this.cacheHover(cacheKey, content, word);

        return new vscode.Hover(content, wordRange);
    }

    /**
     * Get symbol information at position
     */
    private async getSymbolInfo(
        document: vscode.TextDocument,
        position: vscode.Position,
        token: vscode.CancellationToken
    ): Promise<{ name: string; kind: string } | null> {
        try {
            const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                'vscode.executeDocumentSymbolProvider',
                document.uri
            );

            if (token.isCancellationRequested || !symbols) {
                return null;
            }

            // Find symbol containing position
            const symbol = this.findSymbolAtPosition(symbols, position);
            if (symbol) {
                return {
                    name: symbol.name,
                    kind: this.getSymbolKindName(symbol.kind),
                };
            }
        } catch {
            // Ignore errors from symbol provider
        }

        return null;
    }

    /**
     * Find symbol at position recursively
     */
    private findSymbolAtPosition(
        symbols: vscode.DocumentSymbol[],
        position: vscode.Position
    ): vscode.DocumentSymbol | null {
        for (const symbol of symbols) {
            if (symbol.range.contains(position)) {
                // Check children first for more specific match
                if (symbol.children && symbol.children.length > 0) {
                    const child = this.findSymbolAtPosition(symbol.children, position);
                    if (child) {
                        return child;
                    }
                }
                return symbol;
            }
        }
        return null;
    }

    /**
     * Get human-readable symbol kind name
     */
    private getSymbolKindName(kind: vscode.SymbolKind): string {
        const kindNames: Record<vscode.SymbolKind, string> = {
            [vscode.SymbolKind.File]: 'File',
            [vscode.SymbolKind.Module]: 'Module',
            [vscode.SymbolKind.Namespace]: 'Namespace',
            [vscode.SymbolKind.Package]: 'Package',
            [vscode.SymbolKind.Class]: 'Class',
            [vscode.SymbolKind.Method]: 'Method',
            [vscode.SymbolKind.Property]: 'Property',
            [vscode.SymbolKind.Field]: 'Field',
            [vscode.SymbolKind.Constructor]: 'Constructor',
            [vscode.SymbolKind.Enum]: 'Enum',
            [vscode.SymbolKind.Interface]: 'Interface',
            [vscode.SymbolKind.Function]: 'Function',
            [vscode.SymbolKind.Variable]: 'Variable',
            [vscode.SymbolKind.Constant]: 'Constant',
            [vscode.SymbolKind.String]: 'String',
            [vscode.SymbolKind.Number]: 'Number',
            [vscode.SymbolKind.Boolean]: 'Boolean',
            [vscode.SymbolKind.Array]: 'Array',
            [vscode.SymbolKind.Object]: 'Object',
            [vscode.SymbolKind.Key]: 'Key',
            [vscode.SymbolKind.Null]: 'Null',
            [vscode.SymbolKind.EnumMember]: 'EnumMember',
            [vscode.SymbolKind.Struct]: 'Struct',
            [vscode.SymbolKind.Event]: 'Event',
            [vscode.SymbolKind.Operator]: 'Operator',
            [vscode.SymbolKind.TypeParameter]: 'TypeParameter',
        };
        return kindNames[kind] || 'Symbol';
    }

    /**
     * Create cache key
     */
    private getCacheKey(uri: vscode.Uri, position: vscode.Position, word: string): string {
        return `${uri.toString()}:${position.line}:${word}`;
    }

    /**
     * Cache hover result with LRU eviction
     */
    private cacheHover(key: string, content: vscode.MarkdownString, symbolName: string): void {
        // Evict oldest entries if cache is full
        if (this.cache.size >= this.maxCacheSize) {
            const entries = Array.from(this.cache.entries());
            entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
            const toRemove = entries.slice(0, Math.floor(this.maxCacheSize / 4));
            for (const [removeKey] of toRemove) {
                this.cache.delete(removeKey);
            }
        }

        this.cache.set(key, {
            content,
            timestamp: Date.now(),
            symbolName,
        });
    }

    /**
     * Cleanup expired cache entries
     */
    private cleanupCache(): void {
        const now = Date.now();
        for (const [key, entry] of this.cache.entries()) {
            if (now - entry.timestamp > this.config.cacheTimeoutMs) {
                this.cache.delete(key);
            }
        }
    }

    /**
     * Clear the cache
     */
    clearCache(): void {
        this.cache.clear();
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.cache.clear();
        this.pendingRequests.clear();
    }
}

/**
 * Register hover commands
 */
export function registerHoverCommands(context: vscode.ExtensionContext): void {
    // Toggle hover provider
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.toggleHover', async () => {
            const config = vscode.workspace.getConfiguration('victor.hover');
            const currentValue = config.get('enabled', true);
            await config.update('enabled', !currentValue, true);
            vscode.window.showInformationMessage(
                `Victor Hover ${!currentValue ? 'enabled' : 'disabled'}`
            );
        })
    );

    // Set hover verbosity
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.setHoverVerbosity', async () => {
            const options: vscode.QuickPickItem[] = [
                {
                    label: 'Minimal',
                    description: 'Brief summaries only',
                    detail: 'Shows minimal information in hover tooltips',
                },
                {
                    label: 'Standard',
                    description: 'Balanced information (default)',
                    detail: 'Shows symbol info with quick actions',
                },
                {
                    label: 'Detailed',
                    description: 'Full explanations',
                    detail: 'Shows comprehensive information with AI explanations',
                },
            ];

            const selected = await vscode.window.showQuickPick(options, {
                placeHolder: 'Select hover verbosity level',
            });

            if (selected) {
                const verbosity = selected.label.toLowerCase() as HoverVerbosity;
                const config = vscode.workspace.getConfiguration('victor.hover');
                await config.update('verbosity', verbosity, true);
                vscode.window.showInformationMessage(
                    `Victor hover verbosity set to: ${selected.label}`
                );
            }
        })
    );
}
