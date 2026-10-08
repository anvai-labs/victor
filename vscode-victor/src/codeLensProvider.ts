/**
 * CodeLens Provider for AI Suggestions
 *
 * Provides CodeLens actions above functions, classes, and other code symbols:
 * - "Ask Victor" - Open chat with context
 * - "Explain" - Get AI explanation
 * - "Generate Tests" - Create unit tests
 * - "Add Documentation" - Generate docstrings
 * - "Optimize" - Get optimization suggestions
 *
 * Features:
 * - Language-aware symbol detection
 * - Configurable action set
 * - Debounced updates for performance
 * - Caching for large files
 */

import * as vscode from 'vscode';

/**
 * CodeLens action types
 */
export type CodeLensAction =
    | 'ask'
    | 'explain'
    | 'test'
    | 'document'
    | 'optimize'
    | 'refactor'
    | 'review';

/**
 * Symbol information for CodeLens
 */
interface SymbolInfo {
    name: string;
    kind: vscode.SymbolKind;
    range: vscode.Range;
    selectionRange: vscode.Range;
    children?: SymbolInfo[];
}

/**
 * CodeLens configuration
 */
interface CodeLensConfig {
    enabled: boolean;
    actions: CodeLensAction[];
    showOnFunctions: boolean;
    showOnClasses: boolean;
    showOnMethods: boolean;
    showOnInterfaces: boolean;
    maxSymbolsPerFile: number;
    debounceMs: number;
}

/**
 * Default configuration
 */
const DEFAULT_CONFIG: CodeLensConfig = {
    enabled: true,
    actions: ['ask', 'explain', 'test'],
    showOnFunctions: true,
    showOnClasses: true,
    showOnMethods: true,
    showOnInterfaces: true,
    maxSymbolsPerFile: 100,
    debounceMs: 500,
};

/**
 * Action metadata for display
 */
const ACTION_METADATA: Record<CodeLensAction, { title: string; icon: string; command: string }> = {
    ask: { title: 'Ask Victor', icon: '$(comment-discussion)', command: 'victor.askAboutSymbol' },
    explain: { title: 'Explain', icon: '$(question)', command: 'victor.explainSymbol' },
    test: { title: 'Test', icon: '$(beaker)', command: 'victor.generateTestsForSymbol' },
    document: { title: 'Document', icon: '$(book)', command: 'victor.documentSymbol' },
    optimize: { title: 'Optimize', icon: '$(zap)', command: 'victor.optimizeSymbol' },
    refactor: { title: 'Refactor', icon: '$(edit)', command: 'victor.refactorSymbol' },
    review: { title: 'Review', icon: '$(eye)', command: 'victor.reviewSymbol' },
};

/**
 * Symbol kinds to show CodeLens on
 */
const CODELENS_SYMBOL_KINDS = new Set([
    vscode.SymbolKind.Function,
    vscode.SymbolKind.Method,
    vscode.SymbolKind.Class,
    vscode.SymbolKind.Interface,
    vscode.SymbolKind.Constructor,
    vscode.SymbolKind.Module,
    vscode.SymbolKind.Namespace,
]);

/**
 * Supported languages for CodeLens
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
 * CodeLens Provider class
 */
export class VictorCodeLensProvider implements vscode.CodeLensProvider {
    private _onDidChangeCodeLenses = new vscode.EventEmitter<void>();
    readonly onDidChangeCodeLenses = this._onDidChangeCodeLenses.event;

    private config: CodeLensConfig;
    private symbolCache: Map<string, { symbols: SymbolInfo[]; timestamp: number }> = new Map();
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
    private readonly cacheTtlMs = 30000; // 30 seconds

    constructor() {
        this.config = this.loadConfig();

        // Listen for configuration changes
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('victor.codeLens')) {
                this.config = this.loadConfig();
                this._onDidChangeCodeLenses.fire();
            }
        });

        // Clear cache on document changes
        vscode.workspace.onDidChangeTextDocument(e => {
            const key = e.document.uri.toString();
            this.symbolCache.delete(key);
            this.debouncedRefresh(key);
        });
    }

    /**
     * Load configuration from VS Code settings
     */
    private loadConfig(): CodeLensConfig {
        const config = vscode.workspace.getConfiguration('victor.codeLens');
        return {
            enabled: config.get('enabled', DEFAULT_CONFIG.enabled),
            actions: config.get('actions', DEFAULT_CONFIG.actions),
            showOnFunctions: config.get('showOnFunctions', DEFAULT_CONFIG.showOnFunctions),
            showOnClasses: config.get('showOnClasses', DEFAULT_CONFIG.showOnClasses),
            showOnMethods: config.get('showOnMethods', DEFAULT_CONFIG.showOnMethods),
            showOnInterfaces: config.get('showOnInterfaces', DEFAULT_CONFIG.showOnInterfaces),
            maxSymbolsPerFile: config.get('maxSymbolsPerFile', DEFAULT_CONFIG.maxSymbolsPerFile),
            debounceMs: config.get('debounceMs', DEFAULT_CONFIG.debounceMs),
        };
    }

    /**
     * Debounced refresh for performance
     */
    private debouncedRefresh(documentUri: string): void {
        const existing = this.debounceTimers.get(documentUri);
        if (existing) {
            clearTimeout(existing);
        }

        const timer = setTimeout(() => {
            this.debounceTimers.delete(documentUri);
            this._onDidChangeCodeLenses.fire();
        }, this.config.debounceMs);

        this.debounceTimers.set(documentUri, timer);
    }

    /**
     * Provide CodeLenses for a document
     */
    async provideCodeLenses(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<vscode.CodeLens[]> {
        if (!this.config.enabled) {
            return [];
        }

        if (!SUPPORTED_LANGUAGES.has(document.languageId)) {
            return [];
        }

        const symbols = await this.getSymbols(document, token);
        if (token.isCancellationRequested) {
            return [];
        }

        const codeLenses: vscode.CodeLens[] = [];
        const processedSymbols = this.flattenSymbols(symbols).slice(0, this.config.maxSymbolsPerFile);

        for (const symbol of processedSymbols) {
            if (!this.shouldShowCodeLens(symbol)) {
                continue;
            }

            // Create CodeLens for each configured action
            for (const action of this.config.actions) {
                const metadata = ACTION_METADATA[action];
                const codeLens = new vscode.CodeLens(symbol.range, {
                    title: `${metadata.icon} ${metadata.title}`,
                    command: metadata.command,
                    arguments: [document.uri, symbol.range, symbol.name, symbol.kind],
                });
                codeLenses.push(codeLens);
            }
        }

        return codeLenses;
    }

    /**
     * Resolve CodeLens (already resolved in provideCodeLenses)
     */
    resolveCodeLens(codeLens: vscode.CodeLens): vscode.CodeLens {
        return codeLens;
    }

    /**
     * Get symbols from document (with caching)
     */
    private async getSymbols(
        document: vscode.TextDocument,
        token: vscode.CancellationToken
    ): Promise<SymbolInfo[]> {
        const key = document.uri.toString();
        const cached = this.symbolCache.get(key);

        if (cached && Date.now() - cached.timestamp < this.cacheTtlMs) {
            return cached.symbols;
        }

        try {
            const vscodeSymbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
                'vscode.executeDocumentSymbolProvider',
                document.uri
            );

            if (token.isCancellationRequested) {
                return [];
            }

            const symbols = this.convertSymbols(vscodeSymbols || []);
            this.symbolCache.set(key, { symbols, timestamp: Date.now() });
            return symbols;
        } catch {
            return [];
        }
    }

    /**
     * Convert VS Code symbols to our format
     */
    private convertSymbols(vscodeSymbols: vscode.DocumentSymbol[]): SymbolInfo[] {
        return vscodeSymbols.map(s => ({
            name: s.name,
            kind: s.kind,
            range: s.range,
            selectionRange: s.selectionRange,
            children: s.children ? this.convertSymbols(s.children) : undefined,
        }));
    }

    /**
     * Flatten nested symbols
     */
    private flattenSymbols(symbols: SymbolInfo[]): SymbolInfo[] {
        const result: SymbolInfo[] = [];

        const flatten = (syms: SymbolInfo[]) => {
            for (const sym of syms) {
                result.push(sym);
                if (sym.children) {
                    flatten(sym.children);
                }
            }
        };

        flatten(symbols);
        return result;
    }

    /**
     * Check if CodeLens should be shown for a symbol
     */
    private shouldShowCodeLens(symbol: SymbolInfo): boolean {
        if (!CODELENS_SYMBOL_KINDS.has(symbol.kind)) {
            return false;
        }

        switch (symbol.kind) {
            case vscode.SymbolKind.Function:
                return this.config.showOnFunctions;
            case vscode.SymbolKind.Method:
            case vscode.SymbolKind.Constructor:
                return this.config.showOnMethods;
            case vscode.SymbolKind.Class:
                return this.config.showOnClasses;
            case vscode.SymbolKind.Interface:
                return this.config.showOnInterfaces;
            default:
                return true;
        }
    }

    /**
     * Clear all caches
     */
    clearCache(): void {
        this.symbolCache.clear();
        for (const timer of this.debounceTimers.values()) {
            clearTimeout(timer);
        }
        this.debounceTimers.clear();
    }

    /**
     * Dispose resources
     */
    dispose(): void {
        this.clearCache();
        this._onDidChangeCodeLenses.dispose();
    }
}

/**
 * Register CodeLens commands
 */
export function registerCodeLensCommands(context: vscode.ExtensionContext): void {
    // Symbol commands are owned by codeActionProvider and accept CodeLens targets.
    // Toggle CodeLens
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.toggleCodeLens', async () => {
            const config = vscode.workspace.getConfiguration('victor.codeLens');
            const currentValue = config.get('enabled', true);
            await config.update('enabled', !currentValue, true);
            vscode.window.showInformationMessage(
                `Victor CodeLens ${!currentValue ? 'enabled' : 'disabled'}`
            );
        })
    );
}
