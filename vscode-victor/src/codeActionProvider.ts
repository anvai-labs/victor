/**
 * Code Action Provider for Victor AI
 *
 * Provides inline code actions (lightbulb menu) for AI-powered assistance:
 * - Fix with AI - Fix errors and warnings
 * - Explain with AI - Get explanations for selected code
 * - Refactor with AI - Get refactoring suggestions
 * - Generate Tests - Create unit tests
 * - Add Documentation - Generate docstrings
 * - Optimize - Performance suggestions
 *
 * Features:
 * - Context-aware actions based on diagnostics
 * - Selection-based actions
 * - Configurable action set
 * - Quick fix integration with VS Code diagnostics
 */

import * as vscode from 'vscode';

/**
 * Code action kinds provided
 */
const VICTOR_CODE_ACTION_KINDS = [
    vscode.CodeActionKind.QuickFix.append('victor'),
    vscode.CodeActionKind.Refactor.append('victor'),
    vscode.CodeActionKind.Source.append('victor'),
];

/**
 * Supported languages for code actions
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
 * Code action metadata
 */
interface VictorCodeAction {
    title: string;
    kind: vscode.CodeActionKind;
    command: string;
    icon?: string;
    priority: number;
    requiresDiagnostics?: boolean;
    requiresSelection?: boolean;
}

/**
 * Available code actions
 */
const CODE_ACTIONS: VictorCodeAction[] = [
    {
        title: 'Fix with Victor AI',
        kind: vscode.CodeActionKind.QuickFix.append('victor'),
        command: 'victor.fixWithAI',
        icon: '$(lightbulb-autofix)',
        priority: 1,
        requiresDiagnostics: true,
    },
    {
        title: 'Explain with Victor AI',
        kind: vscode.CodeActionKind.Source.append('victor.explain'),
        command: 'victor.explainWithAI',
        icon: '$(question)',
        priority: 2,
        requiresSelection: true,
    },
    {
        title: 'Refactor with Victor AI',
        kind: vscode.CodeActionKind.Refactor.append('victor'),
        command: 'victor.refactorWithAI',
        icon: '$(edit)',
        priority: 3,
        requiresSelection: true,
    },
    {
        title: 'Generate Tests with Victor AI',
        kind: vscode.CodeActionKind.Source.append('victor.test'),
        command: 'victor.generateTestsWithAI',
        icon: '$(beaker)',
        priority: 4,
        requiresSelection: true,
    },
    {
        title: 'Add Documentation with Victor AI',
        kind: vscode.CodeActionKind.Source.append('victor.document'),
        command: 'victor.documentWithAI',
        icon: '$(book)',
        priority: 5,
        requiresSelection: true,
    },
    {
        title: 'Optimize with Victor AI',
        kind: vscode.CodeActionKind.Refactor.append('victor.optimize'),
        command: 'victor.optimizeWithAI',
        icon: '$(zap)',
        priority: 6,
        requiresSelection: true,
    },
    {
        title: 'Review with Victor AI',
        kind: vscode.CodeActionKind.Source.append('victor.review'),
        command: 'victor.reviewWithAI',
        icon: '$(eye)',
        priority: 7,
        requiresSelection: true,
    },
];

/**
 * Victor Code Action Provider
 */
export class VictorCodeActionProvider implements vscode.CodeActionProvider {
    public static readonly providedCodeActionKinds = VICTOR_CODE_ACTION_KINDS;

    private enabled: boolean;

    constructor() {
        this.enabled = this.loadConfig();

        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('victor.codeActions')) {
                this.enabled = this.loadConfig();
            }
        });
    }

    private loadConfig(): boolean {
        const config = vscode.workspace.getConfiguration('victor.codeActions');
        return config.get('enabled', true);
    }

    provideCodeActions(
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        context: vscode.CodeActionContext,
        _token: vscode.CancellationToken
    ): vscode.CodeAction[] {
        if (!this.enabled) {
            return [];
        }

        if (!SUPPORTED_LANGUAGES.has(document.languageId)) {
            return [];
        }

        const actions: vscode.CodeAction[] = [];
        const hasSelection = !range.isEmpty;
        const hasDiagnostics = context.diagnostics.length > 0;
        const selectedText = hasSelection ? document.getText(range) : '';

        for (const actionDef of CODE_ACTIONS) {
            // Skip actions that require diagnostics if none present
            if (actionDef.requiresDiagnostics && !hasDiagnostics) {
                continue;
            }

            // Skip actions that require selection if none present
            if (actionDef.requiresSelection && !hasSelection) {
                continue;
            }

            const action = this.createCodeAction(
                actionDef,
                document,
                range,
                context.diagnostics,
                selectedText
            );

            if (action) {
                actions.push(action);
            }
        }

        // Sort by priority
        actions.sort((a, b) => {
            const aPriority = CODE_ACTIONS.find(ca => ca.title === a.title)?.priority || 99;
            const bPriority = CODE_ACTIONS.find(ca => ca.title === b.title)?.priority || 99;
            return aPriority - bPriority;
        });

        return actions;
    }

    private createCodeAction(
        actionDef: VictorCodeAction,
        document: vscode.TextDocument,
        range: vscode.Range | vscode.Selection,
        diagnostics: readonly vscode.Diagnostic[],
        selectedText: string
    ): vscode.CodeAction | null {
        const action = new vscode.CodeAction(actionDef.title, actionDef.kind);
        action.command = {
            command: actionDef.command,
            title: actionDef.title,
            arguments: [document.uri, range, selectedText, diagnostics],
        };

        // Set as preferred for quick fix with diagnostics
        if (actionDef.requiresDiagnostics && diagnostics.length > 0) {
            action.isPreferred = true;
            action.diagnostics = [...diagnostics];
        }

        return action;
    }
}

/**
 * Register code action commands
 */
export function registerCodeActionProviderCommands(
    context: vscode.ExtensionContext,
    sendToChat: (message: string) => Promise<void>
): void {
    // Fix with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.fixWithAI',
            async (
                uri: vscode.Uri,
                range: vscode.Range,
                _selectedText: string,
                diagnostics: vscode.Diagnostic[]
            ) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = document.getText(range.isEmpty ? undefined : range);

                const diagnosticMessages = diagnostics
                    .map(d => `- [${d.severity === 0 ? 'Error' : 'Warning'}] ${d.message}`)
                    .join('\n');

                const prompt = diagnosticMessages
                    ? `Fix these issues:\n${diagnosticMessages}\n\nIn this ${document.languageId} code:\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                    : `Fix any issues in this ${document.languageId} code:\n\`\`\`${document.languageId}\n${code}\n\`\`\``;

                await sendToChat(prompt);
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Explain with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.explainWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                await sendToChat(
                    `Explain this ${document.languageId} code in detail:\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Refactor with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.refactorWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                const suggestion = await vscode.window.showInputBox({
                    prompt: 'What refactoring would you like?',
                    placeHolder: 'e.g., Extract function, simplify logic, improve readability...',
                });

                if (suggestion) {
                    await sendToChat(
                        `Refactor this ${document.languageId} code (${suggestion}):\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                    );
                    await vscode.commands.executeCommand('victor.chatView.focus');
                }
            }
        )
    );

    // Generate tests with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.generateTestsWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                await sendToChat(
                    `Generate comprehensive unit tests for this ${document.languageId} code:\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Document with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.documentWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                await sendToChat(
                    `Add comprehensive documentation/docstrings to this ${document.languageId} code:\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Optimize with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.optimizeWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                await sendToChat(
                    `Analyze and optimize this ${document.languageId} code for better performance:\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Review with AI
    context.subscriptions.push(
        vscode.commands.registerCommand(
            'victor.reviewWithAI',
            async (uri: vscode.Uri, range: vscode.Range, selectedText: string) => {
                const document = await vscode.workspace.openTextDocument(uri);
                const code = selectedText || document.getText(range);

                await sendToChat(
                    `Review this ${document.languageId} code for:\n- Code quality and best practices\n- Potential bugs\n- Security issues\n- Performance concerns\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        )
    );

    // Toggle code actions
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.toggleCodeActions', async () => {
            const config = vscode.workspace.getConfiguration('victor.codeActions');
            const currentValue = config.get('enabled', true);
            await config.update('enabled', !currentValue, true);
            vscode.window.showInformationMessage(
                `Victor Code Actions ${!currentValue ? 'enabled' : 'disabled'}`
            );
        })
    );

    // Register symbol-based commands (work on symbol at cursor)
    registerSymbolCommands(context, sendToChat);
}

/**
 * Get the symbol range at the current cursor position
 */
async function getSymbolAtCursor(): Promise<{ uri: vscode.Uri; range: vscode.Range; name: string; kind: string } | null> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        return null;
    }

    const document = editor.document;
    const position = editor.selection.active;

    // Get document symbols
    const symbols = await vscode.commands.executeCommand<vscode.DocumentSymbol[]>(
        'vscode.executeDocumentSymbolProvider',
        document.uri
    );

    if (!symbols || symbols.length === 0) {
        return null;
    }

    // Find the smallest symbol containing the cursor
    const findSymbol = (
        symbols: vscode.DocumentSymbol[],
        position: vscode.Position
    ): vscode.DocumentSymbol | null => {
        for (const symbol of symbols) {
            if (symbol.range.contains(position)) {
                // Check children first for more specific match
                const child = findSymbol(symbol.children, position);
                if (child) {
                    return child;
                }
                return symbol;
            }
        }
        return null;
    };

    const symbol = findSymbol(symbols, position);
    if (!symbol) {
        return null;
    }

    return {
        uri: document.uri,
        range: symbol.range,
        name: symbol.name,
        kind: vscode.SymbolKind[symbol.kind],
    };
}

/** Encode the explicit target contract for hover command links. */
export function encodeSymbolTarget(uri: vscode.Uri, range: vscode.Range, name: string): string {
    return encodeURIComponent(JSON.stringify([
        uri.toString(),
        [[range.start.line, range.start.character], [range.end.line, range.end.character]],
        name,
    ]));
}

/** Resolve a CodeLens/hover target or the current cursor for palette commands. */
export async function resolveSymbolTarget(...args: unknown[]): Promise<{
    uri: vscode.Uri; range: vscode.Range; name: string; kind: string;
} | null> {
    if (args.length === 0) {
        return getSymbolAtCursor();
    }
    let [uri, range] = args;
    const name = args[2];
    // Hover links use a URI string and two [line, character] pairs. Decode only
    // this complete shape; malformed explicit targets must never use the cursor.
    if (typeof uri === 'string' && /^[a-z][a-z0-9+.-]*:/i.test(uri)
        && Array.isArray(range) && range.length === 2
        && range.every(p => Array.isArray(p) && p.length === 2
            && p.every(n => Number.isSafeInteger(n) && n >= 0))) {
        const start = new vscode.Position(range[0][0], range[0][1]);
        const end = new vscode.Position(range[1][0], range[1][1]);
        if (start.isAfter(end)) {
            throw new TypeError('Expected an ordered symbol range');
        }
        uri = vscode.Uri.parse(uri, true);
        range = new vscode.Range(start, end);
    }
    if (args.length !== 3 || !(uri instanceof vscode.Uri) || !(range instanceof vscode.Range)
        || typeof name !== 'string' || !name.trim()) {
        throw new TypeError('Expected a symbol URI, range, and non-empty name');
    }
    return { uri, range, name, kind: 'symbol' };
}

/**
 * Register symbol-based AI commands
 */
function registerSymbolCommands(
    context: vscode.ExtensionContext,
    sendToChat: (message: string) => Promise<void>
): void {
    // Ask about symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.askAboutSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const question = await vscode.window.showInputBox({
                prompt: `Ask about ${symbol.name}`,
                placeHolder: 'What does this do? How can I improve it?',
            });

            if (question) {
                const document = await vscode.workspace.openTextDocument(symbol.uri);
                const code = document.getText(symbol.range);

                await sendToChat(
                    `Question about the ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n${question}\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        })
    );

    // Explain symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.explainSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const document = await vscode.workspace.openTextDocument(symbol.uri);
            const code = document.getText(symbol.range);

            await sendToChat(
                `Explain this ${symbol.kind.toLowerCase()} "${symbol.name}" in detail:\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Refactor symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.refactorSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const suggestion = await vscode.window.showInputBox({
                prompt: `Refactor ${symbol.name}`,
                placeHolder: 'e.g., Extract helper function, simplify logic, rename variables...',
            });

            if (suggestion) {
                const document = await vscode.workspace.openTextDocument(symbol.uri);
                const code = document.getText(symbol.range);

                await sendToChat(
                    `Refactor this ${symbol.kind.toLowerCase()} "${symbol.name}" (${suggestion}):\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
                );
                await vscode.commands.executeCommand('victor.chatView.focus');
            }
        })
    );

    // Document symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.documentSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const document = await vscode.workspace.openTextDocument(symbol.uri);
            const code = document.getText(symbol.range);

            await sendToChat(
                `Add comprehensive documentation to this ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Generate tests for symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.generateTestsForSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const document = await vscode.workspace.openTextDocument(symbol.uri);
            const code = document.getText(symbol.range);

            await sendToChat(
                `Generate comprehensive unit tests for this ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Optimize symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.optimizeSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const document = await vscode.workspace.openTextDocument(symbol.uri);
            const code = document.getText(symbol.range);

            await sendToChat(
                `Analyze and optimize this ${symbol.kind.toLowerCase()} "${symbol.name}" for better performance:\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );

    // Review symbol
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.reviewSymbol', async (...args: unknown[]) => {
            const symbol = await resolveSymbolTarget(...args);
            if (!symbol) {
                vscode.window.showWarningMessage('No symbol found at cursor position');
                return;
            }

            const document = await vscode.workspace.openTextDocument(symbol.uri);
            const code = document.getText(symbol.range);

            await sendToChat(
                `Review this ${symbol.kind.toLowerCase()} "${symbol.name}" for:\n- Code quality and best practices\n- Potential bugs\n- Security issues\n- Performance concerns\n\n\`\`\`${document.languageId}\n${code}\n\`\`\``
            );
            await vscode.commands.executeCommand('victor.chatView.focus');
        })
    );
}
