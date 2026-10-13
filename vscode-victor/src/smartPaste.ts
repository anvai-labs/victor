/**
 * Smart Paste Provider
 *
 * Automatically adapts pasted code to the current context.
 * Similar to Windsurf's Smart Paste feature.
 */

import * as vscode from 'vscode';
import { VictorClient, requireCompletedChat, ChatOutcomeError } from './victorClient';

export interface PasteContext {
    language: string;
    imports: string[];
    surroundingCode: string;
    indentation: string;
    cursorPosition: vscode.Position;
}

export interface AdaptedCode {
    original: string;
    adapted: string;
    changes: string[];
}

/**
 * Smart Paste Provider
 *
 * Intercepts paste operations and offers to adapt the code to match:
 * - Current file's language/style
 * - Existing imports and dependencies
 * - Indentation and formatting
 * - Variable naming conventions
 */
export class SmartPasteProvider implements vscode.Disposable {
    private _disposables: vscode.Disposable[] = [];
    private _isProcessing: boolean = false;
    private _lastPastedText: string = '';

    constructor(
        private readonly _client: VictorClient,
        private readonly _log?: vscode.OutputChannel
    ) {
        // Register clipboard paste interception
        this._disposables.push(
            vscode.commands.registerCommand('victor.smartPaste', () => this.smartPaste())
        );
    }

    dispose(): void {
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    /**
     * Perform smart paste operation
     */
    async smartPaste(): Promise<void> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showWarningMessage('No active editor');
            return;
        }

        if (this._isProcessing) {
            return;
        }

        try {
            this._isProcessing = true;

            // Get clipboard content
            const clipboardText = await vscode.env.clipboard.readText();
            if (!clipboardText.trim()) {
                vscode.window.showInformationMessage('Clipboard is empty');
                return;
            }

            // Check if it looks like code
            if (!this._looksLikeCode(clipboardText)) {
                // Just do regular paste for non-code content
                await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
                return;
            }

            // Get context
            const context = this._getPasteContext(editor);
            this._log?.appendLine(`[SmartPaste] Context: ${context.language}, ${context.imports.length} imports`);

            // Show quick pick for user choice
            const choice = await vscode.window.showQuickPick(
                [
                    {
                        label: '$(wand) Smart Paste',
                        description: 'Adapt code to current context',
                        value: 'smart',
                    },
                    {
                        label: '$(clippy) Regular Paste',
                        description: 'Paste without modification',
                        value: 'regular',
                    },
                    {
                        label: '$(preview) Preview Adaptation',
                        description: 'See changes before pasting',
                        value: 'preview',
                    },
                ],
                { placeHolder: 'How would you like to paste?' }
            );

            if (!choice) {
                return;
            }

            if (choice.value === 'regular') {
                await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
                return;
            }

            // Show progress while adapting
            await vscode.window.withProgress(
                {
                    location: vscode.ProgressLocation.Notification,
                    title: 'Adapting pasted code...',
                    cancellable: true,
                },
                async (progress, token) => {
                    const adapted = await this._adaptCode(clipboardText, context, token);

                    if (token.isCancellationRequested) {
                        return;
                    }

                    if (!adapted) {
                        // Fallback to regular paste with user notification
                        vscode.window.showInformationMessage(
                            'Smart Paste: AI adaptation unavailable, using regular paste.',
                            'OK'
                        );
                        await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
                        return;
                    }

                    if (choice.value === 'preview') {
                        // Show preview in diff view
                        await this._showPreview(clipboardText, adapted, editor.document.languageId);
                    } else {
                        // Apply adapted code
                        await this._insertCode(editor, adapted.adapted);
                        this._showAdaptationSummary(adapted);
                    }
                }
            );

        } catch (error) {
            this._log?.appendLine(`[SmartPaste] Error: ${error}`);
            if (error instanceof ChatOutcomeError) {
                void vscode.window.showWarningMessage(error.message);
                return;
            }
            // Fallback to regular paste with error notification
            vscode.window.showWarningMessage(
                `Smart Paste error: ${error instanceof Error ? error.message : 'Unknown error'}. Using regular paste.`
            );
            await vscode.commands.executeCommand('editor.action.clipboardPasteAction');
        } finally {
            this._isProcessing = false;
        }
    }

    /**
     * Get context for paste operation
     */
    private _getPasteContext(editor: vscode.TextEditor): PasteContext {
        const document = editor.document;
        const position = editor.selection.active;

        // Get indentation at cursor
        const line = document.lineAt(position.line);
        const indentMatch = line.text.match(/^(\s*)/);
        const indentation = indentMatch ? indentMatch[1] : '';

        // Get imports from file
        const imports = this._extractImports(document);

        // Get surrounding code (5 lines before and after)
        const startLine = Math.max(0, position.line - 5);
        const endLine = Math.min(document.lineCount - 1, position.line + 5);
        const surroundingCode = document.getText(
            new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length)
        );

        return {
            language: document.languageId,
            imports,
            surroundingCode,
            indentation,
            cursorPosition: position,
        };
    }

    /**
     * Extract imports from document
     */
    private _extractImports(document: vscode.TextDocument): string[] {
        const imports: string[] = [];
        const text = document.getText();

        // Match various import patterns
        const patterns = [
            /^import\s+.+$/gm,                     // JS/TS/Python imports
            /^from\s+.+\s+import\s+.+$/gm,         // Python from imports
            /^require\s*\(.+\)/gm,                 // Node.js requires
            /^use\s+.+;$/gm,                       // Rust use
            /^using\s+.+;$/gm,                     // C# using
            /^#include\s+.+$/gm,                   // C/C++ includes
            /^package\s+.+;?$/gm,                  // Java/Go package
        ];

        for (const pattern of patterns) {
            const matches = text.match(pattern);
            if (matches) {
                imports.push(...matches);
            }
        }

        return imports.slice(0, 20); // Limit to 20 imports
    }

    /**
     * Check if text looks like code
     */
    private _looksLikeCode(text: string): boolean {
        // Simple heuristics for code detection
        const codeIndicators = [
            /\bfunction\b/,
            /\bclass\b/,
            /\bconst\b/,
            /\blet\b/,
            /\bvar\b/,
            /\bdef\b/,
            /\bimport\b/,
            /\breturn\b/,
            /\bif\s*\(/,
            /\bfor\s*\(/,
            /\bwhile\s*\(/,
            /=>/,
            /\(\)\s*{/,
            /;\s*$/m,
            /^\s*#/m,  // Comments
            /^\s*\/\//m,
        ];

        return codeIndicators.some(pattern => pattern.test(text));
    }

    /**
     * Adapt code using AI
     */
    private async _adaptCode(
        code: string,
        context: PasteContext,
        token: vscode.CancellationToken
    ): Promise<AdaptedCode | null> {
        try {
            const prompt = `Adapt this code to match the current context. Make minimal changes to fit the style and conventions.

Current file language: ${context.language}
Current indentation: "${context.indentation}" (${context.indentation.length} spaces/tabs)

Current imports in file:
${context.imports.slice(0, 10).join('\n')}

Surrounding code:
\`\`\`${context.language}
${context.surroundingCode}
\`\`\`

Code to adapt:
\`\`\`
${code}
\`\`\`

Respond with ONLY the adapted code, properly indented to match the context. No explanations.`;

            const response = await this._client.chat([{ role: 'user', content: prompt }]);

            if (token.isCancellationRequested) {
                return null;
            }

            let adapted = requireCompletedChat(response).content?.trim() || code;

            // Clean up markdown code blocks
            if (adapted.startsWith('```')) {
                const lines = adapted.split('\n');
                adapted = lines.slice(1, lines[lines.length - 1] === '```' ? -1 : undefined).join('\n');
            }

            // Apply indentation
            if (context.indentation) {
                adapted = this._applyIndentation(adapted, context.indentation);
            }

            // Detect changes
            const changes = this._detectChanges(code, adapted);

            return {
                original: code,
                adapted,
                changes,
            };
        } catch (error) {
            if (error instanceof ChatOutcomeError) { throw error; }
            this._log?.appendLine(`[SmartPaste] Adaptation error: ${error}`);
            return null;
        }
    }

    /**
     * Apply base indentation to code
     */
    private _applyIndentation(code: string, baseIndent: string): string {
        const lines = code.split('\n');

        // Find minimum indentation in original code
        let minIndent = Infinity;
        for (const line of lines) {
            if (line.trim()) {
                const indent = line.match(/^(\s*)/)?.[1].length || 0;
                minIndent = Math.min(minIndent, indent);
            }
        }

        if (minIndent === Infinity) { minIndent = 0; }

        // Apply new base indentation
        return lines.map((line, _index) => {
            if (!line.trim()) { return line; } // Keep empty lines as-is
            const currentIndent = line.match(/^(\s*)/)?.[1].length || 0;
            const relativeIndent = currentIndent - minIndent;
            const spaces = ' '.repeat(Math.max(0, relativeIndent));
            return baseIndent + spaces + line.trim();
        }).join('\n');
    }

    /**
     * Detect changes between original and adapted code
     */
    private _detectChanges(original: string, adapted: string): string[] {
        const changes: string[] = [];

        // Simple change detection
        if (original.trim() === adapted.trim()) {
            return ['No content changes, only formatting'];
        }

        // Check for variable name changes
        const originalVars: string[] = original.match(/\b[a-z_][a-zA-Z0-9_]*\b/g) || [];
        const adaptedVars: string[] = adapted.match(/\b[a-z_][a-zA-Z0-9_]*\b/g) || [];
        const newVars = adaptedVars.filter((v: string) => !originalVars.includes(v));
        if (newVars.length > 0) {
            changes.push(`Variable naming updated`);
        }

        // Check for import changes
        if (adapted.includes('import') && !original.includes('import')) {
            changes.push('Added imports');
        }

        // Check for type annotation changes
        if (adapted.includes(':') && !original.includes(':')) {
            changes.push('Added type annotations');
        }

        // Check for async/await changes
        if (adapted.includes('async') && !original.includes('async')) {
            changes.push('Made function async');
        }

        // Default changes
        if (changes.length === 0) {
            changes.push('Adapted to match context style');
        }

        return changes;
    }

    /**
     * Show preview of adaptation
     */
    private async _showPreview(original: string, adapted: AdaptedCode, language: string): Promise<void> {
        // Create virtual documents for comparison
        const originalDoc = await vscode.workspace.openTextDocument({
            content: original,
            language,
        });

        const adaptedDoc = await vscode.workspace.openTextDocument({
            content: adapted.adapted,
            language,
        });

        // Show diff
        await vscode.commands.executeCommand(
            'vscode.diff',
            originalDoc.uri,
            adaptedDoc.uri,
            'Original → Adapted (Smart Paste Preview)'
        );

        // Offer to apply
        const apply = await vscode.window.showInformationMessage(
            `Smart Paste made ${adapted.changes.length} change(s): ${adapted.changes.join(', ')}`,
            'Apply',
            'Cancel'
        );

        if (apply === 'Apply') {
            const editor = vscode.window.activeTextEditor;
            if (editor) {
                await this._insertCode(editor, adapted.adapted);
            }
        }
    }

    /**
     * Insert code at cursor position
     */
    private async _insertCode(editor: vscode.TextEditor, code: string): Promise<void> {
        await editor.edit(editBuilder => {
            if (editor.selection.isEmpty) {
                editBuilder.insert(editor.selection.active, code);
            } else {
                editBuilder.replace(editor.selection, code);
            }
        });
    }

    /**
     * Show summary of adaptations made
     */
    private _showAdaptationSummary(adapted: AdaptedCode): void {
        if (adapted.changes.length > 0) {
            vscode.window.showInformationMessage(
                `Smart Paste: ${adapted.changes.join(', ')}`,
                'Undo'
            ).then(action => {
                if (action === 'Undo') {
                    vscode.commands.executeCommand('undo');
                }
            });
        }
    }
}

/**
 * Register smart paste commands and keybindings
 */
export function registerSmartPasteCommands(
    context: vscode.ExtensionContext,
    provider: SmartPasteProvider
): void {
    context.subscriptions.push(provider);

    // Also register as editor command for keybinding
    context.subscriptions.push(
        vscode.commands.registerTextEditorCommand('victor.smartPasteInEditor', async (_editor) => {
            await provider.smartPaste();
        })
    );
}
