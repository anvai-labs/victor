/**
 * Context Provider
 *
 * Provides context gathering for AI prompts including:
 * - @file mentions to include file contents
 * - @symbol mentions to include class/function definitions
 * - @folder mentions to include folder structure
 * - @selection for current editor selection
 * - @diagnostics for current file errors/warnings
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as cp from 'child_process';

export interface ContextItem {
    type: 'file' | 'symbol' | 'folder' | 'selection' | 'diagnostics' | 'git';
    name: string;
    content: string;
    path?: string;
    range?: vscode.Range;
    language?: string;
}

export interface ParsedMention {
    type: 'file' | 'symbol' | 'folder' | 'selection' | 'diagnostics' | 'git';
    query: string;
    startIndex: number;
    endIndex: number;
}

/**
 * Parses @-mentions from user input
 */
export function parseMentions(input: string): ParsedMention[] {
    const mentions: ParsedMention[] = [];

    // Match patterns like @file:path, @symbol:name, @folder:path, @selection, @diagnostics, @git
    const mentionRegex = /@(file|symbol|folder|selection|diagnostics|git)(?::([^\s]+))?/g;
    let match;

    while ((match = mentionRegex.exec(input)) !== null) {
        mentions.push({
            type: match[1] as ParsedMention['type'],
            query: match[2] || '',
            startIndex: match.index,
            endIndex: match.index + match[0].length,
        });
    }

    return mentions;
}

/**
 * Removes @-mentions from input text, returning clean prompt
 */
export function stripMentions(input: string): string {
    return input.replace(/@(file|symbol|folder|selection|diagnostics|git)(?::[^\s]+)?/g, '').trim();
}

/**
 * Context Provider for gathering code context
 */
export class ContextProvider {
    private _workspaceRoot: string;

    constructor() {
        this._workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
    }

    /**
     * Gather all context items from mentions in input
     */
    async gatherContext(input: string): Promise<ContextItem[]> {
        const mentions = parseMentions(input);
        const items: ContextItem[] = [];

        for (const mention of mentions) {
            const contextItems = await this._resolveContext(mention);
            items.push(...contextItems);
        }

        return items;
    }

    /**
     * Resolve a single mention to context items
     */
    private async _resolveContext(mention: ParsedMention): Promise<ContextItem[]> {
        switch (mention.type) {
            case 'file':
                return this._resolveFileContext(mention.query);
            case 'symbol':
                return this._resolveSymbolContext(mention.query);
            case 'folder':
                return this._resolveFolderContext(mention.query);
            case 'selection':
                return this._resolveSelectionContext();
            case 'diagnostics':
                return this._resolveDiagnosticsContext();
            case 'git':
                return this._resolveGitContext(mention.query);
            default:
                return [];
        }
    }

    /**
     * Resolve @file:path mention
     */
    private async _resolveFileContext(query: string): Promise<ContextItem[]> {
        if (!query) {
            // Use current active file
            const editor = vscode.window.activeTextEditor;
            if (!editor) {return [];}

            const content = editor.document.getText();
            const relativePath = vscode.workspace.asRelativePath(editor.document.uri);

            return [{
                type: 'file',
                name: path.basename(editor.document.fileName),
                path: relativePath,
                content,
                language: editor.document.languageId,
            }];
        }

        // Search for matching files
        const files = await vscode.workspace.findFiles(
            `**/${query}*`,
            '**/node_modules/**',
            10
        );

        const items: ContextItem[] = [];
        for (const file of files) {
            try {
                const doc = await vscode.workspace.openTextDocument(file);
                const relativePath = vscode.workspace.asRelativePath(file);

                items.push({
                    type: 'file',
                    name: path.basename(file.fsPath),
                    path: relativePath,
                    content: doc.getText(),
                    language: doc.languageId,
                });
            } catch {
                // Skip files that can't be read
            }
        }

        return items;
    }

    /**
     * Resolve @symbol:name mention using workspace symbols
     */
    private async _resolveSymbolContext(query: string): Promise<ContextItem[]> {
        if (!query) {return [];}

        // Search workspace symbols
        const symbols = await vscode.commands.executeCommand<vscode.SymbolInformation[]>(
            'vscode.executeWorkspaceSymbolProvider',
            query
        );

        if (!symbols || symbols.length === 0) {return [];}

        const items: ContextItem[] = [];

        // Get first 5 matching symbols
        for (const symbol of symbols.slice(0, 5)) {
            try {
                const doc = await vscode.workspace.openTextDocument(symbol.location.uri);
                const range = symbol.location.range;

                // Expand range to include full definition (up to 50 lines)
                const startLine = range.start.line;
                const endLine = Math.min(range.end.line + 20, doc.lineCount - 1);
                const expandedRange = new vscode.Range(startLine, 0, endLine, doc.lineAt(endLine).text.length);

                const content = doc.getText(expandedRange);
                const relativePath = vscode.workspace.asRelativePath(symbol.location.uri);

                items.push({
                    type: 'symbol',
                    name: symbol.name,
                    path: relativePath,
                    content,
                    range: expandedRange,
                    language: doc.languageId,
                });
            } catch {
                // Skip symbols that can't be resolved
            }
        }

        return items;
    }

    /**
     * Resolve @folder:path mention
     */
    private async _resolveFolderContext(query: string): Promise<ContextItem[]> {
        const folderPath = query || '.';
        const absolutePath = path.isAbsolute(folderPath)
            ? folderPath
            : path.join(this._workspaceRoot, folderPath);

        try {
            const uri = vscode.Uri.file(absolutePath);
            const entries = await vscode.workspace.fs.readDirectory(uri);

            // Build tree structure
            const tree = entries
                .map(([name, type]) => {
                    const icon = type === vscode.FileType.Directory ? '📁' : '📄';
                    return `${icon} ${name}`;
                })
                .join('\n');

            return [{
                type: 'folder',
                name: path.basename(absolutePath) || folderPath,
                path: vscode.workspace.asRelativePath(uri),
                content: tree,
            }];
        } catch {
            return [];
        }
    }

    /**
     * Resolve @selection mention
     */
    private async _resolveSelectionContext(): Promise<ContextItem[]> {
        const editor = vscode.window.activeTextEditor;
        if (!editor || editor.selection.isEmpty) {return [];}

        const selection = editor.selection;
        const content = editor.document.getText(selection);
        const relativePath = vscode.workspace.asRelativePath(editor.document.uri);

        return [{
            type: 'selection',
            name: `Selection in ${path.basename(editor.document.fileName)}`,
            path: relativePath,
            content,
            range: selection,
            language: editor.document.languageId,
        }];
    }

    /**
     * Resolve @diagnostics mention
     */
    private async _resolveDiagnosticsContext(): Promise<ContextItem[]> {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {return [];}

        const diagnostics = vscode.languages.getDiagnostics(editor.document.uri);
        if (diagnostics.length === 0) {return [];}

        const content = diagnostics
            .map(d => {
                const severity = this._getSeverityString(d.severity);
                const line = d.range.start.line + 1;
                return `[${severity}] Line ${line}: ${d.message}`;
            })
            .join('\n');

        const relativePath = vscode.workspace.asRelativePath(editor.document.uri);

        return [{
            type: 'diagnostics',
            name: `Diagnostics for ${path.basename(editor.document.fileName)}`,
            path: relativePath,
            content,
        }];
    }

    /**
     * Resolve @git mention
     */
    private async _resolveGitContext(query: string): Promise<ContextItem[]> {
        // Context preview is read-only. Never interpret a mention as a command.
        const gitCommand = query || 'status';
        const commands = new Map<string, string[]>([
            ['status', ['status', '--short']],
            ['diff', ['diff', '--no-ext-diff', '--no-textconv']],
            ['log', ['log', '-20', '--oneline', '--no-decorate']],
            ['show', ['show', '--no-ext-diff', '--no-textconv']],
            ['branch', ['branch', '--list']],
        ]);
        const args = commands.get(gitCommand);
        if (!args) {
            vscode.window.showWarningMessage('Git context supports status, diff, log, show, and branch.');
            return [];
        }

        try {
            const result = await this._executeGitCommand(args);
            return [{
                type: 'git',
                name: `git ${gitCommand}`,
                content: result,
            }];
        } catch {
            return [];
        }
    }

    private async _executeGitCommand(args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            cp.execFile(
                'git',
                ['--no-pager', '-c', 'core.fsmonitor=false', ...args],
                {
                    cwd: this._workspaceRoot,
                    maxBuffer: 1024 * 1024,
                    timeout: 10000,
                    env: { ...process.env, GIT_NO_LAZY_FETCH: '1' },
                },
                (error: Error | null, stdout: string, stderr: string) => {
                    if (error) {
                        reject(error);
                    } else {
                        resolve(stdout || stderr);
                    }
                }
            );
        });
    }

    private _getSeverityString(severity: vscode.DiagnosticSeverity): string {
        switch (severity) {
            case vscode.DiagnosticSeverity.Error:
                return 'ERROR';
            case vscode.DiagnosticSeverity.Warning:
                return 'WARNING';
            case vscode.DiagnosticSeverity.Information:
                return 'INFO';
            case vscode.DiagnosticSeverity.Hint:
                return 'HINT';
            default:
                return 'UNKNOWN';
        }
    }

    /**
     * Format context items for inclusion in prompt
     */
    formatContextForPrompt(items: ContextItem[]): string {
        if (items.length === 0) {return '';}

        const sections = items.map(item => {
            const header = this._formatContextHeader(item);
            return `${header}\n\`\`\`${item.language || ''}\n${item.content}\n\`\`\``;
        });

        return `## Context\n\n${sections.join('\n\n')}`;
    }

    private _formatContextHeader(item: ContextItem): string {
        switch (item.type) {
            case 'file':
                return `### File: ${item.path}`;
            case 'symbol':
                return `### Symbol: ${item.name} (${item.path})`;
            case 'folder':
                return `### Folder: ${item.path}`;
            case 'selection':
                return `### Selection: ${item.name}`;
            case 'diagnostics':
                return `### Diagnostics: ${item.name}`;
            case 'git':
                return `### Git: ${item.name}`;
            default:
                return `### ${item.name}`;
        }
    }
}

/**
 * Completion provider for @-mentions
 */
export class MentionCompletionProvider implements vscode.CompletionItemProvider {
    private _contextProvider: ContextProvider;

    constructor(contextProvider: ContextProvider) {
        this._contextProvider = contextProvider;
    }

    async provideCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        _token: vscode.CancellationToken
    ): Promise<vscode.CompletionItem[]> {
        const lineText = document.lineAt(position).text;
        const linePrefix = lineText.substring(0, position.character);

        // Check if we're after an @
        const atMatch = linePrefix.match(/@(\w*)$/);
        if (!atMatch) {return [];}

        const prefix = atMatch[1].toLowerCase();
        const items: vscode.CompletionItem[] = [];

        // Suggest mention types
        const mentionTypes = [
            { label: 'file', description: 'Include file contents', detail: '@file:filename' },
            { label: 'symbol', description: 'Include symbol definition', detail: '@symbol:name' },
            { label: 'folder', description: 'Include folder structure', detail: '@folder:path' },
            { label: 'selection', description: 'Include current selection', detail: '@selection' },
            { label: 'diagnostics', description: 'Include current file errors', detail: '@diagnostics' },
            { label: 'git', description: 'Include git info', detail: '@git:status' },
        ];

        for (const type of mentionTypes) {
            if (type.label.startsWith(prefix)) {
                const item = new vscode.CompletionItem(type.label, vscode.CompletionItemKind.Keyword);
                item.detail = type.detail;
                item.documentation = type.description;
                item.insertText = type.label + (type.label === 'selection' || type.label === 'diagnostics' ? '' : ':');
                items.push(item);
            }
        }

        return items;
    }
}

/**
 * Quick pick for selecting context items
 */
export async function showContextPicker(): Promise<ContextItem[]> {
    const options: vscode.QuickPickItem[] = [
        { label: '$(file) Current File', description: 'Include current file contents', detail: '@file' },
        { label: '$(selection) Current Selection', description: 'Include selected code', detail: '@selection' },
        { label: '$(warning) Current Diagnostics', description: 'Include errors and warnings', detail: '@diagnostics' },
        { label: '$(git-branch) Git Status', description: 'Include git status', detail: '@git:status' },
        { label: '$(git-commit) Git Diff', description: 'Include uncommitted changes', detail: '@git:diff' },
        { label: '$(search) Search Files...', description: 'Find and include files', detail: '@file:' },
        { label: '$(symbol-class) Search Symbols...', description: 'Find and include symbols', detail: '@symbol:' },
    ];

    const selected = await vscode.window.showQuickPick(options, {
        placeHolder: 'Select context to include',
        canPickMany: true,
    });

    if (!selected) {return [];}

    const contextProvider = new ContextProvider();
    const items: ContextItem[] = [];

    for (const option of selected) {
        if (option.detail === '@file:') {
            // Show file picker
            const files = await vscode.window.showOpenDialog({
                canSelectMany: true,
                openLabel: 'Include',
            });
            if (files) {
                for (const file of files) {
                    const fileItems = await contextProvider.gatherContext(`@file:${file.fsPath}`);
                    items.push(...fileItems);
                }
            }
        } else if (option.detail === '@symbol:') {
            // Show symbol search
            const query = await vscode.window.showInputBox({
                prompt: 'Enter symbol name to search',
                placeHolder: 'e.g., MyClass, handleClick',
            });
            if (query) {
                const symbolItems = await contextProvider.gatherContext(`@symbol:${query}`);
                items.push(...symbolItems);
            }
        } else if (option.detail) {
            const contextItems = await contextProvider.gatherContext(option.detail);
            items.push(...contextItems);
        }
    }

    return items;
}

/**
 * Register context commands
 */
export function registerContextCommands(
    context: vscode.ExtensionContext,
    contextProvider: ContextProvider
): void {
    // Add context command
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.addContext', async () => {
            const items = await showContextPicker();
            if (items.length > 0) {
                const formatted = contextProvider.formatContextForPrompt(items);
                // Copy to clipboard or insert in active editor
                await vscode.env.clipboard.writeText(formatted);
                vscode.window.showInformationMessage(
                    `${items.length} context item(s) copied to clipboard`
                );
            }
        })
    );

    // Show context preview
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.previewContext', async () => {
            const input = await vscode.window.showInputBox({
                prompt: 'Enter text with @-mentions to preview context',
                placeHolder: 'e.g., @file:main.ts @diagnostics',
            });

            if (input) {
                const items = await contextProvider.gatherContext(input);
                if (items.length === 0) {
                    vscode.window.showInformationMessage('No context items found');
                    return;
                }

                // Show in output channel
                const output = vscode.window.createOutputChannel('Victor Context Preview');
                output.clear();
                output.appendLine(contextProvider.formatContextForPrompt(items));
                output.show();
            }
        })
    );
}
