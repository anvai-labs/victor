/**
 * Diff View Provider
 *
 * Provides inline diff preview and application UI for proposed file changes.
 * Shows file modifications before applying them with accept/reject controls.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { resolveWorkspaceFile } from './workspacePaths';

export interface FileChange {
    filePath: string;
    originalContent: string;
    newContent: string;
    changeType: 'create' | 'modify' | 'delete';
    description?: string;
    linesAdded?: number;
    linesRemoved?: number;
    selected?: boolean;  // For selective application
}

export interface DiffSession {
    id: string;
    changes: FileChange[];
    timestamp: Date;
    description: string;
    isDryRun?: boolean;
}

/**
 * Manages diff previews and file change applications
 */
export class DiffViewProvider {
    private _pendingSessions: Map<string, DiffSession> = new Map();
    private _decorationType: vscode.TextEditorDecorationType;
    private _statusBarItem: vscode.StatusBarItem;

    constructor() {
        // Create decoration type for highlighting changes
        this._decorationType = vscode.window.createTextEditorDecorationType({
            backgroundColor: new vscode.ThemeColor('diffEditor.insertedTextBackground'),
            isWholeLine: true,
        });

        // Create status bar item for pending changes
        this._statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            98
        );
        this._statusBarItem.command = 'victor.showPendingChanges';
        this._updateStatusBar();
    }

    /**
     * Create a new diff session with proposed changes
     */
    createSession(changes: FileChange[], description: string): string {
        const sessionId = `diff-${Date.now()}`;
        const session: DiffSession = {
            id: sessionId,
            changes,
            timestamp: new Date(),
            description,
        };
        this._pendingSessions.set(sessionId, session);
        this._updateStatusBar();
        return sessionId;
    }

    /**
     * Show diff preview for a specific file change
     */
    async showDiff(change: FileChange): Promise<void> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        let absolutePath: string;
        try {
            absolutePath = await resolveWorkspaceFile(workspaceRoot, change.filePath);
        } catch (error) {
            vscode.window.showErrorMessage(`Cannot preview proposed file: ${error}`);
            return;
        }

        // Create URIs for diff view
        const originalUri = vscode.Uri.parse(`victor-original:${change.filePath}`);
        const modifiedUri = vscode.Uri.file(absolutePath);

        // For new files, show the new content directly
        if (change.changeType === 'create') {
            const doc = await vscode.workspace.openTextDocument({
                content: change.newContent,
                language: this._getLanguageId(change.filePath),
            });
            await vscode.window.showTextDocument(doc, { preview: true });
            return;
        }

        // For modifications, show side-by-side diff
        const title = `${path.basename(change.filePath)} (Proposed Changes)`;
        await vscode.commands.executeCommand(
            'vscode.diff',
            originalUri,
            modifiedUri,
            title,
            { preview: true }
        );
    }

    /**
     * Show multi-file diff preview panel with selective application
     */
    async showMultiFileDiffPanel(sessionId: string, _context: vscode.ExtensionContext): Promise<void> {
        const session = this._pendingSessions.get(sessionId);
        if (!session) {
            vscode.window.showErrorMessage('Session not found');
            return;
        }

        // Create webview panel
        const panel = vscode.window.createWebviewPanel(
            'victorDiffPreview',
            `Review Changes: ${session.description}`,
            vscode.ViewColumn.One,
            { enableScripts: true }
        );

        // Calculate line stats for each change
        for (const change of session.changes) {
            if (change.originalContent && change.newContent) {
                const origLines = change.originalContent.split('\n').length;
                const newLines = change.newContent.split('\n').length;
                change.linesAdded = Math.max(0, newLines - origLines);
                change.linesRemoved = Math.max(0, origLines - newLines);
            } else if (change.changeType === 'create') {
                change.linesAdded = change.newContent.split('\n').length;
                change.linesRemoved = 0;
            } else if (change.changeType === 'delete') {
                change.linesAdded = 0;
                change.linesRemoved = change.originalContent.split('\n').length;
            }
            change.selected = true; // Default selected
        }

        panel.webview.html = this._getMultiFileDiffHtml(session);

        // Handle messages from webview
        panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'toggleFile': {
                    const change = session.changes.find(c => c.filePath === message.filePath);
                    if (change) {
                        change.selected = message.selected;
                    }
                    break;
                }

                case 'toggleAll': {
                    session.changes.forEach(c => c.selected = message.selected);
                    break;
                }

                case 'showFileDiff': {
                    const fileChange = session.changes.find(c => c.filePath === message.filePath);
                    if (fileChange) {
                        await this.showDiff(fileChange);
                    }
                    break;
                }

                case 'applySelected': {
                    const selectedChanges = session.changes.filter(c => c.selected);
                    if (selectedChanges.length === 0) {
                        vscode.window.showWarningMessage('No files selected');
                        return;
                    }

                    const confirm = await vscode.window.showWarningMessage(
                        `Apply ${selectedChanges.length} selected change(s)?`,
                        { modal: true },
                        'Apply',
                        'Cancel'
                    );

                    if (confirm === 'Apply') {
                        let successCount = 0;
                        for (const change of selectedChanges) {
                            if (await this.applyChange(change)) {
                                successCount++;
                            }
                        }
                        vscode.window.showInformationMessage(`Applied ${successCount}/${selectedChanges.length} changes`);

                        // Remove applied changes from session
                        session.changes = session.changes.filter(c => !c.selected);
                        if (session.changes.length === 0) {
                            this._pendingSessions.delete(sessionId);
                            panel.dispose();
                        } else {
                            panel.webview.html = this._getMultiFileDiffHtml(session);
                        }
                        this._updateStatusBar();
                    }
                    break;
                }

                case 'rejectAll': {
                    this.rejectSession(sessionId);
                    panel.dispose();
                    break;
                }
            }
        });
    }

    private _getMultiFileDiffHtml(session: DiffSession): string {
        const totalAdded = session.changes.reduce((sum, c) => sum + (c.linesAdded || 0), 0);
        const totalRemoved = session.changes.reduce((sum, c) => sum + (c.linesRemoved || 0), 0);
        const selectedCount = session.changes.filter(c => c.selected).length;

        const fileRows = session.changes.map(change => {
            const icon = this._getChangeIcon(change.changeType);
            const stats = change.changeType === 'delete'
                ? `<span class="stat removed">-${change.linesRemoved || 0}</span>`
                : change.changeType === 'create'
                    ? `<span class="stat added">+${change.linesAdded || 0}</span>`
                    : `<span class="stat added">+${change.linesAdded || 0}</span> <span class="stat removed">-${change.linesRemoved || 0}</span>`;

            return `
                <tr class="file-row ${change.changeType}">
                    <td class="checkbox-cell">
                        <input type="checkbox" ${change.selected ? 'checked' : ''} onchange="toggleFile('${change.filePath}', this.checked)">
                    </td>
                    <td class="file-cell" onclick="showFileDiff('${change.filePath}')">
                        <span class="icon">${icon}</span>
                        <span class="filename">${path.basename(change.filePath)}</span>
                        <span class="filepath">${path.dirname(change.filePath)}</span>
                    </td>
                    <td class="stats-cell">${stats}</td>
                    <td class="action-cell">
                        <button onclick="showFileDiff('${change.filePath}')" title="View diff">View</button>
                    </td>
                </tr>
            `;
        }).join('');

        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Review Changes</title>
    <style>
        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            padding: 20px;
            margin: 0;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid var(--vscode-panel-border);
        }

        .title {
            font-size: 1.3em;
            font-weight: 600;
        }

        .summary {
            display: flex;
            gap: 20px;
            color: var(--vscode-descriptionForeground);
        }

        .summary-item {
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .stat {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: 500;
        }

        .stat.added {
            background: var(--vscode-gitDecoration-addedResourceForeground);
            color: #000;
        }

        .stat.removed {
            background: var(--vscode-gitDecoration-deletedResourceForeground);
            color: #fff;
        }

        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .controls label {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            text-align: left;
            padding: 8px;
            background: var(--vscode-editor-inactiveSelectionBackground);
            font-weight: 500;
            border-bottom: 1px solid var(--vscode-panel-border);
        }

        .file-row {
            border-bottom: 1px solid var(--vscode-panel-border);
        }

        .file-row:hover {
            background: var(--vscode-list-hoverBackground);
        }

        .file-row.create {
            background: rgba(0, 200, 83, 0.1);
        }

        .file-row.delete {
            background: rgba(255, 82, 82, 0.1);
        }

        td {
            padding: 10px 8px;
            vertical-align: middle;
        }

        .checkbox-cell {
            width: 30px;
            text-align: center;
        }

        .file-cell {
            cursor: pointer;
        }

        .file-cell:hover .filename {
            text-decoration: underline;
        }

        .icon {
            margin-right: 8px;
        }

        .filename {
            font-weight: 500;
        }

        .filepath {
            color: var(--vscode-descriptionForeground);
            margin-left: 8px;
            font-size: 0.9em;
        }

        .stats-cell {
            width: 120px;
            text-align: center;
        }

        .action-cell {
            width: 80px;
        }

        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none;
            padding: 6px 12px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }

        button:hover {
            background: var(--vscode-button-hoverBackground);
        }

        button.secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        button.danger {
            background: var(--vscode-inputValidation-errorBackground);
        }

        .actions {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            padding-top: 15px;
            border-top: 1px solid var(--vscode-panel-border);
        }

        .actions button {
            padding: 8px 16px;
        }

        .spacer {
            flex: 1;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">${session.description}</div>
        <div class="summary">
            <div class="summary-item">
                <span>${session.changes.length} files</span>
            </div>
            <div class="summary-item">
                <span class="stat added">+${totalAdded}</span>
            </div>
            <div class="summary-item">
                <span class="stat removed">-${totalRemoved}</span>
            </div>
        </div>
    </div>

    <div class="controls">
        <label>
            <input type="checkbox" ${selectedCount === session.changes.length ? 'checked' : ''} onchange="toggleAll(this.checked)">
            Select All (${selectedCount}/${session.changes.length})
        </label>
    </div>

    <table>
        <thead>
            <tr>
                <th></th>
                <th>File</th>
                <th>Changes</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
            ${fileRows}
        </tbody>
    </table>

    <div class="actions">
        <button onclick="applySelected()">Apply Selected (${selectedCount})</button>
        <div class="spacer"></div>
        <button class="secondary danger" onclick="rejectAll()">Discard All</button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function toggleFile(filePath, selected) {
            vscode.postMessage({ type: 'toggleFile', filePath, selected });
        }

        function toggleAll(selected) {
            vscode.postMessage({ type: 'toggleAll', selected });
        }

        function showFileDiff(filePath) {
            vscode.postMessage({ type: 'showFileDiff', filePath });
        }

        function applySelected() {
            vscode.postMessage({ type: 'applySelected' });
        }

        function rejectAll() {
            vscode.postMessage({ type: 'rejectAll' });
        }
    </script>
</body>
</html>`;
    }

    /**
     * Show all pending changes in a quick pick
     */
    async showPendingChanges(): Promise<void> {
        if (this._pendingSessions.size === 0) {
            vscode.window.showInformationMessage('No pending changes');
            return;
        }

        const items: vscode.QuickPickItem[] = [];

        for (const session of this._pendingSessions.values()) {
            items.push({
                label: `$(git-commit) ${session.description}`,
                description: `${session.changes.length} file(s)`,
                detail: session.changes.map(c => `  ${this._getChangeIcon(c.changeType)} ${c.filePath}`).join('\n'),
            });

            // Add individual file items
            for (const change of session.changes) {
                items.push({
                    label: `    ${this._getChangeIcon(change.changeType)} ${path.basename(change.filePath)}`,
                    description: change.filePath,
                    detail: change.description,
                });
            }
        }

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select a change to preview',
            matchOnDescription: true,
        });

        if (selected?.description) {
            // Find the change and show diff
            for (const session of this._pendingSessions.values()) {
                const change = session.changes.find(c => c.filePath === selected.description);
                if (change) {
                    await this.showDiff(change);
                    break;
                }
            }
        }
    }

    /**
     * Apply a single file change
     */
    async applyChange(change: FileChange): Promise<boolean> {
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';
        try {
            const absolutePath = await resolveWorkspaceFile(workspaceRoot, change.filePath);
            const uri = vscode.Uri.file(absolutePath);

            switch (change.changeType) {
                case 'create':
                case 'modify': {
                    const edit = new vscode.WorkspaceEdit();
                    if (change.changeType === 'create') {
                        edit.createFile(uri, { overwrite: false, ignoreIfExists: false });
                    }
                    edit.replace(
                        uri,
                        new vscode.Range(0, 0, Number.MAX_VALUE, 0),
                        change.newContent
                    );
                    if (!await vscode.workspace.applyEdit(edit)) { return false; }
                    break;
                }

                case 'delete':
                    await vscode.workspace.fs.delete(uri);
                    break;
            }

            return true;
        } catch (error) {
            vscode.window.showErrorMessage(`Failed to apply change to ${change.filePath}: ${error}`);
            return false;
        }
    }

    /**
     * Apply all changes in a session
     */
    async applySession(sessionId: string): Promise<boolean> {
        const session = this._pendingSessions.get(sessionId);
        if (!session) {
            vscode.window.showErrorMessage('Session not found');
            return false;
        }

        const confirm = await vscode.window.showWarningMessage(
            `Apply ${session.changes.length} change(s)?`,
            { modal: true },
            'Apply All',
            'Review First'
        );

        if (confirm === 'Review First') {
            await this.showPendingChanges();
            return false;
        }

        if (confirm !== 'Apply All') {
            return false;
        }

        let successCount = 0;
        for (const change of session.changes) {
            if (await this.applyChange(change)) {
                successCount++;
            }
        }

        if (successCount === session.changes.length) {
            vscode.window.showInformationMessage(`Applied all ${successCount} changes`);
            this._pendingSessions.delete(sessionId);
            this._updateStatusBar();
            return true;
        } else {
            vscode.window.showWarningMessage(
                `Applied ${successCount}/${session.changes.length} changes`
            );
            return false;
        }
    }

    /**
     * Reject/discard a session
     */
    rejectSession(sessionId: string): void {
        if (this._pendingSessions.delete(sessionId)) {
            this._updateStatusBar();
            vscode.window.showInformationMessage('Changes discarded');
        }
    }

    /**
     * Get all pending sessions
     */
    getPendingSessions(): DiffSession[] {
        return Array.from(this._pendingSessions.values());
    }

    /**
     * Parse file changes from AI response
     */
    parseChangesFromResponse(response: string): FileChange[] {
        const changes: FileChange[] = [];

        // Match code blocks with file paths
        const codeBlockRegex = /```(?:(\w+)\s+)?(?:([^\n]+\.[\w]+)\n)?([\s\S]*?)```/g;
        let match;

        while ((match = codeBlockRegex.exec(response)) !== null) {
            const language = match[1];
            const filePath = match[2];
            const content = match[3]?.trim();

            if (filePath && content) {
                changes.push({
                    filePath,
                    originalContent: '', // Will be fetched when applying
                    newContent: content,
                    changeType: 'modify', // Default to modify
                    description: `${language || 'code'} changes`,
                });
            }
        }

        // Match diff blocks
        const diffRegex = /```diff\n([\s\S]*?)```/g;
        while ((match = diffRegex.exec(response)) !== null) {
            const diffContent = match[1];
            const fileMatch = diffContent.match(/^(?:---|\+\+\+)\s+([^\n]+)/m);
            if (fileMatch) {
                changes.push({
                    filePath: fileMatch[1].replace(/^[ab]\//, ''),
                    originalContent: '',
                    newContent: diffContent,
                    changeType: 'modify',
                    description: 'Diff patch',
                });
            }
        }

        return changes;
    }

    private _getLanguageId(filePath: string): string {
        const ext = path.extname(filePath).toLowerCase();
        const languageMap: Record<string, string> = {
            '.ts': 'typescript',
            '.tsx': 'typescriptreact',
            '.js': 'javascript',
            '.jsx': 'javascriptreact',
            '.py': 'python',
            '.rs': 'rust',
            '.go': 'go',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.md': 'markdown',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sql': 'sql',
        };
        return languageMap[ext] || 'plaintext';
    }

    private _getChangeIcon(changeType: FileChange['changeType']): string {
        switch (changeType) {
            case 'create': return '$(new-file)';
            case 'modify': return '$(edit)';
            case 'delete': return '$(trash)';
            default: return '$(file)';
        }
    }

    private _updateStatusBar(): void {
        const totalChanges = Array.from(this._pendingSessions.values())
            .reduce((sum, s) => sum + s.changes.length, 0);

        if (totalChanges > 0) {
            this._statusBarItem.text = `$(git-compare) ${totalChanges} pending`;
            this._statusBarItem.tooltip = `${totalChanges} pending file change(s)\nClick to review`;
            this._statusBarItem.show();
        } else {
            this._statusBarItem.hide();
        }
    }

    dispose(): void {
        this._decorationType.dispose();
        this._statusBarItem.dispose();
    }
}

/**
 * Content provider for original file content in diffs
 */
export class OriginalContentProvider implements vscode.TextDocumentContentProvider {
    private _originalContents: Map<string, string> = new Map();

    provideTextDocumentContent(uri: vscode.Uri): string {
        return this._originalContents.get(uri.path) || '';
    }

    setOriginalContent(filePath: string, content: string): void {
        this._originalContents.set(filePath, content);
    }

    clearOriginalContent(filePath: string): void {
        this._originalContents.delete(filePath);
    }
}

/**
 * Register diff view commands
 */
export function registerDiffCommands(
    context: vscode.ExtensionContext,
    diffProvider: DiffViewProvider
): void {
    // Show pending changes (quick pick)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.showPendingChanges', () => {
            diffProvider.showPendingChanges();
        })
    );

    // Review changes in panel (multi-file preview)
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.reviewChanges', async () => {
            const sessions = diffProvider.getPendingSessions();
            if (sessions.length === 0) {
                vscode.window.showInformationMessage('No pending changes to review');
                return;
            }

            // Show the most recent session or let user pick
            if (sessions.length === 1) {
                await diffProvider.showMultiFileDiffPanel(sessions[0].id, context);
            } else {
                const items = sessions.map(s => ({
                    label: s.description,
                    description: `${s.changes.length} file(s)`,
                    detail: s.timestamp.toLocaleString(),
                    sessionId: s.id,
                }));

                const selected = await vscode.window.showQuickPick(items, {
                    placeHolder: 'Select a change set to review',
                });

                if (selected) {
                    await diffProvider.showMultiFileDiffPanel(selected.sessionId, context);
                }
            }
        })
    );

    // Apply all pending changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.applyAllChanges', async () => {
            const sessions = diffProvider.getPendingSessions();
            if (sessions.length === 0) {
                vscode.window.showInformationMessage('No pending changes to apply');
                return;
            }

            for (const session of sessions) {
                await diffProvider.applySession(session.id);
            }
        })
    );

    // Reject all pending changes
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.rejectAllChanges', () => {
            const sessions = diffProvider.getPendingSessions();
            if (sessions.length === 0) {
                vscode.window.showInformationMessage('No pending changes to reject');
                return;
            }

            for (const session of sessions) {
                diffProvider.rejectSession(session.id);
            }
        })
    );

    // Register content provider for original files
    const contentProvider = new OriginalContentProvider();
    context.subscriptions.push(
        vscode.workspace.registerTextDocumentContentProvider('victor-original', contentProvider)
    );
}
