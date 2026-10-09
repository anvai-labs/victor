/**
 * Multi-file Composer
 *
 * Enables complex multi-file edits with a unified interface.
 * Similar to Cursor's Composer feature.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { VictorClient, requireCompletedChat, ChatApprovalRequiredError, PausedChatResponse } from './victorClient';
import { DiffViewProvider, FileChange } from './diffView';

export interface ComposerFile {
    path: string;
    relativePath: string;
    content: string;
    language: string;
    selected: boolean;
}

export interface ComposerSession {
    id: string;
    prompt: string;
    files: ComposerFile[];
    changes: FileChange[];
    status: 'idle' | 'analyzing' | 'generating' | 'ready' | 'applying' | 'completed' | 'error' | 'awaiting_approval';
    pausedResponse?: PausedChatResponse;
    error?: string;
    timestamp: Date;
}

/**
 * Multi-file Composer View Provider
 */
export class ComposerViewProvider implements vscode.WebviewViewProvider, vscode.Disposable {
    public static readonly viewType = 'victor.composerView';

    private _view?: vscode.WebviewView;
    private _session?: ComposerSession;
    private _disposables: vscode.Disposable[] = [];

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _client: VictorClient,
        private readonly _diffProvider: DiffViewProvider,
        private readonly _log?: vscode.OutputChannel
    ) {}

    dispose(): void {
        this._disposables.forEach(d => d.dispose());
        this._disposables = [];
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri],
        };

        webviewView.webview.html = this._getHtml(webviewView.webview);

        // Handle messages from webview
        webviewView.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.type) {
                    case 'compose':
                        await this.compose(message.prompt, message.files);
                        break;
                    case 'addFiles':
                        await this._addFiles();
                        break;
                    case 'removeFile':
                        this._removeFile(message.path);
                        break;
                    case 'toggleFile':
                        this._toggleFile(message.path, message.selected);
                        break;
                    case 'previewChange':
                        await this._previewChange(message.path);
                        break;
                    case 'applyChanges':
                        await this._applyChanges();
                        break;
                    case 'discardChanges':
                        this._discardChanges();
                        break;
                    case 'clear':
                        this._clear();
                        break;
                }
            },
            undefined,
            this._disposables
        );
    }

    /**
     * Start a new composition session
     */
    async compose(prompt: string, filePaths?: string[]): Promise<void> {
        if (!prompt.trim()) {
            this._postMessage({ type: 'error', message: 'Please enter a description of the changes' });
            return;
        }

        // Initialize session
        this._session = {
            id: `composer-${Date.now()}`,
            prompt,
            files: [],
            changes: [],
            status: 'analyzing',
            timestamp: new Date(),
        };

        const session = this._session;
        this._updateView();

        try {
            // Get files to include
            const files = filePaths?.length
                ? await this._loadFiles(filePaths)
                : await this._getContextFiles();
            if (this._session !== session) { return; }

            if (files.length === 0) {
                session.status = 'error';
                session.error = 'No files selected. Add files to the composer first.';
                this._updateView();
                return;
            }

            session.files = files;
            session.status = 'generating';
            this._updateView();

            // Generate changes using AI
            const changes = await this._generateChanges(prompt, files);
            if (this._session !== session) { return; }

            if (changes.length === 0) {
                session.status = 'error';
                session.error = 'No changes generated. Try a more specific prompt.';
                this._updateView();
                return;
            }

            session.changes = changes;
            session.status = 'ready';
            this._updateView();

            this._log?.appendLine(`[Composer] Generated ${changes.length} changes for ${files.length} files`);

        } catch (error) {
            if (this._session !== session) { return; }
            session.status = error instanceof ChatApprovalRequiredError ? 'awaiting_approval' : 'error';
            if (error instanceof ChatApprovalRequiredError) { session.pausedResponse = error.response; }
            session.error = error instanceof Error ? error.message : String(error);
            this._updateView();
            this._log?.appendLine(`[Composer] Error: ${error}`);
        }
    }

    /**
     * Open composer panel and optionally add files
     */
    async openWithFiles(files?: vscode.Uri[]): Promise<void> {
        // Focus the composer view
        await vscode.commands.executeCommand('victor.composerView.focus');

        if (files && files.length > 0) {
            const loadedFiles = await this._loadFiles(files.map(f => f.fsPath));
            if (this._session) {
                this._session.files = [...this._session.files, ...loadedFiles];
            } else {
                this._session = {
                    id: `composer-${Date.now()}`,
                    prompt: '',
                    files: loadedFiles,
                    changes: [],
                    status: 'idle',
                    timestamp: new Date(),
                };
            }
            this._updateView();
        }
    }

    // --- Private Methods ---

    private async _addFiles(): Promise<void> {
        const files = await vscode.window.showOpenDialog({
            canSelectMany: true,
            openLabel: 'Add to Composer',
            filters: {
                'Code Files': ['ts', 'tsx', 'js', 'jsx', 'py', 'java', 'go', 'rs', 'cpp', 'c', 'cs'],
                'All Files': ['*'],
            },
        });

        if (files && files.length > 0) {
            const loadedFiles = await this._loadFiles(files.map(f => f.fsPath));

            if (!this._session) {
                this._session = {
                    id: `composer-${Date.now()}`,
                    prompt: '',
                    files: [],
                    changes: [],
                    status: 'idle',
                    timestamp: new Date(),
                };
            }

            // Deduplicate
            const existingPaths = new Set(this._session.files.map(f => f.path));
            const newFiles = loadedFiles.filter(f => !existingPaths.has(f.path));
            this._session.files = [...this._session.files, ...newFiles];

            this._updateView();
        }
    }

    private _removeFile(filePath: string): void {
        if (this._session) {
            this._session.files = this._session.files.filter(f => f.path !== filePath);
            this._session.changes = this._session.changes.filter(c => c.filePath !== filePath);
            this._updateView();
        }
    }

    private _toggleFile(filePath: string, selected: boolean): void {
        if (this._session) {
            const file = this._session.files.find(f => f.path === filePath);
            if (file) {
                file.selected = selected;
                this._updateView();
            }
        }
    }

    private async _previewChange(filePath: string): Promise<void> {
        if (!this._session) { return; }

        const change = this._session.changes.find(c => c.filePath === filePath);
        if (change) {
            await this._diffProvider.showDiff(change);
        }
    }

    private async _applyChanges(): Promise<void> {
        if (!this._session || this._session.changes.length === 0) { return; }

        this._session.status = 'applying';
        this._updateView();

        try {
            // Create a diff session for the changes
            this._diffProvider.createSession(
                this._session.changes,
                `Composer: ${this._session.prompt.slice(0, 50)}...`
            );

            // Apply all changes
            let successCount = 0;
            for (const change of this._session.changes) {
                if (await this._diffProvider.applyChange(change)) {
                    successCount++;
                }
            }

            this._session.status = 'completed';
            this._updateView();

            vscode.window.showInformationMessage(
                `Composer: Applied ${successCount}/${this._session.changes.length} changes`,
                'Undo All'
            ).then(action => {
                if (action === 'Undo All') {
                    vscode.commands.executeCommand('workbench.action.files.revert');
                }
            });

            this._log?.appendLine(`[Composer] Applied ${successCount} changes`);

        } catch (error) {
            this._session.status = 'error';
            this._session.error = `Apply failed: ${error}`;
            this._updateView();
        }
    }

    private _discardChanges(): void {
        if (this._session) {
            this._session.changes = [];
            this._session.status = 'idle';
            this._updateView();
        }
    }

    private _clear(): void {
        this._session = undefined;
        this._updateView();
    }

    private async _loadFiles(filePaths: string[]): Promise<ComposerFile[]> {
        const files: ComposerFile[] = [];
        const failedFiles: string[] = [];
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';

        for (const filePath of filePaths) {
            try {
                const uri = vscode.Uri.file(filePath);
                const doc = await vscode.workspace.openTextDocument(uri);

                files.push({
                    path: filePath,
                    relativePath: path.relative(workspaceRoot, filePath),
                    content: doc.getText(),
                    language: doc.languageId,
                    selected: true,
                });
            } catch (error) {
                this._log?.appendLine(`[Composer] Failed to load file: ${filePath}`);
                failedFiles.push(path.basename(filePath));
            }
        }

        // Notify user if some files failed to load
        if (failedFiles.length > 0) {
            const message = failedFiles.length === 1
                ? `Could not load file: ${failedFiles[0]}`
                : `Could not load ${failedFiles.length} files: ${failedFiles.slice(0, 3).join(', ')}${failedFiles.length > 3 ? '...' : ''}`;
            vscode.window.showWarningMessage(`Composer: ${message}`);
        }

        return files;
    }

    private async _getContextFiles(): Promise<ComposerFile[]> {
        // Get files from current context (open editors, selection, etc.)
        const files: ComposerFile[] = [];
        const workspaceRoot = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || '';

        // Add currently open text editors
        for (const editor of vscode.window.visibleTextEditors) {
            if (editor.document.uri.scheme === 'file') {
                files.push({
                    path: editor.document.uri.fsPath,
                    relativePath: path.relative(workspaceRoot, editor.document.uri.fsPath),
                    content: editor.document.getText(),
                    language: editor.document.languageId,
                    selected: true,
                });
            }
        }

        return files;
    }

    private async _generateChanges(prompt: string, files: ComposerFile[]): Promise<FileChange[]> {
        const selectedFiles = files.filter(f => f.selected);

        if (selectedFiles.length === 0) {
            return [];
        }

        // Build context for AI
        const fileContext = selectedFiles.map(f => `
=== ${f.relativePath} ===
\`\`\`${f.language}
${f.content.slice(0, 5000)}${f.content.length > 5000 ? '\n// ... (truncated)' : ''}
\`\`\`
`).join('\n');

        const systemPrompt = `You are a code modification assistant. Given files and a request, generate the modified versions of the files.

For each file that needs changes, output it in this exact format:
--- FILE: relative/path/to/file.ext ---
\`\`\`language
<complete new file content>
\`\`\`

Only output files that need changes. Include the complete file content, not just the changes.`;

        const userPrompt = `Here are the files to modify:

${fileContext}

Request: ${prompt}

Generate the modified files.`;

        try {
            const response = await this._client.chat([
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userPrompt },
            ]);

            const content = requireCompletedChat(response).content || '';
            return this._parseGeneratedChanges(content, selectedFiles);

        } catch (error) {
            this._log?.appendLine(`[Composer] AI error: ${error}`);
            throw error;
        }
    }

    private _parseGeneratedChanges(content: string, files: ComposerFile[]): FileChange[] {
        const changes: FileChange[] = [];

        // Parse --- FILE: path --- blocks
        const fileBlockRegex = /---\s*FILE:\s*([^\n]+)\s*---\s*\n```\w*\n([\s\S]*?)```/g;
        let match;

        while ((match = fileBlockRegex.exec(content)) !== null) {
            const filePath = match[1].trim();
            const newContent = match[2].trim();

            // Find the original file
            const originalFile = files.find(f =>
                f.relativePath === filePath ||
                f.path.endsWith(filePath) ||
                filePath.endsWith(f.relativePath)
            );

            if (originalFile) {
                // Only add if content actually changed
                if (originalFile.content.trim() !== newContent) {
                    changes.push({
                        filePath: originalFile.relativePath,
                        originalContent: originalFile.content,
                        newContent,
                        changeType: 'modify',
                        description: 'Modified by Composer',
                    });
                }
            } else {
                // New file
                changes.push({
                    filePath,
                    originalContent: '',
                    newContent,
                    changeType: 'create',
                    description: 'Created by Composer',
                });
            }
        }

        return changes;
    }

    private _postMessage(message: Record<string, unknown>): void {
        this._view?.webview.postMessage(message);
    }

    private _updateView(): void {
        this._postMessage({
            type: 'update',
            session: this._session ? {
                ...this._session,
                files: this._session.files.map(f => ({
                    path: f.path,
                    relativePath: f.relativePath,
                    language: f.language,
                    selected: f.selected,
                    lines: f.content.split('\n').length,
                })),
                changes: this._session.changes.map(c => ({
                    filePath: c.filePath,
                    changeType: c.changeType,
                    linesAdded: c.linesAdded,
                    linesRemoved: c.linesRemoved,
                })),
            } : null,
        });
    }

    private _getHtml(_webview: vscode.Webview): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Composer</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
            color: var(--vscode-foreground);
            background: var(--vscode-editor-background);
            padding: 12px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }

        .header h2 {
            font-size: 14px;
            font-weight: 600;
            flex: 1;
        }

        .prompt-area {
            margin-bottom: 12px;
        }

        textarea {
            width: 100%;
            min-height: 80px;
            padding: 8px;
            border: 1px solid var(--vscode-input-border);
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border-radius: 4px;
            resize: vertical;
            font-family: inherit;
            font-size: inherit;
        }

        textarea:focus {
            outline: 1px solid var(--vscode-focusBorder);
        }

        .files-section {
            flex: 1;
            overflow: auto;
            margin-bottom: 12px;
        }

        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid var(--vscode-panel-border);
            margin-bottom: 8px;
        }

        .section-title {
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            color: var(--vscode-descriptionForeground);
        }

        .file-list {
            list-style: none;
        }

        .file-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 4px;
            border-radius: 3px;
            cursor: pointer;
        }

        .file-item:hover {
            background: var(--vscode-list-hoverBackground);
        }

        .file-item input[type="checkbox"] {
            flex-shrink: 0;
        }

        .file-name {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .file-meta {
            font-size: 11px;
            color: var(--vscode-descriptionForeground);
        }

        .file-remove {
            opacity: 0;
            cursor: pointer;
            color: var(--vscode-errorForeground);
        }

        .file-item:hover .file-remove {
            opacity: 1;
        }

        .changes-section {
            margin-bottom: 12px;
        }

        .change-item {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 4px;
            border-radius: 3px;
            cursor: pointer;
        }

        .change-item:hover {
            background: var(--vscode-list-hoverBackground);
        }

        .change-icon {
            width: 16px;
            text-align: center;
        }

        .change-icon.create { color: var(--vscode-gitDecoration-addedResourceForeground); }
        .change-icon.modify { color: var(--vscode-gitDecoration-modifiedResourceForeground); }
        .change-icon.delete { color: var(--vscode-gitDecoration-deletedResourceForeground); }

        .change-stats {
            font-size: 11px;
        }

        .change-stats .added { color: var(--vscode-gitDecoration-addedResourceForeground); }
        .change-stats .removed { color: var(--vscode-gitDecoration-deletedResourceForeground); }

        .actions {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }

        button {
            padding: 6px 12px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        button.primary {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
        }

        button.primary:hover {
            background: var(--vscode-button-hoverBackground);
        }

        button.secondary {
            background: var(--vscode-button-secondaryBackground);
            color: var(--vscode-button-secondaryForeground);
        }

        button.secondary:hover {
            background: var(--vscode-button-secondaryHoverBackground);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .status {
            padding: 8px;
            border-radius: 4px;
            margin-bottom: 12px;
            font-size: 12px;
        }

        .status.analyzing { background: var(--vscode-inputValidation-infoBackground); }
        .status.generating { background: var(--vscode-inputValidation-warningBackground); }
        .status.ready { background: var(--vscode-inputValidation-infoBackground); }
        .status.error { background: var(--vscode-inputValidation-errorBackground); }
        .status.completed { background: rgba(0, 200, 83, 0.2); }

        .empty-state {
            text-align: center;
            padding: 20px;
            color: var(--vscode-descriptionForeground);
        }

        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid var(--vscode-foreground);
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h2>Composer</h2>
        <button class="secondary" onclick="clear()" title="Clear">Clear</button>
    </div>

    <div class="prompt-area">
        <textarea id="prompt" placeholder="Describe the changes you want to make across files...&#10;&#10;Example: Add error handling to all API calls"></textarea>
    </div>

    <div id="status-area"></div>

    <div class="files-section">
        <div class="section-header">
            <span class="section-title">Files (<span id="file-count">0</span>)</span>
            <button class="secondary" onclick="addFiles()">+ Add Files</button>
        </div>
        <ul class="file-list" id="file-list"></ul>
    </div>

    <div class="changes-section" id="changes-section" style="display: none;">
        <div class="section-header">
            <span class="section-title">Changes (<span id="change-count">0</span>)</span>
        </div>
        <ul class="file-list" id="change-list"></ul>
    </div>

    <div class="actions">
        <button class="primary" id="compose-btn" onclick="compose()">
            <span class="codicon codicon-sparkle"></span>
            Compose
        </button>
        <button class="primary" id="apply-btn" onclick="applyChanges()" style="display: none;">
            Apply Changes
        </button>
        <button class="secondary" id="discard-btn" onclick="discardChanges()" style="display: none;">
            Discard
        </button>
    </div>

    <script>
        const vscode = acquireVsCodeApi();

        function compose() {
            const prompt = document.getElementById('prompt').value;
            vscode.postMessage({ type: 'compose', prompt });
        }

        function addFiles() {
            vscode.postMessage({ type: 'addFiles' });
        }

        function removeFile(path) {
            vscode.postMessage({ type: 'removeFile', path });
        }

        function toggleFile(path, selected) {
            vscode.postMessage({ type: 'toggleFile', path, selected });
        }

        function previewChange(path) {
            vscode.postMessage({ type: 'previewChange', path });
        }

        function applyChanges() {
            vscode.postMessage({ type: 'applyChanges' });
        }

        function discardChanges() {
            vscode.postMessage({ type: 'discardChanges' });
        }

        function clear() {
            document.getElementById('prompt').value = '';
            vscode.postMessage({ type: 'clear' });
        }

        // Handle messages from extension
        window.addEventListener('message', event => {
            const message = event.data;

            if (message.type === 'update') {
                updateUI(message.session);
            } else if (message.type === 'error') {
                alert(message.message);
            }
        });

        function updateUI(session) {
            const fileList = document.getElementById('file-list');
            const changeList = document.getElementById('change-list');
            const changesSection = document.getElementById('changes-section');
            const statusArea = document.getElementById('status-area');
            const composeBtn = document.getElementById('compose-btn');
            const applyBtn = document.getElementById('apply-btn');
            const discardBtn = document.getElementById('discard-btn');

            if (!session) {
                fileList.innerHTML = '<li class="empty-state">No files added. Click "Add Files" to start.</li>';
                document.getElementById('file-count').textContent = '0';
                changesSection.style.display = 'none';
                statusArea.innerHTML = '';
                composeBtn.style.display = '';
                applyBtn.style.display = 'none';
                discardBtn.style.display = 'none';
                return;
            }

            // Update file list
            document.getElementById('file-count').textContent = session.files.length;
            if (session.files.length === 0) {
                fileList.innerHTML = '<li class="empty-state">No files added. Click "Add Files" to start.</li>';
            } else {
                fileList.innerHTML = session.files.map(f => \`
                    <li class="file-item">
                        <input type="checkbox" \${f.selected ? 'checked' : ''} onchange="toggleFile('\${f.path}', this.checked)">
                        <span class="file-name" title="\${f.relativePath}">\${f.relativePath}</span>
                        <span class="file-meta">\${f.lines} lines</span>
                        <span class="file-remove" onclick="removeFile('\${f.path}')" title="Remove">x</span>
                    </li>
                \`).join('');
            }

            // Update status
            const statusMessages = {
                idle: '',
                analyzing: '<div class="status analyzing"><span class="spinner"></span> Analyzing files...</div>',
                generating: '<div class="status generating"><span class="spinner"></span> Generating changes...</div>',
                ready: '<div class="status ready">Changes ready for review</div>',
                applying: '<div class="status analyzing"><span class="spinner"></span> Applying changes...</div>',
                completed: '<div class="status completed">Changes applied successfully!</div>',
                error: '<div class="status error"></div>',
                awaiting_approval: '<div class="status analyzing"></div>',
            };
            statusArea.innerHTML = statusMessages[session.status] || '';
            if (session.status === 'error' || session.status === 'awaiting_approval') {
                statusArea.firstElementChild.textContent = session.error || 'An error occurred';
            }

            // Update changes
            if (session.changes.length > 0) {
                changesSection.style.display = '';
                document.getElementById('change-count').textContent = session.changes.length;
                changeList.innerHTML = session.changes.map(c => {
                    const icon = c.changeType === 'create' ? '+' : c.changeType === 'delete' ? '-' : '~';
                    return \`
                        <li class="change-item" onclick="previewChange('\${c.filePath}')">
                            <span class="change-icon \${c.changeType}">\${icon}</span>
                            <span class="file-name">\${c.filePath}</span>
                            <span class="change-stats">
                                \${c.linesAdded ? '<span class="added">+' + c.linesAdded + '</span>' : ''}
                                \${c.linesRemoved ? '<span class="removed">-' + c.linesRemoved + '</span>' : ''}
                            </span>
                        </li>
                    \`;
                }).join('');
            } else {
                changesSection.style.display = 'none';
            }

            // Update buttons
            const isProcessing = ['analyzing', 'generating', 'applying'].includes(session.status);
            composeBtn.disabled = isProcessing;
            composeBtn.style.display = session.status === 'ready' ? 'none' : '';
            applyBtn.style.display = session.status === 'ready' ? '' : 'none';
            discardBtn.style.display = session.status === 'ready' ? '' : 'none';
        }

        // Initialize
        updateUI(null);
    </script>
</body>
</html>`;
    }
}

/**
 * Register composer commands
 */
export function registerComposerCommands(
    context: vscode.ExtensionContext,
    provider: ComposerViewProvider
): void {
    context.subscriptions.push(
        vscode.commands.registerCommand('victor.openComposer', () => {
            vscode.commands.executeCommand('victor.composerView.focus');
        }),

        vscode.commands.registerCommand('victor.composeWithFiles', async () => {
            const activeEditor = vscode.window.activeTextEditor;
            if (activeEditor) {
                await provider.openWithFiles([activeEditor.document.uri]);
            } else {
                await provider.openWithFiles();
            }
        }),

        vscode.commands.registerCommand('victor.addToComposer', async (uri: vscode.Uri) => {
            if (uri) {
                await provider.openWithFiles([uri]);
            }
        })
    );
}
