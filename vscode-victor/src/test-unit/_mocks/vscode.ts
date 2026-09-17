// Lightweight mock of the subset of the `vscode` API used by the extension's
// logic modules (state store, serverManager, eventBridgeClient, services), so
// they can be unit-tested in a plain Node process via vitest's `vscode` alias.
// Extend this as more modules are brought under unit coverage.

export class EventEmitter<T> {
    private listeners: Array<(e: T) => void> = [];
    event = (listener: (e: T) => void) => {
        this.listeners.push(listener);
        return { dispose: () => { this.listeners = this.listeners.filter((l) => l !== listener); } };
    };
    fire(data: T): void {
        for (const l of [...this.listeners]) {
            l(data);
        }
    }
    dispose(): void {
        this.listeners = [];
    }
}

export class Disposable {
    constructor(private readonly callOnDispose: () => void) {}
    dispose(): void {
        this.callOnDispose?.();
    }
    static from(...disposables: Array<{ dispose: () => unknown }>): Disposable {
        return new Disposable(() => disposables.forEach((d) => d.dispose()));
    }
}

const _configStore = new Map<string, unknown>();

export const workspace = {
    applyEdit: async (_edit: unknown): Promise<boolean> => true,
    getConfiguration: (_section?: string) => ({
        get: <T>(key: string, defaultValue?: T): T | undefined =>
            (_configStore.has(key) ? (_configStore.get(key) as T) : defaultValue),
        update: async (key: string, value: unknown): Promise<void> => { _configStore.set(key, value); },
        has: (key: string): boolean => _configStore.has(key),
        inspect: (_key: string) => undefined,
    }),
    onDidChangeConfiguration: (_listener: (e: unknown) => void) => ({ dispose: () => {} }),
    workspaceFolders: undefined as unknown,
    // Test helper (not part of real vscode): seed config values.
    __setConfig: (key: string, value: unknown) => _configStore.set(key, value),
    __clearConfig: () => _configStore.clear(),
};

export class TreeItem {
    constructor(public label: string, public collapsibleState?: number) {}
}
export enum TreeItemCollapsibleState { None, Collapsed, Expanded }
export class WorkspaceEdit {
    createFile(_uri: unknown, _options?: unknown): void {}
    replace(_uri: unknown, _range: unknown, _text: string): void {}
}
export class Range {
    constructor(..._args: unknown[]) {}
}

export class ThemeIcon {
    constructor(public readonly id: string) {}
}

export const window = {
    createTerminal: (_options: unknown) => ({
        sendText: (_text: string, _newline?: boolean) => {},
        show: (_preserveFocus?: boolean) => {},
        dispose: () => {},
    }),
    onDidCloseTerminal: (_listener: (terminal: unknown) => void) => ({ dispose: () => {} }),
    createOutputChannel: (name: string) => ({
        name,
        appendLine: (_value: string) => {},
        append: (_value: string) => {},
        clear: () => {},
        replace: (_value: string) => {},
        show: (_preserveFocus?: boolean) => {},
        hide: () => {},
        dispose: () => {},
    }),
    showInformationMessage: async (..._args: unknown[]): Promise<undefined> => undefined,
    showWarningMessage: async (..._args: unknown[]): Promise<undefined> => undefined,
    showErrorMessage: async (..._args: unknown[]): Promise<undefined> => undefined,
    withProgress: async <T>(_opts: unknown, task: (...a: unknown[]) => Promise<T>): Promise<T> =>
        task({ report: () => {} }, { isCancellationRequested: false, onCancellationRequested: () => ({ dispose: () => {} }) }),
};

export enum ConfigurationTarget {
    Global = 1,
    Workspace = 2,
    WorkspaceFolder = 3,
}

export enum StatusBarAlignment {
    Left = 1,
    Right = 2,
}

export const commands = {
    executeCommand: async (..._args: unknown[]): Promise<undefined> => undefined,
    registerCommand: (_id: string, _cb: (...a: unknown[]) => unknown) => ({ dispose: () => {} }),
};

export const Uri = {
    parse: (value: string) => ({ toString: () => value, fsPath: value }),
    file: (value: string) => ({ toString: () => value, fsPath: value }),
};
