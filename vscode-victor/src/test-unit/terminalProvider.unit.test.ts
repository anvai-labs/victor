import * as vscode from 'vscode';
import { afterEach, expect, it, vi } from 'vitest';
import { TerminalProvider } from '../terminalProvider';

afterEach(() => vi.restoreAllMocks());

it('passes a special-character working directory as data, never terminal commands', async () => {
    const sent: string[] = [];
    const create = vi.spyOn(vscode.window, 'createTerminal').mockReturnValue({
        sendText: (text: string) => sent.push(text), show: () => {}, dispose: () => {},
    } as unknown as vscode.Terminal);
    const provider = new TerminalProvider();
    const cwd = '/workspace/path$(echo marker)with"quotes';
    try {
        const execution = await provider.executeCommand('git status', { cwd, requireApproval: false });
        expect(execution.status).toBe('completed');
        expect(create).toHaveBeenCalledWith(expect.objectContaining({ cwd }));
        expect(sent).toEqual(['git status']);
    } finally {
        provider.dispose();
    }
});
