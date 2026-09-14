import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import * as vscode from 'vscode';
import { DiffViewProvider } from '../diffView';
import { resolveWorkspaceFile } from '../workspacePaths';

let base: string;
let root: string;
beforeEach(async () => {
    base = await fs.mkdtemp(path.join(os.tmpdir(), 'victor-path-test-'));
    root = path.join(base, 'workspace');
    await fs.mkdir(root);
});
afterEach(async () => { await fs.rm(base, { recursive: true, force: true }); });

it('accepts nested new files and existing files within the workspace', async () => {
    expect(await resolveWorkspaceFile(root, 'new/deep/file.ts')).toBe(path.join(root, 'new/deep/file.ts'));
    await fs.writeFile(path.join(root, 'existing.ts'), 'content');
    expect(await resolveWorkspaceFile(root, 'existing.ts')).toBe(path.join(root, 'existing.ts'));
});

it('rejects parent traversal, absolute outside files and the workspace itself', async () => {
    for (const proposed of ['../../outside.txt', path.join(base, 'outside.txt'), '.']) {
        await expect(resolveWorkspaceFile(root, proposed)).rejects.toThrow('outside');
    }
});

it('rejects an existing symlink to an outside directory', async () => {
    const outside = path.join(base, 'outside');
    await fs.mkdir(outside);
    await fs.symlink(outside, path.join(root, 'link'), 'junction');
    await expect(resolveWorkspaceFile(root, 'link/new/file.txt')).rejects.toThrow('outside');
});

it('fails closed on a dangling symlink', async () => {
    await fs.symlink(path.join(base, 'missing'), path.join(root, 'dangling'), 'junction');
    await expect(resolveWorkspaceFile(root, 'dangling/file.txt')).rejects.toThrow();
});


it('refuses an outside Composer create before applying any editor mutation', async () => {
    const previous = vscode.workspace.workspaceFolders;
    Object.assign(vscode.workspace, { workspaceFolders: [{ uri: vscode.Uri.file(root) }] });
    const apply = vi.spyOn(vscode.workspace, 'applyEdit');
    try {
        const provider = Object.create(DiffViewProvider.prototype) as DiffViewProvider;
        expect(await provider.applyChange({
            filePath: '../../outside-marker.txt', originalContent: '',
            newContent: 'must not be written', changeType: 'create',
        })).toBe(false);
        expect(apply).not.toHaveBeenCalled();
    } finally {
        apply.mockRestore();
        Object.assign(vscode.workspace, { workspaceFolders: previous });
    }
});
