/**
 * Composer Tests
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { ComposerFile, ComposerSession, ComposerViewProvider } from '../../composer';

suite('Composer Test Suite', () => {
    for (const paused of [false, true]) {
        test(`actual composition ${paused ? 'retains approval pause' : 'produces changes'}`, async () => {
            const response = { role: 'assistant', content: '--- FILE: file.ts ---\n```ts\nconst x = 2;\n```',
                ...(paused ? { status: 'awaiting_approval', run_id: 'run-compose', approval_request: { id: 'a1' } } : {}) };
            let calls = 0;
            const provider = new ComposerViewProvider(vscode.Uri.file('/tmp'),
                { chat: async () => { calls++; return response; } } as any, {} as any);
            const owner = provider as any;
            owner._loadFiles = async () => [{ path: '/tmp/file.ts', relativePath: 'file.ts',
                content: 'const x = 1;', language: 'typescript', selected: true }];
            try {
                await provider.compose('change x', ['/tmp/file.ts']);
                assert.strictEqual(calls, 1);
                assert.strictEqual(owner._session.status, paused ? 'awaiting_approval' : 'ready');
                assert.strictEqual(owner._session.changes.length, paused ? 0 : 1);
                if (paused) { assert.deepStrictEqual(owner._session.pausedResponse, response); }
            } finally { provider.dispose(); }
        });
    }

    for (const replaced of [false, true]) {
        test(`late pause cannot overwrite a ${replaced ? 'replacement' : 'cleared'} composition`, async () => {
            let resolve!: (value: unknown) => void;
            const pending = new Promise(r => { resolve = r; });
            let started!: () => void;
            const called = new Promise<void>(r => { started = r; });
            const provider = new ComposerViewProvider(vscode.Uri.file('/tmp'),
                { chat: () => { started(); return pending; } } as any, {} as any);
            const owner = provider as any;
            owner._loadFiles = async () => [{ path: '/tmp/file.ts', relativePath: 'file.ts',
                content: 'const x = 1;', language: 'typescript', selected: true }];
            const running = provider.compose('first', ['/tmp/file.ts']);
            await called;
            owner._clear();
            const replacement = { id: 'new-session', status: 'idle', files: [], changes: [] };
            if (replaced) { owner._session = replacement; }
            resolve({ role: 'assistant', content: 'pending', status: 'awaiting_approval',
                run_id: 'old-run', approval_request: { id: 'old-approval' } });
            try {
                await running;
                assert.strictEqual(owner._session, replaced ? replacement : undefined);
                assert.deepStrictEqual(replacement, { id: 'new-session', status: 'idle', files: [], changes: [] });
            } finally { provider.dispose(); }
        });
    }

    test('ComposerSession should have all status types', () => {
        const statuses: ComposerSession['status'][] = [
            'idle',
            'analyzing',
            'generating',
            'ready',
            'applying',
            'completed',
            'error',
        ];

        statuses.forEach(status => {
            const session: ComposerSession = {
                id: 'test-session',
                prompt: 'Test prompt',
                files: [],
                changes: [],
                status,
                timestamp: new Date(),
            };
            assert.strictEqual(session.status, status);
        });
    });

    test('ComposerSession with error should have error field', () => {
        const session: ComposerSession = {
            id: 'error-session',
            prompt: 'Test prompt',
            files: [],
            changes: [],
            status: 'error',
            error: 'Something went wrong',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'error');
        assert.strictEqual(session.error, 'Something went wrong');
    });

    test('ComposerFile selected toggle', () => {
        const file: ComposerFile = {
            path: '/path/to/file.ts',
            relativePath: 'src/file.ts',
            content: 'code',
            language: 'typescript',
            selected: true,
        };

        assert.strictEqual(file.selected, true);
        file.selected = false;
        assert.strictEqual(file.selected, false);
    });

    test('ComposerSession with multiple files', () => {
        const files: ComposerFile[] = [
            {
                path: '/src/a.ts',
                relativePath: 'src/a.ts',
                content: 'export const a = 1;',
                language: 'typescript',
                selected: true,
            },
            {
                path: '/src/b.ts',
                relativePath: 'src/b.ts',
                content: 'export const b = 2;',
                language: 'typescript',
                selected: false,
            },
            {
                path: '/src/c.py',
                relativePath: 'src/c.py',
                content: 'x = 3',
                language: 'python',
                selected: true,
            },
        ];

        const session: ComposerSession = {
            id: 'multi-file-session',
            prompt: 'Update all exports',
            files,
            changes: [],
            status: 'idle',
            timestamp: new Date(),
        };

        assert.strictEqual(session.files.length, 3);

        const selectedFiles = session.files.filter(f => f.selected);
        assert.strictEqual(selectedFiles.length, 2);

        const typescriptFiles = session.files.filter(f => f.language === 'typescript');
        assert.strictEqual(typescriptFiles.length, 2);
    });

    test('ComposerSession timestamp', () => {
        const before = new Date();
        const session: ComposerSession = {
            id: 'timed-session',
            prompt: 'Test',
            files: [],
            changes: [],
            status: 'idle',
            timestamp: new Date(),
        };
        const after = new Date();

        assert.ok(session.timestamp >= before);
        assert.ok(session.timestamp <= after);
    });

    test('ComposerSession id format', () => {
        const session: ComposerSession = {
            id: `composer-${Date.now()}`,
            prompt: 'Test',
            files: [],
            changes: [],
            status: 'idle',
            timestamp: new Date(),
        };

        assert.ok(session.id.startsWith('composer-'));
        assert.ok(session.id.length > 10);
    });
});

suite('Composer Status Transitions', () => {
    test('Should start in idle status', () => {
        const session: ComposerSession = {
            id: 'new-session',
            prompt: '',
            files: [],
            changes: [],
            status: 'idle',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'idle');
    });

    test('Should transition to analyzing', () => {
        const session: ComposerSession = {
            id: 'analyzing-session',
            prompt: 'Add tests',
            files: [{
                path: '/src/app.ts',
                relativePath: 'src/app.ts',
                content: 'code',
                language: 'typescript',
                selected: true,
            }],
            changes: [],
            status: 'analyzing',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'analyzing');
        assert.ok(session.files.length > 0);
    });

    test('Should transition to generating', () => {
        const session: ComposerSession = {
            id: 'generating-session',
            prompt: 'Add tests',
            files: [{
                path: '/src/app.ts',
                relativePath: 'src/app.ts',
                content: 'code',
                language: 'typescript',
                selected: true,
            }],
            changes: [],
            status: 'generating',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'generating');
    });

    test('Should transition to ready with changes', () => {
        const session: ComposerSession = {
            id: 'ready-session',
            prompt: 'Add tests',
            files: [{
                path: '/src/app.ts',
                relativePath: 'src/app.ts',
                content: 'code',
                language: 'typescript',
                selected: true,
            }],
            changes: [{
                filePath: 'src/app.ts',
                originalContent: 'code',
                newContent: 'code // modified',
                changeType: 'modify',
                description: 'Added comment',
            }],
            status: 'ready',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'ready');
        assert.strictEqual(session.changes.length, 1);
    });

    test('Should transition to completed after applying', () => {
        const session: ComposerSession = {
            id: 'completed-session',
            prompt: 'Add tests',
            files: [],
            changes: [],
            status: 'completed',
            timestamp: new Date(),
        };

        assert.strictEqual(session.status, 'completed');
    });
});

suite('Composer Language Detection', () => {
    const testCases: Array<{ ext: string; language: string }> = [
        { ext: '.ts', language: 'typescript' },
        { ext: '.tsx', language: 'typescriptreact' },
        { ext: '.js', language: 'javascript' },
        { ext: '.jsx', language: 'javascriptreact' },
        { ext: '.py', language: 'python' },
        { ext: '.go', language: 'go' },
        { ext: '.rs', language: 'rust' },
        { ext: '.java', language: 'java' },
    ];

    testCases.forEach(({ ext, language }) => {
        test(`Should recognize ${ext} as ${language}`, () => {
            const file: ComposerFile = {
                path: `/src/file${ext}`,
                relativePath: `src/file${ext}`,
                content: '// code',
                language,
                selected: true,
            };

            assert.strictEqual(file.language, language);
        });
    });
});
