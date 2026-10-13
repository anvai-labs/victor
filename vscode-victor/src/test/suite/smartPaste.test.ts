/**
 * Smart Paste Tests
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { ChatOutcomeError } from '../../victorClient';
import { PasteContext, AdaptedCode, SmartPasteProvider } from '../../smartPaste';

suite('SmartPaste Test Suite', () => {
    test('PasteContext interface', () => {
        const context: PasteContext = {
            language: 'typescript',
            imports: ['import * as vscode from "vscode"'],
            surroundingCode: 'function test() {\n  // code\n}',
            indentation: '    ',
            cursorPosition: new vscode.Position(5, 4),
        };

        assert.strictEqual(context.language, 'typescript');
        assert.strictEqual(context.imports.length, 1);
        assert.strictEqual(context.indentation, '    ');
        assert.strictEqual(context.cursorPosition.line, 5);
    });

    for (const outcome of ['complete', 'paused', 'invalid']) {
        const paused = outcome !== 'complete';
        test(`actual Smart Paste outcome ${outcome} controls insertion and fallback`, async () => {
            const doc = await vscode.workspace.openTextDocument({ language: 'typescript', content: '' });
            await vscode.window.showTextDocument(doc);
            const savedClipboard = await vscode.env.clipboard.readText();
            const quickPick = vscode.window.showQuickPick;
            const warning = vscode.window.showWarningMessage;
            const warnings: string[] = [];
            let calls = 0;
            (vscode.window as any).showQuickPick = async () => ({ value: 'smart' });
            (vscode.window as any).showWarningMessage = async (message: string) => { warnings.push(message); };
            // Avoid registering the already-active extension's command; execute the real method.
            const provider = Object.create(SmartPasteProvider.prototype) as SmartPasteProvider;
            Object.assign(provider, { _client: { chat: async () => {
                calls++;
                if (outcome === 'invalid') { throw new ChatOutcomeError('Invalid approval metadata'); }
                return { role: 'assistant', content: 'const adapted = 2;', ...(paused ? {
                    status: 'awaiting_approval', run_id: 'run-paste', approval_request: { id: 'a1' }
                } : {}) };
            } }, _isProcessing: false });
            try {
                await vscode.env.clipboard.writeText('const original = 1;');
                await provider.smartPaste();
                assert.strictEqual(calls, 1);
                assert.strictEqual(doc.getText(), paused ? '' : 'const adapted = 2;');
                assert.strictEqual((provider as any)._isProcessing, false);
                if (paused) { assert.ok(warnings.some(message => message.includes(outcome === 'invalid' ? 'Invalid approval metadata' : 'run-paste'))); }
            } finally {
                (vscode.window as any).showQuickPick = quickPick;
                (vscode.window as any).showWarningMessage = warning;
                await vscode.env.clipboard.writeText(savedClipboard);
                await vscode.commands.executeCommand('workbench.action.revertAndCloseActiveEditor');
            }
        });
    }

    test('AdaptedCode with multiple changes', () => {
        const adapted: AdaptedCode = {
            original: 'function foo(a, b) { return a + b }',
            adapted: 'const foo = (a: number, b: number): number => a + b;',
            changes: [
                'Converted to arrow function',
                'Added type annotations',
                'Updated formatting',
            ],
        };

        assert.strictEqual(adapted.changes.length, 3);
        assert.ok(adapted.changes.includes('Added type annotations'));
    });

    test('AdaptedCode with no changes', () => {
        const adapted: AdaptedCode = {
            original: 'const x = 1;',
            adapted: 'const x = 1;',
            changes: ['No content changes, only formatting'],
        };

        assert.strictEqual(adapted.original.trim(), adapted.adapted.trim());
        assert.strictEqual(adapted.changes.length, 1);
    });
});

suite('PasteContext Language Detection', () => {
    const languages = [
        'typescript',
        'javascript',
        'python',
        'go',
        'rust',
        'java',
        'csharp',
        'cpp',
    ];

    languages.forEach(lang => {
        test(`Should handle ${lang} context`, () => {
            const context: PasteContext = {
                language: lang,
                imports: [],
                surroundingCode: '',
                indentation: '',
                cursorPosition: new vscode.Position(0, 0),
            };

            assert.strictEqual(context.language, lang);
        });
    });
});

suite('Import Pattern Detection', () => {
    test('Detect JavaScript/TypeScript imports', () => {
        const imports = [
            'import * as vscode from "vscode"',
            'import { Component } from "react"',
            "import React from 'react'",
        ];

        imports.forEach(imp => {
            assert.ok(imp.startsWith('import'));
        });
    });

    test('Detect Python imports', () => {
        const imports = [
            'import os',
            'from typing import List',
            'from pathlib import Path',
        ];

        imports.forEach(imp => {
            assert.ok(imp.startsWith('import') || imp.startsWith('from'));
        });
    });

    test('Detect Rust imports', () => {
        const imports = [
            'use std::collections::HashMap;',
            'use serde::{Serialize, Deserialize};',
        ];

        imports.forEach(imp => {
            assert.ok(imp.startsWith('use'));
        });
    });
});

suite('Indentation Handling', () => {
    test('Detect space indentation', () => {
        const context: PasteContext = {
            language: 'typescript',
            imports: [],
            surroundingCode: '',
            indentation: '    ',  // 4 spaces
            cursorPosition: new vscode.Position(0, 0),
        };

        assert.strictEqual(context.indentation.length, 4);
        assert.ok(!context.indentation.includes('\t'));
    });

    test('Detect tab indentation', () => {
        const context: PasteContext = {
            language: 'typescript',
            imports: [],
            surroundingCode: '',
            indentation: '\t',
            cursorPosition: new vscode.Position(0, 0),
        };

        assert.strictEqual(context.indentation, '\t');
    });

    test('Detect mixed indentation', () => {
        const context: PasteContext = {
            language: 'typescript',
            imports: [],
            surroundingCode: '',
            indentation: '\t  ',  // tab + 2 spaces
            cursorPosition: new vscode.Position(0, 0),
        };

        assert.ok(context.indentation.includes('\t'));
        assert.ok(context.indentation.includes(' '));
    });

    test('Handle no indentation', () => {
        const context: PasteContext = {
            language: 'typescript',
            imports: [],
            surroundingCode: '',
            indentation: '',
            cursorPosition: new vscode.Position(0, 0),
        };

        assert.strictEqual(context.indentation, '');
    });
});

suite('Code Detection Heuristics', () => {
    const codeSnippets = [
        { code: 'function test() {}', reason: 'function keyword' },
        { code: 'class MyClass {}', reason: 'class keyword' },
        { code: 'const x = 1;', reason: 'const keyword' },
        { code: 'let y = 2;', reason: 'let keyword' },
        { code: 'var z = 3;', reason: 'var keyword' },
        { code: 'def foo():', reason: 'def keyword' },
        { code: 'import os', reason: 'import keyword' },
        { code: 'return value;', reason: 'return keyword' },
        { code: 'if (x > 0) {}', reason: 'if statement' },
        { code: 'for (let i = 0; i < 10; i++) {}', reason: 'for loop' },
        { code: 'while (true) {}', reason: 'while loop' },
        { code: '() => {}', reason: 'arrow function' },
        { code: '// comment', reason: 'single-line comment' },
        { code: '# python comment', reason: 'hash comment' },
    ];

    codeSnippets.forEach(({ code, reason }) => {
        test(`Should detect code with ${reason}`, () => {
            // These patterns should indicate code
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
                /^\s*\/\//m,
                /^\s*#/m,
            ];

            const isCode = codeIndicators.some(pattern => pattern.test(code));
            assert.ok(isCode, `Expected "${code}" to be detected as code (${reason})`);
        });
    });

    test('Should not detect plain text as code', () => {
        const plainText = [
            'Hello world',
            'This is a document',
            'Just some text',
        ];

        const codeIndicators = [
            /\bfunction\b/,
            /\bclass\b/,
            /\bconst\b/,
            /\blet\b/,
            /\bvar\b/,
            /\bdef\b/,
        ];

        plainText.forEach(text => {
            const isCode = codeIndicators.some(pattern => pattern.test(text));
            assert.ok(!isCode, `Expected "${text}" to NOT be detected as code`);
        });
    });
});

suite('Adaptation Changes Detection', () => {
    test('Detect variable naming changes', () => {
        const original = 'const my_var = 1';
        const adapted = 'const myVar = 1';

        const changes: string[] = [];
        // Compare as strings explicitly to avoid TS literal type comparison
        if (original.valueOf() !== adapted.valueOf()) {
            changes.push('Variable naming updated');
        }

        assert.ok(changes.includes('Variable naming updated'));
    });

    test('Detect added imports', () => {
        const original = 'const fs = require("fs")';
        const adapted = 'import * as fs from "fs";\nconst data = fs.readFileSync()';

        const changes: string[] = [];
        if (adapted.includes('import') && !original.includes('import')) {
            changes.push('Added imports');
        }

        assert.ok(changes.includes('Added imports'));
    });

    test('Detect type annotation changes', () => {
        const original = 'const x = 1';
        const adapted = 'const x: number = 1';

        const changes: string[] = [];
        if (adapted.includes(':') && !original.includes(':')) {
            changes.push('Added type annotations');
        }

        assert.ok(changes.includes('Added type annotations'));
    });

    test('Detect async/await changes', () => {
        const original = 'function getData() { return fetch(url) }';
        const adapted = 'async function getData() { return await fetch(url) }';

        const changes: string[] = [];
        if (adapted.includes('async') && !original.includes('async')) {
            changes.push('Made function async');
        }

        assert.ok(changes.includes('Made function async'));
    });
});
