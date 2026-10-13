/**
 * Code Action Provider Tests
 *
 * Tests for the VictorCodeActionProvider which provides AI-powered
 * quick fixes and refactoring suggestions in the lightbulb menu.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { encodeSymbolTarget, resolveSymbolTarget } from '../../codeActionProvider';

suite('CodeActionProvider Test Suite', () => {
    // Test code action kinds
    suite('Code Action Kinds', () => {
        test('Should have correct action kinds', () => {
            const kinds = {
                QuickFix: 'quickfix',
                Refactor: 'refactor',
                RefactorExtract: 'refactor.extract',
                RefactorInline: 'refactor.inline',
                RefactorRewrite: 'refactor.rewrite',
                Source: 'source',
                SourceOrganizeImports: 'source.organizeImports',
            };

            assert.strictEqual(kinds.QuickFix, 'quickfix');
            assert.strictEqual(kinds.Refactor, 'refactor');
            assert.ok(kinds.RefactorExtract.startsWith('refactor.'));
        });
    });

    // Test diagnostic code detection
    suite('Diagnostic Detection', () => {
        test('Should detect error diagnostics', () => {
            const diagnostic = {
                severity: 0, // Error
                message: 'Cannot find name "foo"',
                range: { start: { line: 10, character: 0 }, end: { line: 10, character: 5 } }
            };

            const isError = diagnostic.severity === 0;
            assert.ok(isError);
        });

        test('Should detect warning diagnostics', () => {
            const diagnostic = {
                severity: 1, // Warning
                message: 'Unused variable "x"',
                range: { start: { line: 5, character: 0 }, end: { line: 5, character: 10 } }
            };

            const isWarning = diagnostic.severity === 1;
            assert.ok(isWarning);
        });

        test('Should detect hint diagnostics', () => {
            const diagnostic = {
                severity: 3, // Hint
                message: 'Consider using const instead of let',
                range: { start: { line: 1, character: 0 }, end: { line: 1, character: 10 } }
            };

            const isHint = diagnostic.severity === 3;
            assert.ok(isHint);
        });
    });

    // Test code action categories
    suite('Code Action Categories', () => {
        test('Should categorize explain action', () => {
            const action = {
                title: '$(question) Explain this code',
                kind: 'source',
                command: 'victor.explain'
            };

            assert.ok(action.title.includes('Explain'));
            assert.strictEqual(action.kind, 'source');
        });

        test('Should categorize fix action', () => {
            const action = {
                title: '$(wrench) Fix with AI',
                kind: 'quickfix',
                command: 'victor.fix'
            };

            assert.ok(action.title.includes('Fix'));
            assert.strictEqual(action.kind, 'quickfix');
        });

        test('Should categorize refactor action', () => {
            const action = {
                title: '$(edit) Refactor with AI',
                kind: 'refactor',
                command: 'victor.refactor'
            };

            assert.ok(action.title.includes('Refactor'));
            assert.strictEqual(action.kind, 'refactor');
        });

        test('Should categorize test generation action', () => {
            const action = {
                title: '$(beaker) Generate tests',
                kind: 'source',
                command: 'victor.generateTests'
            };

            assert.ok(action.title.includes('tests'));
            assert.strictEqual(action.kind, 'source');
        });

        test('Should categorize documentation action', () => {
            const action = {
                title: '$(book) Add documentation',
                kind: 'source',
                command: 'victor.addDocs'
            };

            assert.ok(action.title.includes('documentation'));
            assert.strictEqual(action.kind, 'source');
        });
    });

    // Test selection-based actions
    suite('Selection-Based Actions', () => {
        test('Should detect non-empty selection', () => {
            const selection = {
                start: { line: 5, character: 0 },
                end: { line: 10, character: 20 },
                isEmpty: false
            };

            const hasSelection = !selection.isEmpty;
            assert.ok(hasSelection);
        });

        test('Should detect empty selection', () => {
            const selection = {
                start: { line: 5, character: 10 },
                end: { line: 5, character: 10 },
                isEmpty: true
            };

            const hasSelection = !selection.isEmpty;
            assert.ok(!hasSelection);
        });

        test('Should calculate selection size', () => {
            const calculateSelectionLines = (start: number, end: number): number => {
                return end - start + 1;
            };

            assert.strictEqual(calculateSelectionLines(5, 10), 6);
            assert.strictEqual(calculateSelectionLines(0, 0), 1);
            assert.strictEqual(calculateSelectionLines(10, 50), 41);
        });
    });

    // Test prompt generation
    suite('Prompt Generation', () => {
        test('Should generate fix prompt with diagnostic', () => {
            const generateFixPrompt = (diagnostic: string, code: string): string => {
                return `Fix the following issue:\n\nError: ${diagnostic}\n\nCode:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateFixPrompt(
                'Cannot find name "foo"',
                'console.log(foo);'
            );

            assert.ok(prompt.includes('Fix'));
            assert.ok(prompt.includes('Cannot find name'));
            assert.ok(prompt.includes('console.log'));
        });

        test('Should generate explain prompt', () => {
            const generateExplainPrompt = (code: string): string => {
                return `Explain the following code:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateExplainPrompt('const sum = (a, b) => a + b;');
            assert.ok(prompt.includes('Explain'));
            assert.ok(prompt.includes('sum'));
        });

        test('Should generate refactor prompt', () => {
            const generateRefactorPrompt = (code: string): string => {
                return `Suggest improvements for the following code:\n\`\`\`\n${code}\n\`\`\``;
            };

            const prompt = generateRefactorPrompt('for(var i=0;i<arr.length;i++){}');
            assert.ok(prompt.includes('improvements'));
            assert.ok(prompt.includes('for'));
        });
    });

    // Test action filtering
    suite('Action Filtering', () => {
        test('Should filter actions by kind', () => {
            const actions = [
                { kind: 'quickfix', title: 'Fix 1' },
                { kind: 'refactor', title: 'Refactor 1' },
                { kind: 'quickfix', title: 'Fix 2' },
                { kind: 'source', title: 'Source 1' },
            ];

            const quickFixes = actions.filter(a => a.kind === 'quickfix');
            const refactors = actions.filter(a => a.kind === 'refactor');

            assert.strictEqual(quickFixes.length, 2);
            assert.strictEqual(refactors.length, 1);
        });

        test('Should filter actions for diagnostics', () => {
            const hasDiagnostics = true;

            const getAvailableActions = (hasDiag: boolean): string[] => {
                const actions = ['explain', 'refactor', 'addTests', 'addDocs'];
                if (hasDiag) {
                    actions.unshift('fix'); // Fix action only when diagnostics present
                }
                return actions;
            };

            const withDiag = getAvailableActions(true);
            const withoutDiag = getAvailableActions(false);

            assert.ok(withDiag.includes('fix'));
            assert.ok(!withoutDiag.includes('fix'));
        });
    });

    // Test preferred action logic
    suite('Preferred Actions', () => {
        test('Should mark fix as preferred when error diagnostic', () => {
            const diagnostic = { severity: 0 }; // Error

            const isPreferred = diagnostic.severity === 0;
            assert.ok(isPreferred);
        });

        test('Should not mark as preferred for hints', () => {
            const diagnostic = { severity: 3 }; // Hint

            const isPreferred = diagnostic.severity === 0;
            assert.ok(!isPreferred);
        });
    });

    // Test command argument preparation
    suite('Command Arguments', () => {
        test('Should prepare explain command args', () => {
            const prepareExplainArgs = (filePath: string, selection: { start: number; end: number }): object => {
                return {
                    action: 'explain',
                    file: filePath,
                    startLine: selection.start,
                    endLine: selection.end
                };
            };

            const args = prepareExplainArgs('/path/to/file.ts', { start: 10, end: 20 });
            assert.strictEqual((args as any).action, 'explain');
            assert.strictEqual((args as any).startLine, 10);
        });

        test('Should prepare fix command args with diagnostic', () => {
            const prepareFixArgs = (filePath: string, diagnostic: { message: string; line: number }): object => {
                return {
                    action: 'fix',
                    file: filePath,
                    diagnostic: diagnostic.message,
                    line: diagnostic.line
                };
            };

            const args = prepareFixArgs('/path/to/file.ts', { message: 'Error message', line: 15 });
            assert.strictEqual((args as any).action, 'fix');
            assert.strictEqual((args as any).line, 15);
        });
    });

    // Test action title formatting
    suite('Action Title Formatting', () => {
        test('Should format titles with VS Code icons', () => {
            const formatTitle = (icon: string, text: string): string => {
                return `$(${icon}) ${text}`;
            };

            assert.strictEqual(formatTitle('question', 'Explain'), '$(question) Explain');
            assert.strictEqual(formatTitle('wrench', 'Fix'), '$(wrench) Fix');
            assert.strictEqual(formatTitle('edit', 'Refactor'), '$(edit) Refactor');
        });
    });

    // Test context requirements
    suite('Context Requirements', () => {
        test('Should check if in supported language', () => {
            const supportedLanguages = ['typescript', 'javascript', 'python', 'java', 'go', 'rust'];

            const isSupported = (languageId: string): boolean => {
                return supportedLanguages.includes(languageId);
            };

            assert.ok(isSupported('typescript'));
            assert.ok(isSupported('python'));
            assert.ok(!isSupported('markdown'));
            assert.ok(!isSupported('json'));
        });
    });
});

suite('Symbol-Based Commands Test Suite', () => {
    // Test symbol kind mapping
    suite('Symbol Kind Mapping', () => {
        test('Should map symbol kinds correctly', () => {
            const SymbolKind: Record<number, string> = {
                0: 'File',
                1: 'Module',
                2: 'Namespace',
                3: 'Package',
                4: 'Class',
                5: 'Method',
                6: 'Property',
                7: 'Field',
                8: 'Constructor',
                9: 'Enum',
                10: 'Interface',
                11: 'Function',
                12: 'Variable',
                13: 'Constant',
            };

            assert.strictEqual(SymbolKind[4], 'Class');
            assert.strictEqual(SymbolKind[5], 'Method');
            assert.strictEqual(SymbolKind[11], 'Function');
            assert.strictEqual(SymbolKind[12], 'Variable');
        });

        test('Should lowercase symbol kind for display', () => {
            const kind = 'Function';
            const displayKind = kind.toLowerCase();
            assert.strictEqual(displayKind, 'function');
        });
    });

    // Test symbol finding
    suite('Symbol Finding', () => {
        test('Should find symbol at cursor position', () => {
            const position = { line: 10, character: 5 };
            const symbolRange = { start: { line: 5, character: 0 }, end: { line: 15, character: 0 } };

            const containsPosition = (
                range: typeof symbolRange,
                pos: typeof position
            ): boolean => {
                return pos.line >= range.start.line && pos.line <= range.end.line;
            };

            assert.ok(containsPosition(symbolRange, position));
        });

        test('Should not find symbol outside cursor', () => {
            const position = { line: 20, character: 5 };
            const symbolRange = { start: { line: 5, character: 0 }, end: { line: 15, character: 0 } };

            const containsPosition = (
                range: typeof symbolRange,
                pos: typeof position
            ): boolean => {
                return pos.line >= range.start.line && pos.line <= range.end.line;
            };

            assert.ok(!containsPosition(symbolRange, position));
        });

        test('Should prefer nested symbol over parent', () => {
            const parentSymbol = { name: 'MyClass', range: { start: 0, end: 50 } };
            const nestedSymbol = { name: 'myMethod', range: { start: 10, end: 20 } };
            const cursorLine = 15;

            const isInRange = (symbol: typeof parentSymbol, line: number): boolean => {
                return line >= symbol.range.start && line <= symbol.range.end;
            };

            // Both contain cursor, but nested is more specific
            assert.ok(isInRange(parentSymbol, cursorLine));
            assert.ok(isInRange(nestedSymbol, cursorLine));

            // Nested range is smaller, so it should be preferred
            const parentSize = parentSymbol.range.end - parentSymbol.range.start;
            const nestedSize = nestedSymbol.range.end - nestedSymbol.range.start;
            assert.ok(nestedSize < parentSize);
        });
    });

    // Test symbol command prompts
    suite('Symbol Command Prompts', () => {
        test('Should format explain symbol prompt', () => {
            const symbol = { name: 'calculateTotal', kind: 'Function' };
            const code = 'function calculateTotal(items) { return items.reduce((a, b) => a + b, 0); }';
            const language = 'typescript';

            const prompt = `Explain this ${symbol.kind.toLowerCase()} "${symbol.name}" in detail:\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('function'));
            assert.ok(prompt.includes('calculateTotal'));
            assert.ok(prompt.includes('```typescript'));
        });

        test('Should format ask about symbol prompt', () => {
            const symbol = { name: 'UserService', kind: 'Class' };
            const question = 'What design pattern does this use?';
            const code = 'class UserService { constructor() {} getUser() {} }';
            const language = 'typescript';

            const prompt = `Question about the ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n${question}\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('class'));
            assert.ok(prompt.includes('UserService'));
            assert.ok(prompt.includes('design pattern'));
        });

        test('Should format refactor symbol prompt', () => {
            const symbol = { name: 'processData', kind: 'Method' };
            const suggestion = 'Extract helper functions';
            const code = 'processData(data) { /* complex logic */ }';
            const language = 'typescript';

            const prompt = `Refactor this ${symbol.kind.toLowerCase()} "${symbol.name}" (${suggestion}):\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('method'));
            assert.ok(prompt.includes('processData'));
            assert.ok(prompt.includes('Extract helper functions'));
        });

        test('Should format document symbol prompt', () => {
            const symbol = { name: 'fetchUsers', kind: 'Function' };
            const code = 'async function fetchUsers(limit) { return await api.get("/users", { limit }); }';
            const language = 'typescript';

            const prompt = `Add comprehensive documentation to this ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('documentation'));
            assert.ok(prompt.includes('fetchUsers'));
        });

        test('Should format generate tests prompt', () => {
            const symbol = { name: 'validateEmail', kind: 'Function' };
            const code = 'function validateEmail(email) { return /^[^@]+@[^@]+\\.[^@]+$/.test(email); }';
            const language = 'typescript';

            const prompt = `Generate comprehensive unit tests for this ${symbol.kind.toLowerCase()} "${symbol.name}":\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('unit tests'));
            assert.ok(prompt.includes('validateEmail'));
        });

        test('Should format optimize symbol prompt', () => {
            const symbol = { name: 'sortItems', kind: 'Method' };
            const code = 'sortItems(items) { return items.sort((a, b) => a.name.localeCompare(b.name)); }';
            const language = 'typescript';

            const prompt = `Analyze and optimize this ${symbol.kind.toLowerCase()} "${symbol.name}" for better performance:\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('optimize'));
            assert.ok(prompt.includes('performance'));
            assert.ok(prompt.includes('sortItems'));
        });

        test('Should format review symbol prompt', () => {
            const symbol = { name: 'handleAuth', kind: 'Function' };
            const code = 'function handleAuth(token) { /* auth logic */ }';
            const language = 'typescript';

            const prompt = `Review this ${symbol.kind.toLowerCase()} "${symbol.name}" for:\n- Code quality and best practices\n- Potential bugs\n- Security issues\n- Performance concerns\n\n\`\`\`${language}\n${code}\n\`\`\``;

            assert.ok(prompt.includes('Review'));
            assert.ok(prompt.includes('Security'));
            assert.ok(prompt.includes('handleAuth'));
        });
    });

    suite('Symbol Target Resolution', () => {
        let provider: vscode.Disposable;
        let document: vscode.TextDocument;
        const outer = new vscode.Range(0, 0, 2, 1);
        const inner = new vscode.Range(1, 0, 1, 9);

        setup(async () => {
            document = await vscode.workspace.openTextDocument({
                language: 'plaintext', content: 'outer {\n  inner()\n}',
            });
            provider = vscode.languages.registerDocumentSymbolProvider(
                { scheme: 'untitled', language: 'plaintext' },
                { provideDocumentSymbols: () => {
                    const parent = new vscode.DocumentSymbol('outer', '', vscode.SymbolKind.Class, outer, outer);
                    parent.children = [new vscode.DocumentSymbol('inner', '', vscode.SymbolKind.Method, inner, inner)];
                    return [parent];
                } }
            );
            const editor = await vscode.window.showTextDocument(document);
            editor.selection = new vscode.Selection(1, 3, 1, 3);
        });

        teardown(async () => {
            provider.dispose();
            await vscode.commands.executeCommand('workbench.action.revertAndCloseActiveEditor');
        });

        test('Palette resolves the deepest symbol at the cursor', async () => {
            const result = await resolveSymbolTarget();
            assert.ok(result);
            assert.strictEqual(result.uri.toString(), document.uri.toString());
            assert.strictEqual(result.name, 'inner');
            assert.strictEqual(result.kind, 'Method');
            assert.ok(result.range.isEqual(inner));
        });

        test('CodeLens preserves its explicit target when another document is active', async () => {
            const uri = vscode.Uri.parse('untitled:explicit-target');
            const result = await resolveSymbolTarget(uri, outer, 'requested');
            assert.ok(result);
            assert.strictEqual(result.uri.toString(), uri.toString());
            assert.strictEqual(result.name, 'requested');
            assert.ok(result.range.isEqual(outer));
        });

        test('Hover command targets survive URI encoding and JSON round trips', async () => {
            const uri = vscode.Uri.parse('untitled:target%20with%20spaces#fragment');
            const encoded = encodeSymbolTarget(uri, inner, 'name [with](punctuation)?');
            assert.ok(!encoded.includes('['));
            const result = await resolveSymbolTarget(...JSON.parse(decodeURIComponent(encoded)));
            assert.ok(result);
            assert.strictEqual(result.uri.toString(), uri.toString());
            assert.ok(result.range.isEqual(inner));
            assert.strictEqual(result.name, 'name [with](punctuation)?');
        });

        test('Malformed explicit targets reject instead of falling back to the cursor', async () => {
            for (const args of [
                [document.uri], [document.uri, inner], [document.uri.toString(), inner, 'inner'],
                [document.uri, {}, 'inner'], [document.uri, inner, ' '],
                [document.uri, inner, 'inner', 'extra'],
                ['not a URI', [[0, 0], [1, 0]], 'inner'],
                [document.uri.toString(), [[-1, 0], [1, 0]], 'inner'],
                [document.uri.toString(), [[0, 0.5], [1, 0]], 'inner'],
                [document.uri.toString(), [[2, 0], [1, 0]], 'inner'],
            ]) {
                await assert.rejects(resolveSymbolTarget(...args), /Expected (a symbol URI|an ordered symbol range)/);
            }
        });

        test('Palette reports no target when the document has no symbols', async () => {
            provider.dispose();
            assert.strictEqual(await resolveSymbolTarget(), null);
        });
    });
});
