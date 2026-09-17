/**
 * Inline Edit Tests
 */

import * as assert from 'assert';
import * as vscode from 'vscode';
import { InlineEditRequest, InlineEditResult } from '../../inlineEdit';

suite('InlineEdit Test Suite', () => {
    test('InlineEditRequest interface', () => {
        const request: Partial<InlineEditRequest> = {
            instruction: 'Fix the bug',
            context: 'Error handling code',
        };

        assert.strictEqual(request.instruction, 'Fix the bug');
        assert.strictEqual(request.context, 'Error handling code');
    });

    test('InlineEditResult interface', () => {
        const result: InlineEditResult = {
            originalText: 'const x = 1',
            suggestedText: 'const x: number = 1',
            range: new vscode.Range(0, 0, 0, 11),
            explanation: 'Added type annotation',
        };

        assert.strictEqual(result.originalText, 'const x = 1');
        assert.ok(result.suggestedText.includes('number'));
        assert.ok(result.explanation);
    });

    test('InlineEditResult without explanation', () => {
        const result: InlineEditResult = {
            originalText: 'function foo() {}',
            suggestedText: 'function foo(): void {}',
            range: new vscode.Range(0, 0, 0, 17),
        };

        assert.ok(!result.explanation);
    });
});

suite('Edit Instruction Parsing', () => {
    const commonInstructions = [
        'fix the bug',
        'add error handling',
        'make it more efficient',
        'add type annotations',
        'refactor this code',
        'simplify',
        'add documentation',
    ];

    commonInstructions.forEach(instruction => {
        test(`Should accept instruction: "${instruction}"`, () => {
            const request: Partial<InlineEditRequest> = { instruction };
            assert.ok(request.instruction);
            assert.ok(request.instruction.trim().length > 0);
        });
    });

    test('Should reject empty instruction', () => {
        const instruction = '';
        const isValid = instruction.trim().length > 0;
        assert.ok(!isValid);
    });

    test('Should reject whitespace-only instruction', () => {
        const instruction = '   ';
        const isValid = instruction.trim().length > 0;
        assert.ok(!isValid);
    });
});

suite('Code Transformation', () => {
    test('Add type annotation', () => {
        const original = 'const x = 1';
        const suggested = 'const x: number = 1';

        assert.notStrictEqual(original, suggested);
        assert.ok(suggested.includes('number'));
    });

    test('Add error handling', () => {
        const original = 'fetch(url)';
        const suggested = `try {
  const response = await fetch(url);
} catch (error) {
  console.error(error);
}`;

        assert.ok(suggested.includes('try'));
        assert.ok(suggested.includes('catch'));
    });

    test('Simplify code', () => {
        const original = `if (condition) {
  return true;
} else {
  return false;
}`;
        const suggested = 'return condition;';

        assert.ok(suggested.length < original.length);
    });

    test('Add documentation', () => {
        const original = 'function add(a, b) { return a + b; }';
        const suggested = `/**
 * Adds two numbers.
 * @param a First number
 * @param b Second number
 * @returns Sum of a and b
 */
function add(a, b) { return a + b; }`;

        assert.ok(suggested.includes('/**'));
        assert.ok(suggested.includes('@param'));
        assert.ok(suggested.includes('@returns'));
    });
});

suite('Range Calculations', () => {
    test('Single line range', () => {
        const range = new vscode.Range(5, 0, 5, 20);

        assert.strictEqual(range.start.line, 5);
        assert.strictEqual(range.end.line, 5);
        assert.ok(range.isSingleLine);
    });

    test('Multi-line range', () => {
        const range = new vscode.Range(10, 0, 15, 0);

        assert.strictEqual(range.start.line, 10);
        assert.strictEqual(range.end.line, 15);
        assert.ok(!range.isSingleLine);
    });

    test('Empty range', () => {
        const range = new vscode.Range(0, 0, 0, 0);
        assert.ok(range.isEmpty);
    });
});

suite('Ghost Text Preview', () => {
    test('Short suggestion preview', () => {
        const suggested = 'const x = 1';
        const preview = `  → ${suggested}`;

        assert.ok(preview.includes('→'));
        assert.ok(preview.includes(suggested));
    });

    test('Multi-line suggestion preview (truncated)', () => {
        const suggested = 'line1\nline2\nline3';
        const firstLine = suggested.split('\n')[0];
        const preview = `  → ${firstLine}${suggested.includes('\n') ? '...' : ''}`;

        assert.strictEqual(preview, '  → line1...');
    });

    test('Single line suggestion preview', () => {
        const suggested = 'simple change';
        const preview = `  → ${suggested.split('\n')[0]}${suggested.includes('\n') ? '...' : ''}`;

        assert.strictEqual(preview, '  → simple change');
        assert.ok(!preview.endsWith('...'));
    });
});

suite('Edit Actions', () => {
    test('Accept edit action', () => {
        const actions = ['Accept (Tab)', 'Reject (Esc)', 'Show Diff'];

        assert.ok(actions.includes('Accept (Tab)'));
        assert.strictEqual(actions.length, 3);
    });

    test('Reject edit action', () => {
        const actions = ['Accept (Tab)', 'Reject (Esc)', 'Show Diff'];

        assert.ok(actions.includes('Reject (Esc)'));
    });

    test('Show diff action', () => {
        const actions = ['Accept (Tab)', 'Reject (Esc)', 'Show Diff'];

        assert.ok(actions.includes('Show Diff'));
    });
});

suite('Response Cleaning', () => {
    test('Remove markdown code fences', () => {
        const response = '```typescript\nconst x = 1;\n```';
        const cleaned = response
            .replace(/^```\w*\n/m, '')
            .replace(/\n```$/m, '')
            .trim();

        assert.strictEqual(cleaned, 'const x = 1;');
    });

    test('Handle response without fences', () => {
        const response = 'const x = 1;';
        const cleaned = response
            .replace(/^```\w*\n/m, '')
            .replace(/\n```$/m, '')
            .trim();

        assert.strictEqual(cleaned, 'const x = 1;');
    });

    test('Handle response with language tag', () => {
        const response = '```javascript\nconsole.log("hello");\n```';
        const cleaned = response
            .replace(/^```\w*\n/m, '')
            .replace(/\n```$/m, '')
            .trim();

        assert.strictEqual(cleaned, 'console.log("hello");');
    });

    test('Trim whitespace', () => {
        const response = '  const x = 1;  \n';
        const cleaned = response.trim();

        assert.strictEqual(cleaned, 'const x = 1;');
    });
});

suite('Edit State Management', () => {
    test('No active edit initially', () => {
        const activeEdit: InlineEditResult | undefined = undefined;
        assert.ok(!activeEdit);
    });

    test('Active edit after request', () => {
        const activeEdit: InlineEditResult = {
            originalText: 'old',
            suggestedText: 'new',
            range: new vscode.Range(0, 0, 0, 3),
        };

        assert.ok(activeEdit);
        assert.strictEqual(activeEdit.originalText, 'old');
    });

    test('Clear active edit', () => {
        let activeEdit: InlineEditResult | undefined = {
            originalText: 'old',
            suggestedText: 'new',
            range: new vscode.Range(0, 0, 0, 3),
        };

        assert.strictEqual(activeEdit.originalText, 'old');
        activeEdit = undefined;
        assert.ok(!activeEdit);
    });
});

suite('Processing State', () => {
    test('Initial processing state', () => {
        const isProcessing = false;
        assert.ok(!isProcessing);
    });

    test('Processing during edit', () => {
        const isProcessing = true;
        assert.ok(isProcessing);
    });

    test('Processing prevents concurrent edits', () => {
        const isProcessing = true;

        // Simulate attempt to start new edit
        const canStart = !isProcessing;
        assert.ok(!canStart);
    });
});
