/**
 * Extension Tests
 *
 * Tests for VS Code extension activation and basic functionality.
 */

import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Extension Test Suite', () => {
    vscode.window.showInformationMessage('Starting extension tests');

    test('Extension should activate in the test host', async () => {
        const extension = vscode.extensions.getExtension('victor-ai.victor-ai');
        assert.ok(extension, 'Victor extension must be loaded by the test host');
        await extension.activate();
        assert.strictEqual(extension.isActive, true);
    });

    test('Should register all commands', async () => {
        const extension = vscode.extensions.getExtension('victor-ai.victor-ai');
        assert.ok(extension, 'Victor extension must be loaded by the test host');
        await extension.activate();
        const commands = await vscode.commands.getCommands(true);
        const declared = extension.packageJSON.contributes.commands as Array<{ command: string }>;
        assert.ok(declared.length > 0, 'Manifest must declare Victor commands');
        const missing = declared.map(({ command }) => command).filter(command => !commands.includes(command));
        assert.deepStrictEqual(missing, [], 'Every advertised command must have a handler');
    });

    test('Configuration should have default values', () => {
        const config = vscode.workspace.getConfiguration('victor');

        // Check default configuration values
        assert.strictEqual(config.get('serverUrl'), '');
        assert.strictEqual(config.get('serverPort'), 8765);
        assert.strictEqual(config.get('profile'), 'default');  // Uses profile system now
        assert.strictEqual(config.get('mode'), 'build');
        assert.strictEqual(config.get('autoStart'), false);
        assert.strictEqual(config.get('serverApiKey'), '');
    });
});
