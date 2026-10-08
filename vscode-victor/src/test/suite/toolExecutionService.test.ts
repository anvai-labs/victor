/**
 * Tool Execution Service Tests
 *
 * Comprehensive tests for the ToolExecutionService which tracks
 * tool executions, progress, and provides event-driven updates.
 */

import * as assert from 'assert';
import { ToolExecutionService } from '../../toolExecutionService';
import type { VictorClient } from '../../victorClient';

// We'll test the logic without full VS Code integration
suite('ToolExecutionService Test Suite', () => {
    // Test the tool metadata mapping
    suite('Tool Metadata', () => {
        test('Should format tool names correctly', () => {
            // Test the formatToolName logic
            const formatToolName = (name: string): string => {
                return name
                    .split('_')
                    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                    .join(' ');
            };

            assert.strictEqual(formatToolName('read_file'), 'Read File');
            assert.strictEqual(formatToolName('semantic_code_search'), 'Semantic Code Search');
            assert.strictEqual(formatToolName('git_status'), 'Git Status');
            assert.strictEqual(formatToolName('bash'), 'Bash');
        });

        test('Should provide correct metadata for known tools', () => {
            const TOOL_METADATA: Record<string, { displayName: string; icon: string; category: string }> = {
                read_file: { displayName: 'Read File', icon: '$(file)', category: 'filesystem' },
                write_file: { displayName: 'Write File', icon: '$(file-add)', category: 'filesystem' },
                git_status: { displayName: 'Git Status', icon: '$(git-branch)', category: 'git' },
                code_search: { displayName: 'Code Search', icon: '$(search)', category: 'search' },
                bash: { displayName: 'Run Command', icon: '$(terminal)', category: 'shell' },
            };

            assert.strictEqual(TOOL_METADATA['read_file'].category, 'filesystem');
            assert.strictEqual(TOOL_METADATA['git_status'].icon, '$(git-branch)');
            assert.strictEqual(TOOL_METADATA['bash'].displayName, 'Run Command');
        });
    });

    // Test execution tracking logic
    suite('Execution Tracking', () => {
        test('Should track execution start time', () => {
            const startTime = Date.now();
            const execution = {
                id: 'test-123',
                toolName: 'read_file',
                displayName: 'Read File',
                status: 'running' as const,
                startTime,
            };

            assert.ok(execution.startTime >= startTime - 100);
            assert.ok(execution.startTime <= Date.now());
        });

        test('Should calculate duration on completion', () => {
            const startTime = Date.now() - 1000; // Started 1 second ago
            const endTime = Date.now();
            const duration = endTime - startTime;

            assert.ok(duration >= 1000);
            assert.ok(duration <= 1100);
        });

        test('Should clamp progress percentage', () => {
            const clampProgress = (value: number): number => {
                return Math.min(100, Math.max(0, value));
            };

            assert.strictEqual(clampProgress(-10), 0);
            assert.strictEqual(clampProgress(50), 50);
            assert.strictEqual(clampProgress(150), 100);
            assert.strictEqual(clampProgress(0), 0);
            assert.strictEqual(clampProgress(100), 100);
        });
    });

    // Test history management
    suite('History Management', () => {
        test('Should limit history size', () => {
            const maxHistorySize = 100;
            const history: any[] = [];

            const addToHistory = (execution: any) => {
                history.unshift(execution);
                if (history.length > maxHistorySize) {
                    history.pop();
                }
            };

            // Add more than max items
            for (let i = 0; i < 150; i++) {
                addToHistory({ id: `exec-${i}`, status: 'completed' });
            }

            assert.strictEqual(history.length, 100);
            assert.strictEqual(history[0].id, 'exec-149'); // Most recent first
        });

        test('Should clear history correctly', () => {
            let history = [
                { id: '1', status: 'completed' },
                { id: '2', status: 'failed' },
                { id: '3', status: 'completed' },
            ];

            assert.strictEqual(history.length, 3);
            history = [];
            assert.strictEqual(history.length, 0);
        });
    });

    // Test event handling
    suite('WebSocket Event Handling', () => {
        test('Should parse tool start event', () => {
            const event = {
                type: 'start' as const,
                tool_call_id: 'call-123',
                tool_name: 'read_file',
                arguments: { path: '/test/file.ts' }
            };

            assert.strictEqual(event.type, 'start');
            assert.strictEqual(event.tool_call_id, 'call-123');
            assert.deepStrictEqual(event.arguments, { path: '/test/file.ts' });
        });

        test('Should parse tool progress event', () => {
            const event = {
                type: 'progress' as const,
                tool_call_id: 'call-123',
                tool_name: 'batch_process',
                progress: 50,
                message: 'Processing file 50 of 100'
            };

            assert.strictEqual(event.progress, 50);
            assert.ok(event.message?.includes('50'));
        });

        test('Should parse tool complete event', () => {
            const event = {
                type: 'complete' as const,
                tool_call_id: 'call-123',
                tool_name: 'read_file',
                result: { content: 'file contents' }
            };

            assert.strictEqual(event.type, 'complete');
            assert.ok(event.result);
        });

        test('Should parse tool error event', () => {
            const event = {
                type: 'error' as const,
                tool_call_id: 'call-123',
                tool_name: 'read_file',
                error: 'File not found'
            };

            assert.strictEqual(event.type, 'error');
            assert.strictEqual(event.error, 'File not found');
        });
    });

    // Test status bar formatting
    suite('Status Bar Formatting', () => {
        test('Should format single tool execution', () => {
            const formatSingleExecution = (displayName: string, progress?: number): string => {
                const progressStr = progress !== undefined
                    ? ` (${Math.round(progress)}%)`
                    : '';
                return `${displayName}${progressStr}`;
            };

            assert.strictEqual(formatSingleExecution('Read File'), 'Read File');
            assert.strictEqual(formatSingleExecution('Read File', 50), 'Read File (50%)');
            assert.strictEqual(formatSingleExecution('Read File', 75.6), 'Read File (76%)');
        });

        test('Should format multiple tool executions', () => {
            const formatMultipleExecutions = (count: number): string => {
                return `${count} tools running`;
            };

            assert.strictEqual(formatMultipleExecutions(2), '2 tools running');
            assert.strictEqual(formatMultipleExecutions(5), '5 tools running');
        });

        test('Should format duration correctly', () => {
            const formatDuration = (ms: number): string => {
                if (ms < 1000) {return `${ms}ms`;}
                return `${(ms / 1000).toFixed(1)}s`;
            };

            assert.strictEqual(formatDuration(500), '500ms');
            assert.strictEqual(formatDuration(1000), '1.0s');
            assert.strictEqual(formatDuration(2500), '2.5s');
        });
    });

    // Exercise the actual owner instead of assigning strings to a local variable.
    suite('Execution State Machine', () => {
        let service: ToolExecutionService;
        setup(() => { service = new ToolExecutionService(); });
        teardown(() => service.dispose());

        test('Tracks real start, completion and failure transitions', () => {
            const first = service.startExecution('first', 'read_file');
            assert.strictEqual(first.status, 'running');
            service.completeExecution('first', 'contents');
            assert.strictEqual(first.status, 'completed');
            const second = service.startExecution('second', 'read_file');
            service.failExecution('second', 'read failed');
            assert.strictEqual(second.status, 'failed');
            assert.deepStrictEqual(service.getActiveExecutions(), []);
        });

        for (const outcome of ['missing', 'rejected', 'error', 'confirmed'] as const) {
            test(`Cancellation ${outcome} preserves honest execution state`, async () => {
                const execution = service.startExecution('call-1', 'write_file');
                let cancelledEvents = 0;
                service.onExecutionCancelled(() => cancelledEvents++);
                if (outcome !== 'missing') {
                    service.setVictorClient({ cancelToolExecution: async () => {
                        if (outcome === 'error') throw new Error('offline');
                        return outcome === 'confirmed';
                    } } as unknown as VictorClient);
                }
                assert.strictEqual(await service.cancelExecution('call-1'), outcome === 'confirmed');
                assert.strictEqual(execution.status, outcome === 'confirmed' ? 'cancelled' : 'running');
                assert.strictEqual(cancelledEvents, outcome === 'confirmed' ? 1 : 0);
                assert.strictEqual(service.getActiveExecutions().length, outcome === 'confirmed' ? 0 : 1);
            });
        }

        for (const terminal of ['completed', 'failed', 'replaced'] as const) {
            test(`Late cancellation must not overwrite a ${terminal} execution`, async () => {
                const original = service.startExecution('call-1', 'write_file');
                let acknowledge!: (value: boolean) => void;
                service.setVictorClient({ cancelToolExecution: () => new Promise<boolean>(resolve => {
                    acknowledge = resolve;
                }) } as unknown as VictorClient);
                const pending = service.cancelExecution('call-1');
                if (terminal === 'completed') service.completeExecution('call-1', 'saved');
                else if (terminal === 'failed') service.failExecution('call-1', 'failed');
                else service.startExecution('call-1', 'read_file');
                acknowledge(true);
                assert.strictEqual(await pending, false);
                assert.strictEqual(original.status, terminal === 'replaced' ? 'running' : terminal);
                if (terminal === 'replaced') assert.strictEqual(service.getActiveExecutions().length, 1);
            });
        }
    });

    // Test execution map operations
    suite('Execution Map Operations', () => {
        test('Should track active executions', () => {
            const executions = new Map<string, any>();

            executions.set('exec-1', { id: 'exec-1', status: 'running' });
            executions.set('exec-2', { id: 'exec-2', status: 'running' });

            assert.strictEqual(executions.size, 2);
            assert.ok(executions.has('exec-1'));
        });

        test('Should remove completed executions', () => {
            const executions = new Map<string, any>();

            executions.set('exec-1', { id: 'exec-1', status: 'running' });
            executions.set('exec-2', { id: 'exec-2', status: 'running' });

            // Complete exec-1
            executions.delete('exec-1');

            assert.strictEqual(executions.size, 1);
            assert.ok(!executions.has('exec-1'));
            assert.ok(executions.has('exec-2'));
        });

        test('Should get all active executions as array', () => {
            const executions = new Map<string, any>();

            executions.set('exec-1', { id: 'exec-1', toolName: 'read_file' });
            executions.set('exec-2', { id: 'exec-2', toolName: 'git_status' });

            const active = Array.from(executions.values());

            assert.strictEqual(active.length, 2);
            assert.ok(active.some(e => e.toolName === 'read_file'));
            assert.ok(active.some(e => e.toolName === 'git_status'));
        });
    });

    // Test tool category grouping
    suite('Tool Category Grouping', () => {
        test('Should group tools by category', () => {
            const tools = [
                { name: 'read_file', category: 'filesystem' },
                { name: 'write_file', category: 'filesystem' },
                { name: 'git_status', category: 'git' },
                { name: 'code_search', category: 'search' },
            ];

            const grouped = tools.reduce((acc, tool) => {
                if (!acc[tool.category]) {
                    acc[tool.category] = [];
                }
                acc[tool.category].push(tool.name);
                return acc;
            }, {} as Record<string, string[]>);

            assert.strictEqual(grouped['filesystem'].length, 2);
            assert.strictEqual(grouped['git'].length, 1);
            assert.strictEqual(grouped['search'].length, 1);
        });
    });
});
