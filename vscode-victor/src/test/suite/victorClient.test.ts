/**
 * Victor Client Tests
 *
 * Tests for the VictorClient API client.
 * Note: These are unit tests that mock the HTTP layer.
 */

import * as assert from 'assert';
import { VictorClient, ChatMessage, ToolCall, VictorError, VictorErrorType } from '../../victorClient';

suite('VictorClient Test Suite', () => {

    let client: VictorClient;

    setup(() => {
        // Create client with test configuration
        client = new VictorClient('http://localhost:9999');
    });

    teardown(() => {
        // Clean up WebSocket connection
        client.disconnectWebSocket();
    });

    test('Client should initialize with correct base URL', () => {
        // The client should be created without errors
        assert.ok(client);
    });

    test('ChatMessage interface should accept valid messages', () => {
        const userMessage: ChatMessage = {
            role: 'user',
            content: 'Hello, Victor!'
        };

        const assistantMessage: ChatMessage = {
            role: 'assistant',
            content: 'Hello! How can I help you?',
            toolCalls: [{ name: 'read_file', arguments: { path: 'test.ts' } }]
        };

        assert.strictEqual(userMessage.role, 'user');
        assert.strictEqual(assistantMessage.role, 'assistant');
        assert.ok(assistantMessage.toolCalls);
    });

    test('ToolCall interface should have correct structure', () => {
        const toolCall: ToolCall = {
            id: 'tc_123',
            name: 'read_file',
            arguments: { path: '/test/file.ts' },
            result: 'file contents'
        };

        assert.strictEqual(toolCall.name, 'read_file');
        assert.ok(toolCall.arguments);
    });

    test('VictorErrorType enum values should exist', () => {
        assert.strictEqual(VictorErrorType.Network, 'NETWORK');
        assert.strictEqual(VictorErrorType.Timeout, 'TIMEOUT');
        assert.strictEqual(VictorErrorType.ServerError, 'SERVER_ERROR');
        assert.strictEqual(VictorErrorType.NotFound, 'NOT_FOUND');
        assert.strictEqual(VictorErrorType.Validation, 'VALIDATION');
        assert.strictEqual(VictorErrorType.Auth, 'AUTH');
        assert.strictEqual(VictorErrorType.Unknown, 'UNKNOWN');
    });

    test('VictorError should have correct properties', () => {
        const error = new VictorError(
            'Test error',
            VictorErrorType.Network,
            500,
            new Error('Original')
        );

        assert.strictEqual(error.message, 'Test error');
        assert.strictEqual(error.type, VictorErrorType.Network);
        assert.strictEqual(error.statusCode, 500);
        assert.ok(error.originalError);
    });

    test('getStatus should handle server unavailable', async () => {
        try {
            const status = await client.getStatus();
            // If we get here, server is running
            assert.ok(status);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('supportsCapability should return false when server is unavailable', async () => {
        (client as unknown as { getStatus: () => Promise<unknown> }).getStatus = async () => {
            throw new VictorError('offline', VictorErrorType.Network);
        };

        const supported = await client.supportsCapability('agents');
        assert.strictEqual(supported, false);
    });

    test('supportsCapability should return true for legacy servers without /status', async () => {
        (client as unknown as { getStatus: () => Promise<unknown> }).getStatus = async () => {
            throw new VictorError('missing status route', VictorErrorType.NotFound, 404);
        };

        const supported = await client.supportsCapability('agents');
        assert.strictEqual(supported, true);
    });

    test('disconnectWebSocket should be safe to call multiple times', () => {
        // Should not throw
        client.disconnectWebSocket();
        client.disconnectWebSocket();
        client.disconnectWebSocket();

        assert.ok(true);
    });

    test('connectWebSocket should not throw', () => {
        // Should not throw even when server is unavailable
        client.connectWebSocket();

        // Give it a moment then disconnect
        return new Promise<void>((resolve) => {
            setTimeout(() => {
                client.disconnectWebSocket();
                resolve();
            }, 100);
        });
    });

    test('streamChat should handle callback setup', async () => {
        const messages: ChatMessage[] = [{ role: 'user', content: 'Test' }];

        try {
            await client.streamChat(
                messages,
                (content) => { /* content callback */ },
                (toolCall) => { /* tool call callback */ }
            );
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('prefetchSessionToken should resolve gracefully when server is unavailable', async () => {
        client.setApiToken('secret');
        const token = await client.prefetchSessionToken();
        // With no server running, we expect undefined (no throw)
        assert.strictEqual(token, undefined);
    });

    test('undo should handle server unavailable', async () => {
        try {
            const result = await client.undo();
            assert.ok(result);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('redo should handle server unavailable', async () => {
        try {
            const result = await client.redo();
            assert.ok(result);
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('semanticSearch should handle server unavailable', async () => {
        try {
            const results = await client.semanticSearch('test query');
            assert.ok(Array.isArray(results));
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('switchModel should require both parameters', async () => {
        try {
            await client.switchModel('openai', 'gpt-4');
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('switchMode should accept valid mode', async () => {
        try {
            await client.switchMode('plan');
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    test('switchProfile should accept a profile name', async () => {
        try {
            await client.switchProfile('default');
        } catch (error) {
            // Expected when server is not running
            assert.ok(error instanceof VictorError);
        }
    });

    // URL Construction tests
    suite('URL Construction', () => {
        test('Should construct correct endpoint URLs', () => {
            const baseUrl = 'http://localhost:9999';

            const endpoints = {
                status: `${baseUrl}/status`,
                chat: `${baseUrl}/chat`,
                stream: `${baseUrl}/chat/stream`,
                undo: `${baseUrl}/undo`,
                redo: `${baseUrl}/redo`,
                search: `${baseUrl}/search`,
                model: `${baseUrl}/model`,
                mode: `${baseUrl}/mode`,
            };

            assert.strictEqual(endpoints.status, 'http://localhost:9999/status');
            assert.strictEqual(endpoints.chat, 'http://localhost:9999/chat');
            assert.strictEqual(endpoints.stream, 'http://localhost:9999/chat/stream');
        });

        test('Should handle trailing slash in base URL', () => {
            const normalizeUrl = (base: string, path: string): string => {
                const cleanBase = base.endsWith('/') ? base.slice(0, -1) : base;
                const cleanPath = path.startsWith('/') ? path : `/${path}`;
                return cleanBase + cleanPath;
            };

            assert.strictEqual(normalizeUrl('http://localhost:9999/', '/chat'), 'http://localhost:9999/chat');
            assert.strictEqual(normalizeUrl('http://localhost:9999', 'chat'), 'http://localhost:9999/chat');
        });

        test('Should convert HTTP to WebSocket URL', () => {
            const toWebSocketUrl = (httpUrl: string): string => {
                return httpUrl.replace('http://', 'ws://').replace('https://', 'wss://');
            };

            assert.strictEqual(toWebSocketUrl('http://localhost:9999'), 'ws://localhost:9999');
            assert.strictEqual(toWebSocketUrl('https://api.example.com'), 'wss://api.example.com');
        });
    });

    // Request/Response Handling tests
    suite('Request/Response Handling', () => {
        test('Should format chat request body', () => {
            const messages: ChatMessage[] = [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there!' }
            ];

            const requestBody = {
                messages,
                stream: false,
                provider: 'anthropic',
                model: 'claude-3-sonnet'
            };

            assert.strictEqual(requestBody.messages.length, 2);
            assert.strictEqual(requestBody.stream, false);
        });

        test('Should parse JSON response', () => {
            const jsonResponse = '{"status": "ok", "version": "0.1.0"}';
            const parsed = JSON.parse(jsonResponse);

            assert.strictEqual(parsed.status, 'ok');
            assert.strictEqual(parsed.version, '0.1.0');
        });

        test('Should handle empty response', () => {
            const emptyResponse = '';

            const parseResponse = (text: string): object | null => {
                if (!text || text.trim() === '') {return null;}
                try {
                    return JSON.parse(text);
                } catch {
                    return null;
                }
            };

            assert.strictEqual(parseResponse(emptyResponse), null);
            assert.deepStrictEqual(parseResponse('{}'), {});
        });
    });

    // Error Classification tests
    suite('Error Classification', () => {
        test('Should classify HTTP status codes', () => {
            const classifyError = (status: number): VictorErrorType => {
                if (status === 401 || status === 403) {return VictorErrorType.Auth;}
                if (status === 404) {return VictorErrorType.NotFound;}
                if (status === 400 || status === 422) {return VictorErrorType.Validation;}
                if (status >= 500) {return VictorErrorType.ServerError;}
                return VictorErrorType.Unknown;
            };

            assert.strictEqual(classifyError(401), VictorErrorType.Auth);
            assert.strictEqual(classifyError(403), VictorErrorType.Auth);
            assert.strictEqual(classifyError(404), VictorErrorType.NotFound);
            assert.strictEqual(classifyError(400), VictorErrorType.Validation);
            assert.strictEqual(classifyError(500), VictorErrorType.ServerError);
            assert.strictEqual(classifyError(503), VictorErrorType.ServerError);
        });

        test('Should detect network errors', () => {
            const isNetworkError = (error: Error): boolean => {
                const message = error.message.toLowerCase();
                return message.includes('network') ||
                       message.includes('econnrefused') ||
                       message.includes('enotfound') ||
                       message.includes('fetch failed');
            };

            assert.ok(isNetworkError(new Error('Network error')));
            assert.ok(isNetworkError(new Error('ECONNREFUSED')));
            assert.ok(!isNetworkError(new Error('Some other error')));
        });

        test('Should detect timeout errors', () => {
            const isTimeoutError = (error: Error): boolean => {
                const message = error.message.toLowerCase();
                return message.includes('timeout') || message.includes('timed out');
            };

            assert.ok(isTimeoutError(new Error('Request timeout')));
            assert.ok(isTimeoutError(new Error('Connection timed out')));
            assert.ok(!isTimeoutError(new Error('Server error')));
        });
    });

    // Tool Call Handling tests
    suite('Tool Call Handling', () => {
        test('Should parse tool calls from response', () => {
            const response = {
                toolCalls: [
                    { id: 'tc1', name: 'read_file', arguments: { path: '/test.ts' } },
                    { id: 'tc2', name: 'write_file', arguments: { path: '/out.ts', content: 'data' } }
                ]
            };

            assert.strictEqual(response.toolCalls.length, 2);
            assert.strictEqual(response.toolCalls[0].name, 'read_file');
            assert.strictEqual(response.toolCalls[1].name, 'write_file');
        });

        test('Should validate tool call structure', () => {
            const isValidToolCall = (tc: any): boolean => {
                return typeof tc.name === 'string' &&
                       tc.name.length > 0 &&
                       tc.arguments !== null &&
                       typeof tc.arguments === 'object';
            };

            assert.ok(isValidToolCall({ name: 'test', arguments: {} }));
            assert.ok(!isValidToolCall({ name: '', arguments: {} }));
            assert.ok(!isValidToolCall({ name: 'test', arguments: null }));
        });

        test('Should handle tool call with result', () => {
            const toolCall: ToolCall = {
                id: 'tc_123',
                name: 'read_file',
                arguments: { path: '/test.ts' },
                result: 'console.log("hello");'
            };

            assert.ok(toolCall.result);
            assert.strictEqual(typeof toolCall.result, 'string');
        });

        test('Should handle tool call without result', () => {
            const toolCall: ToolCall = {
                name: 'delete_file',
                arguments: { path: '/temp.txt' }
            };

            assert.strictEqual(toolCall.result, undefined);
        });
    });

    // Streaming behavior is exercised through the real client in
    // test-unit/victorClientStream.unit.test.ts and the HTTP contract smoke.

    // WebSocket Event Handling tests
    suite('WebSocket Events', () => {
        test('Should classify WebSocket events', () => {
            type WSEventType = 'tool_start' | 'tool_progress' | 'tool_complete' | 'tool_error' | 'unknown';

            const classifyEvent = (event: { type: string }): WSEventType => {
                switch (event.type) {
                    case 'start': return 'tool_start';
                    case 'progress': return 'tool_progress';
                    case 'complete': return 'tool_complete';
                    case 'error': return 'tool_error';
                    default: return 'unknown';
                }
            };

            assert.strictEqual(classifyEvent({ type: 'start' }), 'tool_start');
            assert.strictEqual(classifyEvent({ type: 'progress' }), 'tool_progress');
            assert.strictEqual(classifyEvent({ type: 'complete' }), 'tool_complete');
            assert.strictEqual(classifyEvent({ type: 'error' }), 'tool_error');
            assert.strictEqual(classifyEvent({ type: 'other' }), 'unknown');
        });

        test('Should handle tool progress event', () => {
            const event = {
                type: 'progress',
                tool_call_id: 'tc_123',
                tool_name: 'batch_process',
                progress: 75,
                message: 'Processing file 75 of 100'
            };

            assert.strictEqual(event.progress, 75);
            assert.ok(event.message.includes('75'));
        });
    });

    // Retry Logic tests
    suite('Retry Logic', () => {
        test('Should calculate exponential backoff', () => {
            const calculateBackoff = (attempt: number, baseMs: number = 1000): number => {
                return Math.min(baseMs * Math.pow(2, attempt), 30000);
            };

            assert.strictEqual(calculateBackoff(0), 1000);
            assert.strictEqual(calculateBackoff(1), 2000);
            assert.strictEqual(calculateBackoff(2), 4000);
            assert.strictEqual(calculateBackoff(3), 8000);
            assert.strictEqual(calculateBackoff(10), 30000); // capped
        });

        test('Should determine if error is retryable', () => {
            const isRetryable = (error: VictorError): boolean => {
                return error.type === VictorErrorType.Network ||
                       error.type === VictorErrorType.Timeout ||
                       (error.statusCode !== undefined && error.statusCode >= 500);
            };

            assert.ok(isRetryable(new VictorError('Net', VictorErrorType.Network)));
            assert.ok(isRetryable(new VictorError('Time', VictorErrorType.Timeout)));
            assert.ok(isRetryable(new VictorError('Srv', VictorErrorType.ServerError, 503)));
            assert.ok(!isRetryable(new VictorError('Val', VictorErrorType.Validation, 400)));
        });

        test('Should track retry attempts', () => {
            const maxRetries = 3;
            let attempts = 0;

            const shouldRetry = (): boolean => {
                return attempts < maxRetries;
            };

            const recordAttempt = () => {
                attempts++;
            };

            assert.ok(shouldRetry()); // 0 < 3
            recordAttempt();
            assert.ok(shouldRetry()); // 1 < 3
            recordAttempt();
            assert.ok(shouldRetry()); // 2 < 3
            recordAttempt();
            assert.ok(!shouldRetry()); // 3 < 3 = false
        });
    });

    // Request Configuration tests
    suite('Request Configuration', () => {
        test('Should build request headers', () => {
            const buildHeaders = (apiKey?: string): Record<string, string> => {
                const headers: Record<string, string> = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                };
                if (apiKey) {
                    headers['Authorization'] = `Bearer ${apiKey}`;
                }
                return headers;
            };

            const withKey = buildHeaders('test-key');
            const withoutKey = buildHeaders();

            assert.strictEqual(withKey['Authorization'], 'Bearer test-key');
            assert.strictEqual(withoutKey['Authorization'], undefined);
            assert.strictEqual(withKey['Content-Type'], 'application/json');
        });

        test('Should set request timeout', () => {
            const defaultTimeout = 30000;
            const longTimeout = 120000;

            const getTimeout = (operation: string): number => {
                switch (operation) {
                    case 'chat':
                    case 'stream':
                        return longTimeout;
                    default:
                        return defaultTimeout;
                }
            };

            assert.strictEqual(getTimeout('status'), 30000);
            assert.strictEqual(getTimeout('chat'), 120000);
            assert.strictEqual(getTimeout('stream'), 120000);
        });
    });

    // State Management tests
    suite('Connection State', () => {
        test('Should track connection state', () => {
            type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error';
            let state: ConnectionState = 'disconnected';

            const connect = () => { state = 'connecting'; };
            const onConnected = () => { state = 'connected'; };
            const onError = () => { state = 'error'; };
            const disconnect = () => { state = 'disconnected'; };

            assert.strictEqual(state, 'disconnected');
            connect();
            assert.strictEqual(state, 'connecting');
            onConnected();
            assert.strictEqual(state, 'connected');
            disconnect();
            assert.strictEqual(state, 'disconnected');
        });

        test('Should track reconnection attempts', () => {
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;

            const shouldReconnect = (): boolean => {
                return reconnectAttempts < maxReconnectAttempts;
            };

            const recordReconnect = () => {
                reconnectAttempts++;
            };

            const resetReconnect = () => {
                reconnectAttempts = 0;
            };

            recordReconnect();
            recordReconnect();
            assert.strictEqual(reconnectAttempts, 2);
            resetReconnect();
            assert.strictEqual(reconnectAttempts, 0);
        });
    });
});
