// Unit tests for streamChat's v1 wire-contract consumption (UX foundations L1).
// Runs in plain Node via vitest with axios mocked — the SSE stream is a real
// EventEmitter, so the parse loop, session capture, and termination contract
// are exercised end to end without a server.
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { EventEmitter } from 'node:events';

const mockClient = {
    get: vi.fn(),
    post: vi.fn(),
    delete: vi.fn(),
    put: vi.fn(),
    defaults: { headers: { common: {} as Record<string, string> } },
};

vi.mock('axios', () => {
    const isAxiosError = (e: unknown): boolean =>
        !!(e && typeof e === 'object' && (e as { isAxiosError?: boolean }).isAxiosError === true);
    return {
        default: { create: vi.fn(() => mockClient), isAxiosError },
        isAxiosError,
    };
});

import { VictorClient, VictorError, type StreamEvent, type ToolCall } from '../victorClient';

function sse(events: Array<Record<string, unknown> | string>): EventEmitter {
    const stream = new EventEmitter();
    // Emit on a macrotask so streamChat has attached its 'data' listeners
    // (it awaits the mocked post — a microtask — before wiring the stream).
    setTimeout(() => {
        for (const event of events) {
            const payload = typeof event === 'string' ? event : JSON.stringify(event);
            stream.emit('data', Buffer.from(`data: ${payload}\n\n`));
        }
        stream.emit('end');
    }, 0);
    return stream;
}

function wire(event: string, fields: Record<string, unknown> = {}): Record<string, unknown> {
    return { v: 1, event, ...fields };
}

async function run(
    client: VictorClient,
    events: Array<Record<string, unknown> | string>,
    headers: Record<string, string> = {}
) {
    mockClient.post.mockResolvedValueOnce({ data: sse(events), headers });
    const chunks: string[] = [];
    const toolCalls: ToolCall[] = [];
    const seen: StreamEvent[] = [];
    await client.streamChat(
        [{ role: 'user', content: 'hi' }],
        (c) => chunks.push(c),
        (tc) => toolCalls.push(tc),
        (e) => seen.push(e)
    );
    return { chunks, toolCalls, seen };
}

describe('streamChat v1 wire contract', () => {
    let client: VictorClient;

    beforeEach(() => {
        vi.clearAllMocks();
        client = new VictorClient('http://localhost:8765');
    });

    it('sends both established request shapes and captures the web session header', async () => {
        await run(client, [wire('content', { content: 'a' }), wire('stream_end')], {
            'x-session-id': 'sess-1',
        });
        expect(mockClient.post).toHaveBeenCalledWith(
            '/chat/stream',
            { messages: [{ role: 'user', content: 'hi' }], message: 'hi' },
            { responseType: 'stream' }
        );
        expect(client.getChatSessionId()).toBe('sess-1');

        // Second turn echoes the captured session id back.
        await run(client, [wire('stream_end')]);
        expect(mockClient.post).toHaveBeenLastCalledWith(
            '/chat/stream',
            { messages: [{ role: 'user', content: 'hi' }], message: 'hi', session_id: 'sess-1' },
            { responseType: 'stream' }
        );
    });

    it('resetChatSession forgets the session id', async () => {
        await run(client, [wire('stream_end')], { 'x-session-id': 'sess-2' });
        client.resetChatSession();
        await run(client, [wire('stream_end')]);
        expect(mockClient.post).toHaveBeenLastCalledWith(
            '/chat/stream',
            { messages: [{ role: 'user', content: 'hi' }], message: 'hi' },
            { responseType: 'stream' }
        );
    });

    it('derives both fields from the newest user message without replaying history', async () => {
        mockClient.post.mockResolvedValueOnce({ data: sse(['[DONE]']), headers: {} });
        await client.streamChat([
            { role: 'system', content: 'system context' },
            { role: 'user', content: 'old turn' },
            { role: 'assistant', content: 'old response' },
            { role: 'user', content: 'latest 🧪\nline' },
            { role: 'assistant', content: 'pending response' },
        ], () => undefined);
        expect(mockClient.post).toHaveBeenCalledExactlyOnceWith('/chat/stream', {
            messages: [{ role: 'user', content: 'latest 🧪\nline' }],
            message: 'latest 🧪\nline',
        }, { responseType: 'stream' });
    });

    it.each([401, 422, 503])('propagates HTTP %i without retrying another request shape', async (status) => {
        mockClient.post.mockRejectedValueOnce({
            isAxiosError: true,
            message: `HTTP ${status}`,
            response: { status, data: { detail: 'rejected' } },
        });
        await expect(client.streamChat([{ role: 'user', content: 'hi' }], () => undefined))
            .rejects.toThrowError(VictorError);
        expect(mockClient.post).toHaveBeenCalledTimes(1);
    });

    it('routes all six event types to the right callbacks', async () => {
        const { chunks, toolCalls, seen } = await run(client, [
            wire('thinking', { content: 'pondering' }),
            wire('content', { content: 'Hello ' }),
            wire('tool_call', { tool: 'read', arguments: { path: 'a.py' }, call_id: 'c1' }),
            wire('tool_result', {
                tool: 'read',
                call_id: 'c1',
                success: true,
                result: 'file text',
                elapsed_ms: 120,
                truncated: true,
            }),
            wire('content', { content: 'world' }),
            wire('stream_end'),
        ]);

        expect(chunks).toEqual(['Hello ', 'world']);
        expect(toolCalls).toEqual([
            { id: 'c1', name: 'read', arguments: { path: 'a.py' }, status: 'running' },
        ]);
        expect(seen.map((e) => e.type)).toEqual([
            'thinking',
            'content',
            'tool_call',
            'tool_result',
            'content',
            'stream_end',
        ]);
        const result = seen.find((e) => e.type === 'tool_result')!;
        expect(result.callId).toBe('c1');
        expect(result.success).toBe(true);
        expect(result.content).toBe('file text');
        expect(result.elapsedMs).toBe(120);
        expect(result.truncated).toBe(true);
        expect(seen.find((e) => e.type === 'thinking')!.content).toBe('pondering');
    });

    it('failed tool results carry success=false', async () => {
        const { seen } = await run(client, [
            wire('tool_result', { tool: 'shell', success: false, result: 'exit 1' }),
            wire('stream_end'),
        ]);
        expect(seen[0].success).toBe(false);
    });

    it('rejects on an in-stream error event', async () => {
        mockClient.post.mockResolvedValueOnce({
            data: sse([wire('content', { content: 'partial' }), wire('error', { message: 'provider died' })]),
            headers: {},
        });
        await expect(
            client.streamChat([{ role: 'user', content: 'hi' }], () => undefined)
        ).rejects.toThrowError(VictorError);
    });

    it('resolves on stream_end and ignores anything after it', async () => {
        const { seen } = await run(client, [
            wire('stream_end'),
            wire('error', { message: 'late — must not reject after settle' }),
        ]);
        expect(seen.map(e => e.type)).toEqual(['stream_end']);
    });

    it('ignores unknown additive event types without crashing', async () => {
        const { seen } = await run(client, [
            wire('usage', { tokens: 5 }),
            wire('stream_end'),
        ]);
        expect(seen.map((e) => e.type)).toEqual(['usage', 'stream_end']);
    });

    it('still understands the legacy pre-v1 protocol', async () => {
        const { chunks, seen } = await run(client, [
            { type: 'content', content: 'old-school' },
            '[DONE]',
        ]);
        expect(chunks).toEqual(['old-school']);
        expect(seen.map((e) => e.type)).toEqual(['content', 'done']);
    });

    it.each(['not json', 'null', '[]', '{"v":2,"event":"stream_end"}', '{}'])(
        'rejects malformed or unsupported payload %s before later completion', async (payload) => {
            await expect(run(client, [payload, wire('stream_end')])).rejects.toThrowError(VictorError);
        }
    );

    it('rejects EOF after partial content without a terminator', async () => {
        await expect(run(client, [wire('content', { content: 'partial' })]))
            .rejects.toThrow(/before.*terminator/i);
    });

    it.each(['close', 'aborted'])('rejects premature %s without waiting for end', async (event) => {
        const stream = new EventEmitter();
        mockClient.post.mockResolvedValueOnce({ data: stream, headers: {} });
        const result = client.streamChat([], () => undefined);
        const check = expect(result).rejects.toThrowError(VictorError);
        await Promise.resolve();
        stream.emit(event);
        await check;
    });

    it.each(['\n', '\r\n', '\r'])('decodes split UTF-8 and multiline SSE with %j', async (newline) => {
        const stream = new EventEmitter();
        mockClient.post.mockResolvedValueOnce({ data: stream, headers: {} });
        const chunks: string[] = [];
        const result = client.streamChat([], c => chunks.push(c));
        await Promise.resolve();
        const body = Buffer.from([
            '\uFEFF: heartbeat', '', 'data:{"v":1,"event":"content",',
            'data: "content":"🧪 café"}', '', 'data: [DONE]', '', '',
        ].join(newline));
        for (const byte of body) stream.emit('data', Buffer.from([byte]));
        stream.emit('end');
        await result;
        expect(chunks).toEqual(['🧪 café']);
    });

    it.each(['data: [DONE]\n', 'data: {"v":1,"event":"stream_end"}']) (
        'does not dispatch an unterminated SSE frame at EOF: %s', async (body) => {
            const stream = new EventEmitter();
            mockClient.post.mockResolvedValueOnce({ data: stream, headers: {} });
            const result = client.streamChat([], () => undefined);
            const check = expect(result).rejects.toThrowError(VictorError);
            await Promise.resolve();
            stream.emit('data', Buffer.from(body));
            stream.emit('end');
            await check;
        }
    );

    it('rejects an oversized unterminated frame and releases the response', async () => {
        const stream = Object.assign(new EventEmitter(), { destroy: vi.fn() });
        mockClient.post.mockResolvedValueOnce({ data: stream, headers: {} });
        const result = client.streamChat([], () => undefined);
        const check = expect(result).rejects.toThrow(/limit/i);
        await Promise.resolve();
        stream.emit('data', Buffer.from('data: ' + 'x'.repeat(1024 * 1024)));
        await check;
        expect(stream.destroy).toHaveBeenCalledOnce();
        expect(stream.listenerCount('data')).toBe(0);
    });

    it('propagates callback failures without logging or consuming later content', async () => {
        const seen = vi.fn();
        mockClient.post.mockResolvedValueOnce({
            data: sse([wire('content', { content: 'first' }), wire('stream_end')]), headers: {},
        });
        await expect(client.streamChat([], () => { throw new Error('callback failed'); }, undefined, seen))
            .rejects.toThrow('callback failed');
        expect(seen).not.toHaveBeenCalled();
    });

    it.each([true, false])('handles invalid UTF-8 after/before termination (terminal=%s)', async (terminal) => {
        const stream = new EventEmitter();
        mockClient.post.mockResolvedValueOnce({ data: stream, headers: {} });
        const seen = vi.fn();
        const result = client.streamChat([], () => undefined, undefined, seen);
        const check = terminal ? expect(result).resolves.toBeUndefined() : expect(result).rejects.toThrowError(VictorError);
        await Promise.resolve();
        stream.emit('data', Buffer.concat([
            Buffer.from(terminal ? 'data: [DONE]\n\n' : 'data: '), Buffer.from([0xff]),
        ]));
        stream.emit('end');
        await check;
        expect(seen).toHaveBeenCalledTimes(terminal ? 1 : 0);
    });
});
