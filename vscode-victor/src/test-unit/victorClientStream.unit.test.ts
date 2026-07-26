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

    it('sends {message, session_id?} and captures X-Session-Id', async () => {
        await run(client, [wire('content', { content: 'a' }), wire('stream_end')], {
            'x-session-id': 'sess-1',
        });
        expect(mockClient.post).toHaveBeenCalledWith(
            '/chat/stream',
            { message: 'hi' },
            { responseType: 'stream' }
        );
        expect(client.getChatSessionId()).toBe('sess-1');

        // Second turn echoes the captured session id back.
        await run(client, [wire('stream_end')]);
        expect(mockClient.post).toHaveBeenLastCalledWith(
            '/chat/stream',
            { message: 'hi', session_id: 'sess-1' },
            { responseType: 'stream' }
        );
    });

    it('resetChatSession forgets the session id', async () => {
        await run(client, [wire('stream_end')], { 'x-session-id': 'sess-2' });
        client.resetChatSession();
        await run(client, [wire('stream_end')]);
        expect(mockClient.post).toHaveBeenLastCalledWith(
            '/chat/stream',
            { message: 'hi' },
            { responseType: 'stream' }
        );
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
        expect(seen[0].type).toBe('stream_end');
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

    it('tolerates non-JSON noise between frames', async () => {
        const { chunks } = await run(client, [
            'not json at all',
            wire('content', { content: 'ok' }),
            wire('stream_end'),
        ]);
        expect(chunks).toEqual(['ok']);
    });
});
