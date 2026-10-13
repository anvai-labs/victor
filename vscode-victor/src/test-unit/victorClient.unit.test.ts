// Unit tests for VictorClient (the extension's backend HTTP client) and its error
// mapping. Runs in plain Node via vitest with axios mocked — no VS Code, no live
// server. VictorClient imports zero vscode APIs, so this is the cleanest high-value
// coverage target.
import { describe, it, expect, beforeEach, vi } from 'vitest';

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

import { VictorClient, VictorError, VictorErrorType } from '../victorClient';

function axiosErr(opts: { code?: string; status?: number; message?: string; noResponse?: boolean }): any {
    return {
        isAxiosError: true,
        code: opts.code,
        message: opts.message ?? 'boom',
        response: opts.noResponse
            ? undefined
            : { status: opts.status ?? 500, data: opts.message ? { message: opts.message } : {} },
    };
}

describe('VictorError.fromAxiosError', () => {
    it('maps timeout (ECONNABORTED) -> Timeout', () => {
        const e = VictorError.fromAxiosError(axiosErr({ code: 'ECONNABORTED' }));
        expect(e).toBeInstanceOf(VictorError);
        expect(e.type).toBe(VictorErrorType.Timeout);
    });

    it('maps ERR_NETWORK and missing response -> Network', () => {
        expect(VictorError.fromAxiosError(axiosErr({ code: 'ERR_NETWORK' })).type).toBe(VictorErrorType.Network);
        expect(VictorError.fromAxiosError(axiosErr({ noResponse: true })).type).toBe(VictorErrorType.Network);
    });

    it('maps HTTP statuses to the right error types', () => {
        expect(VictorError.fromAxiosError(axiosErr({ status: 401 })).type).toBe(VictorErrorType.Auth);
        expect(VictorError.fromAxiosError(axiosErr({ status: 403 })).type).toBe(VictorErrorType.Auth);
        expect(VictorError.fromAxiosError(axiosErr({ status: 404 })).type).toBe(VictorErrorType.NotFound);
        expect(VictorError.fromAxiosError(axiosErr({ status: 422 })).type).toBe(VictorErrorType.Validation);
        expect(VictorError.fromAxiosError(axiosErr({ status: 500 })).type).toBe(VictorErrorType.ServerError);
        expect(VictorError.fromAxiosError(axiosErr({ status: 502 })).type).toBe(VictorErrorType.ServerError);
        expect(VictorError.fromAxiosError(axiosErr({ status: 503 })).type).toBe(VictorErrorType.ServerError);
        expect(VictorError.fromAxiosError(axiosErr({ status: 418 })).type).toBe(VictorErrorType.Unknown);
    });

    it('prefers the server-provided message when present', () => {
        const e = VictorError.fromAxiosError(axiosErr({ status: 422, message: 'bad field' }));
        expect(e.message).toBe('bad field');
        expect(e.statusCode).toBe(422);
    });
});

describe('VictorClient connection state', () => {
    let client: VictorClient;
    beforeEach(() => {
        vi.clearAllMocks();
        mockClient.defaults.headers.common = {};
        client = new VictorClient('http://localhost:8765', undefined, 'tok-1');
    });

    it('exposes the server url', () => {
        expect(client.getServerUrl()).toBe('http://localhost:8765');
    });

    it('setApiToken sets/clears the Authorization header', () => {
        client.setApiToken('tok-2');
        expect(mockClient.defaults.headers.common['Authorization']).toBe('Bearer tok-2');
        client.setApiToken(undefined);
        expect(mockClient.defaults.headers.common['Authorization']).toBeUndefined();
    });

    it('setServerUrl normalizes trailing slashes', () => {
        client.setServerUrl('http://localhost:9000///');
        expect(client.getServerUrl()).toBe('http://localhost:9000');
    });
});

describe('VictorClient API methods', () => {
    let client: VictorClient;
    beforeEach(() => {
        vi.clearAllMocks();
        client = new VictorClient('http://localhost:8765');
    });

    it.each([false, 'true', 1, undefined])('does not confirm cancellation from %s', async (cancelled) => {
        mockClient.post.mockResolvedValueOnce({ data: { tool_call_id: 'call-1', cancelled } });
        expect(await client.cancelToolExecution('call-1')).toBe(false);
    });

    it('uses the acknowledged HTTP operation even with an open WebSocket', async () => {
        const send = vi.fn();
        Object.assign(client, { wsConnection: { readyState: 1, send } });
        mockClient.post.mockResolvedValueOnce({ data: { tool_call_id: 'call-1', cancelled: true } });
        expect(await client.cancelToolExecution('call-1')).toBe(true);
        expect(mockClient.post).toHaveBeenCalledExactlyOnceWith('/tools/cancel', { tool_call_id: 'call-1' });
        expect(send).not.toHaveBeenCalled();
    });

    it('rejects a cancellation acknowledgement for another tool call', async () => {
        mockClient.post.mockResolvedValueOnce({ data: { tool_call_id: 'other', cancelled: true } });
        expect(await client.cancelToolExecution('call-1')).toBe(false);
    });

    it('chat() POSTs to /chat and normalizes the response payload', async () => {
        mockClient.post.mockResolvedValue({ data: { role: 'assistant', content: 'hi', tool_calls: [{ id: 't1' }] } });
        const res = await client.chat([{ role: 'user', content: 'hello' } as any]);
        expect(mockClient.post).toHaveBeenCalledWith('/chat', { messages: [{ role: 'user', content: 'hello' }] });
        expect(res).toMatchObject({ role: 'assistant', content: 'hi' });
        expect(res.toolCalls).toEqual([{ id: 't1' }]);
    });

    it('chat() applies defaults when the payload is sparse', async () => {
        mockClient.post.mockResolvedValue({ data: {} });
        const res = await client.chat([{ role: 'user', content: 'x' } as any]);
        expect(res.role).toBe('assistant');
        expect(res.content).toBe('');
    });

    it('preserves a paused run without treating its content as a final answer', async () => {
        const paused = {
            role: 'assistant', content: 'partial draft', status: 'awaiting_approval',
            run_id: 'run-approval', approval_request: { id: 'approval-1', metadata: { hash: 'abc' } },
        };
        mockClient.post.mockResolvedValueOnce({ status: 202, data: paused });
        expect(await client.chat([{ role: 'user', content: 'write' }])).toMatchObject(paused);
        expect(mockClient.post).toHaveBeenCalledTimes(1);
    });

    it.each([
        { status: 'awaiting_approval', approval_request: {} },
        { status: 'awaiting_approval', run_id: ' ', approval_request: {} },
        { status: 'awaiting_approval', run_id: 'run', approval_request: null },
        { status: 'awaiting_approval', run_id: 'run', approval_request: [] },
        { status: 'ok' }, {},
    ])('rejects incomplete or contradictory HTTP 202 outcomes: %j', async data => {
        mockClient.post.mockResolvedValueOnce({ status: 202, data });
        await expect(client.chat([{ role: 'user', content: 'write' }])).rejects.toMatchObject({
            name: 'ChatOutcomeError',
        });
        expect(mockClient.post).toHaveBeenCalledTimes(1);
    });

    it.each([null, [], { status: 'failed' }, { status: null },
        { status: 'ok', run_id: 'run', approval_request: { id: 'a1' } }])(
        'rejects unsupported chat outcomes without retry: %j', async data => {
            mockClient.post.mockResolvedValueOnce({ status: 200, data });
            await expect(client.chat([{ role: 'user', content: 'write' }])).rejects.toMatchObject({
                name: 'ChatOutcomeError',
            });
            expect(mockClient.post).toHaveBeenCalledTimes(1);
        }
    );

    it('semanticSearch() POSTs to /search/semantic and returns results', async () => {
        mockClient.post.mockResolvedValue({ data: { results: [{ file: 'a.ts' }] } });
        const res = await client.semanticSearch('query', 5);
        expect(mockClient.post).toHaveBeenCalledWith('/search/semantic', { query: 'query', max_results: 5 });
        expect(res).toEqual([{ file: 'a.ts' }]);
    });

    it('semanticSearch() throws a typed VictorError on failure', async () => {
        mockClient.post.mockRejectedValue(axiosErr({ status: 404 }));
        await expect(client.semanticSearch('q')).rejects.toBeInstanceOf(VictorError);
        await expect(client.semanticSearch('q')).rejects.toMatchObject({ type: VictorErrorType.NotFound });
    });

    it('switchModel() POSTs to /model/switch', async () => {
        mockClient.post.mockResolvedValue({ data: {} });
        await client.switchModel('anthropic', 'claude');
        expect(mockClient.post).toHaveBeenCalledWith('/model/switch', { provider: 'anthropic', model: 'claude' });
    });

    it('getModels() GETs /models and returns the model list', async () => {
        mockClient.get.mockResolvedValue({ data: { models: [{ provider: 'p', model_id: 'm' }] } });
        const res = await client.getModels();
        expect(mockClient.get).toHaveBeenCalledWith('/models');
        expect(res).toEqual([{ provider: 'p', model_id: 'm' }]);
    });

    it('getModels() degrades gracefully to [] on error (does not throw)', async () => {
        mockClient.get.mockRejectedValue(axiosErr({ status: 500 }));
        await expect(client.getModels()).resolves.toEqual([]);
    });

    it('getHistory() GETs /history with a limit and degrades to [] on error', async () => {
        mockClient.get.mockResolvedValue({ data: { history: [{ id: 'h1' }] } });
        expect(await client.getHistory(3)).toEqual([{ id: 'h1' }]);
        expect(mockClient.get).toHaveBeenCalledWith('/history', { params: { limit: 3 } });

        mockClient.get.mockRejectedValue(axiosErr({ noResponse: true }));
        await expect(client.getHistory()).resolves.toEqual([]);
    });
});
