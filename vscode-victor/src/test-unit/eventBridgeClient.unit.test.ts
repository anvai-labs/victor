import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import * as vscode from 'vscode';

const sockets = vi.hoisted(() => [] as any[]);
vi.mock('ws', async () => {
    const { EventEmitter } = await import('node:events');
    class Socket extends EventEmitter {
        static OPEN = 1;
        static CONNECTING = 0;
        readyState = 0;
        send = vi.fn();
        close = vi.fn();
        terminate = vi.fn();
        constructor(public url: string, public options: any) {
            super();
            sockets.push(this);
        }
    }
    return { default: Socket };
});

import { ConnectionState, EventBridgeClient } from '../eventBridgeClient';
import { VictorClient } from '../victorClient';

describe('EventBridge owned transport', () => {
    let bridge: EventBridgeClient;
    let logs: string[];
    const config = { initialDelayMs: 10, maxDelayMs: 20, multiplier: 2, maxRetries: 2 };
    const open = (socket = sockets.at(-1)) => { socket.readyState = 1; socket.emit('open'); };
    beforeEach(() => {
        vi.useFakeTimers();
        sockets.length = 0;
        logs = [];
        vi.spyOn(vscode.window, 'createOutputChannel').mockReturnValue({
            appendLine: (line: string) => logs.push(line), dispose: vi.fn(), show: vi.fn(),
        } as any);
        bridge = new EventBridgeClient(config);
    });
    afterEach(() => { bridge.dispose(); vi.useRealTimers(); vi.restoreAllMocks(); });

    it('sends credentials only in upgrade headers, with no redirect following', () => {
        bridge.connect('https://example.test/prefix/', undefined, 'secret-key');
        expect(sockets[0].url).toBe('wss://example.test/prefix/ws/events');
        expect(sockets[0].options).toMatchObject({
            headers: { Authorization: 'Bearer secret-key' }, followRedirects: false,
        });
        expect(JSON.stringify(logs)).not.toContain('secret-key');
    });

    it('rotates and clears credentials while an upgrade is pending', () => {
        bridge.connect('http://localhost:8765', undefined, 'old');
        const first = sockets[0];
        bridge.connect('http://localhost:8765', undefined, 'new');
        expect(sockets).toHaveLength(2);
        expect(first.terminate).toHaveBeenCalledOnce();
        expect(sockets[1].options.headers.Authorization).toBe('Bearer new');
        bridge.connect('http://localhost:8765');
        expect(sockets).toHaveLength(3);
        expect(sockets[2].options.headers).toEqual({});
    });

    it('does not redial unchanged configuration; updates subscription without aliasing caller arrays', () => {
        const categories = ['tool.start'];
        bridge.connect('http://localhost:8765', { categories });
        categories.push('tool.error');
        bridge.connect('http://localhost:8765');
        expect(sockets).toHaveLength(1);
        open();
        expect(JSON.parse(sockets[0].send.mock.calls[0][0]).categories).toEqual(['tool.start']);
        bridge.subscribe(['tool.complete'], 'request-1');
        expect(JSON.parse(sockets[0].send.mock.lastCall[0])).toEqual({
            type: 'subscribe', categories: ['tool.complete'], correlation_id: 'request-1',
        });
    });

    it.each(['open', 'message', 'error', 'close'])('ignores obsolete %s callbacks after endpoint replacement', (event) => {
        const deliver = vi.fn();
        bridge.onAny(deliver);
        bridge.connect('http://localhost:8765', undefined, 'old');
        const old = sockets[0];
        open(old);
        bridge.connect('http://localhost:8766', undefined, 'new');
        const current = sockets[1];
        open(current);
        const payload: Record<string, any> = {
            message: Buffer.from(JSON.stringify({ type: 'event', event: { type: 'tool.start', data: {} } })),
            error: new Error('old credential'), close: 1006,
        };
        old.emit(event, payload[event]);
        bridge.subscribe(['tool.complete']);
        expect(bridge.getState()).toBe(ConnectionState.Connected);
        expect(deliver).not.toHaveBeenCalled();
        expect(current.send).toHaveBeenCalledTimes(2);
        vi.advanceTimersByTime(30);
        expect(sockets).toHaveLength(2);
    });

    it.each(['disconnect', 'dispose'] as const)('cannot revive after %s through old callbacks or timers', (operation) => {
        bridge.connect('http://localhost:8765');
        const old = sockets[0];
        open();
        old.emit('close', 1006);
        bridge[operation]();
        old.emit('open');
        old.emit('error', new Error('late'));
        old.emit('close', 1006);
        vi.advanceTimersByTime(60000);
        expect(sockets).toHaveLength(1);
        expect(bridge.getState()).toBe(ConnectionState.Disconnected);
        if (operation === 'dispose') {
            bridge.connect('http://localhost:8765');
            expect(sockets).toHaveLength(1);
        }
    });

    it.each([401, 403, 302])('treats HTTP %s as terminal without retry or anonymous downgrade', (statusCode) => {
        bridge.connect('http://localhost:8765', undefined, 'secret-key');
        const request = { destroy: vi.fn() };
        const response = { statusCode, destroy: vi.fn() };
        sockets[0].emit('unexpected-response', request, response);
        sockets[0].emit('close', 1006);
        vi.advanceTimersByTime(60000);
        bridge.connect('http://localhost:8765', undefined, 'secret-key');
        expect(request.destroy).toHaveBeenCalled();
        expect(response.destroy).toHaveBeenCalled();
        expect(sockets).toHaveLength(1);
        expect(bridge.getState()).toBe(ConnectionState.Error);
        bridge.connect('http://localhost:8765', undefined, 'corrected');
        expect(sockets).toHaveLength(2);
    });

    it('bounds transient reconnects and cancels stale ping activity', () => {
        bridge.connect('http://localhost:8765');
        for (const delay of [10, 20]) {
            sockets.at(-1).emit('error', new Error('offline'));
            sockets.at(-1).emit('close', 1006);
            vi.advanceTimersByTime(delay);
        }
        sockets.at(-1).emit('close', 1006);
        vi.advanceTimersByTime(60000);
        expect(sockets).toHaveLength(3);
        expect(bridge.getState()).toBe(ConnectionState.Error);
        expect(sockets.every(socket => socket.send.mock.calls.length === 0)).toBe(true);
    });

    it.each([
        'http://user:secret@localhost', 'http://localhost?api_key=secret',
        'http://localhost#secret', 'file:///secret', 'not a URL',
    ])('rejects ambiguous credential/endpoint syntax without logging it: %s', (url) => {
        bridge.connect(url, undefined, 'secret');
        expect(sockets).toHaveLength(0);
        expect(bridge.getState()).toBe(ConnectionState.Error);
        expect(logs.join('\n')).not.toContain('secret');
    });

    it('does not log server-controlled errors, close reasons or event payloads', () => {
        bridge.connect('http://localhost:8765', undefined, 'secret');
        open();
        sockets[0].emit('message', Buffer.from(JSON.stringify({type: 'event', event: { type: 'secret', data: { secret: 'secret' } }})));
        sockets[0].emit('error', new Error('secret'));
        sockets[0].emit('close', 1006, Buffer.from('secret'));
        expect(logs.join('\n')).not.toContain('secret');
    });

    it.each([ConnectionState.Connecting, ConnectionState.Connected, ConnectionState.Reconnecting])(
        'honors synchronous disconnect during %s notification', (state) => {
            bridge.onStateChange(next => { if (next === state) {bridge.disconnect();} });
            bridge.connect('http://localhost:8765');
            if (state !== ConnectionState.Connecting) {open();}
            if (state === ConnectionState.Reconnecting) {sockets[0].emit('close', 1006);}
            vi.advanceTimersByTime(60000);
            expect(bridge.getState()).toBe(ConnectionState.Disconnected);
            expect(sockets).toHaveLength(state === ConnectionState.Connecting ? 0 : 1);
            if (state === ConnectionState.Connected) {expect(sockets[0].send).not.toHaveBeenCalled();}
        }
    );

    it('stops dispatching an obsolete event when a handler changes the connection', () => {
        const next = vi.fn();
        bridge.on('tool.start', () => bridge.disconnect());
        bridge.onAny(next);
        bridge.connect('http://localhost:8765');
        open();
        sockets[0].emit('message', Buffer.from(JSON.stringify({ type: 'event', event: { type: 'tool.start', data: {} } })));
        expect(next).not.toHaveBeenCalled();
    });
});

describe('VictorClient connection owner', () => {
    it('isolates failed listeners so key removal and endpoint changes reach every consumer', () => {
        const client = new VictorClient('http://localhost:8765', undefined, 'old');
        const warning = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const snapshots: unknown[] = [];
        client.onConnectionChange(() => { throw new Error('secret-key'); });
        client.onConnectionChange(() => snapshots.push(client.getConnectionConfig()));
        try {
            expect(() => client.setApiToken('new')).not.toThrow();
            expect(() => client.setApiToken(undefined)).not.toThrow();
            expect(() => client.setServerUrl('http://localhost:8766')).not.toThrow();
            expect(snapshots).toEqual([
                { serverUrl: 'http://localhost:8765', apiToken: 'new' },
                { serverUrl: 'http://localhost:8765', apiToken: undefined },
                { serverUrl: 'http://localhost:8766', apiToken: undefined },
            ]);
            expect(warning).toHaveBeenCalledTimes(3);
            expect(JSON.stringify(warning.mock.calls)).not.toContain('secret-key');
        } finally { warning.mockRestore(); }
    });

    it('publishes coherent immutable snapshots only on changes and releases listeners', () => {
        const client = new VictorClient('http://localhost:8765', undefined, 'old');
        const changes = vi.fn();
        const release = client.onConnectionChange(changes);
        const first = client.getConnectionConfig();
        client.setApiToken('new');
        client.setApiToken('new');
        client.setServerUrl('http://localhost:8766');
        expect(changes).toHaveBeenCalledTimes(2);
        expect(first).toEqual({ serverUrl: 'http://localhost:8765', apiToken: 'old' });
        expect(Object.isFrozen(first)).toBe(true);
        expect(client.getConnectionConfig()).toEqual({ serverUrl: 'http://localhost:8766', apiToken: 'new' });
        release.dispose();
        client.setApiToken(undefined);
        expect(changes).toHaveBeenCalledTimes(2);
        expect(client.getConnectionConfig().apiToken).toBeUndefined();
    });
});
