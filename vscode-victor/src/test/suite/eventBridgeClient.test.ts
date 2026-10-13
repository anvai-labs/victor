/** Real upgrade/socket tests in the discovered VS Code host suite. Race schedules
 * are covered separately by controlled sockets/timers in test-unit. */
import * as assert from 'assert';
import * as http from 'http';
import * as vscode from 'vscode';
import { AddressInfo } from 'net';
import WebSocket, { WebSocketServer } from 'ws';
import { EventBridgeClient, ConnectionState, disposeEventBridgeClient, getEventBridgeClient } from '../../eventBridgeClient';
import { VictorClient } from '../../victorClient';
import { ChatViewProvider } from '../../chatViewProvider';

async function until(predicate: () => boolean): Promise<void> {
    const deadline = Date.now() + 3000;
    while (!predicate()) {
        if (Date.now() > deadline) {assert.fail('Timed out waiting for EventBridge outcome');}
        await new Promise(resolve => setTimeout(resolve, 10));
    }
}

suite('EventBridge real extension-host transport', () => {
    let server: http.Server;
    let websocketServer: WebSocketServer;
    let bridge: EventBridgeClient;
    let url: string;
    let expectedKey: string | undefined;
    let status: number;
    let attempts: http.IncomingMessage[];
    let subscriptions: unknown[];

    setup(async () => {
        attempts = [];
        subscriptions = [];
        status = 0;
        expectedKey = 'host-test-key';
        server = http.createServer();
        websocketServer = new WebSocketServer({ noServer: true });
        server.on('upgrade', (request, socket, head) => {
            attempts.push(request);
            const denied = expectedKey && request.headers.authorization !== `Bearer ${expectedKey}`;
            if (denied || status) {
                socket.end(`HTTP/1.1 ${status || 403} Rejected\r\nLocation: /redirect-target\r\nContent-Length: 0\r\nConnection: close\r\n\r\n`);
                return;
            }
            websocketServer.handleUpgrade(request, socket, head, ws => {
                ws.on('message', data => {
                    const message = JSON.parse(data.toString());
                    if (message.type === 'subscribe') {
                        subscriptions.push(message);
                        ws.send(JSON.stringify({ type: 'subscribed' }));
                        ws.send(JSON.stringify({ type: 'event', event: {
                            id: 'host-event', type: 'tool.start', data: {}, timestamp: 1,
                        } }));
                    }
                });
            });
        });
        await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
        url = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
        bridge = new EventBridgeClient({ initialDelayMs: 5, maxDelayMs: 10, multiplier: 2, maxRetries: 1 });
    });

    teardown(async () => {
        bridge.dispose();
        disposeEventBridgeClient();
        for (const socket of websocketServer.clients) {socket.terminate();}
        await new Promise<void>(resolve => websocketServer.close(() => resolve()));
        await new Promise<void>(resolve => server.close(() => resolve()));
    });

    test('authenticates, delivers events and rotates/removes the configured key', async () => {
        let received = 0;
        bridge.onAny(() => { received++; });
        bridge.connect(url, { categories: ['tool.start'] }, expectedKey);
        await until(() => received === 1);
        assert.strictEqual(attempts[0].url, '/ws/events');
        assert.strictEqual(attempts[0].headers.authorization, 'Bearer host-test-key');
        expectedKey = 'rotated-host-key';
        bridge.connect(url, undefined, expectedKey);
        await until(() => received === 2);
        assert.strictEqual(attempts[1].headers.authorization, 'Bearer rotated-host-key');
        expectedKey = undefined;
        bridge.connect(url);
        await until(() => received === 3);
        assert.strictEqual(attempts[2].headers.authorization, undefined);
        assert.ok(attempts.every(request => request.url === '/ws/events'));
    });

    for (const credential of [undefined, 'incorrect']) {
        test(`protected server rejects ${credential ? 'incorrect' : 'missing'} credentials without retry`, async () => {
            bridge.connect(url, undefined, credential);
            await until(() => bridge.getState() === ConnectionState.Error);
            // A new chat with identical configuration must not restart a denied upgrade.
            bridge.connect(url, undefined, credential);
            assert.strictEqual(attempts.length, 1);
            assert.strictEqual(subscriptions.length, 0);
        });
    }

    test('does not follow a redirect carrying the credential', async () => {
        status = 302;
        bridge.connect(url, undefined, expectedKey);
        await until(() => bridge.getState() === ConnectionState.Error);
        assert.strictEqual(attempts.length, 1);
        assert.strictEqual(attempts[0].url, '/ws/events');
        assert.strictEqual(subscriptions.length, 0);
    });

    test('ChatView follows the credential owner and never sends credentials to the webview', async () => {
        const client = new VictorClient(url, undefined, expectedKey);
        client.streamChat = async () => {};
        const provider = new ChatViewProvider(vscode.Uri.file('/nonexistent-victor-extension'), client);
        const messages: unknown[] = [];
        const view = {
            webview: { options: {}, html: '', cspSource: 'test',
                asWebviewUri: (uri: vscode.Uri) => uri,
                postMessage: (message: unknown) => { messages.push(message); return Promise.resolve(true); },
                onDidReceiveMessage: () => new vscode.Disposable(() => {}),
            }, onDidDispose: () => new vscode.Disposable(() => {}),
        } as unknown as vscode.WebviewView;
        try {
            provider.resolveWebviewView(view, {} as vscode.WebviewViewResolveContext, {} as vscode.CancellationToken);
            await provider.sendMessage('hello');
            await until(() => subscriptions.length === 1);
            expectedKey = 'rotated-host-key';
            client.setApiToken(expectedKey);
            await until(() => subscriptions.length === 2);
            expectedKey = undefined;
            client.setApiToken(undefined);
            await until(() => subscriptions.length === 3);
            assert.strictEqual(attempts[2].headers.authorization, undefined);
            const posted = JSON.stringify(messages);
            assert.ok(!posted.includes('host-test-key') && !posted.includes('rotated-host-key'));
            provider.dispose();
            client.setServerUrl('http://127.0.0.1:1');
            assert.strictEqual(getEventBridgeClient().getState(), ConnectionState.Disconnected);
            assert.strictEqual(attempts.length, 3);
        } finally {
            provider.dispose();
        }
    });

    // Local co-design smoke can point the real host at an ephemeral production
    // Victor router. Only a synthetic test key is accepted by this fixture.
    const productionUrl = process.env.VICTOR_EVENTBRIDGE_SMOKE_URL;
    if (productionUrl) {
        test('production Victor /ws/events accepts the host header and rejects a bad key', async () => {
            bridge.connect(productionUrl, undefined, 'contract-test-key');
            await until(() => bridge.getState() === ConnectionState.Connected);
            const ws = new WebSocket(productionUrl.replace(/^http/, 'ws') + '/ws/events', {
                headers: { Authorization: 'Bearer contract-test-key' },
            });
            try {
                await new Promise<void>((resolve, reject) => { ws.once('open', resolve); ws.once('error', reject); });
                const ack = new Promise<WebSocket.RawData>(resolve => ws.once('message', resolve));
                ws.send(JSON.stringify({ type: 'subscribe', categories: ['tool.start'] }));
                assert.strictEqual(JSON.parse((await ack).toString()).type, 'subscribed');
            } finally { ws.terminate(); }
            bridge.connect(productionUrl, undefined, 'incorrect');
            await until(() => bridge.getState() === ConnectionState.Error);
        });
    }
});
