/**
 * EventBridge Client for VS Code
 *
 * Connects to Victor's EventBridge WebSocket endpoint for real-time
 * updates on tool execution, file changes, and other events.
 *
 * Architecture:
 *   Victor Server (EventBus) → EventBridge → WebSocket → This Client
 */

import * as vscode from 'vscode';
import WebSocket from 'ws';

export interface VictorEvent {
    id: string;
    type: string;
    data: Record<string, unknown>;
    timestamp: number;
}

export interface EventBridgeSubscription {
    categories?: string[];
    correlationId?: string;
}

export type EventHandler = (event: VictorEvent) => void;

export enum ConnectionState {
    Disconnected = 'disconnected',
    Connecting = 'connecting',
    Connected = 'connected',
    Reconnecting = 'reconnecting',
    Error = 'error'
}

interface ReconnectConfig {
    initialDelayMs: number;
    maxDelayMs: number;
    multiplier: number;
    maxRetries: number;
}

const DEFAULT_RECONNECT_CONFIG: ReconnectConfig = {
    initialDelayMs: 500,
    maxDelayMs: 30000,
    multiplier: 2,
    maxRetries: 10
};

/**
 * Client for connecting to Victor's EventBridge WebSocket endpoint.
 *
 * Provides:
 * - Automatic reconnection with exponential backoff
 * - Event filtering by type
 * - Connection state tracking
 * - Graceful cleanup
 */
export class EventBridgeClient {
    private ws: WebSocket | null = null;
    private serverUrl: string = '';
    private state: ConnectionState = ConnectionState.Disconnected;
    private reconnectAttempt: number = 0;
    private reconnectConfig: ReconnectConfig;
    private eventHandlers: Map<string, Set<EventHandler>> = new Map();
    private globalHandlers: Set<EventHandler> = new Set();
    private stateChangeHandlers: Set<(state: ConnectionState) => void> = new Set();
    private outputChannel: vscode.OutputChannel;
    private reconnectTimer: NodeJS.Timeout | null = null;
    private pingInterval: NodeJS.Timeout | null = null;
    private subscription: EventBridgeSubscription = { categories: ['all'] };
    private apiToken?: string;
    private generation = 0;
    private desired = false;
    private disposed = false;
    private terminalFailure = false;

    constructor(reconnectConfig: ReconnectConfig = DEFAULT_RECONNECT_CONFIG) {
        this.reconnectConfig = reconnectConfig;
        this.outputChannel = vscode.window.createOutputChannel('Victor Events');
    }

    /**
     * Connect to the EventBridge WebSocket endpoint.
     *
     * @param serverUrl Base URL of the Victor server (e.g., http://127.0.0.1:8765)
     */
    connect(serverUrl: string, subscription?: EventBridgeSubscription, apiToken?: string): void {
        if (this.disposed) {
            return;
        }
        let wsUrl: string;
        try {
            const url = new URL(serverUrl);
            if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password ||
                url.search || url.hash || (apiToken && /[\r\n]/.test(apiToken))) {
                throw new Error('Invalid EventBridge connection configuration');
            }
            url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
            url.pathname = url.pathname.replace(/\/+$/, '') + '/ws/events';
            wsUrl = url.toString();
        } catch {
            this.disconnect();
            if (this.desired || this.disposed) {
                return;
            }
            this.log('Invalid EventBridge connection configuration');
            this.setState(ConnectionState.Error);
            return;
        }
        if (subscription) {
            this.subscription = {
                categories: [...(subscription.categories?.length ? subscription.categories : ['all'])],
                correlationId: subscription.correlationId,
            };
        }
        const changed = this.serverUrl !== wsUrl || this.apiToken !== apiToken || !this.desired;
        if (changed) {
            this.retireSocket();
            this.serverUrl = wsUrl;
            this.apiToken = apiToken;
            this.desired = true;
            this.terminalFailure = false;
            this.reconnectAttempt = 0;
        }
        if (this.terminalFailure) {
            return;
        }
        if (this.ws || this.reconnectTimer) {
            this.sendSubscription();
            return;
        }
        this.doConnect();
    }

    /** Stop intentionally; stale socket callbacks cannot restart the connection. */
    disconnect(): void {
        this.desired = false;
        this.retireSocket();
        this.serverUrl = '';
        this.apiToken = undefined;
        this.setState(ConnectionState.Disconnected);
    }

    private retireSocket(): void {
        ++this.generation;
        this.stopReconnect();
        this.stopPing();
        const old = this.ws;
        this.ws = null;
        if (old?.readyState === WebSocket.CONNECTING) {
            old.terminate();
        }
        else {old?.close();}
    }

    private isActive(generation: number): boolean {
        return this.desired && !this.disposed && this.generation === generation;
    }

    /**
     * Get the current connection state.
     */
    getState(): ConnectionState {
        return this.state;
    }

    getSubscription(): EventBridgeSubscription {
        return {
            categories: [...(this.subscription.categories || ['all'])],
            correlationId: this.subscription.correlationId,
        };
    }

    subscribe(categories: string[] = ['all'], correlationId?: string): void {
        this.subscription = {
            categories: categories.length > 0 ? [...categories] : ['all'],
            correlationId,
        };
        this.sendSubscription();
    }

    /**
     * Subscribe to specific event types.
     *
     * @param eventType Event type to listen for (e.g., "tool.start", "file.modified")
     * @param handler Callback function to handle the event
     * @returns Disposable to unsubscribe
     */
    on(eventType: string, handler: EventHandler): vscode.Disposable {
        if (!this.eventHandlers.has(eventType)) {
            this.eventHandlers.set(eventType, new Set());
        }
        this.eventHandlers.get(eventType)!.add(handler);

        return new vscode.Disposable(() => {
            this.eventHandlers.get(eventType)?.delete(handler);
        });
    }

    /**
     * Subscribe to all events.
     *
     * @param handler Callback function to handle all events
     * @returns Disposable to unsubscribe
     */
    onAny(handler: EventHandler): vscode.Disposable {
        this.globalHandlers.add(handler);
        return new vscode.Disposable(() => {
            this.globalHandlers.delete(handler);
        });
    }

    /**
     * Subscribe to connection state changes.
     *
     * @param handler Callback function for state changes
     * @returns Disposable to unsubscribe
     */
    onStateChange(handler: (state: ConnectionState) => void): vscode.Disposable {
        this.stateChangeHandlers.add(handler);
        return new vscode.Disposable(() => {
            this.stateChangeHandlers.delete(handler);
        });
    }

    /**
     * Show the output channel.
     */
    showOutput(): void {
        this.outputChannel.show();
    }

    /**
     * Dispose of resources.
     */
    dispose(): void {
        if (this.disposed) {
            return;
        }
        this.disposed = true;
        this.disconnect();
        this.eventHandlers.clear();
        this.globalHandlers.clear();
        this.stateChangeHandlers.clear();
        this.outputChannel.dispose();
    }

    // --- Private Methods ---

    private doConnect(): void {
        if (!this.desired || this.disposed || this.terminalFailure || this.ws) {
            return;
        }
        this.stopReconnect();
        const generation = ++this.generation;
        this.setState(ConnectionState.Connecting);
        // A synchronous state listener may disconnect or replace configuration.
        if (!this.isActive(generation)) {
            return;
        }
        this.log('Connecting to EventBridge');
        try {
            const socket = new WebSocket(this.serverUrl, {
                headers: this.apiToken ? { Authorization: `Bearer ${this.apiToken}` } : {},
                followRedirects: false,
                handshakeTimeout: 10000,
            });
            this.ws = socket;
            const current = () => this.isActive(generation) && this.ws === socket;
            socket.on('open', () => {
                if (!current()) {
                    return;
                }
                this.log('Connected to EventBridge');
                this.reconnectAttempt = 0;
                this.setState(ConnectionState.Connected);
                if (!current()) {
                    return;
                }
                this.startPing(socket, current);
                this.sendSubscription();
            });
            socket.on('message', (data: WebSocket.Data) => {
                if (!current()) {
                    return;
                }
                try {
                    const message = JSON.parse(data.toString());
                    if (message?.type === 'event' && message.event && typeof message.event.type === 'string') {
                        this.handleEvent(message.event, current);
                    } else if (message?.type === 'subscribed') {
                        this.log('EventBridge subscription acknowledged');
                    }
                } catch {
                    this.log('Invalid EventBridge message');
                }
            });
            socket.on('unexpected-response', (request, response) => {
                if (!current()) {
                    response.destroy();
                    request.destroy();
                    return;
                }
                const status = response.statusCode ?? 0;
                this.terminalFailure = status >= 300 && status < 500;
                // Taking ownership of unexpected-response requires releasing both
                // streams; otherwise ws leaves the handshake pending indefinitely.
                this.retireSocket();
                response.destroy();
                request.destroy();
                this.log(`EventBridge upgrade rejected (HTTP ${status})`);
                if (this.terminalFailure) {this.setState(ConnectionState.Error);}
                else {this.scheduleReconnect();}
            });
            socket.on('close', (code) => {
                if (!current()) {
                    return;
                }
                this.ws = null;
                this.stopPing();
                this.log(`EventBridge connection closed (code ${code})`);
                if ([1008, 4401, 4403].includes(code)) {
                    this.terminalFailure = true;
                    ++this.generation;
                    this.setState(ConnectionState.Error);
                } else {
                    this.scheduleReconnect();
                }
            });
            socket.on('error', () => {
                if (!current()) {
                    return;
                }
                // Remote diagnostics may echo credentials; keep transport logs structural.
                this.log('EventBridge transport error');
                this.setState(ConnectionState.Error);
            });
        } catch {
            if (!this.isActive(generation)) {
                return;
            }
            this.log('EventBridge connection failed');
            this.scheduleReconnect();
        }
    }

    private handleEvent(event: VictorEvent, current: () => boolean): void {
        this.log('EventBridge event received');
        for (const handler of [...(this.eventHandlers.get(event.type) ?? []), ...this.globalHandlers]) {
            if (!current()) {
                return;
            }
            try {
                handler(event);
            } catch {
                this.log('EventBridge event handler failed');
            }
        }
    }

    private scheduleReconnect(): void {
        if (!this.desired || this.disposed || this.terminalFailure) {
            return;
        }
        this.stopReconnect();
        const generation = this.generation;
        if (this.reconnectAttempt >= this.reconnectConfig.maxRetries) {
            this.log('EventBridge reconnection attempts exhausted');
            this.terminalFailure = true;
            this.setState(ConnectionState.Error);
            return;
        }
        this.reconnectAttempt++;
        const delay = Math.min(
            this.reconnectConfig.initialDelayMs * Math.pow(this.reconnectConfig.multiplier, this.reconnectAttempt - 1),
            this.reconnectConfig.maxDelayMs
        );
        this.setState(ConnectionState.Reconnecting);
        if (!this.isActive(generation)) {
            return;
        }
        this.reconnectTimer = setTimeout(() => {
            if (!this.isActive(generation)) {
                return;
            }
            this.reconnectTimer = null;
            this.doConnect();
        }, delay);
    }

    private stopReconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    private startPing(socket: WebSocket, current: () => boolean): void {
        this.stopPing();
        // Send ping every 30 seconds to keep connection alive
        this.pingInterval = setInterval(() => {
            if (current() && socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    private sendSubscription(): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            return;
        }

        const categories = this.subscription.categories && this.subscription.categories.length > 0
            ? this.subscription.categories
            : ['all'];
        this.ws.send(JSON.stringify({
            type: 'subscribe',
            categories,
            correlation_id: this.subscription.correlationId,
        }));
    }

    private stopPing(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }

    private setState(state: ConnectionState): void {
        if (this.state !== state) {
            this.state = state;
            for (const handler of this.stateChangeHandlers) {
                if (this.state !== state) {
                    break;
                }
                try {
                    handler(state);
                } catch {
                    this.log('EventBridge state handler failed');
                }
            }
        }
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }
}

/**
 * Create an EventBridge client singleton.
 */
let eventBridgeInstance: EventBridgeClient | null = null;

export function getEventBridgeClient(): EventBridgeClient {
    if (!eventBridgeInstance) {
        eventBridgeInstance = new EventBridgeClient();
    }
    return eventBridgeInstance;
}

export function disposeEventBridgeClient(): void {
    if (eventBridgeInstance) {
        eventBridgeInstance.dispose();
        eventBridgeInstance = null;
    }
}
