/**
 * Victor Server Manager
 *
 * Handles automatic spawning, lifecycle management, and health monitoring
 * of the Victor Python backend server.
 *
 * Phase 3 Optimizations:
 * - Server discovery: Detect existing servers on multiple ports
 * - Port fallback: Try alternate ports if primary is occupied
 * - Exponential backoff: Improved reconnection with backoff strategy
 * - Multi-window sharing: PID file for server coordination across VS Code windows
 */

import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as net from 'net';

export interface ServerConfig {
    host: string;
    port: number;
    autoStart: boolean;
    serverUrl?: string;
    pythonPath?: string;
    victorPath?: string;
    fallbackPorts?: number[];
    profile?: string;
    mode?: string;
}

export enum ServerStatus {
    Stopped = 'stopped',
    Starting = 'starting',
    Running = 'running',
    Reconnecting = 'reconnecting',
    Error = 'error'
}

interface ServerCommand {
    executable: string;
    args: string[];
}

interface ServerPidFile {
    pid: number;
    port: number;
    host: string;
    startTime: number;
    workspacePath?: string;
}

// Exponential backoff configuration
const BACKOFF_CONFIG = {
    initialDelayMs: 100,
    maxDelayMs: 30000,
    multiplier: 2,
    maxRetries: 10
};

// Fallback ports to try
const DEFAULT_FALLBACK_PORTS = [8765, 8766, 8767, 8768, 8000];

function normalizeServerUrl(serverUrl?: string): string | null {
    const trimmed = serverUrl?.trim();
    if (!trimmed) {
        return null;
    }
    return trimmed.replace(/\/+$/, '');
}

export class ServerManager {
    private serverProcess: cp.ChildProcess | null = null;
    private status: ServerStatus = ServerStatus.Stopped;
    private statusChangeCallbacks: ((status: ServerStatus) => void)[] = [];
    private config: ServerConfig;
    private outputChannel: vscode.OutputChannel;
    private healthCheckInterval: NodeJS.Timeout | null = null;
    private reconnectAttempt: number = 0;
    private activePort: number;
    private isExternalServer: boolean = false;
    private pidFilePath: string;
    private remoteServerUrl: string | null = null;

    constructor(config: ServerConfig) {
        this.config = {
            ...config,
            fallbackPorts: config.fallbackPorts || DEFAULT_FALLBACK_PORTS
        };
        this.remoteServerUrl = normalizeServerUrl(config.serverUrl);
        this.activePort = config.port;
        if (this.remoteServerUrl) {
            try {
                const parsed = new URL(this.remoteServerUrl);
                this.config.host = parsed.hostname;
                this.activePort = parsed.port
                    ? Number(parsed.port)
                    : parsed.protocol === 'https:' ? 443 : 80;
            } catch {
                // Keep configured host/port if URL parsing fails.
            }
        }
        this.outputChannel = vscode.window.createOutputChannel('Victor Server');
        this.pidFilePath = path.join(os.homedir(), '.victor', 'server.pid');
    }

    /**
     * Get the current server status
     */
    getStatus(): ServerStatus {
        return this.status;
    }

    /**
     * Get the active port (may differ from config if fallback was used)
     */
    getActivePort(): number {
        return this.activePort;
    }

    /**
     * Check if connected to an external (not self-spawned) server
     */
    isExternal(): boolean {
        return this.isExternalServer;
    }

    /**
     * Register a callback for status changes
     */
    onStatusChange(callback: (status: ServerStatus) => void): void {
        this.statusChangeCallbacks.push(callback);
    }

    /**
     * Get the server URL
     */
    getServerUrl(): string {
        return this.remoteServerUrl || `http://${this.config.host}:${this.activePort}`;
    }

    /**
     * Check if server is running by hitting health endpoint
     */
    async checkHealth(port?: number): Promise<boolean> {
        const targetUrl = this.remoteServerUrl && port === undefined
            ? `${this.remoteServerUrl}/health`
            : `http://${this.config.host}:${port || this.activePort}/health`;
        try {
            const response = await fetch(targetUrl, {
                method: 'GET',
                signal: AbortSignal.timeout(2000)
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * Discover an existing running server
     * Checks configured port and fallback ports
     */
    async discoverExistingServer(): Promise<number | null> {
        if (this.remoteServerUrl) {
            return (await this.checkHealth()) ? this.activePort : null;
        }

        const portsToCheck = [this.config.port, ...(this.config.fallbackPorts || [])];
        const uniquePorts = [...new Set(portsToCheck)];

        this.log(`Discovering existing servers on ports: ${uniquePorts.join(', ')}`);

        // First check PID file for existing server
        const pidInfo = this.readPidFile();
        if (pidInfo && await this.checkHealth(pidInfo.port)) {
            this.log(`Found existing server from PID file on port ${pidInfo.port}`);
            return pidInfo.port;
        }

        // Check each port in parallel
        const checks = uniquePorts.map(async (port) => {
            const healthy = await this.checkHealth(port);
            return { port, healthy };
        });

        const results = await Promise.all(checks);
        const runningServer = results.find(r => r.healthy);

        if (runningServer) {
            this.log(`Discovered existing server on port ${runningServer.port}`);
            return runningServer.port;
        }

        return null;
    }

    /**
     * Find an available port from fallback list
     */
    async findAvailablePort(): Promise<number | null> {
        const portsToCheck = [this.config.port, ...(this.config.fallbackPorts || [])];
        const uniquePorts = [...new Set(portsToCheck)];

        for (const port of uniquePorts) {
            const inUse = await this.isPortInUse(port);
            if (!inUse) {
                return port;
            }
            this.log(`Port ${port} is in use, trying next...`);
        }

        return null;
    }

    /**
     * Check if a port is in use
     */
    private async isPortInUse(port: number): Promise<boolean> {
        return new Promise((resolve) => {
            const server = net.createServer();

            server.once('error', (err: NodeJS.ErrnoException) => {
                if (err.code === 'EADDRINUSE') {
                    resolve(true);
                } else {
                    resolve(false);
                }
            });

            server.once('listening', () => {
                server.close();
                resolve(false);
            });

            server.listen(port, this.config.host);
        });
    }

    /**
     * Start or connect to server
     * First tries to discover existing server, then spawns new one if needed
     */
    async start(): Promise<boolean> {
        if (this.remoteServerUrl) {
            this.isExternalServer = true;
            if (await this.checkHealth()) {
                this.setStatus(ServerStatus.Running);
                this.log(`Connected to configured remote server ${this.remoteServerUrl}`);
                this.startHealthCheck();
                return true;
            }

            this.setStatus(ServerStatus.Error);
            this.log(`ERROR: Remote Victor server unavailable at ${this.remoteServerUrl}`);
            return false;
        }

        // First, try to discover an existing server
        const existingPort = await this.discoverExistingServer();
        if (existingPort !== null) {
            this.activePort = existingPort;
            this.isExternalServer = true;
            this.setStatus(ServerStatus.Running);
            this.log(`Connected to existing server on port ${existingPort}`);
            this.startHealthCheck();
            return true;
        }

        // No existing server, spawn a new one
        this.isExternalServer = false;
        return this.spawnServer();
    }

    /**
     * Spawn a new server process
     */
    private async spawnServer(): Promise<boolean> {
        // Find an available port
        const availablePort = await this.findAvailablePort();
        if (availablePort === null) {
            this.setStatus(ServerStatus.Error);
            this.log('ERROR: No available ports found');
            vscode.window.showErrorMessage(
                'Victor: No available ports. Please stop other servers or configure different ports.'
            );
            return false;
        }

        this.activePort = availablePort;
        this.setStatus(ServerStatus.Starting);
        this.log(`Starting Victor server on port ${availablePort}...`);

        // Find the victor executable
        const victorCommand = await this.findVictorCommand();
        if (!victorCommand) {
            this.setStatus(ServerStatus.Error);
            this.log('ERROR: Could not find victor command');
            vscode.window.showErrorMessage(
                'Victor not found. Please install with: pip install -e ".[dev]"'
            );
            return false;
        }

        this.log(`Using executable: ${victorCommand.executable}`);

        // Build spawn arguments
        const args = [
            ...victorCommand.args,
            'serve',
            '--host', this.config.host,
            '--port', availablePort.toString(),
            '--log-level', 'INFO',
            '--profile', this.config.profile || 'default',
            '--mode', this.config.mode || 'build'
        ];

        try {
            // Spawn the server process
            this.serverProcess = cp.spawn(victorCommand.executable, args, {
                shell: false,
                env: {
                    ...process.env,
                    PYTHONUNBUFFERED: '1'
                },
                cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
            });

            // Write PID file for multi-window coordination
            this.writePidFile(availablePort);

            // Handle stdout
            this.serverProcess.stdout?.on('data', (data: Buffer) => {
                this.log(data.toString().trim());
            });

            // Handle stderr
            this.serverProcess.stderr?.on('data', (data: Buffer) => {
                this.log(`[stderr] ${data.toString().trim()}`);
            });

            // Handle process exit
            this.serverProcess.on('exit', (code, signal) => {
                this.log(`Server process exited (code: ${code}, signal: ${signal})`);
                this.serverProcess = null;
                this.removePidFile();
                this.setStatus(ServerStatus.Stopped);
                this.stopHealthCheck();
            });

            // Handle errors
            this.serverProcess.on('error', (err) => {
                this.log(`ERROR: ${err.message}`);
                this.setStatus(ServerStatus.Error);
            });

            // Wait for server to be ready
            const ready = await this.waitForServer(15000);
            if (ready) {
                this.setStatus(ServerStatus.Running);
                this.log(`Server started successfully on port ${availablePort}`);
                this.reconnectAttempt = 0;
                this.startHealthCheck();
                return true;
            } else {
                this.setStatus(ServerStatus.Error);
                this.log('ERROR: Server failed to start within timeout');
                await this.stop();
                return false;
            }

        } catch (error) {
            this.setStatus(ServerStatus.Error);
            this.log(`ERROR: Failed to spawn server: ${error}`);
            return false;
        }
    }

    /**
     * Stop the server
     */
    async stop(): Promise<void> {
        this.stopHealthCheck();

        // Don't stop external servers
        if (this.isExternalServer || this.remoteServerUrl) {
            this.log('Disconnecting from external server (not stopping it)');
            this.setStatus(ServerStatus.Stopped);
            return;
        }

        if (this.serverProcess) {
            this.log('Stopping Victor server...');

            // Try graceful shutdown via API first
            try {
                await fetch(`${this.getServerUrl()}/shutdown`, {
                    method: 'POST',
                    signal: AbortSignal.timeout(2000)
                });
            } catch {
                // Ignore errors, will force kill below
            }

            // Wait a bit for graceful shutdown
            await this.sleep(500);

            // Force kill if still running
            if (this.serverProcess && !this.serverProcess.killed) {
                this.serverProcess.kill('SIGTERM');
                await this.sleep(500);

                if (this.serverProcess && !this.serverProcess.killed) {
                    this.serverProcess.kill('SIGKILL');
                }
            }

            this.serverProcess = null;
            this.removePidFile();
        }

        this.setStatus(ServerStatus.Stopped);
        this.log('Server stopped');
    }

    /**
     * Restart the server
     */
    async restart(): Promise<boolean> {
        await this.stop();
        return this.start();
    }

    /**
     * Attempt reconnection with exponential backoff
     */
    private async attemptReconnect(): Promise<void> {
        if (this.reconnectAttempt >= BACKOFF_CONFIG.maxRetries) {
            this.log(`Max reconnection attempts (${BACKOFF_CONFIG.maxRetries}) reached`);
            this.setStatus(ServerStatus.Error);
            vscode.window.showWarningMessage(
                'Victor server connection lost. Click to restart.',
                'Restart Server'
            ).then(selection => {
                if (selection === 'Restart Server') {
                    this.restart();
                }
            });
            return;
        }

        this.reconnectAttempt++;
        const delay = Math.min(
            BACKOFF_CONFIG.initialDelayMs * Math.pow(BACKOFF_CONFIG.multiplier, this.reconnectAttempt - 1),
            BACKOFF_CONFIG.maxDelayMs
        );

        this.log(`Reconnection attempt ${this.reconnectAttempt}/${BACKOFF_CONFIG.maxRetries} in ${delay}ms`);
        this.setStatus(ServerStatus.Reconnecting);

        await this.sleep(delay);

        // Try to reconnect
        if (await this.checkHealth()) {
            this.log('Server reconnected');
            this.setStatus(ServerStatus.Running);
            this.reconnectAttempt = 0;
        } else {
            // If external server, try to discover another
            if (this.isExternalServer) {
                const newPort = await this.discoverExistingServer();
                if (newPort !== null) {
                    this.activePort = newPort;
                    this.log(`Found server on new port ${newPort}`);
                    this.setStatus(ServerStatus.Running);
                    this.reconnectAttempt = 0;
                    return;
                }
            }
            // Continue retry loop
            this.attemptReconnect();
        }
    }

    /**
     * Dispose of resources
     *
     * Note: This performs a synchronous cleanup. The async stop() is called
     * but not awaited to comply with VS Code's Disposable interface.
     * The server process is killed immediately if running.
     */
    dispose(): void {
        // Stop health check first
        this.stopHealthCheck();

        // Kill the server process immediately (sync cleanup)
        if (this.serverProcess && !this.serverProcess.killed) {
            try {
                this.serverProcess.kill('SIGTERM');
            } catch {
                // Ignore errors during cleanup
            }
            this.serverProcess = null;
            this.removePidFile();
        }

        this.setStatus(ServerStatus.Stopped);
        this.outputChannel.dispose();
    }

    /**
     * Show the output channel
     */
    showOutput(): void {
        this.outputChannel.show();
    }

    // --- PID File Management ---

    private writePidFile(port: number): void {
        try {
            const pidDir = path.dirname(this.pidFilePath);
            if (!fs.existsSync(pidDir)) {
                fs.mkdirSync(pidDir, { recursive: true });
            }

            const pidInfo: ServerPidFile = {
                pid: this.serverProcess?.pid || 0,
                port: port,
                host: this.config.host,
                startTime: Date.now(),
                workspacePath: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath
            };

            fs.writeFileSync(this.pidFilePath, JSON.stringify(pidInfo, null, 2));
            this.log(`PID file written: ${this.pidFilePath}`);
        } catch (error) {
            this.log(`Warning: Could not write PID file: ${error}`);
        }
    }

    private readPidFile(): ServerPidFile | null {
        try {
            if (fs.existsSync(this.pidFilePath)) {
                const content = fs.readFileSync(this.pidFilePath, 'utf-8');
                return JSON.parse(content);
            }
        } catch {
            // Ignore read errors
        }
        return null;
    }

    private removePidFile(): void {
        try {
            if (fs.existsSync(this.pidFilePath)) {
                fs.unlinkSync(this.pidFilePath);
                this.log('PID file removed');
            }
        } catch {
            // Ignore removal errors
        }
    }

    // --- Private methods ---

    private setStatus(status: ServerStatus): void {
        if (this.status !== status) {
            this.status = status;
            for (const callback of this.statusChangeCallbacks) {
                callback(status);
            }
        }
    }

    private log(message: string): void {
        const timestamp = new Date().toISOString();
        this.outputChannel.appendLine(`[${timestamp}] ${message}`);
    }

    /**
     * Find the victor command
     * Tries: victor in PATH, python -m victor.ui.cli, bundled binary
     */
    private async findVictorCommand(): Promise<ServerCommand | null> {
        // A configured path is one executable, including any spaces or shell
        // characters. Interpreter arguments are supplied separately to spawn.
        if (this.config.pythonPath) {
            return { executable: this.config.pythonPath, args: ['-m', 'victor.ui.cli'] };
        }

        for (const executable of ['victor', 'vic']) {
            if (await this.commandExists(executable)) {
                return { executable, args: [] };
            }
        }

        for (const executable of ['python3', 'python']) {
            if (await this.commandExists(executable)) {
                try {
                    cp.execFileSync(executable, ['-c', 'import victor'], { timeout: 5000 });
                    return { executable, args: ['-m', 'victor.ui.cli'] };
                } catch {
                    // Try the next interpreter if Victor is not installed.
                }
            }
        }

        if (this.config.victorPath && fs.existsSync(this.config.victorPath)) {
            return { executable: this.config.victorPath, args: [] };
        }
        return null;
    }

    /** Check a literal executable name in PATH without shell expansion. */
    private async commandExists(command: string): Promise<boolean> {
        try {
            const checkCmd = process.platform === 'win32' ? 'where' : 'which';
            cp.execFileSync(checkCmd, [command], { stdio: 'ignore', timeout: 2000 });
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Wait for server to respond to health checks
     */
    private async waitForServer(timeoutMs: number): Promise<boolean> {
        const startTime = Date.now();
        const checkInterval = 500;

        while (Date.now() - startTime < timeoutMs) {
            if (await this.checkHealth()) {
                return true;
            }
            await this.sleep(checkInterval);
        }

        return false;
    }

    /**
     * Start periodic health checks with exponential backoff on failure
     */
    private startHealthCheck(): void {
        this.stopHealthCheck();
        this.healthCheckInterval = setInterval(async () => {
            const healthy = await this.checkHealth();
            if (!healthy && this.status === ServerStatus.Running) {
                this.log('WARNING: Health check failed, attempting reconnection...');
                this.attemptReconnect();
            } else if (healthy && this.status === ServerStatus.Reconnecting) {
                this.log('Server recovered during health check');
                this.setStatus(ServerStatus.Running);
                this.reconnectAttempt = 0;
            }
        }, 30000); // Check every 30 seconds
    }

    /**
     * Stop periodic health checks
     */
    private stopHealthCheck(): void {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
    }

    private sleep(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

/**
 * Create a server manager with VS Code configuration
 */
export function createServerManager(): ServerManager {
    const config = vscode.workspace.getConfiguration('victor');

    const workspacePath = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const detectedPython = detectWorkspacePython(workspacePath);
    const configuredPython = config.get<string>('pythonPath') || undefined;
    const pythonPath = configuredPython || detectedPython;

    return new ServerManager({
        host: '127.0.0.1',
        port: config.get('serverPort', 8765),
        autoStart: config.get('autoStart', false),
        serverUrl: config.get('serverUrl') || undefined,
        pythonPath,
        victorPath: config.get('victorPath'),
        fallbackPorts: config.get('fallbackPorts', DEFAULT_FALLBACK_PORTS),
        profile: config.get('profile', 'default'),
        mode: config.get('mode', 'build')
    });
}

/**
 * Detect a workspace-local Python (common venv layouts)
 */
function detectWorkspacePython(workspacePath?: string): string | undefined {
    if (!workspacePath) {return undefined;}

    const candidates = [
        path.join(workspacePath, 'venv', 'bin', 'python'),
        path.join(workspacePath, '.venv', 'bin', 'python'),
        path.join(workspacePath, 'venv', 'Scripts', 'python.exe'),
        path.join(workspacePath, '.venv', 'Scripts', 'python.exe'),
    ];

    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    return undefined;
}
