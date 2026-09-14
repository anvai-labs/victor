import { EventEmitter } from 'events';
import * as cp from 'child_process';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ServerManager } from '../serverManager';

vi.mock('child_process', async (importOriginal) => ({
    ...await importOriginal<typeof import('child_process')>(),
    spawn: vi.fn(),
}));

afterEach(() => vi.restoreAllMocks());

describe('backend process startup', () => {
    it.each(['/opt/Python env/bin/python', '/opt/python;echo marker']) (
        'preserves executable and argument boundaries for %s', async (pythonPath) => {
            const manager = new ServerManager({
                host: '127.0.0.1', port: 8765, autoStart: true,
                pythonPath, profile: 'profile;echo marker', mode: 'build',
            });
            const internals = manager as unknown as {
                discoverExistingServer(): Promise<number | null>;
                findAvailablePort(): Promise<number | null>;
                waitForServer(timeout: number): Promise<boolean>;
                writePidFile(port: number): void;
                startHealthCheck(): void;
            };
            vi.spyOn(internals, 'discoverExistingServer').mockResolvedValue(null);
            vi.spyOn(internals, 'findAvailablePort').mockResolvedValue(8765);
            vi.spyOn(internals, 'waitForServer').mockResolvedValue(true);
            vi.spyOn(internals, 'writePidFile').mockImplementation(() => {});
            vi.spyOn(internals, 'startHealthCheck').mockImplementation(() => {});
            vi.mocked(cp.spawn).mockReturnValue(new EventEmitter() as cp.ChildProcess);

            expect(await manager.start()).toBe(true);
            expect(cp.spawn).toHaveBeenCalledWith(pythonPath, [
                '-m', 'victor.ui.cli', 'serve', '--host', '127.0.0.1', '--port', '8765',
                '--log-level', 'INFO', '--profile', 'profile;echo marker', '--mode', 'build',
            ], expect.objectContaining({ shell: false }));
        }
    );
});
