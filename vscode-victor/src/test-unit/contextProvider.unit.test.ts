import * as cp from 'child_process';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ContextProvider } from '../contextProvider';

vi.mock('child_process', async (importOriginal) => ({
    ...await importOriginal<typeof import('child_process')>(),
    exec: vi.fn(),
    execFile: vi.fn(),
}));
afterEach(() => vi.clearAllMocks());

describe('Git context preview', () => {
    it.each(['status;echo${IFS}marker', '$(echo${IFS}marker)', 'checkout', 'config', '--help', '__proto__']) (
        'rejects unsupported commands without launching a process: %s', async (query) => {
            expect(await new ContextProvider().gatherContext(`Preview @git:${query}`)).toEqual([]);
            expect(cp.exec).not.toHaveBeenCalled();
            expect(cp.execFile).not.toHaveBeenCalled();
        }
    );

    it.each(['', ':status', ':diff', ':log', ':show', ':branch']) (
        'runs a fixed read-only argument list for @git%s', async (suffix) => {
            vi.mocked(cp.execFile).mockImplementation(((file, args, options, callback) => {
                const done = callback as (error: Error | null, stdout: string, stderr: string) => void;
                done(null, 'Git context output', '');
                return {};
            }) as typeof cp.execFile);
            const items = await new ContextProvider().gatherContext(`Preview @git${suffix}`);
            expect(items[0].content).toBe('Git context output');
            expect(cp.exec).not.toHaveBeenCalled();
            const [file, args, options] = vi.mocked(cp.execFile).mock.calls[0];
            expect(file).toBe('git');
            expect(args).toEqual(expect.arrayContaining(['--no-pager', '-c', 'core.fsmonitor=false']));
            expect(options).toEqual(expect.objectContaining({ timeout: 10000 }));
            if (suffix === ':diff' || suffix === ':show') {
                expect(args).toEqual(expect.arrayContaining(['--no-ext-diff', '--no-textconv']));
            }
        }
    );
});
