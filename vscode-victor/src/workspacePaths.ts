/** Constrain proposed edits to the workspace, including existing symlinks. */
import * as fs from 'fs/promises';
import * as path from 'path';

function isWithin(root: string, target: string): boolean {
    const relative = path.relative(root, target);
    return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

export async function resolveWorkspaceFile(root: string, requestedPath: string): Promise<string> {
    if (!root || !requestedPath || requestedPath.includes('\0')) {
        throw new Error('A workspace and file path are required');
    }
    const workspace = path.resolve(root);
    const target = path.resolve(workspace, requestedPath);
    if (target === workspace || !isWithin(workspace, target)) {
        throw new Error('Proposed file is outside the workspace');
    }

    // New files may have missing parents. Find the nearest existing path with
    // lstat first, so a dangling symlink cannot be mistaken for a missing file.
    let ancestor = target;
    let exists = false;
    while (!exists) {
        try {
            await fs.lstat(ancestor);
            exists = true;
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'ENOENT') { throw error; }
            const parent = path.dirname(ancestor);
            if (parent === ancestor) { throw error; }
            ancestor = parent;
        }
    }
    const [realRoot, realAncestor] = await Promise.all([fs.realpath(workspace), fs.realpath(ancestor)]);
    if (!isWithin(realRoot, realAncestor)) {
        throw new Error('Proposed file resolves outside the workspace');
    }
    return target;
}
