/**
 * Narrow, leaf-level access to the extension's shared Victor client.
 *
 * Views and services need only the active VictorClient, which extension.ts
 * creates during activate(). Importing it from './extension' made every such
 * module part of a require cycle with extension.ts (which imports their
 * classes to construct them). This module depends on nothing but the client
 * type, so the dependency runs one way: extension.ts sets the client here,
 * and consumers read it from here.
 */

import type { VictorClient } from './victorClient';

/** The subset of the extension's providers that views and services use. */
export interface ClientProviders {
    readonly victorClient: VictorClient;
}

let current: ClientProviders | null = null;

/** Set by extension.ts on activation (and cleared with null on deactivation). */
export function setClientProviders(providers: ClientProviders | null): void {
    current = providers;
}

/** The active client providers, or null before activation / after deactivation. */
export function getProviders(): ClientProviders | null {
    return current;
}
