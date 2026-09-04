import { defineConfig } from 'vitest/config';
import * as path from 'path';

// Fast, non-electron unit-test harness for the extension's logic modules.
// Modules that import the `vscode` API are aliased to a lightweight mock so they
// can be exercised in a plain Node process (the @vscode/test-electron suites in
// src/test/ remain the integration layer). Coverage is collected with v8 (c8).
export default defineConfig({
  test: {
    include: ['src/test-unit/**/*.unit.test.ts'],
    environment: 'node',
    alias: {
      vscode: path.resolve(__dirname, 'src/test-unit/_mocks/vscode.ts'),
    },
    coverage: {
      provider: 'v8',
      reporter: ['text-summary', 'text', 'html', 'lcov'],
      reportsDirectory: './coverage',
      include: ['src/**/*.ts'],
      exclude: [
        'src/test/**',
        'src/test-unit/**',
        '**/*.d.ts',
        'src/extension.ts',
      ],
      thresholds: {
        // Measured hosted-CI baseline (2026-08-26): lines 4.17%, functions
        // 18.75%, branches 60.77%. Ratchet these upward as tests are added.
        lines: 4,
        functions: 18,
        branches: 60,
      },
    },
  },
});
