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
      vscode: path.resolve(import.meta.dirname, 'src/test-unit/_mocks/vscode.ts'),
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
        // Re-measured with the complete src/** denominator under Node 24 and
        // Vitest 5 (2026-09-17): statements 5.33%, lines 5.41%, functions
        // 5.28%, branches 4.98%. Ratchet these floors upward as tests are added.
        statements: 5,
        lines: 5,
        functions: 5,
        branches: 4,
      },
    },
  },
});
