import js from '@eslint/js';
import typescriptEslint from '@typescript-eslint/eslint-plugin';

export default [
  {
    ignores: ['out/**', 'dist/**', '**/*.d.ts'],
  },
  js.configs.recommended,
  ...typescriptEslint.configs['flat/recommended'],
  {
    files: ['src/**/*.ts'],
    rules: {
      '@typescript-eslint/naming-convention': [
        'warn',
        {
          selector: 'variable',
          format: ['camelCase', 'UPPER_CASE', 'PascalCase'],
          leadingUnderscore: 'allow',
        },
      ],
      semi: 'warn',
      curly: 'warn',
      eqeqeq: 'warn',
      'no-throw-literal': 'warn',
      '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    },
  },
  {
    files: ['src/test/**/*.ts', 'src/test-unit/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      'no-control-regex': 'off',
      '@typescript-eslint/no-unused-vars': 'off',
    },
  },
  {
    files: ['src/chatViewProvider.ts'],
    rules: {
      'no-useless-escape': 'off',
    },
  },
];
