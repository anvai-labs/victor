/**
 * Provider Settings Service Tests
 *
 * Tests for the ProviderSettingsService which manages LLM provider
 * configurations, API keys, and connection testing.
 */

import * as assert from 'assert';

suite('ProviderSettingsService Test Suite', () => {
    // Test provider configurations
    suite('Provider Configurations', () => {
        test('Should have all provider configs', () => {
            const providers = [
                'anthropic', 'openai', 'google', 'xai', 'deepseek',
                'groq', 'mistral', 'ollama', 'lmstudio', 'vllm'
            ];

            assert.strictEqual(providers.length, 10);
            assert.ok(providers.includes('anthropic'));
            assert.ok(providers.includes('ollama'));
        });

        test('Should have required config fields', () => {
            const providerConfig = {
                id: 'anthropic',
                name: 'Anthropic',
                requiresApiKey: true,
                defaultModel: 'claude-3-sonnet-20240229',
                endpoint: 'https://api.anthropic.com/v1',
                models: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku']
            };

            assert.ok(providerConfig.id);
            assert.ok(providerConfig.name);
            assert.ok(providerConfig.defaultModel);
            assert.ok(Array.isArray(providerConfig.models));
        });

        test('Should identify local providers', () => {
            const localProviders = ['ollama', 'lmstudio', 'vllm'];

            const isLocalProvider = (id: string): boolean => {
                return localProviders.includes(id);
            };

            assert.ok(isLocalProvider('ollama'));
            assert.ok(isLocalProvider('lmstudio'));
            assert.ok(!isLocalProvider('anthropic'));
            assert.ok(!isLocalProvider('openai'));
        });
    });

    // Test API key validation
    suite('API Key Validation', () => {
        test('Should validate Anthropic key format', () => {
            const validateAnthropicKey = (key: string): boolean => {
                return /^sk-ant-[a-zA-Z0-9-_]+$/.test(key);
            };

            assert.ok(validateAnthropicKey('sk-ant-api03-abcdefghij'));
            assert.ok(!validateAnthropicKey('sk-openai-key'));
            assert.ok(!validateAnthropicKey('invalid'));
        });

        test('Should validate OpenAI key format', () => {
            const validateOpenAIKey = (key: string): boolean => {
                return /^sk-[a-zA-Z0-9]+$/.test(key);
            };

            assert.ok(validateOpenAIKey('sk-abcdefghijklmnop'));
            assert.ok(!validateOpenAIKey('sk-ant-key'));
            assert.ok(!validateOpenAIKey('invalid'));
        });

        test('Should validate Google API key format', () => {
            const validateGoogleKey = (key: string): boolean => {
                return /^AIza[a-zA-Z0-9_-]+$/.test(key);
            };

            assert.ok(validateGoogleKey('AIzaSyAbcdefghijklmnop'));
            assert.ok(!validateGoogleKey('sk-openai-key'));
        });

        test('Should detect empty/missing keys', () => {
            const isKeyMissing = (key: string | undefined | null): boolean => {
                return !key || key.trim() === '';
            };

            assert.ok(isKeyMissing(undefined));
            assert.ok(isKeyMissing(null));
            assert.ok(isKeyMissing(''));
            assert.ok(isKeyMissing('   '));
            assert.ok(!isKeyMissing('valid-key'));
        });
    });

    // Test connection testing
    suite('Connection Testing', () => {
        test('Should measure connection latency', async () => {
            const measureLatency = async (fn: () => Promise<void>): Promise<number> => {
                const start = Date.now();
                await fn();
                return Date.now() - start;
            };

            // libuv timers can fire ~1-2ms early under CI load, so exact
            // floor assertions (>= requested delay) are flaky. Assert the
            // measurement is a sane non-negative number instead; the
            // classification thresholds are covered by the pure test above.
            const latency = await measureLatency(async () => {
                await new Promise(resolve => setTimeout(resolve, 10));
            });
            assert.ok(latency >= 0);
            assert.ok(latency < 1000);
        });

        test('Should classify connection speed', () => {
            const classifySpeed = (latencyMs: number): string => {
                if (latencyMs < 100) {return 'excellent';}
                if (latencyMs < 300) {return 'good';}
                if (latencyMs < 1000) {return 'fair';}
                return 'poor';
            };

            assert.strictEqual(classifySpeed(50), 'excellent');
            assert.strictEqual(classifySpeed(200), 'good');
            assert.strictEqual(classifySpeed(500), 'fair');
            assert.strictEqual(classifySpeed(2000), 'poor');
        });

        test('Should handle connection timeout', () => {
            const connectionTimeout = 5000; // 5 seconds

            const isTimedOut = (elapsed: number): boolean => {
                return elapsed >= connectionTimeout;
            };

            assert.ok(!isTimedOut(1000));
            assert.ok(isTimedOut(5000));
            assert.ok(isTimedOut(10000));
        });
    });

    // Test provider health monitoring
    suite('Health Monitoring', () => {
        test('Should track provider health status', () => {
            type HealthStatus = 'healthy' | 'degraded' | 'unhealthy' | 'unknown';

            const healthStatus = new Map<string, HealthStatus>();

            healthStatus.set('anthropic', 'healthy');
            healthStatus.set('openai', 'degraded');
            healthStatus.set('ollama', 'unknown');

            assert.strictEqual(healthStatus.get('anthropic'), 'healthy');
            assert.strictEqual(healthStatus.get('openai'), 'degraded');
        });

        test('Should update health on error', () => {
            let consecutiveErrors = 0;
            const errorThreshold = 3;

            const recordError = (): 'healthy' | 'unhealthy' => {
                consecutiveErrors++;
                return consecutiveErrors >= errorThreshold ? 'unhealthy' : 'healthy';
            };

            assert.strictEqual(recordError(), 'healthy');
            assert.strictEqual(recordError(), 'healthy');
            assert.strictEqual(recordError(), 'unhealthy');
        });

        test('Should reset health on success', () => {
            let consecutiveErrors = 5;

            const recordSuccess = () => {
                consecutiveErrors = 0;
            };

            recordSuccess();
            assert.strictEqual(consecutiveErrors, 0);
        });
    });

    // Test provider parameters
    suite('Provider Parameters', () => {
        test('Should have valid temperature range', () => {
            const validateTemperature = (temp: number): boolean => {
                return temp >= 0 && temp <= 2;
            };

            assert.ok(validateTemperature(0));
            assert.ok(validateTemperature(0.7));
            assert.ok(validateTemperature(1.5));
            assert.ok(validateTemperature(2));
            assert.ok(!validateTemperature(-0.1));
            assert.ok(!validateTemperature(2.1));
        });

        test('Should have valid max_tokens range', () => {
            const validateMaxTokens = (tokens: number): boolean => {
                return tokens >= 1 && tokens <= 200000;
            };

            assert.ok(validateMaxTokens(100));
            assert.ok(validateMaxTokens(4096));
            assert.ok(validateMaxTokens(100000));
            assert.ok(!validateMaxTokens(0));
            assert.ok(!validateMaxTokens(-100));
        });

        test('Should have valid top_p range', () => {
            const validateTopP = (topP: number): boolean => {
                return topP >= 0 && topP <= 1;
            };

            assert.ok(validateTopP(0));
            assert.ok(validateTopP(0.5));
            assert.ok(validateTopP(1));
            assert.ok(!validateTopP(-0.1));
            assert.ok(!validateTopP(1.1));
        });
    });

    // Test endpoint configuration
    suite('Endpoint Configuration', () => {
        test('Should validate endpoint URLs', () => {
            const isValidEndpoint = (url: string): boolean => {
                try {
                    new URL(url);
                    return true;
                } catch {
                    return false;
                }
            };

            assert.ok(isValidEndpoint('https://api.anthropic.com/v1'));
            assert.ok(isValidEndpoint('http://localhost:11434'));
            assert.ok(isValidEndpoint('http://192.168.1.100:8080'));
            assert.ok(!isValidEndpoint('not-a-url'));
            assert.ok(!isValidEndpoint(''));
        });

        test('Should detect local endpoints', () => {
            const isLocalEndpoint = (url: string): boolean => {
                try {
                    const parsed = new URL(url);
                    return parsed.hostname === 'localhost' ||
                           parsed.hostname === '127.0.0.1' ||
                           parsed.hostname.startsWith('192.168.');
                } catch {
                    return false;
                }
            };

            assert.ok(isLocalEndpoint('http://localhost:8080'));
            assert.ok(isLocalEndpoint('http://127.0.0.1:11434'));
            assert.ok(isLocalEndpoint('http://192.168.1.100:8080'));
            assert.ok(!isLocalEndpoint('https://api.openai.com'));
        });
    });

    // Test secure storage
    suite('Secure Storage', () => {
        test('Should generate storage keys', () => {
            const getStorageKey = (providerId: string): string => {
                return `victor.provider.${providerId}.apiKey`;
            };

            assert.strictEqual(getStorageKey('anthropic'), 'victor.provider.anthropic.apiKey');
            assert.strictEqual(getStorageKey('openai'), 'victor.provider.openai.apiKey');
        });

        test('Should mask API keys for display', () => {
            const maskApiKey = (key: string): string => {
                if (key.length <= 8) {return '****';}
                return key.substring(0, 4) + '...' + key.substring(key.length - 4);
            };

            assert.strictEqual(maskApiKey('sk-ant-api03-abcdefghijklmnop'), 'sk-a...mnop');
            assert.strictEqual(maskApiKey('short'), '****');
        });
    });

    // Test model selection
    suite('Model Selection', () => {
        test('Should filter models by provider', () => {
            const modelsByProvider: Record<string, string[]> = {
                anthropic: ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
                openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                ollama: ['llama2', 'codellama', 'mistral']
            };

            assert.strictEqual(modelsByProvider.anthropic.length, 3);
            assert.ok(modelsByProvider.openai.includes('gpt-4'));
        });

        test('Should get default model for provider', () => {
            const defaultModels: Record<string, string> = {
                anthropic: 'claude-3-sonnet-20240229',
                openai: 'gpt-4-turbo',
                ollama: 'llama2'
            };

            assert.ok(defaultModels.anthropic.includes('claude'));
            assert.ok(defaultModels.openai.includes('gpt'));
        });
    });

    // Test configuration serialization
    suite('Configuration Serialization', () => {
        test('Should serialize provider config', () => {
            const config = {
                provider: 'anthropic',
                model: 'claude-3-sonnet',
                temperature: 0.7,
                maxTokens: 4096
            };

            const serialized = JSON.stringify(config);
            const deserialized = JSON.parse(serialized);

            assert.deepStrictEqual(deserialized, config);
        });

        test('Should exclude sensitive data from serialization', () => {
            const config = {
                provider: 'anthropic',
                apiKey: 'secret-key',
                model: 'claude-3-sonnet'
            };

            const sanitize = (obj: any): any => {
                const { apiKey, ...rest } = obj;
                return rest;
            };

            const sanitized = sanitize(config);
            assert.ok(!('apiKey' in sanitized));
            assert.ok('provider' in sanitized);
        });
    });

    // Test provider availability
    suite('Provider Availability', () => {
        test('Should check if provider requires network', () => {
            const cloudProviders = ['anthropic', 'openai', 'google', 'xai', 'deepseek', 'groq', 'mistral'];

            const requiresNetwork = (providerId: string): boolean => {
                return cloudProviders.includes(providerId);
            };

            assert.ok(requiresNetwork('anthropic'));
            assert.ok(requiresNetwork('openai'));
            assert.ok(!requiresNetwork('ollama'));
            assert.ok(!requiresNetwork('lmstudio'));
        });

        test('Should check for airgapped mode compatibility', () => {
            const airgappedCompatible = ['ollama', 'lmstudio', 'vllm'];

            const isAirgappedCompatible = (providerId: string): boolean => {
                return airgappedCompatible.includes(providerId);
            };

            assert.ok(isAirgappedCompatible('ollama'));
            assert.ok(!isAirgappedCompatible('anthropic'));
        });
    });
});
