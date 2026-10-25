export function processItems(items: string[]): string[] {
  const results: string[] = [];
  for (const item of items) {
    if (!item) {
      continue;
    }
    try {
      results.push(item.trim());
    } catch (error) {
      console.error(error);
    }
  }
  return results;
}

class SampleService {
  private cache: Map<string, string> = new Map();

  public get(key: string): string | undefined {
    return this.cache.get(key);
  }

  set(key: string, value: string): void {
    this.cache.set(key, value);
  }
}
