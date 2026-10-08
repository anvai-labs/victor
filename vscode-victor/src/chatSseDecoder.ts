/** Bounded framing for the existing chat SSE transports (no reconnect/replay).
 * https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
 */
export class ChatSseDecoder {
    private readonly decoder = new TextDecoder('utf-8', { fatal: true });
    private line = '';
    private data: string[] = [];
    private frameBytes = 0;
    private skipLF = false;
    private static readonly limit = 1024 * 1024;

    /** Returning false from onPayload ends consumption immediately. */
    push(chunk: Uint8Array, onPayload: (payload: string) => boolean): void {
        let start = 0;
        for (let i = 0; i < chunk.length; i++) {
            const byte = chunk[i];
            if (this.skipLF) {
                this.skipLF = false;
                if (byte === 10) {
                    // Include paired CRLF bytes within a pending frame. After
                    // its blank delimiter the next frame has not started yet.
                    if (this.frameBytes && ++this.frameBytes > ChatSseDecoder.limit) {
                        throw new Error('Chat SSE frame exceeds the 1 MiB limit');
                    }
                    start = i + 1;
                    continue;
                }
            }
            if (this.frameBytes + i - start + 1 > ChatSseDecoder.limit) {
                throw new Error('Chat SSE frame exceeds the 1 MiB limit');
            }
            if (byte !== 13 && byte !== 10) continue;
            this.append(chunk.subarray(start, i + 1));
            const line = this.line.slice(0, -1);
            this.line = '';
            this.skipLF = byte === 13;
            start = i + 1;
            if (line === '') {
                const payload = this.data.length ? this.data.join('\n') : undefined;
                this.data = [];
                this.frameBytes = 0;
                if (payload !== undefined && !onPayload(payload)) return;
            } else if (line === 'data' || line.startsWith('data:')) {
                const value = line === 'data' ? '' : line.slice(5);
                this.data.push(value.startsWith(' ') ? value.slice(1) : value);
            }
            // SSE comments, id, event and retry do not grant a chat outcome.
        }
        this.append(chunk.subarray(start));
    }

    private append(chunk: Uint8Array): void {
        this.frameBytes += chunk.length;
        this.line += this.decoder.decode(chunk, { stream: true });
    }
}
