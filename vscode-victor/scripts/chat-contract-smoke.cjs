// Launched by the existing Python contract suite against ephemeral production
// routers. No Axios mock: exercise the compiled extension client over real HTTP.
const assert = require('node:assert/strict');
const { VictorClient } = require('../out/victorClient.js');

async function main() {
    const [url, kind] = process.argv.slice(2);
    const client = new VictorClient(url, undefined, 'contract-test-key');
    const chunks = [];
    const requestIds = [];
    const history = [
        { role: 'system', content: 'context' },
        { role: 'user', content: 'old turn' },
        { role: 'assistant', content: 'old response' },
        { role: 'user', content: 'latest 🧪\nline' },
        { role: 'assistant', content: 'pending response' },
    ];
    const onEvent = (event) => {
        if (event.type === 'request') requestIds.push(event.requestId);
    };
    await client.streamChat(history, c => chunks.push(c), undefined, onEvent);
    const session = client.getChatSessionId();
    assert.equal(Boolean(session), kind === 'web');
    await client.streamChat([{ role: 'user', content: 'second turn' }],
        c => chunks.push(c), undefined, onEvent);
    assert.equal(client.getChatSessionId(), session);
    // Authentication failure must propagate without changing body/retrying.
    const unauthorized = new VictorClient(url, undefined, 'incorrect-test-key');
    await assert.rejects(unauthorized.streamChat(
        [{ role: 'user', content: 'must not execute' }], () => assert.fail('unexpected chunk')));
    process.stdout.write(JSON.stringify({ chunks, request_ids: requestIds, session_id: session }));
}

main().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
