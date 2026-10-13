// Launched by the existing Python contract suite against ephemeral production
// routers. No Axios mock: exercise the compiled extension client over real HTTP.
const assert = require('node:assert/strict');
const { VictorClient } = require('../out/victorClient.js');

async function main() {
    const [url, kind, scenario] = process.argv.slice(2);
    const client = new VictorClient(url, undefined, 'contract-test-key');
    if (scenario === 'paused') {
        const { requireCompletedChat, ChatApprovalRequiredError } = require('../out/victorClient.js');
        const response = await client.chat([{ role: 'user', content: 'pause this turn' }]);
        assert.equal(response.status, 'awaiting_approval');
        assert.equal(response.run_id, 'run-http-approval');
        assert.deepEqual(response.approval_request, { id: 'approval-http', metadata: { hash: 'abc' } });
        assert.throws(() => requireCompletedChat(response), error =>
            error instanceof ChatApprovalRequiredError && error.response === response);
        const unauthorized = new VictorClient(url, undefined, 'incorrect-test-key');
        await assert.rejects(unauthorized.chat([{ role: 'user', content: 'must not execute' }]));
        process.stdout.write(JSON.stringify({ outcome: response.status }));
        return;
    }
    if (scenario === 'truncated') {
        const chunks = [];
        await assert.rejects(client.streamChat([{ role: 'user', content: 'partial' }],
            c => chunks.push(c)), /before an explicit terminator/);
        assert.deepEqual(chunks, ['echo:partial']);
        process.stdout.write(JSON.stringify({ outcome: 'interrupted' }));
        return;
    }
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
    if (kind === 'core') {
        assert.equal(await unauthorized.cancelToolExecution('smoke-pending'), false);
        assert.equal(await client.cancelToolExecution('unknown-call'), false);
        assert.equal(await client.cancelToolExecution('smoke-pending'), true);
        assert.equal(await client.cancelToolExecution('smoke-pending'), false);
    }
    process.stdout.write(JSON.stringify({ chunks, request_ids: requestIds, session_id: session }));
}

main().catch(error => {
    console.error(error);
    process.exitCode = 1;
});
