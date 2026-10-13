"""Workflow API reports the compatibility executor's actual result contract."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from victor.integrations.api.routes import workflow_routes
from victor.workflows.context import WorkflowContext, WorkflowResult
from victor.workflows.definition import TransformNode, WorkflowDefinition
from victor.workflows.registry import WorkflowRegistry
from victor_contracts.workflows import ExecutorNodeStatus, NodeResult


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "failed", "paused", "exception"])
async def test_execution_reports_result_and_pause_without_attribute_errors(monkeypatch, outcome):
    workflow = WorkflowDefinition(
        name="test",
        nodes={
            "first": TransformNode(id="first", name="first"),
            "second": TransformNode(id="second", name="second"),
        },
        start_node="first",
    )
    node_result = NodeResult(
        node_id="first",
        status=(ExecutorNodeStatus.FAILED if outcome == "failed" else ExecutorNodeStatus.COMPLETED),
        duration_seconds=0.125,
    )
    result = WorkflowResult(
        workflow_name="test",
        success=outcome != "failed",
        context=WorkflowContext(data={"answer": 0}, node_results={"first": node_result}),
        error="node failed" if outcome == "failed" else None,
    )
    result.interrupted = outcome == "paused"
    result.interrupt_node = "second" if result.interrupted else None
    executor = Mock(execute=AsyncMock(return_value=result))
    if outcome == "exception":
        executor.execute.side_effect = RuntimeError("engine failed")
    factory = Mock(return_value=executor)
    monkeypatch.setattr(
        "victor.workflows.runtime_executor_factory.create_legacy_workflow_executor", factory
    )
    monkeypatch.setattr(
        "victor.workflows.get_global_registry", lambda: Mock(get=Mock(return_value=workflow))
    )
    server = SimpleNamespace(
        _workflow_executions={}, _get_orchestrator=AsyncMock(return_value=object())
    )
    finished = asyncio.Event()

    async def broadcast(event):
        if event["event"] != "workflow_started":
            finished.set()

    server._broadcast_ws = AsyncMock(side_effect=broadcast)
    app = FastAPI()
    app.include_router(workflow_routes.create_router(server))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/workflows/execute", json={"template_id": "test", "parameters": {"input": 7}}
        )
        assert response.status_code == 200
        await asyncio.wait_for(finished.wait(), timeout=2)

    execution = server._workflow_executions[response.json()["execution_id"]]
    expected_status = "failed" if outcome == "exception" else outcome
    assert execution["status"] == expected_status
    assert server._broadcast_ws.call_args.args[0]["event"] == f"workflow_{expected_status}"
    factory.assert_called_once_with(server._get_orchestrator.return_value)
    executor.execute.assert_awaited_once_with(workflow, initial_context={"input": 7})
    if outcome == "exception":
        assert execution["error"] == "engine failed"
        return
    assert execution["output"] == "{'answer': 0}"
    assert execution["steps"][0]["duration"] == 0.125
    assert execution["error"] == result.error
    if outcome == "paused":
        assert execution["end_time"] is None
        assert execution["progress"] == 50
        assert execution["interrupt_node"] == "second"
        assert execution["steps"][1]["status"] == "pending"
    else:
        assert execution["end_time"] is not None


@pytest.mark.asyncio
async def test_template_routes_use_public_registry_and_definition_contracts(monkeypatch):
    registry = WorkflowRegistry()
    registry.register(
        WorkflowDefinition(
            name="real-template",
            nodes={"start": TransformNode(id="start", name="start")},
            start_node="start",
        )
    )
    monkeypatch.setattr("victor.workflows.get_global_registry", lambda: registry)
    app = FastAPI()
    app.include_router(workflow_routes.create_router(SimpleNamespace()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        listing = await client.get("/workflows/templates")
        details = await client.get("/workflows/templates/real-template")
        missing = await client.get("/workflows/templates/missing")

    assert listing.status_code == details.status_code == 200
    assert listing.json()["templates"][0]["id"] == "real-template"
    assert details.json()["steps"][0]["type"] == "transform"
    assert missing.status_code == 404


@pytest.mark.asyncio
async def test_visualizer_loads_only_packaged_same_origin_scripts(monkeypatch):
    from html.parser import HTMLParser
    from urllib.parse import urljoin

    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []

        def handle_starttag(self, tag, attrs):
            if tag == "script" and "src" in dict(attrs):
                self.sources.append(dict(attrs)["src"])

    app = FastAPI()
    app.include_router(workflow_routes.create_router(SimpleNamespace()))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        page = await client.get("/workflows/visualize/preview")
        assert page.status_code == 200
        scripts = Scripts()
        scripts.feed(page.text)
        assert scripts.sources, "wheel must serve graph dependencies locally"
        for source in scripts.sources:
            url = urljoin(str(page.url), source)
            assert url.startswith("http://test/workflows/assets/"), source
            response = await client.get(url)
            assert response.status_code == 200
            assert "javascript" in response.headers["content-type"]
            assert response.headers["x-content-type-options"] == "nosniff"
            assert len(response.content) > 1000
        for invalid in ["missing.js", "LICENSE", "%2e%2e%2fworkflow_routes.py"]:
            assert (await client.get(f"/workflows/assets/{invalid}")).status_code == 404
        monkeypatch.setattr(workflow_routes.Path, "is_file", lambda _: False)
        unavailable = await client.get("/workflows/assets/cytoscape.min.js")
        assert unavailable.status_code == 503


def test_visualizer_browser_logic_handles_missing_state_and_inert_node_text():
    """Exercise the shipped script without network, a DOM parser, or provider calls."""
    import json
    from pathlib import Path
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the visualizer JavaScript smoke")
    template = Path(workflow_routes.__file__).parent.parent / "templates/workflow_visualizer.html"
    # The final inline script owns the UI; external assets are tested by HTTP above.
    script = template.read_text().rsplit("<script>", 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
class Element {
    constructor() { this.children = []; this.style = {}; this.handlers = {}; }
    set innerHTML(value) { throw new Error('Untrusted HTML sink'); }
    appendChild(child) { this.children.push(child); }
    append(...children) { this.children.push(...children); }
    replaceChildren() { this.children = []; }
    setAttribute() {}
    addEventListener(event, callback) { this.handlers[event] = callback; }
}
const elements = new Map();
const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
let reloads = 0;
// Load the real bundle before adding the fake DOM: library and test objects
// share one realm, while Cytoscape uses its supported headless environment.
const context = vm.createContext({setTimeout, clearTimeout, console});
vm.runInContext(fs.readFileSync(VENDOR, 'utf8'), context);
Object.assign(context, {
    getComputedStyle: () => ({getPropertyValue: name => ({'--warning':'#b54708','--success':'#067647'}[name] || '#101828')}),
    console: {error() {}, warn() {}},
    document: {getElementById: get, createElement: () => new Element()},
    window: {location: {pathname:'/workflows/visualize/test', reload:()=>reloads++}, addEventListener() {}},
    fetch: async () => ({ok:false, status:503}),
});
vm.runInContext(SCRIPT, context);
(async () => {
    await vm.runInContext('initGraph()', context);
    assert.equal(get('status-text').textContent, 'Unavailable');
    get('btn-fit').handlers.click();
    get('btn-refresh').handlers.click();
    assert.equal(reloads, 1);
    await vm.runInContext('fetchExecutionState()', context);
    assert.equal(get('status-text').textContent, 'State unavailable');
    vm.runInContext(`
        const malicious = '<img src=x onerror=alert(1)>';
        const node = {id: () => 'n', data: () => malicious};
        showNodeDetails(node);
        workflowState = {node_execution_path:[{node_id:'n', status:'failed', error:malicious, duration_seconds:0}]};
        showNodeDetails(node);
    `, context);
    const texts = element => [element.textContent, ...element.children.flatMap(texts)];
    assert(texts(get('node-details')).includes('<img src=x onerror=alert(1)>'));
    assert(texts(get('node-details')).includes('0s'));
    vm.runInContext(`
        cy = cytoscape({headless:true, styleEnabled:true, elements:[{data:{id:'n'}}]});
        applyGraphTheme();
    `, context);
    const count = vm.runInContext('cy.style().length', context);
    vm.runInContext('for (let i=0; i<100; i++) applyGraphTheme()', context);
    assert.equal(vm.runInContext('cy.style().length', context), count);
    assert.equal(vm.runInContext("cy.getElementById('n').addClass('running').style('border-color')", context), 'rgb(181,71,8)');
    assert.equal(vm.runInContext("cy.getElementById('n').removeClass('running').addClass('completed').style('border-color')", context), 'rgb(6,118,71)');
})().catch(error => { console.error(error); process.exitCode = 1; })
    .finally(() => vm.runInContext('if (cy) cy.destroy()', context));
"""
    subprocess.run(
        [
            node,
            "-e",
            harness.replace("SCRIPT", json.dumps(script)).replace(
                "VENDOR", json.dumps(str(template.parent / "vendor/cytoscape.min.js"))
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_visualizer_theme_preference_survives_blocked_storage_and_tracks_system():
    import json
    from pathlib import Path
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the visualizer JavaScript smoke")
    html = (
        Path(workflow_routes.__file__).parent.parent / "templates/workflow_visualizer.html"
    ).read_text()
    assert '<script id="theme-bootstrap">' in html
    script = html.split('<script id="theme-bootstrap">', 1)[1].split("</script>", 1)[0]
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
for (const blocked of [false, true]) {
    const handlers = {};
    const selector = {value:'system', addEventListener: (type, fn) => handlers[type] = fn};
    const root = {dataset:{}};
    let saved = 'invalid', changed = 0, ready;
    const media = {matches:false, addEventListener(type, fn) {this.change=fn;}};
    const storage = {
        getItem() {if(blocked) throw Error('denied'); return saved;},
        setItem(key,value) {if(blocked) throw Error('denied'); saved=value;},
        removeItem() {if(blocked) throw Error('denied'); saved=null;},
    };
    const context = vm.createContext({
        Event: class {},
        document: {documentElement:root, getElementById:()=>selector, addEventListener:(type,fn)=>ready=fn},
        window: {localStorage:storage, matchMedia:()=>media, dispatchEvent:()=>changed++, addEventListener() {}},
    });
    vm.runInContext(SCRIPT, context);
    assert.equal(root.dataset.theme, 'system'); ready();
    selector.value='dark'; handlers.change();
    assert.equal(root.dataset.theme, 'dark');
    if(!blocked) assert.equal(saved, 'dark');
    media.matches=true; media.change();
    assert.equal(root.dataset.theme, 'dark');
    selector.value='light'; handlers.change();
    assert.equal(root.dataset.theme, 'light');
    selector.value='system'; handlers.change();
    assert.equal(root.dataset.theme, 'system');
    if(!blocked) assert.equal(saved, null);
    assert(changed >= 4);
}
"""
    subprocess.run(
        [node, "-e", harness.replace("SCRIPT", json.dumps(script))],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
