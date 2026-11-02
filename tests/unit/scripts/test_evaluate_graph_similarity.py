"""The evaluation command must reject ambiguous data and destructive destinations."""

import importlib.util
import json
from pathlib import Path

import pytest

import shutil


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[3] / "scripts" / "evaluate_graph_similarity.py"
    spec = importlib.util.spec_from_file_location("similarity_eval_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_args(tmp_path):
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "graph_similarity"
    data = json.loads((fixture / "controls.json").read_text())
    units = {unit["id"]: unit for unit in data["units"]}
    data["pairs"] = [
        pair
        for pair in data["pairs"]
        if units[pair["left"]]["language"] == units[pair["right"]]["language"] == "python"
    ]
    referenced = {pair[side] for pair in data["pairs"] for side in ("left", "right")}
    data["units"] = [unit for unit in data["units"] if unit["id"] in referenced]
    roots = {}
    for project in data["projects"]:
        roots[project["id"]] = tmp_path / project["id"]
        shutil.copytree(fixture / project["id"], roots[project["id"]])
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps(data))
    output = tmp_path / "report.json"
    args = ["--corpus", str(manifest), "--output", str(output)]
    for key, path in roots.items():
        args.extend(["--project", f"{key}={path}"])
    return args, manifest, output


def test_command_reports_unqualified_control_without_claiming_production_quality(tmp_path, cli):
    args, _, output = fixture_args(tmp_path)
    assert cli.main(args) == 0
    report = json.loads(output.read_text())
    from victor.core.graph_rag.similarity_evaluation import Corpus

    assert Corpus.from_dict(report["corpus_manifest"]).identity == report["corpus"]
    assert report["coverage_complete"] and not report["decision_complete"]
    assert not report["production_quality_qualified"]
    assert report["test"]["overall"]["precision"] is None
    assert cli.main([*args, "--require-decisions"]) == 1


@pytest.mark.parametrize("destination", ["manifest", "source", "cache"])
def test_inputs_and_report_cannot_be_overwritten(tmp_path, cli, destination):
    args, manifest, output = fixture_args(tmp_path)
    source = tmp_path / "controls-test" / "control_test_0_0_left.py"
    if destination == "cache":
        args.extend(["--cache", str(output)])
    else:
        args[args.index("--output") + 1] = str(manifest if destination == "manifest" else source)
    before = {path: path.read_bytes() for path in (manifest, source)}
    assert cli.main(args) == 2
    assert all(path.read_bytes() == content for path, content in before.items())


def test_duplicate_manifest_fields_are_rejected(tmp_path, cli):
    path = tmp_path / "corpus.json"
    path.write_text('{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        cli.load_corpus(path)


def test_invalid_threshold_is_checked_before_source_read(tmp_path, cli):
    args, _, output = fixture_args(tmp_path)
    assert cli.main([*args, "--minimum-precision", "nan"]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "profile",
    [
        {"model_id": "test/model", "revision": "main", "dimension": 3},
        {"model_id": "test/model", "revision": "0" * 40, "dimension": True},
        {"model_id": "test/model", "revision": "0" * 40, "dimension": 3, "trust_remote_code": True},
    ],
)
def test_profile_must_be_pinned_typed_and_non_executable(tmp_path, cli, profile):
    args, _, output = fixture_args(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    assert cli.main([*args, "--channel", "body", "--body-profile", str(path)]) == 2
    assert not output.exists()


def test_body_profile_is_passed_to_the_production_scorer(tmp_path, cli, monkeypatch):
    from victor.core.graph_rag import similarity_pair_scoring
    from victor.core.graph_rag.body_clones import BodyOptions, LocalBodyEncoder

    args, _, output = fixture_args(tmp_path)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({"model_id": "test/model", "revision": "0" * 40, "dimension": 3}))
    seen = []
    original = similarity_pair_scoring.score_corpus

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 0

        def __call__(self, text, **kwargs):
            return {"input_ids": [0], "offset_mapping": [(0, len(text))]}

    class Model:
        tokenizer = Tokenizer()
        max_seq_length = 512

        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            return [[1, 1, 1] for _ in texts]

    def run(*args, **kwargs):
        options = kwargs["body_options"]
        seen.append(options)
        kwargs["encoder"] = LocalBodyEncoder(options, model_factory=lambda *a, **k: Model())
        return original(*args, **kwargs)

    monkeypatch.setattr(cli, "score_corpus", run)
    assert cli.main([*args, "--channel", "body", "--body-profile", str(path)]) == 0
    assert seen == [BodyOptions(model_id="test/model", revision="0" * 40, dimension=3)]
    assert not json.loads(output.read_text())["production_quality_qualified"]


def test_cache_cannot_repurpose_the_project_graph_database(tmp_path, cli):
    args, _, output = fixture_args(tmp_path)
    database = tmp_path / "controls-test" / ".victor" / "project.db"
    database.parent.mkdir()
    database.write_bytes(b"existing graph must remain untouched")
    assert cli.main([*args, "--channel", "body", "--cache", str(database)]) == 2
    assert database.read_bytes() == b"existing graph must remain untouched"
    assert not output.exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("destination", ["output", "cache"])
def test_project_database_sidecars_are_protected(tmp_path, cli, suffix, destination):
    args, _, output = fixture_args(tmp_path)
    target = tmp_path / "controls-test" / ".victor" / ("project.db" + suffix)
    target.parent.mkdir()
    target.write_bytes(b"existing graph state")
    if destination == "output":
        args[args.index("--output") + 1] = str(target)
    else:
        args.extend(["--channel", "body", "--cache", str(target)])
    assert cli.main(args) == 2
    assert target.read_bytes() == b"existing graph state"


def test_live_wal_is_not_replaced_by_a_report(tmp_path, cli):
    import sqlite3

    args, _, output = fixture_args(tmp_path)
    database = tmp_path / "controls-test" / ".victor" / "project.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute("CREATE TABLE protected_graph(value)")
        writer.execute("INSERT INTO protected_graph VALUES ('durable')")
        writer.commit()
        wal = Path(str(database) + "-wal")
        before = wal.read_bytes()
        args[args.index("--output") + 1] = str(wal)
        assert cli.main(args) == 2
        assert wal.read_bytes() == before
        with sqlite3.connect(database) as reader:
            assert reader.execute("SELECT value FROM protected_graph").fetchone() == ("durable",)


def test_hardlink_cache_alias_cannot_mutate_graph_database(tmp_path, cli):
    import os
    import sqlite3

    args, _, output = fixture_args(tmp_path)
    database = tmp_path / "controls-test" / ".victor" / "project.db"
    database.parent.mkdir()
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE protected_graph(value)")
    before = database.read_bytes()
    alias = tmp_path / "innocent-cache.sqlite"
    os.link(database, alias)
    assert cli.main([*args, "--channel", "body", "--cache", str(alias)]) == 2
    assert database.read_bytes() == before
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [
            ("protected_graph",)
        ]


def test_case_alias_of_not_yet_created_sqlite_sidecar_is_protected(tmp_path, cli):
    args, _, output = fixture_args(tmp_path)
    database = tmp_path / "controls-test" / ".victor" / "project.db"
    database.parent.mkdir()
    database.write_bytes(b"existing database")
    destination = database.with_name("PROJECT.DB-WAL")
    args[args.index("--output") + 1] = str(destination)
    assert cli.main(args) == 2
    assert not destination.exists()


@pytest.mark.parametrize("missed_negative", [False, True])
def test_explicit_held_out_precision_gate(tmp_path, cli, monkeypatch, missed_negative):
    from dataclasses import replace
    from victor.core.graph_rag.similarity_evaluation import Observation

    args, _, output = fixture_args(tmp_path)
    original = cli.score_corpus

    def scores(data, *args, **kwargs):
        run = original(data, *args, **kwargs)
        observations = []
        raised = False
        for pair in data.pairs:
            score = 0.99 if pair.positive else 0.1
            if missed_negative and data.split(pair) == "test" and not pair.positive and not raised:
                score, raised = 1.0, True
            observations.append(Observation(pair.id, score))
        return replace(run, observations=tuple(observations))

    monkeypatch.setattr(cli, "score_corpus", scores)
    assert cli.main([*args, "--minimum-test-precision", "0.95"]) == int(missed_negative)
    result = json.loads(output.read_text())
    assert result["requested_gates_passed"] is (not missed_negative)
    assert not result["production_quality_qualified"]


def test_precision_gate_requires_held_out_negative_support(tmp_path, cli, monkeypatch):
    from dataclasses import replace
    from victor.core.graph_rag.similarity_evaluation import Observation

    args, manifest, output = fixture_args(tmp_path)
    raw = json.loads(manifest.read_text())
    remove = next(
        pair["id"]
        for pair in raw["pairs"]
        if pair["id"].startswith("test-") and not pair["positive"]
    )
    raw["pairs"] = [pair for pair in raw["pairs"] if pair["id"] != remove]
    ids = {pair[side] for pair in raw["pairs"] for side in ("left", "right")}
    raw["units"] = [unit for unit in raw["units"] if unit["id"] in ids]
    manifest.write_text(json.dumps(raw))
    original = cli.score_corpus

    def perfect(data, *args, **kwargs):
        run = original(data, *args, **kwargs)
        return replace(
            run,
            observations=tuple(
                Observation(pair.id, 0.99 if pair.positive else 0.1) for pair in data.pairs
            ),
        )

    monkeypatch.setattr(cli, "score_corpus", perfect)
    assert cli.main([*args, "--minimum-test-precision", "0.95"]) == 1
    result = json.loads(output.read_text())
    assert result["test"]["overall"]["precision"] == 1.0
    assert not result["requested_gates_passed"]
