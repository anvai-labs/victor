# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Owner labels and liveness checks shared by the sandbox reapers."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from unittest.mock import patch

import victor.tools.sandbox_ownership as so
from victor.tools.sandbox_ownership import (
    OWNER_HOST_LABEL,
    OWNER_PID_LABEL,
    OWNER_START_LABEL,
    created_age_seconds,
    host_identity,
    owner_alive,
    owner_gone,
    owner_labels,
    process_start_marker,
)

MODULE = "victor.tools.sandbox_ownership"


class TestOwnerAlive:
    def test_self_is_alive(self):
        assert owner_alive(os.getpid(), process_start_marker(os.getpid()))

    def test_missing_pid_is_dead(self):
        assert not owner_alive(0, "")
        with patch(f"{MODULE}.os.kill", side_effect=ProcessLookupError):
            assert not owner_alive(424242, "")

    def test_reused_pid_is_dead(self):
        pid = os.getpid()
        assert not owner_alive(pid, "not-" + process_start_marker(pid))

    def test_unknown_start_marker_is_treated_as_alive(self):
        with patch(f"{MODULE}.process_start_marker", return_value=""):
            assert owner_alive(os.getpid(), "12345")

    def test_permission_denied_means_alive(self):
        with patch(f"{MODULE}.os.kill", side_effect=PermissionError):
            assert owner_alive(1, "")

    def test_undecidable_kill_error_means_alive(self):
        with patch(f"{MODULE}.os.kill", side_effect=OSError):
            assert owner_alive(424242, "")

    def test_start_marker_parses_comm_with_spaces_and_parens(self, tmp_path, monkeypatch):
        stat = "4242 (weird (name) x) S " + " ".join(str(i) for i in range(4, 60))
        real_path = so.Path

        def fake_path(p):
            if p == "/proc/4242/stat":
                f = tmp_path / "stat"
                f.write_text(stat)
                return f
            return real_path(p)

        monkeypatch.setattr(so, "Path", fake_path)
        assert process_start_marker(4242) == "22"  # field 22: starttime

    def test_start_marker_of_missing_process_is_empty(self):
        with patch(f"{MODULE}.Path.read_text", side_effect=OSError):
            assert process_start_marker(424242) == ""


class TestOwnerLabelsAndGone:
    def test_labels_identify_this_process(self):
        labels = owner_labels()
        assert labels[OWNER_HOST_LABEL] == host_identity()
        assert labels[OWNER_PID_LABEL] == str(os.getpid())
        assert labels[OWNER_START_LABEL] == process_start_marker(os.getpid())

    def test_own_labels_are_not_gone(self):
        assert owner_gone(owner_labels()) is False

    def test_unlabeled_is_undecided(self):
        assert owner_gone({}) is None
        assert owner_gone({"victor.sandbox": "eval"}) is None

    def test_foreign_host_is_never_gone(self):
        labels = {OWNER_HOST_LABEL: "another-boot-id", OWNER_PID_LABEL: "999999"}
        assert owner_gone(labels) is False

    def test_dead_owner_on_this_host_is_gone(self):
        labels = {OWNER_HOST_LABEL: host_identity(), OWNER_PID_LABEL: "999999"}
        with patch(f"{MODULE}.os.kill", side_effect=ProcessLookupError):
            assert owner_gone(labels) is True

    def test_reused_pid_is_gone(self):
        labels = owner_labels()
        labels[OWNER_START_LABEL] = "older-process"
        assert owner_gone(labels) is True

    def test_malformed_pid_is_gone(self):
        assert owner_gone({OWNER_HOST_LABEL: host_identity(), OWNER_PID_LABEL: "x"}) is True

    def test_host_identity_includes_pid_namespace(self):
        assert host_identity().endswith(":" + os.readlink("/proc/self/ns/pid"))

    def test_other_pid_namespace_on_same_boot_is_foreign(self):
        """A WSL2 sibling distro shares the boot id; its PIDs mean nothing here."""
        boot = host_identity().split(":", 1)[0]
        labels = {OWNER_HOST_LABEL: f"{boot}:pid:[1]", OWNER_PID_LABEL: "999999"}
        assert owner_gone(labels) is False

    def test_host_identity_without_namespace_is_boot_id(self):
        with patch(f"{MODULE}.os.readlink", side_effect=OSError):
            assert ":" not in host_identity()

    def test_host_identity_falls_back_to_hostname(self):
        with patch(f"{MODULE}.Path.read_text", side_effect=OSError):
            assert host_identity() == so.socket.gethostname()


class TestCreatedAge:
    def test_nanosecond_timestamp(self):
        created = datetime.fromtimestamp(time.time() - 120, timezone.utc)
        stamp = created.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"
        assert 110 < created_age_seconds(stamp) < 130

    def test_whole_seconds_timestamp(self):
        created = datetime.fromtimestamp(time.time() - 60, timezone.utc)
        assert 50 < created_age_seconds(created.strftime("%Y-%m-%dT%H:%M:%SZ")) < 70

    def test_unparsable_is_zero(self):
        assert created_age_seconds("") == 0.0
        assert created_age_seconds("not-a-date") == 0.0
