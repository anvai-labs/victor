# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ADR-036 field-data CLI commands.

Commands:
    summary - Aggregate the opt-in field-data JSONL store into the ADR-036
              evidence table (per flag: sessions exposed, outcome
              distribution, durations/turns/tool-call profile).

Example:
    victor evaluation field-data summary
    victor evaluation field-data summary --store-dir ~/.victor/evaluation --json
"""

from pathlib import Path
from typing import Optional

import typer
from victor.ui.console import get_console

field_data_app = typer.Typer(
    name="field-data",
    help="ADR-036 flag-graduation field-data store utilities.",
)

console = get_console()


@field_data_app.command("summary")
def summary(
    store_dir: Optional[Path] = typer.Option(
        None,
        "--store-dir",
        help="Field-data store directory (default: ~/.victor/evaluation).",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit the aggregated evidence as JSON instead of the markdown table.",
    ),
):
    """Aggregate the field-data store into the ADR-036 evidence table.

    Reads the opt-in recorder's append-only JSONL (one file per day,
    size-capped rotation) and prints, per recorded flag × exposure arm:
    sessions exposed, outcome distribution (completed/cancelled/error/no-end),
    and mean duration / turns / tool calls — the evidence the ADR-036 queue's
    graduation decisions read.

    Example:
        victor evaluation field-data summary
        victor evaluation field-data summary --json
    """
    from victor.evaluation.field_data import (
        aggregate_field_data,
        default_store_dir,
        format_evidence_table,
        read_store,
    )

    root = Path(store_dir) if store_dir is not None else default_store_dir()
    records = read_store(root)
    if not records:
        console.print(
            f"[yellow]No field data found in {root}. "
            "Collection is opt-in: export VICTOR_FIELD_DATA_ENABLED=true "
            "(CLI sessions read env/.env only; the settings.yaml evaluation "
            "group is not on the chat path).[/]"
        )
        raise typer.Exit(0)

    aggregated = aggregate_field_data(records)
    if as_json:
        import json

        console.print_json(json.dumps({**aggregated, "store_dir": str(root)}))
    else:
        console.print(format_evidence_table(aggregated))
        console.print(f"\n[dim]store: {root}[/]")
