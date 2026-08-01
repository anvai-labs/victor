# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shared fixtures for victor-dataanalysis tests.

Importing the victor runtime here (before any test module is collected)
pins the victor_contracts promoted-type identities: victor_contracts
re-resolves some symbols once the runtime is importable, so loading the
runtime first keeps `issubclass`/`isinstance` checks stable regardless of
test collection order.
"""

import victor.framework.extensions  # noqa: F401
