# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Service providers for different execution backends.

Providers handle the lifecycle of services on specific platforms:
- docker: Local Docker containers
- kubernetes: Kubernetes deployments
- local: OS subprocess
- external: Connect to existing services
- aws_*: AWS managed services (RDS, ElastiCache, etc.)
- gcp_*: GCP managed services
- azure_*: Azure managed services
"""

from victor.workflows.services.providers.base import BaseServiceProvider
from victor.workflows.services.providers.docker import DockerServiceProvider
from victor.workflows.services.providers.external import ExternalServiceProvider
from victor.workflows.services.providers.kubernetes import KubernetesServiceProvider
from victor.workflows.services.providers.local import LocalProcessProvider
from victor.workflows.services.providers.aws import AWSServiceProvider

__all__ = [
    "BaseServiceProvider",
    "DockerServiceProvider",
    "ExternalServiceProvider",
    "KubernetesServiceProvider",
    "LocalProcessProvider",
    "AWSServiceProvider",
]
