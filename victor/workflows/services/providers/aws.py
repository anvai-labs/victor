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

"""AWS service provider for managed services (RDS, ElastiCache, MSK, etc.).

Manages AWS resources as workflow services with proper lifecycle control.
Uses boto3 for AWS API interactions and credential management.

Example:
    from victor.workflows.services.credentials import get_credential_manager

    creds = get_credential_manager().get_aws("production")
    provider = AWSServiceProvider(credentials=creds)

    config = ServiceConfig(
        name="ml-db",
        provider="aws_rds",
        aws_engine="postgres",
        aws_instance_class="db.t3.micro",
        aws_db_name="ml_training",
    )

    handle = await provider.start(config)
    # ... use RDS instance ...
    await provider.stop(handle)  # Stops (not deletes) RDS instance
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from victor.workflows.services.definition import (
    HealthCheckConfig,
    HealthCheckType,
    ServiceConfig,
    ServiceHandle,
    ServiceHealthError,
    ServiceStartError,
    ServiceState,
)
from victor.workflows.services.providers.base import BaseServiceProvider

logger = logging.getLogger(__name__)

# Optional boto3 import
try:
    import boto3
    from botocore.exceptions import ClientError, WaiterError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None  # type: ignore

    class ClientError(Exception):
        """Fallback botocore-style client error used in tests without boto3."""

        def __init__(self, error_response: Dict[str, Any], operation_name: str):
            super().__init__(f"{operation_name}: {error_response}")
            self.response = error_response
            self.operation_name = operation_name

    class WaiterError(Exception):
        """Fallback waiter error used in tests without boto3."""

        def __init__(self, name: str, reason: str, last_response: Optional[Dict[str, Any]] = None):
            super().__init__(f"{name}: {reason}")
            self.name = name
            self.reason = reason
            self.last_response = last_response


class AWSServiceProvider(BaseServiceProvider):
    """Service provider for AWS managed services.

    Supports:
    - RDS (PostgreSQL, MySQL, Aurora)
    - ElastiCache (Redis, Memcached)
    - MSK (Kafka)
    - DynamoDB (tables)
    - SQS (queues)
    - Secrets Manager (for credentials)

    Attributes:
        session: boto3 Session for AWS API calls
        region: AWS region
    """

    def __init__(
        self,
        credentials: Optional[Any] = None,  # AWSCredentials
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        """Initialize AWS service provider.

        Args:
            credentials: AWSCredentials object
            region: AWS region override
            profile: AWS profile name
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 not available. Install with: pip install boto3")

        self._credentials = credentials
        self._region = region
        self._profile = profile
        self._session: Optional[boto3.Session] = None

    @property
    def session(self) -> boto3.Session:
        """Get or create boto3 session."""
        if self._session is None:
            if self._credentials:
                self._session = boto3.Session(
                    aws_access_key_id=self._credentials.access_key_id,
                    aws_secret_access_key=self._credentials.secret_access_key,
                    aws_session_token=self._credentials.session_token,
                    region_name=self._region or self._credentials.region,
                )
            elif self._profile:
                self._session = boto3.Session(
                    profile_name=self._profile,
                    region_name=self._region,
                )
            else:
                self._session = boto3.Session(region_name=self._region)
        return self._session

    async def _do_start(self, config: ServiceConfig) -> ServiceHandle:
        """Start AWS service based on provider type."""
        provider = config.provider

        if provider == "aws_rds":
            return await self._start_rds(config)
        elif provider == "aws_elasticache":
            return await self._start_elasticache(config)
        elif provider == "aws_msk":
            return await self._start_msk(config)
        elif provider == "aws_dynamodb":
            return await self._start_dynamodb(config)
        elif provider == "aws_sqs":
            return await self._start_sqs(config)
        else:
            raise ServiceStartError(config.name, f"Unknown AWS provider: {provider}")

    async def _do_stop(self, handle: ServiceHandle, grace_period: float) -> None:
        """Stop AWS service."""
        provider = handle.config.provider

        if provider == "aws_rds":
            await self._stop_rds(handle)
        elif provider == "aws_elasticache":
            await self._stop_elasticache(handle)
        # DynamoDB, SQS, MSK don't typically need stopping

    async def _do_cleanup(self, handle: ServiceHandle) -> None:
        """Cleanup AWS resources (if needed)."""
        # Most AWS services don't need cleanup beyond stop
        pass

    async def get_logs(self, handle: ServiceHandle, tail: int = 100) -> str:
        """Get CloudWatch logs for AWS service."""
        # CloudWatch log retrieval — requires boto3, not yet implemented
        return "[AWS CloudWatch logs - use AWS Console or CLI]"

    async def _run_command_in_service(
        self,
        handle: ServiceHandle,
        command: str,
    ) -> Tuple[int, str]:
        """Cannot run commands in AWS managed services."""
        raise NotImplementedError("Cannot run commands in AWS managed services")

    # =========================================================================
    # RDS (PostgreSQL, MySQL, Aurora)
    # =========================================================================

    async def _start_rds(self, config: ServiceConfig) -> ServiceHandle:
        """Start or connect to RDS instance."""
        handle = ServiceHandle.create(config)

        rds = self.session.client("rds")
        instance_id = config.aws_cluster_id or f"victor-{config.name}"

        try:
            # Check if instance exists
            try:
                response = await asyncio.to_thread(
                    rds.describe_db_instances,
                    DBInstanceIdentifier=instance_id,
                )
                instance = response["DBInstances"][0]
                status = instance["DBInstanceStatus"]

                if status == "available":
                    logger.info(f"RDS instance '{instance_id}' already available")
                elif status == "stopped":
                    logger.info(f"Starting stopped RDS instance '{instance_id}'")
                    await asyncio.to_thread(
                        rds.start_db_instance,
                        DBInstanceIdentifier=instance_id,
                    )
                    await self._wait_for_rds_available(rds, instance_id)
                else:
                    logger.info(f"RDS instance '{instance_id}' in state: {status}")
                    await self._wait_for_rds_available(rds, instance_id)

                # Refresh instance info
                response = await asyncio.to_thread(
                    rds.describe_db_instances,
                    DBInstanceIdentifier=instance_id,
                )
                instance = response["DBInstances"][0]

            except ClientError as e:
                if e.response["Error"]["Code"] == "DBInstanceNotFound":
                    # Create new instance
                    logger.info(f"Creating new RDS instance '{instance_id}'")
                    instance = await self._create_rds_instance(rds, config, instance_id)
                else:
                    raise

            # Extract connection info
            endpoint = instance["Endpoint"]
            handle.host = endpoint["Address"]
            handle.ports[5432] = endpoint["Port"]
            handle.metadata["instance_id"] = instance_id
            handle.metadata["engine"] = instance["Engine"]
            handle.metadata["instance_class"] = instance["DBInstanceClass"]

            # Set exports
            db_name = config.aws_db_name or "postgres"
            user = config.environment.get("POSTGRES_USER", "postgres")
            password = config.environment.get("POSTGRES_PASSWORD", "")

            handle.connection_info["DATABASE_URL"] = (
                f"postgresql://{user}:{password}@{handle.host}:{endpoint['Port']}/{db_name}"
            )
            handle.connection_info["POSTGRES_HOST"] = handle.host
            handle.connection_info["POSTGRES_PORT"] = str(endpoint["Port"])

            handle.state = ServiceState.STARTING
            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    async def _create_rds_instance(
        self,
        rds: Any,
        config: ServiceConfig,
        instance_id: str,
    ) -> Dict[str, Any]:
        """Create a new RDS instance."""
        params = {
            "DBInstanceIdentifier": instance_id,
            "DBInstanceClass": config.aws_instance_class or "db.t3.micro",
            "Engine": config.aws_engine or "postgres",
            "MasterUsername": config.environment.get("POSTGRES_USER", "postgres"),
            "MasterUserPassword": config.environment.get("POSTGRES_PASSWORD", "postgres"),
            "AllocatedStorage": 20,
            "PubliclyAccessible": True,  # For development; disable in prod
            "Tags": [
                {"Key": "victor:managed", "Value": "true"},
                {"Key": "victor:service", "Value": config.name},
            ],
        }

        if config.aws_db_name:
            params["DBName"] = config.aws_db_name

        if config.aws_engine_version:
            params["EngineVersion"] = config.aws_engine_version

        await asyncio.to_thread(rds.create_db_instance, **params)

        # Wait for instance to be available
        await self._wait_for_rds_available(rds, instance_id)

        # Get instance details
        response = await asyncio.to_thread(
            rds.describe_db_instances,
            DBInstanceIdentifier=instance_id,
        )
        return response["DBInstances"][0]

    async def _wait_for_rds_available(
        self,
        rds: Any,
        instance_id: str,
        timeout: int = 900,
    ) -> None:
        """Wait for RDS instance to become available."""
        logger.info(f"Waiting for RDS instance '{instance_id}' to be available...")

        waiter = rds.get_waiter("db_instance_available")
        try:
            await asyncio.to_thread(
                waiter.wait,
                DBInstanceIdentifier=instance_id,
                WaiterConfig={"Delay": 30, "MaxAttempts": timeout // 30},
            )
        except WaiterError as e:
            raise ServiceHealthError(instance_id, f"RDS instance not available: {e}")

    async def _stop_rds(self, handle: ServiceHandle) -> None:
        """Stop RDS instance (doesn't delete)."""
        rds = self.session.client("rds")
        instance_id = handle.metadata.get("instance_id")

        if not instance_id:
            return

        try:
            logger.info(f"Stopping RDS instance '{instance_id}'")
            await asyncio.to_thread(
                rds.stop_db_instance,
                DBInstanceIdentifier=instance_id,
            )
        except ClientError as e:
            if "InvalidDBInstanceState" not in str(e):
                raise

    # =========================================================================
    # ElastiCache (Redis, Memcached)
    # =========================================================================

    async def _start_elasticache(self, config: ServiceConfig) -> ServiceHandle:
        """Start or connect to ElastiCache cluster."""
        handle = ServiceHandle.create(config)

        elasticache = self.session.client("elasticache")
        cluster_id = config.aws_cluster_id or f"victor-{config.name}"

        try:
            # Check if cluster exists
            try:
                response = await asyncio.to_thread(
                    elasticache.describe_cache_clusters,
                    CacheClusterId=cluster_id,
                    ShowCacheNodeInfo=True,
                )
                cluster = response["CacheClusters"][0]
                status = cluster["CacheClusterStatus"]

                if status != "available":
                    logger.info(f"ElastiCache cluster '{cluster_id}' in state: {status}")
                    # Wait for available
                    await self._wait_for_elasticache_available(elasticache, cluster_id)

                    # Refresh
                    response = await asyncio.to_thread(
                        elasticache.describe_cache_clusters,
                        CacheClusterId=cluster_id,
                        ShowCacheNodeInfo=True,
                    )
                    cluster = response["CacheClusters"][0]

            except ClientError as e:
                if "CacheClusterNotFound" in str(e):
                    # Create new cluster
                    logger.info(f"Creating new ElastiCache cluster '{cluster_id}'")
                    cluster = await self._create_elasticache_cluster(
                        elasticache, config, cluster_id
                    )
                else:
                    raise

            # Extract connection info
            node = cluster["CacheNodes"][0]
            endpoint = node["Endpoint"]
            handle.host = endpoint["Address"]
            handle.ports[6379] = endpoint["Port"]
            handle.metadata["cluster_id"] = cluster_id
            handle.metadata["engine"] = cluster["Engine"]

            # Set exports
            password = config.environment.get("REDIS_PASSWORD", "")
            if password:
                handle.connection_info["REDIS_URL"] = (
                    f"redis://:{password}@{handle.host}:{endpoint['Port']}/0"
                )
            else:
                handle.connection_info["REDIS_URL"] = f"redis://{handle.host}:{endpoint['Port']}/0"
            handle.connection_info["REDIS_HOST"] = handle.host
            handle.connection_info["REDIS_PORT"] = str(endpoint["Port"])

            handle.state = ServiceState.STARTING
            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    async def _create_elasticache_cluster(
        self,
        elasticache: Any,
        config: ServiceConfig,
        cluster_id: str,
    ) -> Dict[str, Any]:
        """Create a new ElastiCache cluster."""
        params = {
            "CacheClusterId": cluster_id,
            "Engine": "redis",
            "CacheNodeType": config.aws_instance_class or "cache.t3.micro",
            "NumCacheNodes": 1,
            "Tags": [
                {"Key": "victor:managed", "Value": "true"},
                {"Key": "victor:service", "Value": config.name},
            ],
        }

        await asyncio.to_thread(elasticache.create_cache_cluster, **params)

        # Wait for cluster to be available
        await self._wait_for_elasticache_available(elasticache, cluster_id)

        # Get cluster details
        response = await asyncio.to_thread(
            elasticache.describe_cache_clusters,
            CacheClusterId=cluster_id,
            ShowCacheNodeInfo=True,
        )
        return response["CacheClusters"][0]

    async def _wait_for_elasticache_available(
        self,
        elasticache: Any,
        cluster_id: str,
        timeout: int = 600,
    ) -> None:
        """Wait for ElastiCache cluster to become available."""
        logger.info(f"Waiting for ElastiCache cluster '{cluster_id}' to be available...")

        waiter = elasticache.get_waiter("cache_cluster_available")
        try:
            await asyncio.to_thread(
                waiter.wait,
                CacheClusterId=cluster_id,
                WaiterConfig={"Delay": 15, "MaxAttempts": timeout // 15},
            )
        except WaiterError as e:
            raise ServiceHealthError(cluster_id, f"ElastiCache not available: {e}")

    async def _stop_elasticache(self, handle: ServiceHandle) -> None:
        """ElastiCache clusters can't be stopped, only deleted."""
        logger.info(
            f"ElastiCache cluster '{handle.metadata.get('cluster_id')}' "
            "cannot be stopped (only deleted). Leaving running."
        )

    # =========================================================================
    # MSK (Kafka)
    # =========================================================================

    async def _start_msk(self, config: ServiceConfig) -> ServiceHandle:
        """Connect to MSK cluster."""
        handle = ServiceHandle.create(config)

        kafka = self.session.client("kafka")
        cluster_arn = config.aws_cluster_id

        if not cluster_arn:
            raise ServiceStartError(config.name, "MSK cluster ARN required")

        try:
            # Get bootstrap brokers
            response = await asyncio.to_thread(
                kafka.get_bootstrap_brokers,
                ClusterArn=cluster_arn,
            )

            bootstrap_servers = response.get("BootstrapBrokerString")
            if not bootstrap_servers:
                bootstrap_servers = response.get("BootstrapBrokerStringTls")

            if not bootstrap_servers:
                raise ServiceStartError(config.name, "No bootstrap brokers found")

            # Parse first broker for host/port
            first_broker = bootstrap_servers.split(",")[0]
            host, port = first_broker.rsplit(":", 1)

            handle.host = host
            handle.ports[9092] = int(port)
            handle.metadata["cluster_arn"] = cluster_arn
            handle.metadata["bootstrap_servers"] = bootstrap_servers

            handle.connection_info["KAFKA_BOOTSTRAP_SERVERS"] = bootstrap_servers

            handle.state = ServiceState.STARTING
            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    # =========================================================================
    # DynamoDB
    # =========================================================================

    async def _start_dynamodb(self, config: ServiceConfig) -> ServiceHandle:
        """Ensure DynamoDB table exists."""
        handle = ServiceHandle.create(config)

        dynamodb = self.session.client("dynamodb")
        table_name = config.aws_db_name or config.name

        try:
            # Check if table exists
            try:
                response = await asyncio.to_thread(
                    dynamodb.describe_table,
                    TableName=table_name,
                )
                status = response["Table"]["TableStatus"]

                if status != "ACTIVE":
                    logger.info(f"DynamoDB table '{table_name}' in state: {status}")
                    # Wait for active
                    waiter = dynamodb.get_waiter("table_exists")
                    await asyncio.to_thread(
                        waiter.wait,
                        TableName=table_name,
                    )

            except ClientError as e:
                if "ResourceNotFoundException" in str(e):
                    raise ServiceStartError(
                        config.name,
                        f"DynamoDB table '{table_name}' not found. "
                        "Create it first or use infrastructure-as-code.",
                    )
                raise

            handle.metadata["table_name"] = table_name
            handle.metadata["region"] = self.session.region_name

            handle.connection_info["DYNAMODB_TABLE"] = table_name
            handle.connection_info["AWS_REGION"] = self.session.region_name

            handle.state = ServiceState.HEALTHY
            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))

    # =========================================================================
    # SQS
    # =========================================================================

    async def _start_sqs(self, config: ServiceConfig) -> ServiceHandle:
        """Ensure SQS queue exists and get URL."""
        handle = ServiceHandle.create(config)

        sqs = self.session.client("sqs")
        queue_name = config.aws_db_name or config.name

        try:
            # Get queue URL (creates if doesn't exist with these attributes)
            try:
                response = await asyncio.to_thread(
                    sqs.get_queue_url,
                    QueueName=queue_name,
                )
                queue_url = response["QueueUrl"]
            except ClientError as e:
                if "NonExistentQueue" in str(e):
                    # Create queue
                    logger.info(f"Creating SQS queue '{queue_name}'")
                    response = await asyncio.to_thread(
                        sqs.create_queue,
                        QueueName=queue_name,
                        tags={"victor:managed": "true"},
                    )
                    queue_url = response["QueueUrl"]
                else:
                    raise

            handle.metadata["queue_name"] = queue_name
            handle.metadata["queue_url"] = queue_url

            handle.connection_info["SQS_QUEUE_URL"] = queue_url
            handle.connection_info["SQS_QUEUE_NAME"] = queue_name

            handle.state = ServiceState.HEALTHY
            return handle

        except Exception as e:
            handle.state = ServiceState.FAILED
            handle.error = str(e)
            raise ServiceStartError(config.name, str(e))
