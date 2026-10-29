"""Unit tests for DockerFormatter."""

from victor.tools.formatters.docker import DockerFormatter


class TestDockerFormatter:
    """Test DockerFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = DockerFormatter()

        assert formatter.validate_input({"containers": []}) is True
        assert formatter.validate_input({"images": []}) is True
        assert formatter.validate_input({"volumes": []}) is True
        assert formatter.validate_input({"services": []}) is True
        assert formatter.validate_input({"output": "test"}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = DockerFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_containers_running(self):
        """Test formatting running containers."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": "abc123",
                    "name": "web-server",
                    "image": "nginx:latest",
                    "state": "running",
                    "ports": "80:80",
                }
            ]
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        assert "[green]●[/]" in result.content
        assert "web-server" in result.content
        assert "nginx:latest" in result.content
        assert result.summary == "1 containers"

    def test_format_containers_stopped(self):
        """Test formatting stopped containers."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": "def456",
                    "name": "db-server",
                    "image": "postgres:13",
                    "state": "stopped",
                    "ports": "",
                }
            ]
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        assert "[red]○[/]" in result.content
        assert "db-server" in result.content

    def test_format_containers_paused(self):
        """Test formatting paused containers."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": "ghi789",
                    "name": "cache",
                    "image": "redis:alpine",
                    "state": "paused",
                    "ports": "6379:6379",
                }
            ]
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        assert "[yellow]◐[/]" in result.content

    def test_format_containers_multiple(self):
        """Test formatting multiple containers."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": "abc123",
                    "name": "web",
                    "image": "nginx",
                    "state": "running",
                    "ports": "80:80",
                },
                {
                    "id": "def456",
                    "name": "db",
                    "image": "postgres",
                    "state": "running",
                    "ports": "5432:5432",
                },
            ]
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        assert "2 running" in result.content
        assert "2 total" in result.content

    def test_format_containers_max_items(self):
        """Test max_items parameter limits output."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": f"id{i}",
                    "name": f"container{i}",
                    "image": "img",
                    "state": "running",
                    "ports": "",
                }
                for i in range(25)
            ]
        }

        result = formatter.format(data, operation="ps", max_items=20)

        assert result.contains_markup is True
        assert "... and 5 more containers" in result.content

    def test_format_images(self):
        """Test formatting Docker images."""
        formatter = DockerFormatter()
        data = {
            "images": [
                {
                    "id": "sha256:abc123",
                    "tags": ["nginx:latest", "nginx:1.21"],
                    "size": "133MB",
                    "created": "2 days ago",
                }
            ]
        }

        result = formatter.format(data, operation="images")

        assert result.contains_markup is True
        assert "nginx:latest" in result.content
        assert "133MB" in result.content
        assert result.summary == "1 images"

    def test_format_images_no_tags(self):
        """Test formatting images without tags."""
        formatter = DockerFormatter()
        data = {
            "images": [
                {
                    "id": "sha256:def456",
                    "tags": [],
                    "size": "50MB",
                    "created": "1 week ago",
                }
            ]
        }

        result = formatter.format(data, operation="images")

        assert result.contains_markup is True
        assert "<none>" in result.content

    def test_format_volumes(self):
        """Test formatting Docker volumes."""
        formatter = DockerFormatter()
        data = {
            "volumes": [
                {
                    "name": "data-volume",
                    "driver": "local",
                    "mountpoint": "/var/lib/docker/volumes/data-volume/_data",
                }
            ]
        }

        result = formatter.format(data, operation="volumes")

        assert result.contains_markup is True
        assert "data-volume" in result.content
        assert "local" in result.content
        assert result.summary == "1 volumes"

    def test_format_services(self):
        """Test formatting Docker Compose services."""
        formatter = DockerFormatter()
        data = {
            "services": [
                {
                    "name": "web",
                    "state": "running",
                    "replicas": 3,
                },
                {
                    "name": "worker",
                    "state": "running",
                    "replicas": 2,
                },
            ]
        }

        result = formatter.format(data, operation="services")

        assert result.contains_markup is True
        assert "[green]●[/]" in result.content
        assert "3 replicas" in result.content
        assert "2 replicas" in result.content
        assert result.summary == "2 services"

    def test_format_generic_output(self):
        """Test formatting generic Docker output."""
        formatter = DockerFormatter()
        data = {
            "output": "Some raw Docker output\n",
        }

        result = formatter.format(data, operation="custom")

        assert result.contains_markup is False  # Plain text
        assert "Some raw Docker output" in result.content

    def test_format_no_containers(self):
        """Test formatting when no containers exist."""
        formatter = DockerFormatter()
        data = {
            "containers": [],
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        assert "[dim]No containers[/]" in result.content

    def test_format_missing_optional_fields(self):
        """Test formatting with missing optional fields."""
        formatter = DockerFormatter()
        data = {
            "containers": [
                {
                    "id": "abc123",
                    # Missing name, image, state, ports
                }
            ]
        }

        result = formatter.format(data, operation="ps")

        assert result.contains_markup is True
        # Should still format something
        assert len(result.content) > 0

    def test_format_singular_items(self):
        """Test formatting singular counts."""
        formatter = DockerFormatter()

        # Test containers - summary shows "1 containers" even though it's singular
        result = formatter.format(
            {
                "containers": [
                    {
                        "id": "abc",
                        "name": "test",
                        "image": "img",
                        "state": "running",
                        "ports": "",
                    }
                ]
            },
            operation="ps",
        )
        assert "1 containers" in result.summary  # Summary field has the count

        # Test images
        result = formatter.format(
            {"images": [{"id": "abc", "tags": ["test"], "size": "100MB", "created": "now"}]},
            operation="images",
        )
        assert "1 images" in result.summary  # Summary field has the count
