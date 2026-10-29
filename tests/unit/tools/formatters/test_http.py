"""Unit tests for HTTPFormatter."""

from victor.tools.formatters.http import HTTPFormatter


class TestHTTPFormatter:
    """Test HTTPFormatter."""

    def test_validate_input_valid(self):
        """Test validate_input() with valid data."""
        formatter = HTTPFormatter()

        assert formatter.validate_input({"status_code": 200}) is True
        assert formatter.validate_input({"body": "content"}) is True
        assert formatter.validate_input({"status_code": 200, "body": "test"}) is True

    def test_validate_input_invalid(self):
        """Test validate_input() with invalid data."""
        formatter = HTTPFormatter()

        assert formatter.validate_input({}) is False
        assert formatter.validate_input(None) is False

    def test_format_success_response(self):
        """Test formatting successful 2xx response."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 150,
            "headers": {},
            "body": "Response content",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green bold]200 OK[/]" in result.content
        assert "150ms" in result.content
        assert result.summary == "200 OK"

    def test_format_redirect_response(self):
        """Test formatting 3xx redirect response."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 301,
            "status": "Moved Permanently",
            "duration_ms": 50,
            "headers": {},
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow bold]301 Moved Permanently[/]" in result.content

    def test_format_client_error(self):
        """Test formatting 4xx client error."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 404,
            "status": "Not Found",
            "duration_ms": 25,
            "headers": {},
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red bold]404 Not Found[/]" in result.content

    def test_format_server_error(self):
        """Test formatting 5xx server error."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 500,
            "status": "Internal Server Error",
            "duration_ms": 100,
            "headers": {},
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[red bold]500 Internal Server Error[/]" in result.content

    def test_format_with_headers(self):
        """Test formatting response with headers."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {
                "Content-Type": "application/json",
                "Content-Length": "1234",
            },
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[cyan]Content-Type:[/]" in result.content
        assert "application/json" in result.content

    def test_format_max_headers(self):
        """Test max_headers parameter limits header display."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {f"Header-{i}": f"Value-{i}" for i in range(15)},
        }

        result = formatter.format(data, max_headers=10)

        assert result.contains_markup is True
        assert "... and 5 more headers" in result.content

    def test_format_with_json_body(self):
        """Test formatting JSON response body."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {},
            "body": {"key": "value", "number": 42},
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow]" in result.content  # JSON is highlighted
        assert '"key": "value"' in result.content

    def test_format_with_list_body(self):
        """Test formatting list response body."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {},
            "body": ["item1", "item2", "item3"],
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[yellow]" in result.content

    def test_format_with_string_body(self):
        """Test formatting string response body."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {},
            "body": "Plain text response",
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "Plain text response" in result.content

    def test_format_max_body_length(self):
        """Test max_body_length parameter truncates body."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            "status": "OK",
            "duration_ms": 100,
            "headers": {},
            "body": "A" * 1000,  # Long body
        }

        result = formatter.format(data, max_body_length=500)

        assert result.contains_markup is True
        assert "..." in result.content
        assert len(result.content) < 600

    def test_format_no_body(self):
        """Test formatting response without body."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 204,
            "status": "No Content",
            "duration_ms": 50,
            "headers": {},
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        # Should not have body section
        assert "[dim]Body:[/]" not in result.content

    def test_format_missing_optional_fields(self):
        """Test formatting with missing optional fields."""
        formatter = HTTPFormatter()
        data = {
            "status_code": 200,
            # Missing status, duration_ms, headers, body
        }

        result = formatter.format(data)

        assert result.contains_markup is True
        assert "[green bold]200 [/]" in result.content
