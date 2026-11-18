class TestGenerateApiKey:
    def test_serve_has_generate_api_key_param(self):
        import inspect

        from victor.ui.commands.serve import serve

        params = inspect.signature(serve).parameters
        assert "generate_api_key" in params

    def test_serve_passes_flag_to_serve_impl(self):
        import inspect

        from victor.ui.commands.serve import _serve

        params = inspect.signature(_serve).parameters
        assert "generate_api_key" in params
        assert params["generate_api_key"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_print_quickstart_contains_key_and_curl(self, capsys):
        """The key must be visible (the operator needs to copy it) and the
        curl example must use it (60-second round trip)."""
        from victor.ui.commands.serve import _print_agent_api_quickstart

        _print_agent_api_quickstart("127.0.0.1", 8765, "victor-testkey")
        captured = capsys.readouterr()
        assert "victor-testkey" in captured.out
        assert "curl" in captured.out
        assert "/chat" in captured.out
