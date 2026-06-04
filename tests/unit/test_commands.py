"""Tests for the command parser and executor."""

from doc_assistant.commands import execute_command, parse_command


class TestParseCommand:
    def test_parses_slash_command(self):
        assert parse_command("/help") == ("help", "")

    def test_parses_command_with_arg(self):
        assert parse_command("/library broken") == ("library", "broken")

    def test_strips_whitespace(self):
        assert parse_command("  /document abc123  ") == ("document", "abc123")

    def test_returns_none_for_non_command(self):
        assert parse_command("What is in my library?") is None

    def test_returns_none_for_empty(self):
        assert parse_command("") is None

    def test_lowercases_command(self):
        assert parse_command("/HELP") == ("help", "")

    def test_lowercases_arg(self):
        assert parse_command("/library BROKEN") == ("library", "broken")


class TestExecuteCommand:
    def test_help_returns_help(self):
        result = execute_command("help", "")
        assert "Available commands" in result

    def test_synthesis_reports_mode(self):
        result = execute_command("synthesis", "")
        # Reports whichever mode is active (default ai) and explains the modes.
        assert "Synthesis mode" in result
        assert "`ai`" in result or "`human`" in result

    def test_help_lists_synthesis(self):
        assert "/synthesis" in execute_command("help", "")

    def test_unknown_command(self):
        result = execute_command("foobar", "")
        assert "Unknown command" in result

    def test_unknown_library_filter(self):
        result = execute_command("library", "badfilter")
        assert "Unknown library filter" in result

    def test_document_no_arg(self):
        result = execute_command("document", "")
        assert "Usage" in result
