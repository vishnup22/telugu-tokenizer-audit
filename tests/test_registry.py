from unittest.mock import patch

from telugu_audit.tokenizers.registry import get_tokenizer_versions, load_tokenizers


def test_load_tokenizers_stub_only():
    tokenizers = load_tokenizers(include={"stub-local"})
    assert "stub-local" in tokenizers
    assert tokenizers["stub-local"]("hello world") >= 1


def test_load_tokenizers_warns_on_missing():
    with patch("telugu_audit.tokenizers.registry.logger") as mock_logger:
        tokenizers = load_tokenizers(include={"nonexistent-tokenizer-xyz"})
        assert tokenizers == {}
        mock_logger.warning.assert_called()


def test_version_info_scoped_to_include():
    load_tokenizers(include={"stub-local"})
    versions = get_tokenizer_versions()
    assert "stub-local" in versions
    assert "openai-gpt4o" not in versions
    assert "hf-gpt2" not in versions


def test_registry_skips_failing_adapter():
    bad = type(
        "BadAdapter",
        (),
        {
            "get_tokenizers": staticmethod(
                lambda: (_ for _ in ()).throw(RuntimeError("boom"))
            )
        },
    )
    with patch(
        "telugu_audit.tokenizers.registry._ADAPTER_MODULES",
        [bad, __import__("telugu_audit.tokenizers.adapters.stub_adapter", fromlist=["stub_adapter"])],
    ):
        tokenizers = load_tokenizers(include={"stub-local"})
        assert "stub-local" in tokenizers
