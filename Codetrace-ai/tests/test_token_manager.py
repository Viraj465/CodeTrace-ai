from src.core.agents.token_manager import resolve_tier


def test_unknown_cloud_model_never_uses_the_8k_local_fallback():
    tier = resolve_tier(model_name="unregistered-cloud-model")

    assert tier.context_window == 32_768
    assert tier.max_output_tokens == 4_096


def test_unknown_cloud_model_uses_explicit_limits():
    tier = resolve_tier(
        model_name="z-ai/glm-5.2",
        context_window_override=131_072,
        max_output_tokens_override=8_192,
    )

    assert tier.context_window == 131_072
    assert tier.max_output_tokens == 8_192


def test_unknown_cloud_model_uses_configured_default():
    tier = resolve_tier(
        model_name="z-ai/glm-5.2",
        default_context_window=32_768,
    )

    assert tier.context_window == 32_768


def test_unknown_ollama_model_does_not_use_cloud_override():
    tier = resolve_tier(
        model_name="unknown-local-model",
        ollama_base_url="http://localhost:11434/v1",
        context_window_override=131_072,
    )

    assert tier.context_window == 8_192
