import os
from src.core import load_config


def _format_features(features: list, feature_names: list) -> str:
    """Pair feature values with names from config, or fall back to indexed keys."""
    if feature_names:
        pairs = zip(feature_names, features)
    else:
        pairs = ((f"feature_{i}", v) for i, v in enumerate(features))
    return ", ".join(f"{name}: {value}" for name, value in pairs)


def explain(prediction: dict, cfg: dict = None) -> str:
    if cfg is None:
        cfg = load_config()

    lcfg = cfg["llm"]
    label_name = lcfg["label_names"].get(str(prediction["label"]), str(prediction["label"]))
    feature_str = _format_features(prediction["features"], lcfg.get("feature_names") or [])

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            f"[Mock LLM] Predicted '{label_name}' with {prediction['probability']:.0%} confidence. "
            f"Set ANTHROPIC_API_KEY to get a real explanation from Claude."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=lcfg["model"],
        max_tokens=lcfg["max_tokens"],
        system=[{"type": "text", "text": lcfg["system_prompt"], "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                f"Prediction: {label_name} ({prediction['probability']:.0%} confident)\n"
                f"Features: {feature_str}"
            ),
        }],
    )
    return response.content[0].text
