import os

_SYSTEM = (
    "You are a concise ML model explainer. "
    "Given customer features and a prediction, explain in 2 sentences "
    "why the model made that prediction."
)


def explain(prediction: dict) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        label = "convert" if prediction["label"] == 1 else "no convert"
        return (
            f"[Mock LLM] Predicted '{label}' with {prediction['probability']:.0%} confidence. "
            f"Set ANTHROPIC_API_KEY to get a real explanation from Claude."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    label = "convert" if prediction["label"] == 1 else "no convert"
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=[{"type": "text", "text": _SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": (
                f"Prediction: {label} ({prediction['probability']:.0%} confident)\n"
                f"Features (age_norm, spend_norm, visits_norm, recency_norm): {prediction['features']}"
            ),
        }],
    )
    return response.content[0].text
