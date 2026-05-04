# Architecture Decision Records

---

## ADR-001: ML predictions and LLM reasoning are kept separate

**Decision:** The ML model returns a structured dict. The LLM receives that dict and produces an explanation. They are not merged into a single call.

**Why:** Keeping them separate means each layer can be tested, swapped, or replaced independently. If the LLM call fails or is slow, the prediction is still available. If the model is retrained, the LLM layer needs no changes.

**Tradeoff:** Requires an explicit integration step (the orchestrator) instead of one monolithic call.

---

## ADR-002: Agents are plain classes, not a framework

**Decision:** `OrchestratorAgent` and `RetrainingAgent` are standard Python classes with no dependency on LangChain, AutoGen, or any agent framework.

**Why:** The agents here do specific, bounded things. Introducing a framework adds indirection without adding capability at this scope. The pattern is readable to anyone who knows Python.

**Tradeoff:** Does not provide built-in memory, tool-calling loops, or multi-agent messaging. Those can be added when the use case requires them.

---

## ADR-003: Retraining logic is injected, not embedded

**Decision:** `RetrainingAgent.check_and_act()` accepts a `retrain_fn` callback rather than knowing how to retrain internally.

**Why:** Keeps the agent's responsibility narrow (detect drift, decide to act) while allowing the caller to supply whatever retraining logic is appropriate — a simple `train()` call in the demo, a full pipeline trigger in production.

**Tradeoff:** The caller must always supply the callback if retraining is expected; easy to forget in tests.

---

## ADR-004: LLM call degrades gracefully without an API key

**Decision:** If `ANTHROPIC_API_KEY` is not set, `explain()` returns a clearly labelled mock string instead of raising an exception.

**Why:** The system should be runnable out of the box for demonstration and development. The mock makes it obvious that the real call is not happening without silently returning empty output.

**Tradeoff:** A caller that doesn't check could treat the mock as a real explanation. The mock string is intentionally verbose to make this hard to miss.

---

## ADR-005: Drift is measured as mean feature-mean difference

**Decision:** `detect_drift()` computes the mean absolute difference in per-feature means between two datasets.

**Why:** It is fast, interpretable, and sufficient to demonstrate the pattern. A team can read the number and immediately understand what it measures.

**Tradeoff:** Not statistically rigorous. For production use, replace with PSI, KS-test, or a proper drift library. The function signature stays the same — only the implementation needs to change.

---

## ADR-006: All tuneable values live in a single config.yaml, read once at startup

**Decision:** `src/core/__init__.py` exposes `load_config()` which reads `configs/config.yaml` with `yaml.safe_load` and returns a plain dict. All modules accept `cfg: dict` as a parameter; none hardcode paths, thresholds, model names, or prompt text.

**Why:** A template must be adaptable without code changes. Centralising configuration means a team can clone the repo, edit one file, and have a system tuned to their domain — different model path, different LLM prompt, different label names, different drift threshold.

**Tradeoff:** No hot-reload; a config change requires a process restart. No schema validation at load time — an invalid key silently returns `None` rather than failing fast. Both are acceptable for a template; add `pydantic-settings` or `jsonschema` validation if the project grows.

---

## ADR-007: LLM prompt text and label names are config, not code

**Decision:** The LLM system prompt (`llm.system_prompt`), class label names (`llm.label_names`), and feature names (`llm.feature_names`) are all in `config.yaml`. The `explain()` function reads them at call time and constructs the user message dynamically.

**Why:** The original implementation hardcoded "customer", "convert/no convert", and "age_norm, spend_norm, visits_norm, recency_norm" directly in the source. This made the LLM layer domain-specific and required code edits to reuse the template for any other problem (churn, fraud, risk scoring, etc.).

**Tradeoff:** Prompt engineering happens in YAML rather than Python, which is less flexible for complex multi-turn prompts. If a use case needs structured LLM output or tool use, the `explain()` function should be extended — the config-driven approach remains for the common case.

---

## ADR-008: Missing retrain_fn raises RuntimeError instead of silently doing nothing

**Decision:** If `RetrainingAgent.check_and_act()` detects drift but `retrain_fn` is `None`, it raises `RuntimeError` by default. A `warn_only=True` constructor flag restores the silent behaviour for callers who genuinely want drift detection without triggering retraining.

**Why:** Silent failures in a retraining pipeline are dangerous. Drift is detected, nothing happens, and the model silently degrades. Making the failure loud forces the caller to make an explicit choice.

**Tradeoff:** Callers who use `check_and_act` purely for monitoring must now pass `warn_only=True`. This is a small, explicit cost for a significant safety improvement.
