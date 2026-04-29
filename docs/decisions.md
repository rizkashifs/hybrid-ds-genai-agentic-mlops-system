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
