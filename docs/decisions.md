# Architecture Decisions

## ADR-001: ML predictions and LLM reasoning remain distinct

Predictive models produce structured signals. LLMs provide reasoning, summarization, and orchestration support. Their outputs should be combined through explicit contracts.

## ADR-002: Agent automation requires human approval boundaries

Automated MLOps workflows should include approval checkpoints for deployment, rollback, policy changes, and high-impact actions.
