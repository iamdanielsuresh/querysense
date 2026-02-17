Use This As The Master Recovery Brief For All Agents

Goal: convert Text2SQL from prompt-heavy prototype into an evaluable, safe, production-track system.

Success criteria:

Evaluation path matches runtime path exactly.
Pre-execution SQL safety and cost controls are enforced.
Decomposition is true planner-executor behavior, not prompt decoration.
Metrics are reproducible and include failure taxonomy.
Core pipeline has tests, structured logs, and no hardcoded tenant/project assumptions.
Non-goals for this recovery window:

Multi-turn chat memory.
Streaming UX.
Batch inference APIs.
New model training experiments unless they support core blockers