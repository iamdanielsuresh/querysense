Critical Findings (Highest Severity First)

P0: Your benchmark does not evaluate the same architecture you claim to be building. spider_eval.py (line 191) passes full schema directly to generation (schema["columns"]), while the runtime pipeline uses retrieval and BigQuery execution in pipeline.py (line 51) and pipeline.py (line 95). Reported EX is not representative of production behavior.
P0: No SQL safety/cost guardrails before execution. pipeline.py (line 95) executes immediately; bq_executor.py (line 18) runs raw query. No dry-run, no bytes cap, no timeout policy, no query class restrictions.
P1: “Decomposition” is not true decomposition execution. You generate sub-questions, then still ask for one final SQL directly (query_decomposer.py (line 435)). This is prompt augmentation, not a multi-step planner-executor.
P1: Prompt quality is internally inconsistent and contains likely-wrong join supervision. prompts.py (line 181) joins order_product.product_id to category_tree.category_id; this can teach wrong patterns.
P1: BigQuery executor is hardcoded and not multi-tenant-safe. bq_executor.py (line 6) pins one project ID.
P1: Evaluation comparator can produce false positives/negatives. spider_eval.py (line 113) uses set normalization, which drops duplicate-row semantics and order-sensitive intent.
P1: Retry logic is domain-coupled and brittle. pipeline.py (line 114) hardcodes product_id vs category_id hints globally.
P2: Architecture has script-level side effects and weak boundaries. Global initialization at import time in pipeline.py (line 24), pipeline.py (line 31), pipeline.py (line 39).
P2: Testing discipline is weak. test files are ad-hoc scripts; one is functionally broken (test_retriever.py (line 22) treats tuple result as dict).
P2: Iteration plan is internally contradictory and stale. iteration_plan.md (line 9) says evaluation pipeline not implemented, while iteration_plan.md (line 108) says Phase 1 complete; prompt modularization unchecked at iteration_plan.md (line 47) though already present in prompts.py.