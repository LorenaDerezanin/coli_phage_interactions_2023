# Test Fixture Policy

- Store test fixtures as separate raw files under `lyzortx/tests/fixtures/`, not as inline string variables chopped up
  inside test files.
- Fixture files should be real outputs copied from actual pipeline runs when possible. This catches schema drift and
  format issues that synthetic data misses.
- Organize fixtures by feature block or module name (e.g., `fixtures/host_defense/`, `fixtures/plan/`).
- When the pipeline produces directory-structured outputs (e.g., `{host_id}/host_defense_gene_counts.csv`), mirror that
  directory structure in fixtures so the test layout matches reality.
- Tests should reference fixtures via `Path(__file__).parent / "fixtures" / ...` for portability.
