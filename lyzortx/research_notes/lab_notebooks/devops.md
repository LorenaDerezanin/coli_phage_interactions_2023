### 2026-03-22: CI token usage baseline — 100-run snapshot

#### Summary

First comprehensive token/cost analysis across all LLM-invoking CI workflows using the `ci-token-usage` skill.
Covers 100 most recent workflow runs (2026-03-21 to 2026-03-22).

#### Report

```
Run ID       Workflow              Date        Status     Model    Cost       PR/Issue
-----------  --------------------  ----------  ---------  -------  ---------  ------------------
23393507574  Codex Implement Task  2026-03-22  failure    gpt-5.4  $1.23      TG04
23393126080  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.36      TG03
23392903619  Codex Implement Task  2026-03-22  success    gpt-5.4  $0.96      TG02
23392501930  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.46      TG01
23392240588  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.29      TE03
23392054053  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.17      TE02
23391859832  Codex Implement Task  2026-03-22  success    gpt-5.4  $1.51      TE01
23391685029  Codex Implement Task  2026-03-22  success    gpt-5.4  $0.66      TD03
23401448088  Codex PR Lifecycle    2026-03-22  skipped             skipped    PR#110
23401446130  Codex PR Lifecycle    2026-03-22  failure             ?          PR#111
23401099955  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#110
23400156019  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#108
23399190036  Codex PR Lifecycle    2026-03-22  failure             no LLM     PR#107
23399162621  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#107
23393491419  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#105 / Issue#104
23393109495  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#103 / Issue#102
23392886948  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#101 / Issue#100
23392484693  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#99 / Issue#98
23392445365  Codex PR Lifecycle    2026-03-22  skipped             skipped    PR#99 / Issue#98
23392445346  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392445342  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392408551  Codex PR Lifecycle    2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392222905  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#97 / Issue#96
23392037136  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#95 / Issue#94
23391842693  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#93 / Issue#92
23391664643  Codex PR Lifecycle    2026-03-22  success             no LLM     PR#91 / Issue#90
23401661800  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401564745  Claude PR Review      2026-03-22  skipped             skipped    PR#109
23401448139  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401446395  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401446310  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401433093  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401425335  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401377585  Claude PR Review      2026-03-22  skipped             skipped    PR#111
23401364528  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401099963  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23401063163  Claude PR Review      2026-03-22  skipped             skipped    PR#110
23400672332  Claude PR Review      2026-03-22  skipped             skipped    PR#109
23400156013  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23400156012  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23400099350  Claude PR Review      2026-03-22  skipped             skipped    PR#108
23399190076  Claude PR Review      2026-03-22  skipped             skipped    PR#107
23399190041  Claude PR Review      2026-03-22  skipped             skipped    PR#107
23399143141  Claude PR Review      2026-03-22  skipped             skipped
23399138290  Claude PR Review      2026-03-22  success             $0.38      PR#107
23393461248  Claude PR Review      2026-03-22  skipped             skipped    TG03
23393459493  Claude PR Review      2026-03-22  cancelled           cancelled  TG03
23393458339  Claude PR Review      2026-03-22  success             $0.84      PR#105 / Issue#104
23393109763  Claude PR Review      2026-03-22  skipped             skipped    PR#103 / Issue#102
23393109483  Claude PR Review      2026-03-22  cancelled           cancelled  PR#103 / Issue#102
23393079767  Claude PR Review      2026-03-22  cancelled           cancelled  TG02
23393076800  Claude PR Review      2026-03-22  cancelled           cancelled  TG02
23393075508  Claude PR Review      2026-03-22  success             $0.69      PR#103 / Issue#102
23392886997  Claude PR Review      2026-03-22  skipped             skipped    PR#101 / Issue#100
23392886969  Claude PR Review      2026-03-22  cancelled           cancelled  PR#101 / Issue#100
23392886955  Claude PR Review      2026-03-22  cancelled           cancelled  PR#101 / Issue#100
23392834873  Claude PR Review      2026-03-22  cancelled           cancelled  TG01
23392831636  Claude PR Review      2026-03-22  cancelled           cancelled  TG01
23392830381  Claude PR Review      2026-03-22  success             $1.47      PR#101 / Issue#100
23392445476  Claude PR Review      2026-03-22  skipped             skipped    PR#99 / Issue#98
23392445394  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392445372  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392443487  Claude PR Review      2026-03-22  success             $0.40      PR#99 / Issue#98
23392408605  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392408604  Claude PR Review      2026-03-22  skipped             skipped    PR#99 / Issue#98
23392408584  Claude PR Review      2026-03-22  cancelled           cancelled  PR#99 / Issue#98
23392367636  Claude PR Review      2026-03-22  cancelled           cancelled  TE03
23392365847  Claude PR Review      2026-03-22  cancelled           cancelled  TE03
23392364225  Claude PR Review      2026-03-22  success             $1.01      PR#99 / Issue#98
23392222717  Claude PR Review      2026-03-22  skipped             skipped    PR#97 / Issue#96
23392188368  Claude PR Review      2026-03-22  cancelled           cancelled  TE02
23392186591  Claude PR Review      2026-03-22  cancelled           cancelled  TE02
23392184746  Claude PR Review      2026-03-22  success             $0.87      PR#97 / Issue#96
23392037153  Claude PR Review      2026-03-22  skipped             skipped    PR#95 / Issue#94
23392003520  Claude PR Review      2026-03-22  cancelled           cancelled  TE01
23392001064  Claude PR Review      2026-03-22  cancelled           cancelled  TE01
23391999833  Claude PR Review      2026-03-22  success             $0.95      PR#95 / Issue#94
23391842750  Claude PR Review      2026-03-22  skipped             skipped    PR#93 / Issue#92
23391842735  Claude PR Review      2026-03-22  cancelled           cancelled  PR#93 / Issue#92
23391789152  Claude PR Review      2026-03-22  cancelled           cancelled  TD03
23391785627  Claude PR Review      2026-03-22  success             $1.15      PR#93 / Issue#92
23391466934  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.85      TD02
23390906065  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.45
23390432373  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.85      TC04
23390282751  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.96      TC03
23390122019  Codex Implement Task  2026-03-21  success    gpt-5.4  $0.74      TC02
23376412851  Codex Implement Task  2026-03-21  success    gpt-5.4  $1.09
23368200679  Codex Implement Task  2026-03-21  failure             no LLM     TI05
23367963808  Codex Implement Task  2026-03-21  failure    gpt-5.4  $0.83
23391593132  Codex PR Lifecycle    2026-03-21  success    gpt-5.4  $0.34      PR#91 / Issue#90
23391450667  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#87 / Issue#21
23391423648  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#87 / Issue#21
23391423647  Codex PR Lifecycle    2026-03-21  cancelled           cancelled  PR#87 / Issue#21
23391345012  Codex PR Lifecycle    2026-03-21  cancelled  gpt-5.4  $0.39      PR#87 / Issue#21
23391247403  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#86 / Issue#83
23391242099  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#87 / Issue#21
23391221579  Codex PR Lifecycle    2026-03-21  skipped             skipped    PR#86 / Issue#83
23391208905  Codex PR Lifecycle    2026-03-21  cancelled           cancelled  PR#87 / Issue#21
23391182628  Codex PR Lifecycle    2026-03-21  cancelled  gpt-5.4  $0.28      PR#86 / Issue#83
23390764607  Codex PR Lifecycle    2026-03-21  success             no LLM     PR#85

Total estimated cost: $26.18
  Codex:   $18.42  (18 runs, avg $1.02)
  Claude:  $7.76  (9 runs, avg $0.86)

  ⚠ Codex costs are estimates (blended 30% in / 70% out rate)
```

#### Interpretation

**Cost breakdown.** Total LLM spend across 100 runs: **$26.18** (Codex $18.42, Claude $7.76). Codex costs are
estimates using a 30/70 input/output blended rate against gpt-5.4 pricing; Claude costs are exact values reported by
`anthropics/claude-code-action`.

**Codex implementation runs** average $1.02 per run (range $0.66–$1.85). All use gpt-5.4. One failure (TI05) never
reached the LLM — correctly shows `no LLM`. One failure (run 23367963808) did consume tokens ($0.83) before failing.

**Claude review runs** average $0.86 per successful review (range $0.38–$1.47). Most expensive was PR#101/Issue#100
at $1.47. Many runs show `skipped` or `cancelled` — this is expected concurrency behavior when multiple pushes
trigger the review workflow in quick succession; only the latest run executes.

**Lifecycle runs** correctly split between `no LLM` (auto-merge jobs) and Codex-detected runs (address-feedback jobs
that invoke Codex, e.g. PR#91 at $0.34). Log-based detection handles this mixed-workflow case well.

**Concurrency waste.** PR#99/Issue#98 triggered 7 Claude review runs but only 2 completed ($0.40 + $1.01). The rest
were cancelled or skipped. Similarly, 4 lifecycle runs were cancelled for PR#87/Issue#21. This is not token waste
(cancelled runs do not consume LLM resources) but does consume GitHub Actions minutes.

#### Tool used

Report generated with: `python -m lyzortx.orchestration.ci_token_usage --runs 100`
(see `.agents/skills/ci-token-usage/` for skill documentation and design decisions).
