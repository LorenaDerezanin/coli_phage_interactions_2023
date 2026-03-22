# Skill Vendoring Policy

- Never modify vendored skill files directly. Upstream files live under `<skill>/vendor/<source>/` and must stay
  pristine so they can be updated without merge conflicts.
- Project-specific customizations go in the skill's own `SKILL.md` at the skill root, which references the vendored
  material and adds overrides.
- Each vendored directory must include the upstream `LICENSE` and a `NOTICE` file with attribution (repository URL,
  version, copyright holder).
- The `vendor/` subdirectory must not contain a `SKILL.md` at a path that the skill scanner would pick up as a separate
  skill. Keep vendored `SKILL.md` files nested deep enough (e.g., `vendor/<source>/SKILL.md`) so they are not at the
  `.agents/skills/*/SKILL.md` level.

## Directory Structure

```
.agents/skills/<skill-name>/
  SKILL.md                          # Project skill — triggers, overrides, references vendor/
  vendor/<upstream-source>/         # Pristine upstream files
    LICENSE
    NOTICE
    SKILL.md                        # Upstream original (kept for provenance, not scanned)
    references/                     # Upstream reference docs
```
