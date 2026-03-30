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

# Skill Maintenance

- Skills are living documents. Update a skill's `SKILL.md` whenever there is substantial new learning about how it
  should work — e.g., encountering errors when using it, discovering new edge cases, learning new information about the
  domain, or finding new applications.
- Don't wait for a dedicated "skill improvement" task. If you're using a skill and it's missing guidance that would have
  prevented a mistake, add that guidance now as part of the current work.

## Directory Structure

```text
.agents/skills/<skill-name>/
  SKILL.md                          # Project skill — triggers, overrides, references vendor/
  vendor/<upstream-source>/         # Pristine upstream files
    LICENSE
    NOTICE
    SKILL.md                        # Upstream original (kept for provenance, not scanned)
    references/                     # Upstream reference docs
```
