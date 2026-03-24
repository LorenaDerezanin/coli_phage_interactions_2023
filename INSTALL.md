# Getting Started

This repository uses `micromamba` to manage environments. You can install it using the official docs
[here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

Then create and activate the `micromamba` environment using the following commands:

```bash
micromamba create -f environment.yml
micromamba activate phage_env
```

To enable running Jupyter notebooks using this environment, create a dedicated Jupyter kernel for it:

```bash
python -m ipykernel install --user --name phage_env --display-name phage_env
```

Now when you run notebooks in `jupyter lab`, make sure to select the `phage_env` kernel.

## Optional: automatic env activation with direnv
This repository includes a tracked `.envrc` that activates `phage_env` when you enter the repository directory.

1. Install `direnv` and enable its shell integration.
2. For `zsh`, add the hook to `~/.zshrc` if needed:
   ```bash
   eval "$(direnv hook zsh)"
   ```
3. Reload your shell and allow the repo env file:
   ```bash
   direnv allow
   ```

## Git hooks

This repository uses `pre-commit` for automated checks. Install both hook types once per clone:

```bash
pre-commit install
pre-commit install --hook-type pre-push
```

This activates:

- **pre-commit stage:** ruff linting/formatting, gitignore enforcement, pymarkdown fixes.
- **pre-push stage:** verifies your branch is rebased on `origin/main` before allowing `git push`.

## Markdown linting

Commands for manual runs:

```bash
pymarkdown --config .pymarkdown.yaml fix -r .
pre-commit run pymarkdown --all-files
```

After auto-fixes, stage updated files before committing.
