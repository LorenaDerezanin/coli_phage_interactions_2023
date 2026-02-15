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

## Markdown formatting + linting

This repository uses `pre-commit` with:

- `prettier` to auto-wrap Markdown prose at 120 chars.
- `markdownlint` as an optional manual validator.

Commands:

```bash
micromamba activate phage_env
```

```bash
pre-commit run prettier --all-files
pre-commit run markdownlint --all-files
```

After auto-fixes, stage updated files before committing.
