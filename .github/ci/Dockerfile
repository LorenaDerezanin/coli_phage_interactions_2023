FROM ubuntu:24.04

ARG TARGETARCH
ARG CI_IMAGE_PROFILE=base

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/micromamba/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bzip2 \
        ca-certificates \
        curl \
        git \
        gh \
        jq \
        nodejs \
        sudo \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Let sudo inherit PATH so tools installed by GitHub Actions (e.g., setup-node
# placing node at /__t/node/.../bin) are visible to "sudo -n node ...".
RUN echo 'Defaults env_keep += "PATH"' > /etc/sudoers.d/keep-path \
    && echo 'Defaults !secure_path' >> /etc/sudoers.d/keep-path \
    && chmod 0440 /etc/sudoers.d/keep-path

# Create a "runner" user with passwordless sudo, matching what the
# codex-action drop-sudo safety strategy expects inside the container.
RUN groupadd -g 1001 sudo-runner \
    && useradd -m -u 1001 -g sudo-runner -G sudo -s /bin/bash runner \
    && echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner \
    && chmod 0440 /etc/sudoers.d/runner

RUN case "${TARGETARCH}" in \
      amd64) MICROMAMBA_ARCH=64 ;; \
      arm64) MICROMAMBA_ARCH=aarch64 ;; \
      *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -Ls "https://micro.mamba.pm/api/micromamba/linux-${MICROMAMBA_ARCH}/latest" \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

SHELL ["/bin/bash", "-lc"]

COPY environment.yml /tmp/build/environment.yml
COPY requirements.txt /tmp/build/requirements.txt
COPY environment.phage-annotation-tools.yml /tmp/build/environment.phage-annotation-tools.yml
COPY environment.phylogroup-caller.yml /tmp/build/environment.phylogroup-caller.yml
COPY environment.serotype-caller.yml /tmp/build/environment.serotype-caller.yml
COPY environment.sequence-type-caller.yml /tmp/build/environment.sequence-type-caller.yml

# Keep this profile allowlist in sync with
# lyzortx/orchestration/ci_image_profiles.py::VALID_CI_IMAGE_PROFILES.
RUN case "${CI_IMAGE_PROFILE}" in \
      base|host-typing|full-bio) ;; \
      *) echo "Unsupported CI_IMAGE_PROFILE=${CI_IMAGE_PROFILE}" >&2; exit 1 ;; \
    esac \
    && micromamba create -y -n phage_env -f /tmp/build/environment.yml \
    && if [[ "${CI_IMAGE_PROFILE}" == "host-typing" || "${CI_IMAGE_PROFILE}" == "full-bio" ]]; then \
         micromamba create -y -n phylogroup_caller -f /tmp/build/environment.phylogroup-caller.yml \
         && micromamba create -y -n serotype_caller -f /tmp/build/environment.serotype-caller.yml \
         && micromamba create -y -n sequence_type_caller -f /tmp/build/environment.sequence-type-caller.yml; \
       fi \
    && if [[ "${CI_IMAGE_PROFILE}" == "full-bio" ]]; then \
         micromamba create -y -n phage_annotation_tools -f /tmp/build/environment.phage-annotation-tools.yml; \
       fi \
    && micromamba clean --all --yes

# pharokka.py is only baked into the full-bio amd64 image for now; see the
# publish workflow's amd64-only note before broadening platform support.
RUN micromamba run -n phage_env python --version \
    && if [[ "${CI_IMAGE_PROFILE}" == "host-typing" || "${CI_IMAGE_PROFILE}" == "full-bio" ]]; then \
         micromamba run -n phylogroup_caller python --version \
         && micromamba run -n serotype_caller ectyper -V \
         && micromamba run -n sequence_type_caller mlst --version; \
       fi \
    && if [[ "${CI_IMAGE_PROFILE}" == "full-bio" ]]; then \
         micromamba run -n phage_annotation_tools pharokka.py --version; \
       fi

WORKDIR /repo
