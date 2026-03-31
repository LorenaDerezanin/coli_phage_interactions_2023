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
        wget \
    && rm -rf /var/lib/apt/lists/*

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
