#!/usr/bin/env python3
"""Run DEPLOY03 host-surface feature derivation on all 403 Picard assemblies in parallel.

Uses pyhmmer for in-process HMMER searches (no subprocess overhead) and translates
O-antigen DNA alleles to protein for ~12x faster phmmer search vs nhmmer DNA scan.

Usage:
    python -m lyzortx.pipeline.deployment_paired_features.run_all_host_surface [--max-workers N]

Pre-computed per-host outputs are skipped automatically (presence of host_surface_features.csv).
After all hosts complete, aggregates results into a single checked-in CSV at
``lyzortx/data/deployment_paired_features/403_host_surface_features.csv``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pyhmmer

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.deployment_paired_features.derive_host_surface_features import (
    build_host_surface_feature_row,
    build_host_surface_schema,
    prepare_host_surface_runtime_inputs,
    summarize_o_antigen_result,
    summarize_receptor_scores,
    _column_names_from_schema,
)
from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15

LOGGER = logging.getLogger(__name__)

ASSEMBLIES_DIR = Path("lyzortx/data/assemblies/picard")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_surface")
EXPECTED_HOST_COUNT = 403
AGGREGATED_CSV_PATH = Path("lyzortx/data/deployment_paired_features/403_host_surface_features.csv")
FAST_PATH_RUNTIME_ID = "deploy07_pyhmmer_surface_fast_path_v1"
_CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def _translate_o_antigen_alleles(dna_query_path: Path, output_path: Path) -> int:
    """Translate O-antigen DNA allele queries to protein for fast phmmer search.

    Returns the number of protein sequences written.
    """

    def iter_fasta_records() -> list[tuple[str, str]]:
        records: list[tuple[str, str]] = []
        current_id = ""
        sequence_parts: list[str] = []
        with dna_query_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id:
                        records.append((current_id, "".join(sequence_parts)))
                    current_id = line[1:].split()[0]
                    sequence_parts = []
                    continue
                sequence_parts.append(line)
        if current_id:
            records.append((current_id, "".join(sequence_parts)))
        return records

    def translate_dna(sequence: str) -> str:
        protein: list[str] = []
        clean_sequence = sequence.upper().replace("U", "T")
        for start in range(0, len(clean_sequence) - 2, 3):
            protein.append(_CODON_TABLE.get(clean_sequence[start : start + 3], "X"))
        return "".join(protein).rstrip("*")

    alleles = iter_fasta_records()
    with output_path.open("w", encoding="utf-8") as out:
        for allele_id, sequence in alleles:
            protein = translate_dna(sequence)
            out.write(f">{allele_id}\n{protein}\n")
    return len(alleles)


# Per-worker cache to avoid reloading queries/HMMs for every host
_worker_cache: dict[str, Any] = {}


def _get_worker_cache(assets: dict) -> dict:
    """Load and cache shared query/HMM data once per worker process."""
    global _worker_cache
    if not _worker_cache:
        amino = pyhmmer.easel.Alphabet.amino()
        with pyhmmer.easel.SequenceFile(assets["o_antigen_prot_path"], digital=True, alphabet=amino) as sf:
            _worker_cache["o_queries"] = list(sf)
        with pyhmmer.easel.SequenceFile(assets["omp_reference_path"], digital=True, alphabet=amino) as sf:
            _worker_cache["omp_queries"] = list(sf)
        with pyhmmer.plan7.HMMFile(assets["capsule_hmm_path"]) as hf:
            _worker_cache["hmms"] = list(hf)
        _worker_cache["amino"] = amino
    return _worker_cache


def build_surface_feature_row_from_scan_results(
    *,
    bacteria_id: str,
    o_antigen_result: dict[str, object],
    receptor_scores: dict[str, float],
    capsule_scores: dict[str, float],
    lps_lookup: dict[str, dict[str, object]],
    capsule_profile_names: Sequence[str],
    include_lps_core_type: bool = True,
) -> dict[str, Any]:
    schema = build_host_surface_schema(
        capsule_profile_names,
        include_lps_core_type=include_lps_core_type,
    )
    return build_host_surface_feature_row(
        bacteria=bacteria_id,
        schema=schema,
        o_antigen_type=str(o_antigen_result.get("o_type", "")),
        o_antigen_score=float(o_antigen_result.get("continuous_score", 0.0) or 0.0),
        lps_core_type=(
            str(lps_lookup.get(str(o_antigen_result.get("o_type", "")), {}).get("proxy_type", ""))
            if include_lps_core_type
            else ""
        ),
        receptor_scores=receptor_scores,
        capsule_profile_scores=capsule_scores,
    )


def _process_one_host(
    bacteria_id: str,
    proteins_path_str: str,
    assets: dict,
) -> tuple[str, bool, dict[str, Any] | str]:
    """Run all 3 pyhmmer scans for one host. Returns raw scan summaries or an error string."""
    try:
        cache = _get_worker_cache(assets)
        amino = cache["amino"]
        proteins_path = Path(proteins_path_str)

        with pyhmmer.easel.SequenceFile(str(proteins_path), digital=True, alphabet=amino) as sf:
            targets = sf.read_block()

        def _name(obj: object) -> str:
            n = obj.name  # type: ignore[union-attr]
            return n if isinstance(n, str) else n.decode()

        # 1. O-antigen phmmer (protein-translated alleles)
        o_antigen_hits: list[tl15.HmmerHit] = []
        for top_hits in pyhmmer.hmmer.phmmer(cache["o_queries"], targets, cpus=1):
            query_name = _name(top_hits.query)
            for hit in top_hits:
                if hit.evalue < 1e-5:
                    o_antigen_hits.append(
                        tl15.HmmerHit(
                            target_name=_name(hit),
                            query_name=query_name,
                            evalue=hit.evalue,
                            score=hit.score,
                            description="",
                        )
                    )
        o_antigen_result = summarize_o_antigen_result(
            hits=o_antigen_hits,
            references=assets["references"],
            o_type_contract=assets["o_type_contract"],
        )

        # 2. Receptor phmmer
        receptor_raw_hits = []
        for top_hits in pyhmmer.hmmer.phmmer(cache["omp_queries"], targets, cpus=1):
            query_name = _name(top_hits.query)
            for hit in top_hits:
                receptor_raw_hits.append(
                    tl15.HmmerHit(
                        target_name=_name(hit),
                        query_name=query_name,
                        evalue=hit.evalue,
                        score=hit.score,
                        description="",
                    )
                )
        receptor_scores = summarize_receptor_scores(receptor_raw_hits)

        # 3. Capsule hmmscan
        capsule_scores: dict[str, float] = {}
        for top_hits in pyhmmer.hmmer.hmmscan(targets, cache["hmms"], cpus=1):
            for hit in top_hits:
                if hit.evalue < tl15.HMMSCAN_EVALUE_THRESHOLD:
                    name = _name(hit)
                    capsule_scores[name] = max(capsule_scores.get(name, 0.0), hit.score)
        return (
            bacteria_id,
            True,
            {
                "o_antigen_result": o_antigen_result,
                "receptor_scores": receptor_scores,
                "capsule_scores": capsule_scores,
            },
        )

    except Exception as exc:
        return bacteria_id, False, str(exc)


def _predict_proteins_one(args: tuple[str, str, str]) -> tuple[str, bool, str]:
    """Predict proteins for one host using pyrodigal. Returns (bacteria_id, success, message)."""
    bacteria_id, assembly_path_str, output_path_str = args
    try:
        output_path = Path(output_path_str)
        if output_path.exists() and output_path.stat().st_size > 0:
            return bacteria_id, True, "cached"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        meta = tl15.predict_proteins(Path(assembly_path_str), output_path)
        return bacteria_id, True, f"{len(meta)} proteins"
    except Exception as exc:
        return bacteria_id, False, str(exc)


def aggregate_host_surface_csvs(
    rows: list[dict],
    aggregated_csv_path: Path,
    schema: dict,
) -> int:
    """Write aggregated surface features CSV. Returns the number of rows written."""
    columns = _column_names_from_schema(schema)
    rows_sorted = sorted(rows, key=lambda r: r.get("bacteria", ""))

    aggregated_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({col: row.get(col, "") for col in columns})

    LOGGER.info("Wrote %d rows to %s", len(rows_sorted), aggregated_csv_path)
    return len(rows_sorted)


def build_host_surface_rows_fast_path(
    *,
    assemblies: Sequence[Path],
    output_dir: Path,
    max_workers: int,
    include_lps_core_type: bool = True,
) -> dict[str, Any]:
    if not assemblies:
        raise ValueError("Host-surface fast path requires at least one assembly.")

    LOGGER.info(
        "Building host-surface rows via %s for %d assemblies with %d workers",
        FAST_PATH_RUNTIME_ID,
        len(assemblies),
        max_workers,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.now(timezone.utc)
    protein_tasks = [
        (assembly.stem, str(assembly), str(output_dir / assembly.stem / "predicted_proteins.faa"))
        for assembly in assemblies
    ]
    protein_cache_hits = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_predict_proteins_one, task): task[0] for task in protein_tasks}
        for future in as_completed(futures):
            bacteria_id, ok, message = future.result()
            if not ok:
                raise RuntimeError(f"Protein prediction failed for {bacteria_id}: {message}")
            if message == "cached":
                protein_cache_hits += 1

    prodigal_elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    LOGGER.info(
        "Host-surface protein prediction finished in %.1fs (%d/%d cached)",
        prodigal_elapsed,
        protein_cache_hits,
        len(protein_tasks),
    )

    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    runtime_inputs = prepare_host_surface_runtime_inputs(
        assets_output_dir=assets_dir,
        picard_metadata_path=tl15.DEFAULT_PICARD_METADATA_PATH,
        o_type_output_path=tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
        o_type_allele_path=tl15.DEFAULT_O_TYPE_ALLELE_PATH,
        o_antigen_override_path=tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
        abc_capsule_profile_dir=tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
        omp_reference_path=tl15.DEFAULT_OMP_REFERENCE_PATH,
    )
    o_antigen_prot_path = assets_dir / "o_antigen_protein_queries.faa"
    allele_count = _translate_o_antigen_alleles(runtime_inputs.o_antigen_query_path, o_antigen_prot_path)

    serializable_assets = {
        "o_antigen_prot_path": str(o_antigen_prot_path),
        "omp_reference_path": str(tl15.DEFAULT_OMP_REFERENCE_PATH),
        "capsule_hmm_path": str(runtime_inputs.capsule_hmm_bundle_path),
        "capsule_profile_names": list(runtime_inputs.capsule_profile_names),
        "lps_lookup": {key: dict(value) for key, value in runtime_inputs.lps_lookup.items()},
        "references": list(runtime_inputs.references),
        "o_type_contract": dict(runtime_inputs.o_type_contract),
    }

    scan_start = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_one_host,
                assembly.stem,
                str(output_dir / assembly.stem / "predicted_proteins.faa"),
                serializable_assets,
            ): assembly.stem
            for assembly in assemblies
        }
        for future in as_completed(futures):
            bacteria_id, ok, result = future.result()
            if not ok:
                failures.append((bacteria_id, str(result)))
                continue
            scan_result = dict(result)
            rows.append(
                build_surface_feature_row_from_scan_results(
                    bacteria_id=bacteria_id,
                    o_antigen_result=scan_result["o_antigen_result"],
                    receptor_scores=scan_result["receptor_scores"],
                    capsule_scores=scan_result["capsule_scores"],
                    lps_lookup=serializable_assets["lps_lookup"],
                    capsule_profile_names=runtime_inputs.capsule_profile_names,
                    include_lps_core_type=include_lps_core_type,
                )
            )

    if failures:
        formatted = "; ".join(f"{bacteria}: {message}" for bacteria, message in sorted(failures))
        raise RuntimeError(f"Host-surface fast path failed for {len(failures)} assemblies: {formatted}")

    scan_elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
    total_elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    schema = build_host_surface_schema(
        runtime_inputs.capsule_profile_names,
        include_lps_core_type=include_lps_core_type,
    )
    return {
        "rows": sorted(rows, key=lambda row: str(row["bacteria"])),
        "schema": schema,
        "runtime_metadata": {
            "runtime_id": FAST_PATH_RUNTIME_ID,
            "legacy_nhmmer_path_forbidden": True,
            "host_count": len(assemblies),
            "protein_cache_hit_count": protein_cache_hits,
            "o_antigen_allele_count": allele_count,
            "include_lps_core_type": include_lps_core_type,
            "prodigal_elapsed_seconds": round(prodigal_elapsed, 6),
            "scan_elapsed_seconds": round(scan_elapsed, 6),
            "total_elapsed_seconds": round(total_elapsed, 6),
        },
    }


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not ASSEMBLIES_DIR.exists():
        raise FileNotFoundError(f"Assemblies directory not found: {ASSEMBLIES_DIR}")

    assemblies = sorted(ASSEMBLIES_DIR.glob("*.fasta"))
    if len(assemblies) != EXPECTED_HOST_COUNT:
        LOGGER.warning("Expected %d assemblies, found %d", EXPECTED_HOST_COUNT, len(assemblies))
    result = build_host_surface_rows_fast_path(
        assemblies=assemblies,
        output_dir=args.output_dir,
        max_workers=args.max_workers,
        include_lps_core_type=True,
    )
    rows = result["rows"]
    schema = result["schema"]
    runtime_metadata = result["runtime_metadata"]

    count = aggregate_host_surface_csvs(rows, AGGREGATED_CSV_PATH, schema)
    if count != EXPECTED_HOST_COUNT:
        LOGGER.warning("Expected %d hosts in aggregated CSV, got %d", EXPECTED_HOST_COUNT, count)
    LOGGER.info(
        "All done in %.0fs: prodigal %.0fs + scans %.0fs, %d rows written to %s",
        runtime_metadata["total_elapsed_seconds"],
        runtime_metadata["prodigal_elapsed_seconds"],
        runtime_metadata["scan_elapsed_seconds"],
        count,
        AGGREGATED_CSV_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
