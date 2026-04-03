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
    RECEPTOR_SCORE_COLUMNS,
    build_host_surface_schema,
    prepare_host_surface_runtime_inputs,
    summarize_receptor_scores,
    _capsule_score_column_name,
    _column_names_from_schema,
)
from lyzortx.pipeline.track_l.steps import build_raw_host_surface_projector as tl15

LOGGER = logging.getLogger(__name__)

ASSEMBLIES_DIR = Path("lyzortx/data/assemblies/picard")
DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/deployment_paired_features/host_surface")
EXPECTED_HOST_COUNT = 403
AGGREGATED_CSV_PATH = Path("lyzortx/data/deployment_paired_features/403_host_surface_features.csv")


def _translate_o_antigen_alleles(dna_query_path: Path, output_path: Path) -> int:
    """Translate O-antigen DNA allele queries to protein for fast phmmer search.

    Returns the number of protein sequences written.
    """
    codon_table = {
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
            protein.append(codon_table.get(clean_sequence[start : start + 3], "X"))
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


def best_o_antigen_call(o_hits: dict[str, float]) -> tuple[str, float]:
    """Pick the best O-antigen allele hit and extract O-type.

    Returns (o_type, best_score).  Empty string and 0.0 when no hits pass threshold.
    """
    if not o_hits:
        return "", 0.0
    best_allele = max(o_hits, key=o_hits.get)
    best_score = o_hits[best_allele]
    o_type = best_allele.split("__")[0] if best_allele else ""
    return o_type, round(float(best_score), 6)


def build_surface_feature_row(
    *,
    bacteria_id: str,
    o_hits: dict[str, float],
    receptor_scores: dict[str, float],
    capsule_scores: dict[str, float],
    lps_lookup: dict[str, dict[str, object]],
    capsule_profile_names: Sequence[str],
) -> dict[str, Any]:
    """Build a single feature-row dict from scan results.  Pure function — no I/O."""
    o_type, o_score = best_o_antigen_call(o_hits)
    lps_entry = lps_lookup.get(o_type, {})

    row: dict[str, Any] = {
        "bacteria": bacteria_id,
        "host_o_antigen_type": o_type,
        "host_o_antigen_score": o_score,
        "host_lps_core_type": str(lps_entry.get("proxy_type", "")),
    }
    for receptor_name, col_name in RECEPTOR_SCORE_COLUMNS:
        row[col_name] = round(float(receptor_scores.get(receptor_name, 0.0)), 6)
    for pname in capsule_profile_names:
        row[_capsule_score_column_name(pname)] = round(float(capsule_scores.get(pname, 0.0)), 6)
    return row


def _process_one_host(
    bacteria_id: str,
    proteins_path_str: str,
    assets: dict,
) -> tuple[str, bool, dict | str]:
    """Run all 3 pyhmmer scans for one host. Returns (bacteria_id, success, row_or_error)."""
    try:
        cache = _get_worker_cache(assets)
        amino = cache["amino"]
        proteins_path = Path(proteins_path_str)

        with pyhmmer.easel.SequenceFile(str(proteins_path), digital=True, alphabet=amino) as sf:
            targets = sf.read_block()

        # 1. O-antigen phmmer (protein-translated alleles)
        o_hits: dict[str, float] = {}
        for top_hits in pyhmmer.hmmer.phmmer(cache["o_queries"], targets, cpus=1):
            query_name = top_hits.query.name
            for hit in top_hits:
                if hit.evalue < 1e-5:
                    if query_name not in o_hits or hit.score > o_hits[query_name]:
                        o_hits[query_name] = hit.score

        # 2. Receptor phmmer
        receptor_raw_hits = []
        for top_hits in pyhmmer.hmmer.phmmer(cache["omp_queries"], targets, cpus=1):
            query_name = top_hits.query.name
            for hit in top_hits:
                receptor_raw_hits.append(
                    tl15.HmmerHit(
                        target_name=hit.name,
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
                    name = hit.name
                    capsule_scores[name] = max(capsule_scores.get(name, 0.0), hit.score)

        row = build_surface_feature_row(
            bacteria_id=bacteria_id,
            o_hits=o_hits,
            receptor_scores=receptor_scores,
            capsule_scores=capsule_scores,
            lps_lookup=assets["lps_lookup"],
            capsule_profile_names=assets["capsule_profile_names"],
        )
        return bacteria_id, True, row

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

    # Phase 1: Predict proteins for all hosts
    LOGGER.info("Phase 1: Predicting proteins for %d hosts with %d workers", len(assemblies), args.max_workers)
    start = datetime.now(timezone.utc)

    protein_tasks = [(a.stem, str(a), str(args.output_dir / a.stem / "predicted_proteins.faa")) for a in assemblies]
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_predict_proteins_one, t): t[0] for t in protein_tasks}
        done = 0
        for f in as_completed(futures):
            done += 1
            bid, ok, msg = f.result()
            if not ok:
                LOGGER.error("Prodigal failed for %s: %s", bid, msg)
                return 1
            if done % 50 == 0 or done == len(futures):
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()
                LOGGER.info("  Prodigal: %d/%d done (%.0fs)", done, len(futures), elapsed)

    prodigal_elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    LOGGER.info("Phase 1 done: %.0fs", prodigal_elapsed)

    # Prepare shared assets
    LOGGER.info("Preparing shared assets (O-antigen queries, capsule HMMs)...")
    assets_dir = args.output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ri = prepare_host_surface_runtime_inputs(
        assets_output_dir=assets_dir,
        picard_metadata_path=tl15.DEFAULT_PICARD_METADATA_PATH,
        o_type_output_path=tl15.DEFAULT_O_TYPE_OUTPUT_PATH,
        o_type_allele_path=tl15.DEFAULT_O_TYPE_ALLELE_PATH,
        o_antigen_override_path=tl15.DEFAULT_O_ANTIGEN_OVERRIDE_PATH,
        abc_capsule_profile_dir=tl15.DEFAULT_ABC_CAPSULE_PROFILE_DIR,
        omp_reference_path=tl15.DEFAULT_OMP_REFERENCE_PATH,
    )

    # Translate O-antigen alleles to protein
    o_antigen_prot_path = assets_dir / "o_antigen_protein_queries.faa"
    allele_count = _translate_o_antigen_alleles(ri.o_antigen_query_path, o_antigen_prot_path)
    LOGGER.info("Translated %d O-antigen alleles to protein", allele_count)

    schema = build_host_surface_schema(ri.capsule_profile_names)

    # Serializable assets dict for worker processes
    serializable_assets = {
        "o_antigen_prot_path": str(o_antigen_prot_path),
        "omp_reference_path": str(tl15.DEFAULT_OMP_REFERENCE_PATH),
        "capsule_hmm_path": str(ri.capsule_hmm_bundle_path),
        "capsule_profile_names": list(ri.capsule_profile_names),
        "lps_lookup": {k: dict(v) for k, v in ri.lps_lookup.items()},
    }

    # Phase 2: pyhmmer scans for all hosts
    LOGGER.info("Phase 2: pyhmmer scans for %d hosts with %d workers", len(assemblies), args.max_workers)
    scan_start = datetime.now(timezone.utc)

    rows: list[dict] = []
    failed = 0
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(
                _process_one_host,
                a.stem,
                str(args.output_dir / a.stem / "predicted_proteins.faa"),
                serializable_assets,
            ): a.stem
            for a in assemblies
        }
        done = 0
        for f in as_completed(futures):
            done += 1
            bid, ok, result = f.result()
            if ok:
                rows.append(result)
            else:
                failed += 1
                failures.append((bid, str(result)))
                LOGGER.error("FAILED %s: %s", bid, result)
            if done % 50 == 0 or done == len(futures):
                elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
                LOGGER.info("  Scans: %d/%d done (%.0fs, %d failed)", done, len(futures), elapsed, failed)

    scan_elapsed = (datetime.now(timezone.utc) - scan_start).total_seconds()
    LOGGER.info("Phase 2 done: %.0fs (%d succeeded, %d failed)", scan_elapsed, len(rows), failed)

    if failures:
        LOGGER.error("Failed hosts:")
        for bid, msg in failures:
            LOGGER.error("  %s: %s", bid, msg)
        return 1

    # Aggregate into checked-in CSV
    count = aggregate_host_surface_csvs(rows, AGGREGATED_CSV_PATH, schema)
    if count != EXPECTED_HOST_COUNT:
        LOGGER.warning("Expected %d hosts in aggregated CSV, got %d", EXPECTED_HOST_COUNT, count)

    total_elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    LOGGER.info(
        "All done in %.0fs: prodigal %.0fs + scans %.0fs, %d rows written to %s",
        total_elapsed,
        prodigal_elapsed,
        scan_elapsed,
        count,
        AGGREGATED_CSV_PATH,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
