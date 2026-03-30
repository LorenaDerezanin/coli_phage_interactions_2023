#!/usr/bin/env python3
"""TL09: Validate generalized inference on Virus-Host DB positive pairs."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from random import Random
from typing import Mapping, Sequence

import pandas as pd

from lyzortx.log_config import setup_logging
from lyzortx.pipeline.steel_thread_v0.io.write_outputs import ensure_directory, write_csv, write_json
from lyzortx.pipeline.steel_thread_v0.steps._io_helpers import read_csv_rows
from lyzortx.pipeline.track_d.steps.build_phage_protein_sets import read_panel_phages
from lyzortx.pipeline.track_l.steps import build_generalized_inference_bundle
from lyzortx.pipeline.track_l.steps import generalized_inference

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("lyzortx/generated_outputs/track_l/vhdb_generalized_inference_validation")
DEFAULT_RAW_DIRNAME = "raw_downloads"
DEFAULT_HOST_ASSEMBLY_DIRNAME = "host_assemblies"
DEFAULT_PHAGE_DIRNAME = "vhdb_phage_fastas"
DEFAULT_PANEL_HOSTS_PATH = Path("data/metadata/370+host_cross_validation_groups_1e-4.csv")
DEFAULT_PANEL_PHAGE_DIR = Path("data/genomics/phages/FNA")
DEFAULT_PANEL_PHAGE_METADATA_PATH = Path("data/genomics/phages/guelin_collection.csv")
DEFAULT_ST02_PAIR_TABLE_PATH = Path("lyzortx/generated_outputs/steel_thread_v0/intermediate/st02_pair_table.csv")
DEFAULT_BUNDLE_DIR = Path("lyzortx/generated_outputs/track_l/generalized_inference_bundle")
DEFAULT_BUNDLE_PATH = DEFAULT_BUNDLE_DIR / build_generalized_inference_bundle.BUNDLE_FILENAME
DEFAULT_VHDB_URL = "https://www.genome.jp/ftp/db/virushostdb/virushostdb.tsv"
DEFAULT_ASSEMBLY_SUMMARY_URL = "https://ftp.ncbi.nlm.nih.gov/genomes/ASSEMBLY_REPORTS/assembly_summary_refseq.txt"
ENTREZ_FASTA_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_NOVEL_HOST_COUNT = 10
DEFAULT_MIN_POSITIVE_PHAGES_PER_HOST = 5
DEFAULT_RANDOM_SEED = 42
DEFAULT_BASE_RATE = 0.30
ENTREZ_MIN_REQUEST_INTERVAL_SECONDS = 0.4
ENTREZ_MAX_RETRIES = 5
ROUNDTRIP_PANEL_HOSTS = ("LF82", "EDL933", "55989", "536", "BL21")
ASSEMBLY_LEVEL_PRIORITY = {
    "Complete Genome": 0,
    "Chromosome": 1,
    "Scaffold": 2,
    "Contig": 3,
}
SUMMARY_FIELDNAMES = ("metric", "value")
_LAST_ENTREZ_REQUEST_TIME = 0.0


@dataclass(frozen=True)
class PositivePair:
    host_tax_id: str
    host_name: str
    phage_accession: str
    source_virus_name: str


@dataclass(frozen=True)
class AssemblyRecord:
    assembly_accession: str
    taxid: str
    organism_name: str
    infraspecific_name: str
    isolate: str
    version_status: str
    assembly_level: str
    refseq_category: str
    ftp_path: str


@dataclass(frozen=True)
class HostCandidate:
    host_tax_id: str
    host_name: str
    positive_pair_count: int
    unique_phage_count: int
    panel_match: str
    is_panel_host: bool
    assembly_accession: str
    assembly_level: str
    assembly_organism_name: str
    assembly_ftp_path: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vhdb-url", default=DEFAULT_VHDB_URL)
    parser.add_argument("--assembly-summary-url", default=DEFAULT_ASSEMBLY_SUMMARY_URL)
    parser.add_argument("--panel-hosts-path", type=Path, default=DEFAULT_PANEL_HOSTS_PATH)
    parser.add_argument("--panel-phage-dir", type=Path, default=DEFAULT_PANEL_PHAGE_DIR)
    parser.add_argument("--panel-phage-metadata-path", type=Path, default=DEFAULT_PANEL_PHAGE_METADATA_PATH)
    parser.add_argument("--st02-pair-table-path", type=Path, default=DEFAULT_ST02_PAIR_TABLE_PATH)
    parser.add_argument("--bundle-path", type=Path, default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--novel-host-count", type=int, default=DEFAULT_NOVEL_HOST_COUNT)
    parser.add_argument("--min-positive-phages-per-host", type=int, default=DEFAULT_MIN_POSITIVE_PHAGES_PER_HOST)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args(argv)


def _normalize_host_name(name: str) -> str:
    value = name.lower().strip()
    for token in ("escherichia coli", "e. coli", "str.", "substr.", "subsp."):
        value = value.replace(token, "")
    value = re.sub(r"\b(?:strain|isolate)\b", "", value)
    value = value.replace("[alpha]", "alpha")
    return re.sub(r"[^a-z0-9]+", "", value)


def _split_accessions(value: str) -> list[str]:
    if not value.strip():
        return []
    tokens = [token.strip() for token in value.replace("|", ",").replace(";", ",").split(",")]
    return [token for token in tokens if token]


def _download_text(url: str) -> str:
    return _download_with_retry(url).decode("utf-8", errors="replace")


def _download_binary(url: str) -> bytes:
    return _download_with_retry(url)


def _maybe_rate_limit_entrez(url: str) -> None:
    global _LAST_ENTREZ_REQUEST_TIME
    if "eutils.ncbi.nlm.nih.gov" not in url:
        return
    elapsed = time.monotonic() - _LAST_ENTREZ_REQUEST_TIME
    if elapsed < ENTREZ_MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(ENTREZ_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    _LAST_ENTREZ_REQUEST_TIME = time.monotonic()


def _download_with_retry(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Codex TL09 generalized inference validation/1.0"})
    last_error: Exception | None = None
    for attempt in range(ENTREZ_MAX_RETRIES):
        try:
            _maybe_rate_limit_entrez(url)
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504} or attempt == ENTREZ_MAX_RETRIES - 1:
                raise
            sleep_seconds = 2**attempt
            LOGGER.warning("Retrying %s after HTTP %d in %ss", url, error.code, sleep_seconds)
            time.sleep(sleep_seconds)
        except urllib.error.URLError as error:
            last_error = error
            if attempt == ENTREZ_MAX_RETRIES - 1:
                raise
            sleep_seconds = 2**attempt
            LOGGER.warning("Retrying %s after network error in %ss: %s", url, sleep_seconds, error)
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Download unexpectedly failed without raising for {url}")


def load_panel_hosts(path: Path) -> tuple[set[str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None or "bacteria" not in reader.fieldnames:
            raise ValueError(f"Missing required 'bacteria' column in {path}")
        panel_hosts = {str(row["bacteria"]).strip() for row in reader if str(row.get("bacteria", "")).strip()}
    if not panel_hosts:
        raise ValueError(f"No panel hosts were found in {path}")
    normalized_lookup = {_normalize_host_name(host): host for host in panel_hosts}
    return panel_hosts, normalized_lookup


def match_panel_host_name(host_name: str, panel_lookup: Mapping[str, str]) -> str:
    normalized = _normalize_host_name(host_name)
    if not normalized:
        return ""
    exact = panel_lookup.get(normalized)
    if exact:
        return exact
    candidates = [
        panel_name
        for token, panel_name in panel_lookup.items()
        if len(token) >= 4 and (token in normalized or normalized in token)
    ]
    if not candidates:
        return ""
    return sorted(candidates, key=lambda value: (-len(_normalize_host_name(value)), value))[0]


def parse_vhdb_positive_pairs(vhdb_text: str) -> list[PositivePair]:
    reader = csv.DictReader(StringIO(vhdb_text), delimiter="\t")
    pairs: list[PositivePair] = []
    for row in reader:
        host_tax_id = (row.get("host tax id") or "").strip()
        host_name = (row.get("host name") or "").strip()
        host_lineage = (row.get("host lineage") or "").lower()
        if not host_tax_id or host_tax_id == "562":
            continue
        if "escherichia coli" not in host_name.lower() and "escherichia coli" not in host_lineage:
            continue
        virus_name = (row.get("virus name") or "").strip()
        for accession in _split_accessions(row.get("refseq id", "")):
            pairs.append(
                PositivePair(
                    host_tax_id=host_tax_id,
                    host_name=host_name,
                    phage_accession=accession,
                    source_virus_name=virus_name,
                )
            )
    if not pairs:
        raise ValueError("Virus-Host DB filtering produced zero E. coli strain-level positive pairs.")
    return pairs


def summarize_positive_pairs(pairs: Sequence[PositivePair]) -> dict[str, int]:
    return {
        "positive_pair_count": len({(pair.host_tax_id, pair.host_name, pair.phage_accession) for pair in pairs}),
        "host_count": len({(pair.host_tax_id, pair.host_name) for pair in pairs}),
        "phage_accession_count": len({pair.phage_accession for pair in pairs}),
    }


def parse_assembly_summary(summary_text: str) -> dict[str, list[AssemblyRecord]]:
    header: list[str] | None = None
    by_taxid: dict[str, list[AssemblyRecord]] = defaultdict(list)
    for raw_line in summary_text.splitlines():
        if raw_line.startswith("#assembly_accession"):
            header = raw_line[1:].split("\t")
            continue
        if raw_line.startswith("#") or not raw_line.strip():
            continue
        if header is None:
            raise ValueError("Assembly summary header was not found.")
        fields = raw_line.split("\t")
        if len(fields) != len(header):
            continue
        row = dict(zip(header, fields))
        if row.get("ftp_path", "") in {"", "na"}:
            continue
        record = AssemblyRecord(
            assembly_accession=row["assembly_accession"],
            taxid=row["taxid"],
            organism_name=row["organism_name"],
            infraspecific_name=row["infraspecific_name"],
            isolate=row["isolate"],
            version_status=row["version_status"],
            assembly_level=row["assembly_level"],
            refseq_category=row["refseq_category"],
            ftp_path=row["ftp_path"],
        )
        by_taxid[record.taxid].append(record)
    if not by_taxid:
        raise ValueError("Assembly summary parsing produced zero usable assembly records.")
    return by_taxid


def choose_best_assembly(records: Sequence[AssemblyRecord]) -> AssemblyRecord:
    if not records:
        raise ValueError("At least one assembly record is required.")
    return sorted(
        records,
        key=lambda record: (
            0 if record.version_status == "latest" else 1,
            ASSEMBLY_LEVEL_PRIORITY.get(record.assembly_level, 9),
            0 if record.refseq_category in {"reference genome", "representative genome"} else 1,
            record.assembly_accession,
        ),
    )[0]


def build_host_candidates(
    pairs: Sequence[PositivePair],
    assemblies_by_taxid: Mapping[str, Sequence[AssemblyRecord]],
    panel_lookup: Mapping[str, str],
) -> list[HostCandidate]:
    phages_by_host: dict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in pairs:
        phages_by_host[(pair.host_tax_id, pair.host_name)].add(pair.phage_accession)

    candidates: list[HostCandidate] = []
    for (host_tax_id, host_name), phages in sorted(phages_by_host.items()):
        assembly_records = assemblies_by_taxid.get(host_tax_id, [])
        if not assembly_records:
            continue
        best_assembly = choose_best_assembly(assembly_records)
        panel_match = match_panel_host_name(host_name, panel_lookup)
        candidates.append(
            HostCandidate(
                host_tax_id=host_tax_id,
                host_name=host_name,
                positive_pair_count=len(phages),
                unique_phage_count=len(phages),
                panel_match=panel_match,
                is_panel_host=bool(panel_match),
                assembly_accession=best_assembly.assembly_accession,
                assembly_level=best_assembly.assembly_level,
                assembly_organism_name=best_assembly.organism_name,
                assembly_ftp_path=best_assembly.ftp_path,
            )
        )
    if not candidates:
        raise ValueError("No host candidates remained after assembly availability and pair-count filtering.")
    return candidates


def select_validation_hosts(
    host_candidates: Sequence[HostCandidate],
    *,
    novel_host_count: int,
    min_positive_phages_per_host: int,
) -> tuple[list[HostCandidate], list[HostCandidate]]:
    novel_hosts = sorted(
        (
            candidate
            for candidate in host_candidates
            if not candidate.is_panel_host and candidate.positive_pair_count >= min_positive_phages_per_host
        ),
        key=lambda candidate: (
            ASSEMBLY_LEVEL_PRIORITY.get(candidate.assembly_level, 9),
            candidate.positive_pair_count,
            candidate.host_name,
        ),
    )[:novel_host_count]
    if len(novel_hosts) < novel_host_count:
        raise ValueError(f"Requested {novel_host_count} novel hosts, but only {len(novel_hosts)} were available.")

    roundtrip_by_panel_name: dict[str, HostCandidate] = {}
    for candidate in host_candidates:
        if candidate.panel_match and candidate.panel_match in ROUNDTRIP_PANEL_HOSTS:
            existing = roundtrip_by_panel_name.get(candidate.panel_match)
            if existing is None or candidate.positive_pair_count > existing.positive_pair_count:
                roundtrip_by_panel_name[candidate.panel_match] = candidate
    roundtrip_hosts = [
        roundtrip_by_panel_name[name] for name in ROUNDTRIP_PANEL_HOSTS if name in roundtrip_by_panel_name
    ]
    if not roundtrip_hosts:
        raise ValueError("No round-trip panel hosts were found in Virus-Host DB.")
    return novel_hosts, roundtrip_hosts


def download_host_assembly(host: HostCandidate, host_dir: Path) -> Path:
    ensure_directory(host_dir)
    output_path = host_dir / f"{host.assembly_accession}.fna"
    if output_path.exists():
        return output_path
    ftp_basename = Path(host.assembly_ftp_path).name
    base_url = host.assembly_ftp_path.replace("ftp://", "https://", 1)
    fasta_url = f"{base_url}/{ftp_basename}_genomic.fna.gz"
    LOGGER.info("Starting host assembly download for %s", host.host_name)
    compressed = _download_binary(fasta_url)
    fasta_text = gzip.decompress(compressed).decode("utf-8")
    output_path.write_text(fasta_text, encoding="utf-8")
    LOGGER.info("Finished host assembly download for %s -> %s", host.host_name, output_path)
    return output_path


def download_phage_fasta(accession: str, phage_dir: Path) -> Path:
    ensure_directory(phage_dir)
    output_path = phage_dir / f"{accession}.fna"
    if output_path.exists():
        return output_path
    query = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": accession,
            "rettype": "fasta",
            "retmode": "text",
        }
    )
    fasta_url = f"{ENTREZ_FASTA_URL}?{query}"
    LOGGER.info("Starting phage FASTA download for %s", accession)
    fasta_text = _download_text(fasta_url)
    if not fasta_text.startswith(">"):
        raise ValueError(f"NCBI FASTA fetch for {accession} did not return FASTA content.")
    output_path.write_text(fasta_text, encoding="utf-8")
    LOGGER.info("Finished phage FASTA download for %s -> %s", accession, output_path)
    return output_path


def collect_phages_for_hosts(pairs: Sequence[PositivePair], hosts: Sequence[HostCandidate]) -> dict[str, list[str]]:
    selected_taxids = {host.host_tax_id for host in hosts}
    phages_by_taxid: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if pair.host_tax_id in selected_taxids:
            phages_by_taxid[pair.host_tax_id].add(pair.phage_accession)
    missing = [host.host_name for host in hosts if not phages_by_taxid.get(host.host_tax_id)]
    if missing:
        raise ValueError(f"Selected hosts had zero associated phages: {', '.join(sorted(missing))}")
    return {taxid: sorted(accessions) for taxid, accessions in phages_by_taxid.items()}


def ensure_bundle(bundle_path: Path) -> Path:
    if bundle_path.exists():
        return bundle_path
    LOGGER.info("TL08 bundle missing at %s; rebuilding prerequisites and bundle", bundle_path)
    ensure_directory(bundle_path.parent)
    args = build_generalized_inference_bundle.parse_args(["--output-dir", str(bundle_path.parent)])
    build_generalized_inference_bundle.ensure_prerequisite_outputs(args)
    lightgbm_params = build_generalized_inference_bundle.load_locked_lightgbm_params(args.tg01_summary_path)
    build_generalized_inference_bundle.build_model_bundle(
        st02_pair_table_path=args.st02_pair_table_path,
        st03_split_assignments_path=args.st03_split_assignments_path,
        defense_subtypes_path=args.defense_subtypes_path,
        phage_kmer_feature_path=args.phage_kmer_feature_path,
        phage_kmer_svd_path=args.phage_kmer_svd_path,
        output_dir=args.output_dir,
        lightgbm_params=lightgbm_params,
        random_state=args.random_state,
        calibration_fold=args.calibration_fold,
    )
    if not bundle_path.exists():
        raise FileNotFoundError(f"Expected TL08 bundle at {bundle_path} after rebuild, but it was not created.")
    return bundle_path


def compute_panel_base_rate(st02_pair_table_path: Path) -> float:
    rows = read_csv_rows(st02_pair_table_path)
    if not rows:
        raise ValueError(f"ST02 pair table was empty at {st02_pair_table_path}")
    labels = [int(row["label_hard_any_lysis"]) for row in rows if str(row.get("label_hard_any_lysis", "")).strip()]
    if not labels:
        raise ValueError(f"No resolved hard-label rows were found in {st02_pair_table_path}")
    return sum(labels) / len(labels)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Median requires at least one value.")
    return float(statistics.median(values))


def evaluate_positive_only_metrics(
    *,
    prediction_frames: Sequence[pd.DataFrame],
    host_metadata: Mapping[str, HostCandidate],
    known_phages_by_taxid: Mapping[str, Sequence[str]],
    base_rate: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rng = Random(random_seed)
    positive_rows: list[dict[str, object]] = []
    random_rows: list[dict[str, object]] = []
    host_summary_rows: list[dict[str, object]] = []

    for frame in prediction_frames:
        host_tax_id = str(frame["host_tax_id"].iloc[0])
        host = host_metadata[host_tax_id]
        positives = set(known_phages_by_taxid[host_tax_id])
        positive_frame = frame[frame["phage"].isin(positives)].copy()
        if positive_frame.empty:
            raise ValueError(f"No positive phages were found in predictions for {host.host_name}")
        positive_frame["host_name"] = host.host_name
        positive_frame["panel_match"] = host.panel_match
        positive_rows.extend(positive_frame.to_dict("records"))

        candidate_pool = [phage for phage in frame["phage"].tolist() if phage not in positives]
        if len(candidate_pool) < len(positive_frame):
            raise ValueError(
                f"Host {host.host_name} had only {len(candidate_pool)} non-positive candidates; "
                f"need at least {len(positive_frame)} for random-pair comparison."
            )
        sampled = set(rng.sample(candidate_pool, len(positive_frame)))
        random_frame = frame[frame["phage"].isin(sampled)].copy()
        random_frame["host_name"] = host.host_name
        random_frame["panel_match"] = host.panel_match
        random_rows.extend(random_frame.to_dict("records"))

        percentile_scores = [
            1.0 - ((int(rank) - 1) / (len(frame) - 1)) if len(frame) > 1 else 1.0
            for rank in positive_frame["rank"].tolist()
        ]
        host_summary_rows.append(
            {
                "host_tax_id": host_tax_id,
                "host_name": host.host_name,
                "panel_match": host.panel_match,
                "candidate_phage_count": len(frame),
                "positive_pair_count": len(positive_frame),
                "positive_median_p_lysis": _median([float(value) for value in positive_frame["p_lysis"].tolist()]),
                "positive_median_rank": _median([float(value) for value in positive_frame["rank"].tolist()]),
                "positive_median_rank_percentile": _median(percentile_scores),
                "random_median_p_lysis": _median([float(value) for value in random_frame["p_lysis"].tolist()]),
            }
        )

    positive_df = pd.DataFrame(positive_rows)
    random_df = pd.DataFrame(random_rows)
    host_summary_df = pd.DataFrame(host_summary_rows).sort_values(["panel_match", "host_name"]).reset_index(drop=True)
    overall_metrics = {
        "base_rate": base_rate,
        "positive_pair_count": float(len(positive_df)),
        "positive_host_count": float(host_summary_df["host_tax_id"].nunique()),
        "positive_median_p_lysis": _median([float(value) for value in positive_df["p_lysis"].tolist()]),
        "random_median_p_lysis": _median([float(value) for value in random_df["p_lysis"].tolist()]),
        "host_median_positive_rank_percentile": _median(
            [float(value) for value in host_summary_df["positive_median_rank_percentile"].tolist()]
        ),
        "host_count_above_median_rank": float(
            sum(float(value) > 0.5 for value in host_summary_df["positive_median_rank_percentile"].tolist())
        ),
        "host_count_positive_median_above_base_rate": float(
            sum(float(value) > base_rate for value in host_summary_df["positive_median_p_lysis"].tolist())
        ),
    }
    return positive_df, random_df, host_summary_df, overall_metrics


def build_roundtrip_comparison(
    *,
    prediction_frames: Sequence[pd.DataFrame],
    host_metadata: Mapping[str, HostCandidate],
    panel_predictions_path: Path,
) -> pd.DataFrame:
    reference = pd.read_csv(panel_predictions_path)
    rows: list[dict[str, object]] = []
    for frame in prediction_frames:
        host_tax_id = str(frame["host_tax_id"].iloc[0])
        host = host_metadata[host_tax_id]
        if not host.panel_match:
            continue
        observed = frame[["phage", "p_lysis", "rank"]].rename(
            columns={"p_lysis": "observed_p_lysis", "rank": "observed_rank"}
        )
        expected = reference[reference["bacteria"] == host.panel_match][
            ["phage", "pred_lightgbm_isotonic", "rank_lightgbm_isotonic"]
        ].rename(
            columns={
                "pred_lightgbm_isotonic": "expected_p_lysis",
                "rank_lightgbm_isotonic": "expected_rank",
            }
        )
        merged = observed.merge(expected, on="phage", how="inner")
        if merged.empty:
            raise ValueError(f"No overlapping panel phages for round-trip host {host.panel_match}")
        rows.append(
            {
                "host_tax_id": host_tax_id,
                "host_name": host.host_name,
                "panel_match": host.panel_match,
                "overlap_panel_phage_count": len(merged),
                "median_abs_probability_delta": _median(
                    [abs(float(a) - float(b)) for a, b in zip(merged["observed_p_lysis"], merged["expected_p_lysis"])]
                ),
                "max_abs_probability_delta": max(
                    abs(float(a) - float(b)) for a, b in zip(merged["observed_p_lysis"], merged["expected_p_lysis"])
                ),
                "identical_rank_count": int((merged["observed_rank"] == merged["expected_rank"]).sum()),
            }
        )
    if not rows:
        raise ValueError("Round-trip comparison produced zero rows.")
    return pd.DataFrame(rows).sort_values(["panel_match", "host_name"]).reset_index(drop=True)


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write empty DataFrame to {path}")
    write_csv(path, list(frame.columns), frame.to_dict("records"))


def filter_roundtrip_hosts_for_reference(
    roundtrip_hosts: Sequence[HostCandidate],
    panel_predictions_path: Path,
) -> list[HostCandidate]:
    with panel_predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        reference_hosts = {str(row["bacteria"]).strip() for row in reader if str(row.get("bacteria", "")).strip()}
    filtered = [host for host in roundtrip_hosts if host.panel_match in reference_hosts]
    if not filtered:
        raise ValueError("None of the round-trip panel hosts were present in the TL08 reference prediction table.")
    return filtered


def main(argv: Sequence[str] | None = None) -> int:
    setup_logging()
    args = parse_args(argv)
    ensure_directory(args.output_dir)
    LOGGER.info("Starting TL09 Virus-Host DB generalized inference validation")

    raw_dir = args.output_dir / DEFAULT_RAW_DIRNAME
    host_dir = args.output_dir / DEFAULT_HOST_ASSEMBLY_DIRNAME
    phage_dir = args.output_dir / DEFAULT_PHAGE_DIRNAME
    ensure_directory(raw_dir)
    ensure_directory(host_dir)
    ensure_directory(phage_dir)

    vhdb_path = raw_dir / "virushostdb.tsv"
    if not vhdb_path.exists():
        LOGGER.info("Downloading Virus-Host DB from %s", args.vhdb_url)
        vhdb_path.write_text(_download_text(args.vhdb_url), encoding="utf-8")
    assembly_summary_path = raw_dir / "assembly_summary_refseq.txt"
    if not assembly_summary_path.exists():
        LOGGER.info("Downloading RefSeq assembly summary from %s", args.assembly_summary_url)
        assembly_summary_path.write_text(_download_text(args.assembly_summary_url), encoding="utf-8")

    _, panel_lookup = load_panel_hosts(args.panel_hosts_path)
    positive_pairs = parse_vhdb_positive_pairs(vhdb_path.read_text(encoding="utf-8"))
    cohort_summary = summarize_positive_pairs(positive_pairs)
    LOGGER.info(
        "Filtered Virus-Host DB to %d strain-level E. coli hosts, %d phage accessions, %d positive pairs",
        cohort_summary["host_count"],
        cohort_summary["phage_accession_count"],
        cohort_summary["positive_pair_count"],
    )

    assemblies_by_taxid = parse_assembly_summary(assembly_summary_path.read_text(encoding="utf-8"))
    host_candidates = build_host_candidates(
        positive_pairs,
        assemblies_by_taxid,
        panel_lookup,
    )
    novel_hosts, roundtrip_hosts = select_validation_hosts(
        host_candidates,
        novel_host_count=args.novel_host_count,
        min_positive_phages_per_host=args.min_positive_phages_per_host,
    )
    bundle_path = ensure_bundle(args.bundle_path)
    runtime = generalized_inference.load_runtime(bundle_path)
    panel_predictions_path = runtime.bundle_path.parent / str(runtime.bundle["artifacts"]["panel_predictions_filename"])
    roundtrip_hosts = filter_roundtrip_hosts_for_reference(roundtrip_hosts, panel_predictions_path)
    selected_hosts = [*novel_hosts, *roundtrip_hosts]
    host_metadata = {host.host_tax_id: host for host in selected_hosts}
    phages_by_taxid = collect_phages_for_hosts(positive_pairs, selected_hosts)

    panel_base_rate = (
        compute_panel_base_rate(args.st02_pair_table_path) if args.st02_pair_table_path.exists() else DEFAULT_BASE_RATE
    )

    host_fasta_paths = {host.host_tax_id: download_host_assembly(host, host_dir) for host in selected_hosts}

    panel_phages = read_panel_phages(args.panel_phage_metadata_path, expected_panel_count=96)
    panel_phage_paths = [args.panel_phage_dir / f"{phage}.fna" for phage in panel_phages]
    missing_panel_fastas = [str(path) for path in panel_phage_paths if not path.exists()]
    if missing_panel_fastas:
        raise FileNotFoundError(
            f"Missing panel phage FASTAs for {len(missing_panel_fastas)} phages; first missing: {missing_panel_fastas[0]}"
        )
    panel_phage_feature_rows = generalized_inference.project_phage_features(panel_phage_paths, bundle_path)
    unique_external_accessions = sorted(
        {accession for accessions in phages_by_taxid.values() for accession in accessions}
    )
    external_phage_paths = {
        accession: download_phage_fasta(accession, phage_dir) for accession in unique_external_accessions
    }
    external_phage_feature_rows = {
        row["phage"]: row
        for row in generalized_inference.project_phage_features(
            [external_phage_paths[accession] for accession in unique_external_accessions],
            bundle_path,
        )
    }

    prediction_frames: list[pd.DataFrame] = []
    roundtrip_panel_prediction_frames: list[pd.DataFrame] = []
    for host in selected_hosts:
        LOGGER.info("Starting TL09 inference for %s", host.host_name)
        host_row = generalized_inference.project_host_features(
            host_fasta_paths[host.host_tax_id],
            bundle_path,
            bacteria_id=host.panel_match or host.host_name,
        )
        candidate_phage_rows = list(panel_phage_feature_rows)
        candidate_phage_rows.extend(
            external_phage_feature_rows[accession] for accession in phages_by_taxid[host.host_tax_id]
        )
        predictions = generalized_inference.score_projected_features(host_row, candidate_phage_rows, bundle_path)
        predictions["host_tax_id"] = host.host_tax_id
        predictions["host_name"] = host.host_name
        predictions["panel_match"] = host.panel_match
        predictions["is_known_positive"] = predictions["phage"].isin(phages_by_taxid[host.host_tax_id]).astype(int)
        prediction_frames.append(predictions)
        if host.panel_match:
            roundtrip_predictions = generalized_inference.score_projected_features(
                host_row, panel_phage_feature_rows, bundle_path
            )
            roundtrip_predictions["host_tax_id"] = host.host_tax_id
            roundtrip_predictions["host_name"] = host.host_name
            roundtrip_predictions["panel_match"] = host.panel_match
            roundtrip_panel_prediction_frames.append(roundtrip_predictions)
        LOGGER.info(
            "Completed TL09 inference for %s (%d candidates, %d known positives)",
            host.host_name,
            len(predictions),
            len(phages_by_taxid[host.host_tax_id]),
        )

    positive_df, random_df, host_summary_df, overall_metrics = evaluate_positive_only_metrics(
        prediction_frames=prediction_frames,
        host_metadata=host_metadata,
        known_phages_by_taxid=phages_by_taxid,
        base_rate=panel_base_rate,
        random_seed=args.random_seed,
    )
    roundtrip_df = build_roundtrip_comparison(
        prediction_frames=roundtrip_panel_prediction_frames,
        host_metadata=host_metadata,
        panel_predictions_path=panel_predictions_path,
    )

    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    host_candidates_df = pd.DataFrame([asdict(candidate) for candidate in host_candidates]).sort_values(
        ["is_panel_host", "positive_pair_count", "host_name"],
        ascending=[True, True, True],
    )
    novel_hosts_df = pd.DataFrame([asdict(candidate) for candidate in novel_hosts])
    roundtrip_hosts_df = pd.DataFrame([asdict(candidate) for candidate in roundtrip_hosts])

    _write_dataframe(
        args.output_dir / "vhdb_filtered_positive_pairs.csv", pd.DataFrame([asdict(pair) for pair in positive_pairs])
    )
    _write_dataframe(args.output_dir / "vhdb_host_candidates.csv", host_candidates_df)
    _write_dataframe(args.output_dir / "selected_novel_hosts.csv", novel_hosts_df)
    _write_dataframe(args.output_dir / "selected_roundtrip_hosts.csv", roundtrip_hosts_df)
    _write_dataframe(args.output_dir / "all_candidate_predictions.csv", all_predictions_df)
    _write_dataframe(args.output_dir / "known_positive_prediction_rows.csv", positive_df)
    _write_dataframe(args.output_dir / "random_candidate_prediction_rows.csv", random_df)
    _write_dataframe(args.output_dir / "host_validation_summary.csv", host_summary_df)
    _write_dataframe(args.output_dir / "roundtrip_panel_comparison.csv", roundtrip_df)
    write_csv(
        args.output_dir / "overall_validation_metrics.csv",
        list(SUMMARY_FIELDNAMES),
        [{"metric": key, "value": value} for key, value in overall_metrics.items()],
    )

    manifest = {
        "task_id": "TL09",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "vhdb_url": args.vhdb_url,
            "assembly_summary_url": args.assembly_summary_url,
            "panel_hosts_path": str(args.panel_hosts_path),
            "panel_phage_dir": str(args.panel_phage_dir),
            "bundle_path": str(bundle_path),
            "st02_pair_table_path": str(args.st02_pair_table_path),
        },
        "cohort_summary": cohort_summary,
        "selection": {
            "novel_host_count_requested": args.novel_host_count,
            "novel_host_count_selected": len(novel_hosts),
            "roundtrip_host_count_selected": len(roundtrip_hosts),
            "min_positive_phages_per_host": args.min_positive_phages_per_host,
            "panel_roundtrip_hosts_requested": list(ROUNDTRIP_PANEL_HOSTS),
        },
        "metrics": overall_metrics,
        "selected_novel_hosts": [asdict(host) for host in novel_hosts],
        "selected_roundtrip_hosts": [asdict(host) for host in roundtrip_hosts],
    }
    write_json(args.output_dir / "tl09_validation_manifest.json", manifest)
    LOGGER.info("Completed TL09 Virus-Host DB generalized inference validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
