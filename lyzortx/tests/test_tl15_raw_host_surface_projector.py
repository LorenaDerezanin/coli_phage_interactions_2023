from lyzortx.pipeline.track_l.steps.build_raw_host_surface_projector import (
    DIRECT,
    NOT_CALLABLE,
    PRESENT,
    PROXY,
    CapsuleCall,
    CapsuleModel,
    build_lps_proxy_lookup,
    build_o_antigen_reference_contract,
    build_projected_feature_row,
    choose_capsule_call,
    load_o_antigen_override_references,
    parse_nhmmer_tblout,
)


def test_build_o_antigen_reference_contract_keeps_only_o_antigen_gene_queries() -> None:
    output_rows = [
        {
            "O-type": "O157",
            "AlleleKeys": "O157-1-wzx-origin;O157-118-wzy;",
        },
        {
            "O-type": "Oneg",
            "AlleleKeys": "H7-1-fliC-origin;",
        },
    ]
    allele_rows = [
        {
            "name": "O157-1-wzx-origin",
            "gene": "wzx",
            "sseq": "ATGC",
            "antigen": "O157",
            "type": "O",
        },
        {
            "name": "O157-118-wzy",
            "gene": "wzy",
            "sseq": "ATGG",
            "antigen": "O157",
            "type": "O",
        },
        {
            "name": "H7-1-fliC-origin",
            "gene": "fliC",
            "sseq": "ATGA",
            "antigen": "H7",
            "type": "H",
        },
    ]

    references, contract = build_o_antigen_reference_contract(
        o_type_output_rows=output_rows,
        o_type_allele_rows=allele_rows,
    )

    assert [reference.query_id for reference in references] == [
        "O157__wzx__O157-1-wzx-origin",
        "O157__wzy__O157-118-wzy",
    ]
    assert contract == {
        "O157": {
            "wzx": ("O157__wzx__O157-1-wzx-origin",),
            "wzy": ("O157__wzy__O157-118-wzy",),
        }
    }


def test_build_lps_proxy_lookup_leaves_ambiguous_o_types_untyped() -> None:
    rows = [
        {"O-type": "O1", "LPS_type": "R1"},
        {"O-type": "O1", "LPS_type": "R1"},
        {"O-type": "O2", "LPS_type": "R3"},
        {"O-type": "O2", "LPS_type": "R4"},
    ]

    lookup = build_lps_proxy_lookup(rows)

    assert lookup["O1"]["proxy_type"] == "R1"
    assert lookup["O2"]["proxy_type"] == ""
    assert lookup["O2"]["type_counts"] == {"R3": 1, "R4": 1}


def test_load_o_antigen_override_references_extracts_fasta_coordinates(tmp_path) -> None:
    fasta_path = tmp_path / "host.fasta"
    fasta_path.write_text(">contig1\nAACCGGTTAACC\n", encoding="utf-8")
    override_path = tmp_path / "overrides.tsv"
    override_path.write_text(
        "\n".join(
            [
                "allele_key\to_type\tgene_family\tsource_fasta_path\trecord_name\tstart\tend\tstrand",
                f"O157-1-wzx-origin\tO157\twzx\t{fasta_path}\tcontig1\t3\t8\t+",
                f"O157-118-wzy\tO157\twzy\t{fasta_path}\tcontig1\t9\t4\t-",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    overrides = load_o_antigen_override_references(override_path)

    assert overrides["O157-1-wzx-origin"] == ("CCGGTT", "wzx", "O157")
    assert overrides["O157-118-wzy"] == ("TAACCG", "wzy", "O157")


def test_choose_capsule_call_prefers_model_with_more_mandatory_support() -> None:
    models = [
        CapsuleModel(
            model_id="K1",
            mandatory_genes=frozenset({"NeuA", "NeuB", "NeuE"}),
            all_genes=frozenset({"NeuA", "NeuB", "NeuE", "KpsE"}),
            min_mandatory_genes_required=2,
            min_genes_required=3,
            inter_gene_max_space=5,
        ),
        CapsuleModel(
            model_id="K5",
            mandatory_genes=frozenset({"KfiA", "KfiB", "KfiC"}),
            all_genes=frozenset({"KfiA", "KfiB", "KfiC", "KpsE"}),
            min_mandatory_genes_required=3,
            min_genes_required=4,
            inter_gene_max_space=5,
        ),
    ]
    hits = [
        type("Hit", (), {"target_name": "NeuA", "query_name": "gene_1", "score": 100.0, "evalue": 0.0}),
        type("Hit", (), {"target_name": "NeuB", "query_name": "gene_2", "score": 90.0, "evalue": 0.0}),
        type("Hit", (), {"target_name": "NeuE", "query_name": "gene_3", "score": 80.0, "evalue": 0.0}),
        type("Hit", (), {"target_name": "KpsE", "query_name": "gene_4", "score": 70.0, "evalue": 0.0}),
    ]
    protein_metadata = {
        "gene_1": {"contig": "c1", "order": 1},
        "gene_2": {"contig": "c1", "order": 2},
        "gene_3": {"contig": "c1", "order": 3},
        "gene_4": {"contig": "c1", "order": 4},
    }

    call = choose_capsule_call(models=models, hits=hits, protein_metadata=protein_metadata)

    assert call.capsule_type == "K1"
    assert call.projection_mode == DIRECT
    assert set(call.matched_genes) == {"KpsE", "NeuA", "NeuB", "NeuE"}


def test_build_projected_feature_row_keeps_not_callable_distinct_from_absent() -> None:
    row, status_rows = build_projected_feature_row(
        bacteria="LF82",
        o_antigen_type="O83",
        capsule_call=CapsuleCall(
            capsule_type="K5",
            matched_genes=("KfiA", "KfiB", "KfiC"),
            core_profile_hits=("KpsE",),
            evidence="model=K5",
            projection_mode=DIRECT,
        ),
        lps_lookup={"O83": {"proxy_type": "R1", "support_fraction": 1.0, "type_counts": {"R1": 5}}},
        receptor_presence_calls={"BTUB": (PRESENT, "best_hit=gene_1")},
    )

    assert row["host_o_antigen_present"] == 1
    assert row["host_k_antigen_type"] == "K5"
    assert row["host_lps_core_type"] == "R1"
    assert row["host_receptor_btub_present"] == 1
    assert row["host_receptor_fadL_present"] == ""

    status_by_column = {status["column_name"]: status for status in status_rows}
    assert status_by_column["host_receptor_fadL_present"]["call_state"] == NOT_CALLABLE
    assert status_by_column["host_receptor_btub_variant"]["projection_mode"] == "unsupported"
    assert status_by_column["host_surface_klebsiella_capsule_type"]["projection_mode"] == "unsupported"


def test_build_projected_feature_row_marks_proxy_only_capsule_signal() -> None:
    row, status_rows = build_projected_feature_row(
        bacteria="B1",
        o_antigen_type="",
        capsule_call=CapsuleCall(
            capsule_type="",
            matched_genes=(),
            core_profile_hits=("KpsE", "KpsM"),
            evidence="core_profiles=KpsE|KpsM",
            projection_mode=PROXY,
        ),
        lps_lookup={},
        receptor_presence_calls={},
    )

    assert row["host_k_antigen_proxy_present"] == 1
    assert row["host_k_antigen_present"] == ""
    status_by_column = {status["column_name"]: status for status in status_rows}
    assert status_by_column["host_k_antigen_proxy_present"]["projection_mode"] == PROXY
    assert status_by_column["host_lps_core_type"]["call_state"] == NOT_CALLABLE


def test_parse_nhmmer_tblout_reads_nhmmer_evalue_and_score_columns(tmp_path) -> None:
    tblout_path = tmp_path / "sample.tbl"
    tblout_path.write_text(
        "\n".join(
            [
                "# target name accession query name accession hmmfrom hmm to alifrom ali to envfrom env to sq len strand E-value score bias description of target",
                "NODE_1 - O83__wzx__O83-1-wzx-origin - 2 1442 75187 73747 75188 73746 194951 - 0 1397.3 136.5 -",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hits = parse_nhmmer_tblout(tblout_path)

    assert len(hits) == 1
    assert hits[0].query_name == "O83__wzx__O83-1-wzx-origin"
    assert hits[0].evalue == 0.0
    assert hits[0].score == 1397.3
