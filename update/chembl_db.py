import os

import sqlite3
import pandas as pd


def build_chembl_activity_datasets(
    chembl_db: str,
    pchembl_threshold: float = 6.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build two ChEMBL-based datasets:

    1) df_relations:
        chem_comp_id | pfam_id | uniprot_id | pchembl | mechanism | activity_comment | source='chembl'

        - One row per (compound, domain, protein).
        - Includes:
            * compounds with pChEMBL >= threshold
            * compounds with no pChEMBL but activity_comment = 'Active'
            * compounds coming only from drug_mechanism (no qualifying activity).

        - Targets restricted to SINGLE PROTEIN.

        Nota:
        - En filas provenientes de drug_mechanism (sin actividad asociada),
          activity_comment será NULL/NaN.

    2) df_smiles:
        chem_comp_id | smiles | source='chembl'

        - One row per unique compound present in df_relations.
        - SMILES taken from compound_structures.canonical_smiles.

    Parameters
    ----------
    chembl_db : str
        Path to the ChEMBL SQLite database file.
    pchembl_threshold : float, default 6.0
        Minimum pChEMBL value to consider for the activity-based subset.

    Returns
    -------
    df_relations : pd.DataFrame
    df_smiles : pd.DataFrame
    """

    # ------------------------------------------------------------------
    # Query 1: activities-based subset
    #   - pChEMBL >= threshold
    #   - OR no pChEMBL but activity_comment = 'Active'
    #   - SINGLE PROTEIN targets only
    #   - Joined to Pfam domains, UniProt IDs and SMILES
    # ------------------------------------------------------------------
    query_activities = """
        SELECT
            md.chembl_id                    AS chem_comp_id,
            csq.accession                   AS uniprot_id,
            d.source_domain_id              AS pfam_id,
            MAX(a.pchembl_value)            AS pchembl,
            MIN(dm.mechanism_of_action)     AS mechanism,
            MIN(a.activity_comment)         AS activity_comment,
            MIN(cs_struct.canonical_smiles) AS smiles
        FROM activities AS a
        JOIN assays AS ass
            ON a.assay_id = ass.assay_id
        JOIN target_dictionary AS td
            ON ass.tid = td.tid
        JOIN molecule_dictionary AS md
            ON a.molregno = md.molregno
        JOIN compound_structures AS cs_struct
            ON md.molregno = cs_struct.molregno
        JOIN target_components AS tc
            ON td.tid = tc.tid
        JOIN component_sequences AS csq
            ON tc.component_id = csq.component_id
        JOIN component_domains AS cd
            ON tc.component_id = cd.component_id
        JOIN domains AS d
            ON cd.domain_id = d.domain_id
        LEFT JOIN drug_mechanism AS dm
            ON dm.molregno = a.molregno
           AND dm.tid = td.tid
        WHERE
            (
                (a.pchembl_value IS NOT NULL AND a.pchembl_value >= ?)
                OR
                (a.pchembl_value IS NULL AND LOWER(a.activity_comment) = 'active')
            )
            AND td.target_type = 'SINGLE PROTEIN'
            AND cs_struct.canonical_smiles IS NOT NULL
        GROUP BY
            md.chembl_id,
            csq.accession,
            d.source_domain_id;
    """

    # ------------------------------------------------------------------
    # Query 2: mechanisms-only subset
    #
    # Pairs (molregno, tid) present in drug_mechanism that DO NOT have
    # any activity fulfilling the same condition used above.
    # We then attach Pfam, UniProt and SMILES in the same way.
    # ------------------------------------------------------------------
    query_mechanisms = """
        WITH valid_activity_pairs AS (
            SELECT DISTINCT
                a.molregno,
                ass.tid
            FROM activities AS a
            JOIN assays AS ass
                ON a.assay_id = ass.assay_id
            JOIN target_dictionary AS td2
                ON ass.tid = td2.tid
            WHERE
                (
                    (a.pchembl_value IS NOT NULL AND a.pchembl_value >= ?)
                    OR
                    (a.pchembl_value IS NULL AND LOWER(a.activity_comment) = 'active')
                )
                AND td2.target_type = 'SINGLE PROTEIN'
        )
        SELECT
            md.chembl_id                    AS chem_comp_id,
            csq.accession                   AS uniprot_id,
            d.source_domain_id              AS pfam_id,
            NULL                            AS pchembl,
            MIN(dm.mechanism_of_action)     AS mechanism,
            NULL                            AS activity_comment,
            MIN(cs_struct.canonical_smiles) AS smiles
        FROM drug_mechanism AS dm
        JOIN molecule_dictionary AS md
            ON dm.molregno = md.molregno
        JOIN target_dictionary AS td
            ON dm.tid = td.tid
        JOIN target_components AS tc
            ON td.tid = tc.tid
        JOIN component_sequences AS csq
            ON tc.component_id = csq.component_id
        JOIN component_domains AS cd
            ON tc.component_id = cd.component_id
        JOIN domains AS d
            ON cd.domain_id = d.domain_id
        JOIN compound_structures AS cs_struct
            ON md.molregno = cs_struct.molregno
        LEFT JOIN valid_activity_pairs AS vap
            ON vap.molregno = dm.molregno
           AND vap.tid = dm.tid
        WHERE
            td.target_type = 'SINGLE PROTEIN'
            AND cs_struct.canonical_smiles IS NOT NULL
            AND vap.molregno IS NULL   -- keep only mechanism-only pairs
        GROUP BY
            md.chembl_id,
            csq.accession,
            d.source_domain_id;
    """

    with sqlite3.connect(chembl_db) as conn:
        # Activities-based subset
        df_act = pd.read_sql_query(
            query_activities,
            conn,
            params=(pchembl_threshold,),
        )

        # Mechanisms-only subset
        df_mech = pd.read_sql_query(
            query_mechanisms,
            conn,
            params=(pchembl_threshold,),
        )

    # ------------------------------------------------------------------
    # Combine both subsets
    # ------------------------------------------------------------------
    df_all = pd.concat([df_act, df_mech], ignore_index=True)

    # Basic normalization
    df_all["chem_comp_id"] = df_all["chem_comp_id"].astype(str)
    df_all["uniprot_id"] = df_all["uniprot_id"].astype(str)
    df_all["pfam_id"] = df_all["pfam_id"].astype(str)
    df_all["source"] = "chembl"

    # Drop exact duplicates if any
    # (no incluimos activity_comment en el subset para seguir teniendo
    # una sola fila por relación, incluso si hubiera varios comentarios)
    df_all = df_all.drop_duplicates(
        subset=["chem_comp_id", "uniprot_id", "pfam_id", "pchembl", "mechanism"]
    )

    # ------------------------------------------------------------------
    # 1) Relations table
    # ------------------------------------------------------------------
    df_relations = df_all[
        [
            "chem_comp_id",
            "pfam_id",
            "uniprot_id",
            "pchembl",
            "mechanism",
            "activity_comment",
            "source",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # 2) SMILES table: one row per compound
    # ------------------------------------------------------------------
    df_smiles = (
        df_all[["chem_comp_id", "smiles"]]
        .dropna(subset=["smiles"])
        .drop_duplicates(subset=["chem_comp_id"])
        .copy()
    )
    df_smiles["source"] = "chembl"

    return df_relations, df_smiles


def generate_chembl_database(chembl_db_path: str, output_dir: str = "databases"):
    """
    Build and save the ChEMBL-derived datasets:
      - curated relations  (ligand binds a protein with a single Pfam domain)
      - possible relations (ligand binds proteins with multiple domains)
      - ligand SMILES table

    The function saves the results in `output_dir` and returns nothing.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Build base activity datasets
    df_relations, df_smiles = build_chembl_activity_datasets(chembl_db_path)

    # Step 2: Count Pfam domains per (ligand, protein)
    pfam_counts = (
        df_relations.groupby(["chem_comp_id", "uniprot_id"])["pfam_id"]
        .nunique()
        .reset_index(name="n_pfams")
    )

    # Step 3: Merge counts back into the relations table
    df_rel = df_relations.merge(pfam_counts, on=["chem_comp_id", "uniprot_id"], how="left")

    # Step 4: Split curated vs possible domain assignments
    df_relations_curated = df_rel[df_rel["n_pfams"] == 1].drop(columns=["n_pfams"])
    df_relations_possible = df_rel[df_rel["n_pfams"] > 1].drop(columns=["n_pfams"])

    # Step 5: Save to disk
    curated_path = os.path.join(output_dir, "chembl_binding_data_curated.parquet")
    possible_path = os.path.join(output_dir, "chembl_binding_data_possible.parquet")
    smiles_path = os.path.join(output_dir, "chembl_ligand_smiles.parquet")

    df_relations_curated.to_parquet(curated_path, index=False)
    df_relations_possible.to_parquet(possible_path, index=False)
    df_smiles.to_parquet(smiles_path, index=False)

    print(f"[OK] Saved curated relations to:   {curated_path}")
    print(f"[OK] Saved possible relations to:  {possible_path}")
    print(f"[OK] Saved ligand SMILES to:       {smiles_path}")