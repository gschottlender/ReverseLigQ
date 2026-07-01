from __future__ import annotations

import json
import re
import shutil
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

from rev_ligq import (
    _ensure_dataset,
    _safe_dirname,
    build_searcher,
    load_batch_csv,
    run_one_query,
)
from update.add_organisms import prepare_local_organism_data

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "databases"
COMPOUND_DIR = DATA_ROOT / "compound_data" / "pdb_chembl"
REV_DIR = DATA_ROOT / "rev_ligq"
LOCAL_ORG_DIR = DATA_ROOT / "local_organism_data"
RUNTIME_DIR = ROOT / "web" / "runtime" / "jobs"
LOGO_PATH = ROOT / "logo_reverse_ligq.png"
FRONTEND_DIST = ROOT / "web" / "frontend" / "dist"

BUILT_IN_ORGANISMS = [
    {"id": "1", "label": "Bartonella bacilliformis", "uploaded": False},
    {"id": "2", "label": "Klebsiella pneumoniae", "uploaded": False},
    {"id": "3", "label": "Mycobacterium tuberculosis", "uploaded": False},
    {"id": "4", "label": "Trypanosoma cruzi", "uploaded": False},
    {"id": "5", "label": "Staphylococcus aureus RF122", "uploaded": False},
    {"id": "6", "label": "Streptococcus uberis 0140J", "uploaded": False},
    {"id": "7", "label": "Enterococcus faecium", "uploaded": False},
    {"id": "8", "label": "Escherichia coli MG1655", "uploaded": False},
    {"id": "9", "label": "Streptococcus agalactiae NEM316", "uploaded": False},
    {"id": "10", "label": "Pseudomonas syringae", "uploaded": False},
    {"id": "11", "label": "DENV (Dengue virus)", "uploaded": False},
    {"id": "12", "label": "SARS-CoV-2", "uploaded": False},
    {"id": "13", "label": "Homo sapiens", "uploaded": False},
]

JobStatus = Literal["queued", "running", "succeeded", "failed"]

app = FastAPI(title="ReverseLigQ Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


class MoleculeRenderRequest(BaseModel):
    smiles: str
    width: int = 260
    height: int = 190


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_dir(job_id: str) -> Path:
    return RUNTIME_DIR / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.astype(object).where(pd.notna(df), None)
    return cleaned.to_dict(orient="records")


def _write_job(job: dict[str, Any]) -> None:
    job_dir = _job_dir(job["id"])
    job_dir.mkdir(parents=True, exist_ok=True)
    _status_path(job["id"]).write_text(json.dumps(job, indent=2), encoding="utf-8")
    with _jobs_lock:
        _jobs[job["id"]] = job


def _update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    with _jobs_lock:
        job = dict(_jobs.get(job_id) or _load_job(job_id))
    job.update(changes)
    _write_job(job)
    return job


def _load_job(job_id: str) -> dict[str, Any]:
    path = _status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _create_job(kind: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "message": "Queued",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "metadata": metadata or {},
    }
    _write_job(job)
    return job


def _safe_upload_name(filename: str | None, fallback: str) -> str:
    name = filename or fallback
    name = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or fallback


def molecule_svg(smiles: str, width: int = 260, height: int = 190) -> str:
    mol = Chem.MolFromSmiles((smiles or "").strip())
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.rdDepictor.Compute2DCoords(mol)
    return str(
        Draw.MolsToGridImage(
            [mol],
            molsPerRow=1,
            subImgSize=(int(width), int(height)),
            useSVG=True,
        )
    )


def _try_svg(smiles: Any, width: int = 220, height: int = 160) -> str | None:
    if smiles is None:
        return None
    try:
        return molecule_svg(str(smiles), width=width, height=height)
    except Exception:
        return None


def _uploaded_organisms() -> list[dict[str, Any]]:
    if not LOCAL_ORG_DIR.exists():
        return []
    rows = []
    for path in sorted(LOCAL_ORG_DIR.iterdir()):
        if path.is_dir() and (path / "ligand_lists.pkl").exists() and (path / "fam_prot_dict.pkl").exists():
            rows.append({"id": path.name, "label": path.name, "uploaded": True})
    return rows


def _resolve_protein_base(organism: str, uploaded: bool) -> tuple[Path, Path]:
    if not uploaded:
        return REV_DIR, REV_DIR
    proteins_base_dir = LOCAL_ORG_DIR / organism
    if not proteins_base_dir.exists():
        raise FileNotFoundError(f"Uploaded organism directory not found: {proteins_base_dir}")
    return proteins_base_dir, proteins_base_dir


def _run_search_job(
    job_id: str,
    organism: str,
    uploaded_organism: bool,
    search_type: str,
    min_score: float,
    max_domain_ranks: int | None,
    query_smiles: str | None,
    query_csv_path: str | None,
) -> None:
    try:
        _update_job(job_id, status="running", started_at=_now(), message="Preparing ReverseLigQ dataset")
        job_dir = _job_dir(job_id)
        results_dir = job_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        _ensure_dataset(compound_dir=COMPOUND_DIR, rev_dir=REV_DIR)
        proteins_base_dir, searcher_rev_dir = _resolve_protein_base(organism, uploaded_organism)

        _update_job(job_id, message="Loading ligand representations")
        searcher = build_searcher(
            organism=organism,
            compound_dir=COMPOUND_DIR,
            rev_dir=searcher_rev_dir,
            search_type=search_type,
            min_score=min_score,
            k_max_ligands=1000,
            chunk_size=50_000,
            chemberta_model="seyonec/ChemBERTa-zinc-base-v1",
            device=None,
            max_length=256,
        )

        query_rows: list[dict[str, str]]
        if query_csv_path:
            batch_df = load_batch_csv(query_csv_path)
            query_rows = [
                {"ligand_id": str(row["lig_id"]), "smiles": str(row["smiles"])}
                for _, row in batch_df.iterrows()
            ]
        else:
            query_rows = [{"ligand_id": "query", "smiles": query_smiles or ""}]

        result_payload = []
        total = len(query_rows)
        for idx, row in enumerate(query_rows, start=1):
            ligand_id = row["ligand_id"]
            smiles = row["smiles"]
            safe_id = _safe_dirname(ligand_id, fallback=f"query_{idx}")
            q_dir = results_dir / safe_id
            q_dir.mkdir(parents=True, exist_ok=True)
            _update_job(job_id, message=f"Running target search {idx}/{total}: {ligand_id}")

            target_df, lig_df = run_one_query(
                searcher=searcher,
                query_smiles=smiles,
                organism=organism,
                rev_dir=REV_DIR,
                proteins_base_dir=proteins_base_dir,
                max_domain_ranks=max_domain_ranks,
                include_only_curated=False,
                only_proteins_with_description=False,
            )

            target_df.to_csv(q_dir / "predicted_targets.csv", index=False)
            lig_df.to_csv(q_dir / "similarity_search_results.csv", index=False)

            target_records = _records(target_df)
            similarity_records = _records(lig_df)
            for item in similarity_records:
                item["structure_svg"] = _try_svg(item.get("smiles"))

            result_payload.append(
                {
                    "ligand_id": ligand_id,
                    "safe_id": safe_id,
                    "smiles": smiles,
                    "query_svg": _try_svg(smiles, width=260, height=190),
                    "predicted_targets": target_records,
                    "similarity_search_results": similarity_records,
                }
            )

        (job_dir / "results.json").write_text(
            json.dumps({"queries": result_payload}, indent=2),
            encoding="utf-8",
        )
        _update_job(job_id, status="succeeded", finished_at=_now(), message="Search completed")
    except Exception as exc:
        (_job_dir(job_id) / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _update_job(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc),
            message="Search failed",
        )


def _run_proteome_job(job_id: str, org_name: str, fasta_path: str, cpu: int) -> None:
    try:
        _update_job(job_id, status="running", started_at=_now(), message="Scanning proteome with Pfam")
        paths = prepare_local_organism_data(
            org_name=org_name,
            fasta_path=Path(fasta_path),
            data_dir=DATA_ROOT,
            temp_base_dir=ROOT / "temp_data" / "new_proteomes",
            cpu=int(cpu),
        )
        payload = {
            "organism": org_name,
            "paths": {key: str(path) for key, path in paths.items()},
        }
        (_job_dir(job_id) / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _update_job(job_id, status="succeeded", finished_at=_now(), message="Proteome uploaded")
    except Exception as exc:
        (_job_dir(job_id) / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        _update_job(
            job_id,
            status="failed",
            finished_at=_now(),
            error=str(exc),
            message="Proteome upload failed",
        )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/assets/logo")
def logo() -> FileResponse:
    if not LOGO_PATH.exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    return FileResponse(LOGO_PATH)


@app.get("/api/organisms")
def organisms() -> dict[str, list[dict[str, Any]]]:
    return {"organisms": BUILT_IN_ORGANISMS + _uploaded_organisms()}


@app.post("/api/molecules/render")
def render_molecule(request: MoleculeRenderRequest) -> dict[str, Any]:
    try:
        return {"valid": True, "svg": molecule_svg(request.smiles, request.width, request.height)}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


@app.post("/api/search-jobs")
async def create_search_job(
    background_tasks: BackgroundTasks,
    organism: str = Form(...),
    uploaded_organism: bool = Form(False),
    search_type: str = Form("morgan_fp_tanimoto"),
    min_score: float = Form(...),
    max_domain_ranks: str = Form("20"),
    query_smiles: str | None = Form(None),
    query_csv: UploadFile | None = File(None),
) -> dict[str, str]:
    if search_type not in {"morgan_fp_tanimoto", "chemberta"}:
        raise HTTPException(status_code=400, detail="Unsupported search type")
    if bool(query_smiles and query_smiles.strip()) == bool(query_csv):
        raise HTTPException(status_code=400, detail="Provide either a SMILES string or a CSV file")

    parsed_max_domain_ranks = None if max_domain_ranks.lower() in {"none", "all", ""} else int(max_domain_ranks)
    if parsed_max_domain_ranks is not None and parsed_max_domain_ranks < 1:
        raise HTTPException(status_code=400, detail="Maximum domain ranks must be positive")

    job = _create_job(
        "search",
        {
            "organism": organism,
            "uploaded_organism": uploaded_organism,
            "search_type": search_type,
            "min_score": min_score,
            "max_domain_ranks": parsed_max_domain_ranks,
        },
    )
    job_dir = _job_dir(job["id"])

    csv_path: str | None = None
    if query_csv is not None:
        uploads_dir = job_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        csv_file = uploads_dir / _safe_upload_name(query_csv.filename, "queries.csv")
        with csv_file.open("wb") as fh:
            shutil.copyfileobj(query_csv.file, fh)
        csv_path = str(csv_file)

    background_tasks.add_task(
        _run_search_job,
        job["id"],
        organism,
        uploaded_organism,
        search_type,
        float(min_score),
        parsed_max_domain_ranks,
        query_smiles.strip() if query_smiles else None,
        csv_path,
    )
    return {"job_id": job["id"]}


@app.post("/api/proteome-jobs")
async def create_proteome_job(
    background_tasks: BackgroundTasks,
    org_name: str = Form(...),
    cpu: int = Form(4),
    fasta: UploadFile = File(...),
) -> dict[str, str]:
    clean_org = re.sub(r"[^A-Za-z0-9_]+", "_", org_name.strip()).strip("_")
    if not clean_org:
        raise HTTPException(status_code=400, detail="Organism name must contain letters, numbers, or underscores")
    if cpu < 1:
        raise HTTPException(status_code=400, detail="CPU count must be positive")

    job = _create_job("proteome", {"organism": clean_org, "cpu": int(cpu)})
    uploads_dir = _job_dir(job["id"]) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = uploads_dir / _safe_upload_name(fasta.filename, f"{clean_org}.fasta")
    with fasta_path.open("wb") as fh:
        shutil.copyfileobj(fasta.file, fh)

    background_tasks.add_task(_run_proteome_job, job["id"], clean_org, str(fasta_path), int(cpu))
    return {"job_id": job["id"]}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id]
    return _load_job(job_id)


@app.get("/api/jobs/{job_id}/results")
def job_results(job_id: str) -> dict[str, Any]:
    job = job_status(job_id)
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail="Job has not succeeded")
    results_path = _job_dir(job_id) / "results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    return json.loads(results_path.read_text(encoding="utf-8"))


if FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}")
    def frontend_spa(path: str) -> FileResponse:
        candidate = FRONTEND_DIST / path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
