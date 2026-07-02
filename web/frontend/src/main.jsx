import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronDown,
  Dna,
  Download,
  FileSpreadsheet,
  FlaskConical,
  Info,
  Loader2,
  Play,
  Search,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_BASE = "";

const SEARCH_TYPES = {
  morgan_fp_tanimoto: {
    label: "ECFP4 + Tanimoto",
    threshold: 0.4,
  },
  chemberta: {
    label: "ChemBERTa + Cosine",
    threshold: 0.85,
  },
};

const PREDICTED_TARGET_TOOLTIPS = {
  domain_id: "Pfam protein domain associated with the candidate target. ReverseLigQ uses this domain as the evidence bridge between similar known ligands and proteins in the selected organism.",
  domain_tag: "Evidence type for the ligand-domain link. Curated means experimentally supported domain evidence; possible means the reference ligand binds a multi-domain protein but the exact binding domain is not resolved.",
  reference_ligand_id: "Known ligand from the ReverseLigQ reference data that is chemically similar to your query and connects this prediction to the reported domain evidence.",
  reference_ligand_score: "Similarity score between your query ligand and the reference ligand under the selected search method. Higher values indicate closer chemical similarity.",
};

const FORM_TOOLTIPS = {
  searchMethod: "Choose how ReverseLigQ compares your query ligand against known ligands. ECFP4 + Tanimoto uses molecular fingerprints and is fast; ChemBERTa + Cosine uses learned SMILES embeddings and can capture broader chemical similarity.",
  similarityThreshold: "Minimum similarity score required for a reference ligand to be considered. Higher values make the search stricter; lower values may return more candidates but can include weaker matches.",
  maxDomainRanks: "Limits how many ranked Pfam domain candidates are kept when building the predicted target table. Lower values keep the strongest domain evidence; use Include all domain ranks for a broader, larger result table.",
  cpuThreads: "Number of CPU threads HMMER uses while scanning the uploaded proteome against Pfam. More threads can speed up large proteomes, but should not exceed the cores available on the machine.",
};

function cx(...items) {
  return items.filter(Boolean).join(" ");
}

async function apiJson(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function usePolling(jobId, onDone) {
  const [job, setJob] = useState(null);
  const onDoneRef = useRef(onDone);

  useEffect(() => {
    onDoneRef.current = onDone;
  }, [onDone]);

  useEffect(() => {
    if (!jobId) return undefined;
    let stopped = false;

    async function tick() {
      try {
        const next = await apiJson(`/api/jobs/${jobId}`);
        if (stopped) return;
        setJob(next);
        if (next.status === "succeeded" || next.status === "failed") {
          onDoneRef.current?.(next);
          return;
        }
        window.setTimeout(tick, 1800);
      } catch (error) {
        if (!stopped) setJob({ status: "failed", error: error.message, message: "Unable to read job status" });
      }
    }

    tick();
    return () => {
      stopped = true;
    };
  }, [jobId]);

  return job;
}

function App() {
  const [screen, setScreen] = useState("search");
  const [organisms, setOrganisms] = useState([]);

  async function refreshOrganisms() {
    const payload = await apiJson("/api/organisms");
    setOrganisms(payload.organisms);
  }

  useEffect(() => {
    refreshOrganisms().catch(() => setOrganisms([]));
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <img src="/api/assets/logo" alt="ReverseLigQ logo" />
        </div>
        <nav className="nav">
          <button className={cx(screen === "search" && "active")} onClick={() => setScreen("search")} aria-label="Target Search">
            <Search size={18} />
            <span className="nav-label">Target Search</span>
          </button>
          <button className={cx(screen === "upload" && "active")} onClick={() => setScreen("upload")} aria-label="Proteome Upload">
            <Upload size={18} />
            <span className="nav-label">Proteome Upload</span>
          </button>
        </nav>
      </aside>

      <main className="main">
        {screen === "search" ? (
          <SearchScreen organisms={organisms} refreshOrganisms={refreshOrganisms} />
        ) : (
          <ProteomeUploadScreen refreshOrganisms={refreshOrganisms} />
        )}
        <AppFooter />
      </main>
    </div>
  );
}

function AppFooter() {
  return (
    <footer className="app-footer">
      <a href="https://github.com/gschottlender/ReverseLigQ" target="_blank" rel="noreferrer">
        github.com/gschottlender/ReverseLigQ
      </a>
      <span>
        Schottlender et al. (2022), Frontiers in Drug Discovery ·{" "}
        <a href="https://doi.org/10.3389/fddsv.2022.969983" target="_blank" rel="noreferrer">
          doi:10.3389/fddsv.2022.969983
        </a>
      </span>
    </footer>
  );
}

function SearchScreen({ organisms }) {
  const defaultOrganism = organisms.find((org) => org.id === "13") || organisms[0];
  const [organismId, setOrganismId] = useState(defaultOrganism?.id || "13");
  const [queryMode, setQueryMode] = useState("single");
  const [smiles, setSmiles] = useState("CC(=O)OC1=CC=CC=C1C(=O)O");
  const [csvFile, setCsvFile] = useState(null);
  const [batchRows, setBatchRows] = useState([]);
  const [searchType, setSearchType] = useState("morgan_fp_tanimoto");
  const [minScore, setMinScore] = useState(SEARCH_TYPES.morgan_fp_tanimoto.threshold);
  const [allDomains, setAllDomains] = useState(false);
  const [maxDomainRanks, setMaxDomainRanks] = useState(20);
  const [moleculePreview, setMoleculePreview] = useState({ state: "idle" });
  const [jobId, setJobId] = useState(null);
  const [results, setResults] = useState(null);
  const [selectedResult, setSelectedResult] = useState(0);
  const [activeTab, setActiveTab] = useState("targets");
  const [submitError, setSubmitError] = useState(null);

  const selectedOrganism = organisms.find((org) => org.id === organismId);
  const job = usePolling(jobId, async (doneJob) => {
    if (doneJob.status === "succeeded") {
      const payload = await apiJson(`/api/jobs/${doneJob.id}/results`);
      const queries = payload.queries || [];
      const firstSuccessful = queries.findIndex((result) => !resultHasError(result));
      setResults(queries);
      setSelectedResult(firstSuccessful === -1 ? 0 : firstSuccessful);
      setActiveTab("targets");
    }
  });

  useEffect(() => {
    const next = organisms.find((org) => org.id === "13") || organisms[0];
    if (next && !organisms.some((org) => org.id === organismId)) {
      setOrganismId(next.id);
    }
  }, [organisms, organismId]);

  useEffect(() => {
    setMinScore(SEARCH_TYPES[searchType].threshold);
  }, [searchType]);

  useEffect(() => {
    if (!smiles.trim()) {
      setMoleculePreview({ state: "idle" });
      return undefined;
    }
    const timer = window.setTimeout(async () => {
      setMoleculePreview({ state: "loading" });
      try {
        const payload = await apiJson("/api/molecules/render", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ smiles, width: 300, height: 210 }),
        });
        setMoleculePreview(payload.valid ? { state: "valid", svg: payload.svg } : { state: "invalid", error: payload.error });
      } catch (error) {
        setMoleculePreview({ state: "invalid", error: error.message });
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [smiles]);

  async function handleCsv(file) {
    setCsvFile(file);
    setBatchRows([]);
    if (!file) return;
    const text = await file.text();
    const rows = parseCsvPreview(text).slice(0, 50);
    setBatchRows(rows.map((row) => ({ ...row, preview: { state: "loading" } })));
    const rendered = await Promise.all(
      rows.map(async (row) => {
        try {
          const payload = await apiJson("/api/molecules/render", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ smiles: row.smiles, width: 160, height: 120 }),
          });
          return { ...row, preview: payload.valid ? { state: "valid", svg: payload.svg } : { state: "invalid" } };
        } catch {
          return { ...row, preview: { state: "invalid" } };
        }
      })
    );
    setBatchRows(rendered);
  }

  async function submitSearch(event) {
    event.preventDefault();
    setSubmitError(null);
    setResults(null);

    if (queryMode === "single" && moleculePreview.state === "invalid") {
      setSubmitError("The SMILES string is invalid.");
      return;
    }
    if (queryMode === "batch" && !csvFile) {
      setSubmitError("Upload a CSV file with lig_id and smiles columns.");
      return;
    }

    const form = new FormData();
    form.append("organism", organismId);
    form.append("uploaded_organism", String(Boolean(selectedOrganism?.uploaded)));
    form.append("search_type", searchType);
    form.append("min_score", String(minScore));
    form.append("max_domain_ranks", allDomains ? "none" : String(maxDomainRanks));
    if (queryMode === "single") {
      form.append("query_smiles", smiles);
    } else {
      form.append("query_csv", csvFile);
    }

    try {
      const payload = await apiJson("/api/search-jobs", { method: "POST", body: form });
      setJobId(payload.job_id);
    } catch (error) {
      setSubmitError(error.message);
    }
  }

  const currentResult = results?.[selectedResult];

  return (
    <section className="workspace">
      <header className="page-header">
        <div>
          <span className="eyebrow">Target Search</span>
          <h1>Ligand-driven target discovery</h1>
        </div>
        <StatusPill job={job} />
      </header>

      <form className="search-layout" onSubmit={submitSearch}>
        <div className="panel controls-panel">
          <Field label="Organism">
            <div className="select-wrap">
              <select value={organismId} onChange={(event) => setOrganismId(event.target.value)}>
                <optgroup label="Preloaded organisms">
                  {organisms.filter((org) => !org.uploaded).map((org) => (
                    <option key={org.id} value={org.id}>{org.id}. {org.label}</option>
                  ))}
                </optgroup>
                {organisms.some((org) => org.uploaded) && (
                  <optgroup label="Uploaded organisms">
                    {organisms.filter((org) => org.uploaded).map((org) => (
                      <option key={org.id} value={org.id}>{org.label}</option>
                    ))}
                  </optgroup>
                )}
              </select>
              <ChevronDown size={18} />
            </div>
          </Field>

          <div className="segmented">
            <button type="button" className={cx(queryMode === "single" && "selected")} onClick={() => setQueryMode("single")}>
              <FlaskConical size={16} /> Single SMILES
            </button>
            <button type="button" className={cx(queryMode === "batch" && "selected")} onClick={() => setQueryMode("batch")}>
              <FileSpreadsheet size={16} /> Batch CSV
            </button>
          </div>

          {queryMode === "single" ? (
            <Field label="Compound SMILES">
              <textarea value={smiles} onChange={(event) => setSmiles(event.target.value)} rows={4} />
            </Field>
          ) : (
            <Field label="Batch CSV">
              <input type="file" accept=".csv,text/csv" onChange={(event) => handleCsv(event.target.files?.[0] || null)} />
            </Field>
          )}

          <Field label="Similarity method" tooltip={FORM_TOOLTIPS.searchMethod}>
            <div className="segmented">
              {Object.entries(SEARCH_TYPES).map(([key, item]) => (
                <button type="button" key={key} className={cx(searchType === key && "selected")} onClick={() => setSearchType(key)}>
                  {item.label}
                </button>
              ))}
            </div>
          </Field>

          <div className="control-grid">
            <Field label="Similarity threshold" tooltip={FORM_TOOLTIPS.similarityThreshold}>
              <input type="number" min="0" max="1" step="0.01" value={minScore} onChange={(event) => setMinScore(event.target.value)} />
            </Field>
            <Field label="Maximum domain ranks" tooltip={FORM_TOOLTIPS.maxDomainRanks}>
              <input type="number" min="1" step="1" value={maxDomainRanks} disabled={allDomains} onChange={(event) => setMaxDomainRanks(event.target.value)} />
            </Field>
          </div>

          <label className="check-row">
            <input type="checkbox" checked={allDomains} onChange={(event) => setAllDomains(event.target.checked)} />
            Include all domain ranks
          </label>

          {submitError && <Alert text={submitError} tone="error" />}
          {job?.status === "failed" && <Alert text={job.error || "Search failed"} tone="error" />}

          <button className="primary-action" type="submit" disabled={job?.status === "running" || job?.status === "queued"}>
            {job?.status === "running" || job?.status === "queued" ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Run Target Search
          </button>
        </div>

        <div className="panel preview-panel">
          {queryMode === "single" ? (
            <MoleculePreview preview={moleculePreview} />
          ) : (
            <BatchPreview rows={batchRows} />
          )}
        </div>
      </form>

      {results && (
        <ResultsPanel
          results={results}
          currentResult={currentResult}
          selectedResult={selectedResult}
          setSelectedResult={setSelectedResult}
          activeTab={activeTab}
          setActiveTab={setActiveTab}
        />
      )}
    </section>
  );
}

function ProteomeUploadScreen({ refreshOrganisms }) {
  const [orgName, setOrgName] = useState("");
  const [fasta, setFasta] = useState(null);
  const [cpu, setCpu] = useState(4);
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);
  const job = usePolling(jobId, async (doneJob) => {
    if (doneJob.status === "succeeded") {
      const payload = await apiJson(`/api/jobs/${doneJob.id}/results`);
      setResult(payload);
      refreshOrganisms();
    }
  });

  async function submitProteome(event) {
    event.preventDefault();
    setError(null);
    setResult(null);
    if (!orgName.trim() || !fasta) {
      setError("Provide an organism name and a FASTA file.");
      return;
    }
    const form = new FormData();
    form.append("org_name", orgName);
    form.append("cpu", String(cpu));
    form.append("fasta", fasta);
    try {
      const payload = await apiJson("/api/proteome-jobs", { method: "POST", body: form });
      setJobId(payload.job_id);
    } catch (submitError) {
      setError(submitError.message);
    }
  }

  return (
    <section className="workspace upload-workspace">
      <header className="page-header">
        <div>
          <span className="eyebrow">Proteome Upload</span>
          <h1>Add a searchable organism</h1>
        </div>
        <StatusPill job={job} />
      </header>

      <form className="upload-layout panel" onSubmit={submitProteome}>
        <div className="upload-copy">
          <Dna size={34} />
          <h2>Upload FASTA proteome</h2>
        </div>
        <Field label="Organism name">
          <input value={orgName} onChange={(event) => setOrgName(event.target.value)} placeholder="siniae" />
        </Field>
        <Field label="FASTA file">
          <input ref={fileRef} type="file" accept=".fasta,.fa,.faa,.txt" onChange={(event) => setFasta(event.target.files?.[0] || null)} />
        </Field>
        <Field label="CPU threads" tooltip={FORM_TOOLTIPS.cpuThreads}>
          <input type="number" min="1" step="1" value={cpu} onChange={(event) => setCpu(event.target.value)} />
        </Field>
        {error && <Alert text={error} tone="error" />}
        {job?.status === "failed" && <Alert text={job.error || "Upload failed"} tone="error" />}
        {result && <Alert text={`${result.organism} is now available for target searches.`} tone="success" />}
        <button className="primary-action" type="submit" disabled={job?.status === "running" || job?.status === "queued"}>
          {job?.status === "running" || job?.status === "queued" ? <Loader2 className="spin" size={18} /> : <Upload size={18} />}
          Upload Proteome
        </button>
      </form>
    </section>
  );
}

function Field({ label, tooltip, children }) {
  return (
    <label className="field">
      <span className="field-label">
        <span>{label}</span>
        {tooltip && <HelpTooltip label={label} tooltip={tooltip} />}
      </span>
      {children}
    </label>
  );
}

function Alert({ text, tone }) {
  const Icon = tone === "success" ? CheckCircle2 : AlertCircle;
  return (
    <div className={cx("alert", tone)}>
      <Icon size={17} />
      <span>{text}</span>
    </div>
  );
}

function StatusPill({ job }) {
  if (!job) return <span className="status-pill idle"><Activity size={15} /> Ready</span>;
  return (
    <span className={cx("status-pill", job.status)}>
      {job.status === "running" || job.status === "queued" ? <Loader2 className="spin" size={15} /> : <Activity size={15} />}
      {job.message || job.status}
    </span>
  );
}

function MoleculePreview({ preview }) {
  return (
    <div className="molecule-preview">
      <div className="preview-heading">
        <span>Structure Preview</span>
      </div>
      <div className="structure-box">
        {preview.state === "loading" && <Loader2 className="spin" size={24} />}
        {preview.state === "valid" && <div dangerouslySetInnerHTML={{ __html: preview.svg }} />}
        {preview.state === "invalid" && <Alert text={preview.error || "Invalid SMILES"} tone="error" />}
        {preview.state === "idle" && <span className="muted">Enter a SMILES string</span>}
      </div>
    </div>
  );
}

function BatchPreview({ rows }) {
  return (
    <div className="batch-preview">
      <div className="preview-heading">
        <span>Batch Ligands</span>
        <small>{rows.length ? `${rows.length} previewed` : "CSV requires lig_id, smiles"}</small>
      </div>
      <div className="batch-list">
        {rows.length === 0 && <span className="muted">Upload a CSV to preview ligands</span>}
        {rows.map((row) => (
          <div className="batch-row" key={`${row.lig_id}-${row.smiles}`}>
            <div className="batch-structure">
              {row.preview?.state === "valid" && <div dangerouslySetInnerHTML={{ __html: row.preview.svg }} />}
              {row.preview?.state === "invalid" && <span>Invalid SMILES</span>}
              {row.preview?.state !== "valid" && row.preview?.state !== "invalid" && <span>No structure</span>}
            </div>
            <div>
              <strong>{row.lig_id}</strong>
              <span>{row.smiles}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultsPanel({ results, currentResult, selectedResult, setSelectedResult, activeTab, setActiveTab }) {
  const currentFailed = resultHasError(currentResult);
  const failedResults = results.filter(resultHasError);
  const allResultsFailed = failedResults.length > 0 && failedResults.length === results.length;
  const rows = activeTab === "targets" ? currentResult?.predicted_targets || [] : currentResult?.similarity_search_results || [];
  const columns = activeTab === "targets"
    ? ["rank", "protein_id", "protein_description", "domain_id", "domain_tag", "reference_ligand_id", "reference_ligand_score"]
    : ["structure_svg", "rank", "comp_id", "score", "curated_domains", "possible_domains", "domain_summary"];
  const targetExportColumns = ["rank", "protein_id", "protein_description", "domain_id", "domain_tag", "reference_ligand_id", "reference_ligand_score", "reference_ligand_smiles"];
  const similarityExportColumns = ["rank", "comp_id", "score", "smiles", "curated_domains", "possible_domains", "domain_summary"];

  return (
    <section className="results-panel panel">
      <div className="results-header">
        <div>
          <span className="eyebrow">Results</span>
          <h2>{currentResult?.ligand_id || "Query"}</h2>
        </div>
        {results.length > 1 && (
          <div className="select-wrap compact">
            <select value={selectedResult} onChange={(event) => setSelectedResult(Number(event.target.value))}>
              {results.map((result, index) => (
                <option key={result.safe_id} value={index}>
                  {result.ligand_id}{resultHasError(result) ? " (error)" : ""}
                </option>
              ))}
            </select>
            <ChevronDown size={16} />
          </div>
        )}
      </div>

      {failedResults.length > 0 && (
        <Alert
          tone="error"
          text={
            allResultsFailed
              ? `${failedResults.length === 1 ? "The compound" : "All compounds"} failed.`
              : `${failedResults.length} compound${failedResults.length === 1 ? "" : "s"} failed. Successful compounds are still shown.`
          }
        />
      )}

      <div className={cx("result-summary", currentFailed && "error")}>
        {currentResult?.query_svg && <div className="query-structure" dangerouslySetInnerHTML={{ __html: currentResult.query_svg }} />}
        <div>
          <strong>{currentResult?.smiles}</strong>
          <span>
            {currentFailed
              ? "Search failed for this compound"
              : `${currentResult?.predicted_targets?.length || 0} target rows · ${currentResult?.similarity_search_results?.length || 0} ligand hits`}
          </span>
        </div>
      </div>

      {currentFailed ? (
        <Alert tone="error" text={`${currentResult?.ligand_id || "Query"}: ${currentResult?.error || "Search failed"}`} />
      ) : (
        <>
          <div className="tabs">
            <button className={cx(activeTab === "targets" && "active")} onClick={() => setActiveTab("targets")}>Predicted Targets</button>
            <button className={cx(activeTab === "similarity" && "active")} onClick={() => setActiveTab("similarity")}>Similarity Search</button>
          </div>

          <div className="export-row">
            <button
              type="button"
              className="secondary-action"
              onClick={() => downloadCsv(
                `${safeFilePart(currentResult?.ligand_id || "query")}_predicted_targets.csv`,
                currentResult?.predicted_targets || [],
                targetExportColumns,
              )}
            >
              <Download size={16} />
              Export Predicted Targets
            </button>
            <button
              type="button"
              className="secondary-action"
              onClick={() => downloadCsv(
                `${safeFilePart(currentResult?.ligand_id || "query")}_similarity_search_results.csv`,
                currentResult?.similarity_search_results || [],
                similarityExportColumns,
              )}
            >
              <Download size={16} />
              Export Similarity Search
            </button>
          </div>

          <DataTable
            rows={rows}
            columns={columns}
            columnTooltips={activeTab === "targets" ? PREDICTED_TARGET_TOOLTIPS : {}}
          />
        </>
      )}
    </section>
  );
}

function resultHasError(result) {
  return Boolean(result?.error || result?.status === "error");
}

function DataTable({ rows, columns, columnTooltips = {} }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>
                <ColumnHeader column={column} tooltip={columnTooltips[column]} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 && (
            <tr>
              <td colSpan={columns.length} className="empty-cell">No rows returned</td>
            </tr>
          )}
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column) => (
                <td key={column}>
                  {column === "structure_svg" && row[column] ? (
                    <div className="table-structure" dangerouslySetInnerHTML={{ __html: row[column] }} />
                  ) : (
                    formatCell(row[column])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ColumnHeader({ column, tooltip }) {
  const label = column.replaceAll("_", " ");
  if (!tooltip) return label;

  return (
    <span className="column-header has-tooltip">
      <span>{label}</span>
      <HelpTooltip label={label} tooltip={tooltip} />
    </span>
  );
}

function HelpTooltip({ label, tooltip }) {
  return (
    <span className="help-tooltip" tabIndex="0" aria-label={`${label}: ${tooltip}`}>
      <Info className="help-tooltip-icon" size={13} aria-hidden="true" />
      <span className="help-tooltip-panel" role="tooltip">{tooltip}</span>
    </span>
  );
}

function formatCell(value) {
  if (value === null || value === undefined || value === "") return <span className="muted">-</span>;
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(4);
  return String(value);
}

function downloadCsv(filename, rows, columns) {
  const csv = toCsv(rows, columns);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function toCsv(rows, columns) {
  const header = columns.join(",");
  const body = rows.map((row) => columns.map((column) => csvCell(row[column])).join(","));
  return [header, ...body].join("\n") + "\n";
}

function csvCell(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

function safeFilePart(value) {
  return String(value).trim().replace(/[^A-Za-z0-9._-]+/g, "_") || "query";
}

function parseCsvPreview(text) {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]).map((item) => item.trim());
  const ligIndex = headers.indexOf("lig_id");
  const smilesIndex = headers.indexOf("smiles");
  if (ligIndex === -1 || smilesIndex === -1) return [];
  return lines.slice(1).map((line, index) => {
    const cells = splitCsvLine(line);
    return {
      lig_id: cells[ligIndex] || `query_${index + 1}`,
      smiles: cells[smilesIndex] || "",
    };
  }).filter((row) => row.smiles.trim());
}

function splitCsvLine(line) {
  const result = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current);
  return result.map((item) => item.trim().replace(/^"|"$/g, ""));
}

createRoot(document.getElementById("root")).render(<App />);
