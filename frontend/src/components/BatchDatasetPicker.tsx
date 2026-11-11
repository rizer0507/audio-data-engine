import { useEffect, useState } from "react";
import { api } from "../api";
import type { DatasetBatch, DatasetSummary } from "../types";
import { datasetLabel } from "../workflows";
import ErrorBanner from "./ErrorBanner";

type CatalogResponse = { batches: DatasetBatch[]; items: DatasetSummary[] };
export const UNGROUPED = "__ungrouped__";

export function useDatasetCatalog(initialId = "", initialBatch = "") {
  const [batch, setBatch] = useState(initialBatch);
  const [batches, setBatches] = useState<DatasetBatch[]>([]);
  const [rows, setRows] = useState<{ batch: string; items: DatasetSummary[] }>();
  const [initialItem, setInitialItem] = useState<DatasetSummary>();
  const [loading, setLoading] = useState(true);
  const [catalogError, setCatalogError] = useState<unknown>(null);
  const [itemsError, setItemsError] = useState<unknown>(null);
  const [revision, setRevision] = useState(0);

  useEffect(() => setBatch(initialBatch), [initialBatch]);

  useEffect(() => {
    const abort = new AbortController();
    setLoading(true);
    setCatalogError(null);
    void Promise.all([
      api.get<CatalogResponse>("/api/v1/datasets?catalog_only=true", abort.signal),
      initialId
        ? api.get<CatalogResponse>(`/api/v1/datasets?dataset_id=${encodeURIComponent(initialId)}`, abort.signal)
        : null,
    ]).then(([catalog, detail]) => {
      if (abort.signal.aborted) return;
      setBatches(catalog.batches || []);
      const item = detail?.items[0];
      if (item) {
        setInitialItem(item);
        setBatch((current) => current || item.batch || UNGROUPED);
      }
    }).catch((error) => {
      if (!abort.signal.aborted) setCatalogError(error);
    }).finally(() => {
      if (!abort.signal.aborted) setLoading(false);
    });
    return () => abort.abort();
  }, [initialId, revision]);

  useEffect(() => {
    setRows(undefined);
    setItemsError(null);
    if (!batch) return;
    const abort = new AbortController();
    const value = batch === UNGROUPED ? "" : batch;
    void api.get<CatalogResponse>(`/api/v1/datasets?batch=${encodeURIComponent(value)}`, abort.signal)
      .then((result) => {
        if (!abort.signal.aborted) setRows({ batch, items: result.items });
      })
      .catch((error) => {
        if (!abort.signal.aborted) setItemsError(error);
      });
    return () => abort.abort();
  }, [batch, revision]);

  return {
    batch, setBatch, batches, initialItem, items: rows?.batch === batch ? rows.items : [],
    loading, loadingItems: Boolean(batch) && rows?.batch !== batch && !itemsError,
    error: catalogError || itemsError, refresh: () => setRevision((value) => value + 1),
  };
}

type Catalog = ReturnType<typeof useDatasetCatalog>;

export function DatasetBatchSelect({ catalog, onChange }: {
  catalog: Catalog; onChange: (value: string) => void;
}) {
  return (
    <label className="field">
      <span>批次</span>
      <select value={catalog.batch} disabled={catalog.loading} onChange={(event) => onChange(event.target.value)}>
        <option value="">{catalog.loading ? "正在加载批次…" : "选择批次"}</option>
        {catalog.batches.map(({ batch, count }) => (
          <option key={batch ?? UNGROUPED} value={batch ?? UNGROUPED}>
            {batch ?? "未归类"} · {count} 份清单
          </option>
        ))}
      </select>
    </label>
  );
}

export default function BatchDatasetPicker({ catalog, value, onChange, onBatchChange, label = "输入清单" }: {
  catalog: Catalog; value: string; onChange: (value: string) => void;
  onBatchChange?: () => void; label?: string;
}) {
  const [manual, setManual] = useState(false);
  const [search, setSearch] = useState("");
  const filtered = catalog.items.filter((item) =>
    `${datasetLabel(item)} ${item.path}`.toLowerCase().includes(search.trim().toLowerCase()),
  );
  const current = catalog.items.find((item) => item.dataset_id === value) ??
    (catalog.initialItem?.dataset_id === value ? catalog.initialItem : undefined);
  const pinned = current && !filtered.some((item) => item.dataset_id === value) ? current : null;

  return (
    <div className="form-stack">
      <ErrorBanner error={catalog.error} />
      {Boolean(catalog.error) && <button type="button" onClick={catalog.refresh}>重试加载</button>}
      <div className="form-columns">
        <DatasetBatchSelect catalog={catalog} onChange={(batch) => {
          catalog.setBatch(batch);
          setManual(false);
          setSearch("");
          onChange("");
          onBatchChange?.();
        }} />
        <label className="field">
          <span>{label}</span>
          {manual ? (
            <input value={value} onChange={(event) => onChange(event.target.value)} placeholder="服务器清单路径" />
          ) : (
            <select value={value} disabled={!catalog.batch || catalog.loadingItems} onChange={(event) => onChange(event.target.value)}>
              <option value="">{!catalog.batch ? "先选择批次" : catalog.loadingItems ? "正在加载清单…" : "选择清单"}</option>
              {pinned && <option value={pinned.dataset_id}>{datasetLabel(pinned)}</option>}
              {value && !current && <option value={value}>{value}</option>}
              {filtered.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{datasetLabel(item)}</option>)}
            </select>
          )}
        </label>
      </div>
      {!manual && catalog.items.length > 12 && (
        <label className="field">
          <span>批次内搜索</span>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="文件名、阶段或路次" />
          <span className="help">{filtered.length} 份匹配清单</span>
        </label>
      )}
      <div className="row selection-hint">
        {current && <span className="help path-text" title={current.path}>{current.path}</span>}
        {!current && catalog.batch && !catalog.loadingItems && !catalog.items.length && !catalog.error && (
          <span className="help">这个批次没有可用清单。</span>
        )}
        <button type="button" className="linkish" onClick={() => {
          if (manual && !current) onChange("");
          setManual(!manual);
        }}>{manual ? "从批次选择" : "手动填写路径"}</button>
      </div>
    </div>
  );
}
