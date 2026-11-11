import { Link, useSearchParams } from "react-router-dom";
import { formatDateTime } from "../api";
import { DatasetBatchSelect, useDatasetCatalog } from "../components/BatchDatasetPicker";
import ErrorBanner from "../components/ErrorBanner";
import { basename, DATASET_STAGE } from "../workflows";

export default function Datasets() {
  const [query, setQuery] = useSearchParams();
  const catalog = useDatasetCatalog("", query.get("batch") || "");
  const search = query.get("q") || "";
  const items = catalog.items.filter((item) =>
    `${item.path} ${DATASET_STAGE[item.stage || ""] || ""} ${item.asr_route || ""}`.toLowerCase().includes(search.trim().toLowerCase()),
  );

  function changeBatch(batch: string) {
    catalog.setBatch(batch);
    setQuery(batch ? { batch } : {});
  }

  return (
    <div className="work-page">
      <header className="page-heading"><div><h2>数据集</h2><p>按批次查找各处理阶段的清单和结果。</p></div></header>
      <section className="panel work-panel">
        <div className="section-line"><h3>清单目录</h3><button type="button" onClick={catalog.refresh}>刷新</button></div>
        <div className="form-columns">
          <DatasetBatchSelect catalog={catalog} onChange={changeBatch} />
          <label className="field">
            <span>批次内搜索</span>
            <input value={search} disabled={!catalog.batch} placeholder="文件名、阶段或路次" onChange={(event) => {
              const next = new URLSearchParams(query);
              if (event.target.value) next.set("q", event.target.value);
              else next.delete("q");
              setQuery(next, { replace: true });
            }} />
          </label>
        </div>
        <ErrorBanner error={catalog.error} />
        {catalog.loadingItems && <p className="muted">正在加载批次清单…</p>}
        {!catalog.batch && !catalog.loading && <div className="empty-state"><strong>{catalog.batches.length ? "先选择一个批次" : "暂无已发现的清单"}</strong></div>}
        {catalog.batch && !catalog.loadingItems && !catalog.error && !items.length && (
          <div className="empty-state"><strong>{search ? "没有符合搜索的清单" : "当前批次暂无清单"}</strong></div>
        )}
        {items.length > 0 && (
          <div className="table-scroll">
            <p className="help">{items.length} 份清单</p>
            <table>
              <thead><tr><th>清单</th><th>阶段</th><th>来源</th></tr></thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.dataset_id}>
                    <td>
                      <Link className="table-title" to={`/datasets/${encodeURIComponent(item.dataset_id)}`}>{basename(item.path)}</Link>
                      <div className="help path-text" title={item.path}>{item.path}</div>
                      {item.has_work_copy && <span className="help">有人工工作副本</span>}
                    </td>
                    <td>{DATASET_STAGE[item.stage || ""] || "其他清单"}{item.asr_route && ` · ${item.asr_route}`}</td>
                    <td>{item.job_id ? <Link to={`/jobs/${item.job_id}`}>{formatDateTime(item.published_at)}</Link> : "服务器目录"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
