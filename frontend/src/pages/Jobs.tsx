import { useEffect, useState } from "react";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import { api, formatDateTime } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import TaskStatus from "../components/TaskStatus";
import type { Job, ListResponse, NodeProgress, PluginDescriptor } from "../types";
import { ACTIVE_JOBS, JOB_STATUS, basename, nodeLabel } from "../workflows";

export default function Jobs() {
  const [params, setParams] = useSearchParams();
  const [page, setPage] = useState<ListResponse<Job>>();
  const [error, setError] = useState<unknown>(null);
  const [revision, setRevision] = useState(0);
  const [names, setNames] = useState<Record<string, string>>({});
  const status = params.get("status") || "";
  const cursor = Number(params.get("cursor") || 0) || 0;
  const inputLink = params.has("source_id") || params.has("dataset_id");

  useEffect(() => {
    if (inputLink) return;
    const abort = new AbortController();
    void api.get<ListResponse<PluginDescriptor>>("/api/v1/plugins?kind=pipeline", abort.signal)
      .then((result) => setNames(Object.fromEntries(result.items.map((item) => [item.id, item.name]))))
      .catch(() => undefined);
    return () => abort.abort();
  }, [inputLink]);

  useEffect(() => {
    if (inputLink) return;
    const abort = new AbortController();
    const query = new URLSearchParams({ cursor: String(cursor), limit: "25" });
    if (status) query.set("status", status);
    let timer: number | undefined;
    setPage(undefined);
    setError(null);
    async function load() {
      try {
        const result = await api.get<ListResponse<Job>>(`/api/v1/jobs?${query}`, abort.signal);
        if (abort.signal.aborted) return;
        setPage(result);
        setError(null);
        if (result.items.some((job) => ACTIVE_JOBS.has(job.status))) timer = window.setTimeout(load, 7000);
      } catch (failure) {
        if (!abort.signal.aborted) setError(failure);
      }
    }
    void load();
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [cursor, inputLink, revision, status]);

  if (inputLink) return <Navigate to={`/jobs/new?${params}`} replace />;

  function goTo(next: number) {
    const query = new URLSearchParams();
    if (status) query.set("status", status);
    if (next > 0) query.set("cursor", String(next));
    setParams(query);
  }

  return (
    <div className="work-page">
      <header className="page-heading">
        <div><h2>任务工作台</h2><p>创建处理任务，查看进度和已发布产物。</p></div>
        <Link className="button primary" to="/jobs/new">新建任务</Link>
      </header>
      <section className="panel work-panel">
        <div className="section-line">
          <h3>任务列表</h3>
          <div className="row">
            <select className="compact-control" aria-label="任务状态" value={status} onChange={(event) => {
              const value = event.target.value;
              setParams(value ? { status: value } : {});
            }}>
              <option value="">全部状态</option>
              {Object.entries(JOB_STATUS).filter(([value]) => value !== "pending").map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
            <button type="button" onClick={() => setRevision((value) => value + 1)}>刷新</button>
          </div>
        </div>
        <ErrorBanner error={error} />
        {!page && !error && <p className="muted">正在加载任务…</p>}
        {page && !page.items.length && (
          <div className="empty-state"><strong>{status ? "没有符合此状态的任务" : "还没有任务"}</strong>
            <p>{status ? "可以切换状态筛选。" : "从工序一完整流程开始，或选择一条单步流水线。"}</p>
            {!status && <Link to="/jobs/new">新建任务</Link>}
          </div>
        )}
        {Boolean(page?.items.length) && (
          <div className="table-scroll">
            <table>
              <thead><tr><th>任务 / 批次</th><th>状态</th><th>当前进度</th><th>创建时间</th></tr></thead>
              <tbody>
                {page?.items.map((job) => {
                  const plugin = job.request.root_plugin as { id: string } | null;
                  const input = job.request.input as { dataset_id?: string } | undefined;
                  const progress = job.progress?.nodes as Record<string, NodeProgress> | undefined;
                  const active = job.nodes?.filter((node) => progress?.[node.node_id]?.status === "running").map(nodeLabel);
                  const complete = Object.values(progress || {}).filter((node) => node.status === "succeeded").length;
                  return (
                    <tr key={job.job_id}>
                      <td>
                        <Link className="table-title" to={`/jobs/${job.job_id}`}>{plugin ? names[plugin.id] || plugin.id : "规则打标"}</Link>
                        <div className="help">{job.batch || (input?.dataset_id ? basename(input.dataset_id) : job.job_id)}</div>
                      </td>
                      <td><TaskStatus status={job.status} /></td>
                      <td>{job.status === "running" && active?.length ? active.join("、") : progress && job.nodes?.length
                        ? `${complete} / ${job.nodes.length} 节点` : "—"}</td>
                      <td className="nowrap">{formatDateTime(job.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        {page && (cursor > 0 || page.next_cursor != null) && (
          <div className="row page-actions">
            <button type="button" disabled={cursor <= 0} onClick={() => goTo(cursor - 25)}>上一页</button>
            <button type="button" disabled={page.next_cursor == null} onClick={() => goTo(Number(page.next_cursor))}>下一页</button>
          </div>
        )}
      </section>
    </div>
  );
}
