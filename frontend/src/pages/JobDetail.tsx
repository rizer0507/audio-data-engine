import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, copyText, durationBetween, formatDateTime, publicPath } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import TaskStages from "../components/TaskStages";
import TaskStatus from "../components/TaskStatus";
import type { Job, JobLogsResponse, NodeProgress, PlanNode } from "../types";
import { ACTIVE_JOBS, DATASET_STAGE, basename, nodeLabel } from "../workflows";

type Artifact = { artifact_id?: string; artifact_kind?: string; path?: string; dataset_id?: string; stage?: string };
type Attempt = { attempt_no: number; status: string; started_at?: string; ended_at?: string };

function RunLog({ endpoint, nodeId, active }: { endpoint: string; nodeId: string; active: boolean }) {
  const [logs, setLogs] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [reload, setReload] = useState(0);
  const [follow, setFollow] = useState(true);
  const [copied, setCopied] = useState(false);
  const cursor = useRef("");
  const logElement = useRef<HTMLPreElement>(null);
  const maxLog = 160_000;
  useEffect(() => {
    const abort = new AbortController();
    let timer: number | undefined;
    async function read() {
      const query = new URLSearchParams();
      if (cursor.current) query.set("cursor", cursor.current);
      if (nodeId) query.set("node_id", nodeId);
      try {
        const result = await api.get<JobLogsResponse>(`${endpoint}/logs?${query}`, abort.signal);
        if (abort.signal.aborted) return;
        setError(null);
        const previous = cursor.current;
        cursor.current = result.next_cursor || previous;
        const chunk = result.chunks.map((line) => line.text).join("");
        if (chunk) setLogs((current) => (current + chunk).slice(-maxLog));
        if (!result.eof && cursor.current !== previous && chunk) timer = window.setTimeout(read, 100);
        else if (active) timer = window.setTimeout(read, document.hidden ? 6500 : 2200);
      } catch (failure) {
        if (!abort.signal.aborted) {
          setError(failure);
          if (active) timer = window.setTimeout(read, 5000);
        }
      }
    }
    void read();
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [active, endpoint, nodeId, reload]);

  useEffect(() => {
    if (follow && logElement.current) logElement.current.scrollTop = logElement.current.scrollHeight;
  }, [follow, logs]);

  return (
    <div className="form-stack">
      <div className="row log-controls">
        <label className="check-row"><input type="checkbox" checked={follow} onChange={(event) => setFollow(event.target.checked)} />自动滚动</label>
        <button type="button" onClick={() => setReload((value) => value + 1)}>刷新日志</button>
        <button type="button" disabled={!logs} onClick={() => void copyText(logs).then(setCopied)}>{copied ? "已复制" : "复制日志"}</button>
      </div>
      <ErrorBanner error={error} />
      <pre className="logbox run-log" ref={logElement}>{logs || "暂无日志。"}</pre>
      {logs.length === maxLog && <p className="help">仅显示最近的日志。</p>}
    </div>
  );
}

function ResultFiles({ files }: { files: Artifact[] }) {
  return (
    <ul className="result-files">
      {files.map((file, index) => (
        <li key={file.artifact_id || file.path || index}>
          <div>
            <strong>{DATASET_STAGE[file.stage || ""] || (file.artifact_kind === "report" ? "汇总报告" : basename(file.path || "产物"))}</strong>
            <small className="help path-text" title={file.path}>{basename(file.path || "")}</small>
          </div>
          <div className="row">
            {file.dataset_id && <Link to={`/datasets/${encodeURIComponent(file.dataset_id)}`}>打开数据集</Link>}
            {file.artifact_id && <a href={publicPath(`/api/v1/artifacts/${encodeURIComponent(file.artifact_id)}/download`)}>下载</a>}
          </div>
        </li>
      ))}
    </ul>
  );
}

function JobRun({ jobId }: { jobId: string }) {
  const endpoint = `/api/v1/jobs/${encodeURIComponent(jobId)}`;
  const navigate = useNavigate();
  const [query, setQuery] = useSearchParams();
  const nodeId = query.get("node_id") || "";
  const [job, setJob] = useState<Job>();
  const [loadError, setLoadError] = useState<unknown>(null);
  const [actionError, setActionError] = useState<unknown>(null);
  const [reload, setReload] = useState(0);
  const [action, setAction] = useState<"cancel" | "resume" | "rerun">();
  const [showLogs, setShowLogs] = useState(Boolean(nodeId));
  const [snapshot, setSnapshot] = useState<Record<string, unknown>>();
  const logSection = useRef<HTMLElement>(null);

  useEffect(() => {
    const abort = new AbortController();
    let timer: number | undefined;
    async function refresh() {
      try {
        const data = await api.get<Job>(endpoint, abort.signal);
        if (abort.signal.aborted) return;
        setJob(data);
        setLoadError(null);
        if (ACTIVE_JOBS.has(data.status)) timer = window.setTimeout(refresh, document.hidden ? 6500 : 2000);
      } catch (failure) {
        if (!abort.signal.aborted) {
          setLoadError(failure);
          timer = window.setTimeout(refresh, 6000);
        }
      }
    }
    void refresh();
    return () => {
      abort.abort();
      window.clearTimeout(timer);
    };
  }, [endpoint, reload]);

  async function runAction(next: "cancel" | "resume" | "rerun") {
    if (action) return;
    setAction(next);
    setActionError(null);
    try {
      if (next === "cancel") await api.post(`${endpoint}/cancel`);
      else {
        const response = await api.post<{ job: Job }>(`${endpoint}/retry`, { mode: next });
        if (next === "rerun") {
          navigate(`/jobs/${response.job.job_id}`);
          return;
        }
        setJob(response.job);
      }
      setReload((value) => value + 1);
    } catch (failure) {
      setActionError(failure);
    } finally {
      setAction(undefined);
    }
  }

  function selectLog(node: string, scroll = false) {
    const next = new URLSearchParams(query);
    if (node) next.set("node_id", node);
    else next.delete("node_id");
    setQuery(next, { replace: true });
    setShowLogs(true);
    if (scroll) logSection.current?.scrollIntoView({ block: "nearest" });
  }

  if (!job) return (
    <div className="work-page">
      <Link className="back-link" to="/jobs">← 任务列表</Link>
      <h2>任务详情</h2>
      {loadError ? <ErrorBanner error={loadError} /> : <p className="muted">正在加载任务…</p>}
    </div>
  );

  const active = ACTIVE_JOBS.has(job.status);
  const progress = job.progress?.nodes as Record<string, NodeProgress> | undefined;
  const nodes = job.nodes?.length ? job.nodes : Object.keys(progress || {}).map<PlanNode>((node_id) => ({
    node_id, label: node_id, depends_on: [],
  }));
  const knownNodes = nodes.filter((node) => progress?.[node.node_id]?.started_at);
  const live = nodes.filter((node) => progress?.[node.node_id]?.status === "running");
  const failed = nodes.filter((node) => progress?.[node.node_id]?.status === "failed");
  const attempts = (job.attempts || []) as Attempt[];
  const current = attempts.find((item) => item.attempt_no === job.current_attempt);
  const canResume = job.capabilities?.supports_resume === true && ["failed", "cancelled", "interrupted"].includes(job.status);
  const canCancel = job.status === "queued" || (job.status === "running" && job.capabilities?.supports_cancel === true);
  const files = (job.artifacts || []) as Artifact[];
  const finalFiles = files.filter((file) => file.stage === "tags" || file.artifact_kind !== "manifest");
  const mainFiles = finalFiles.length ? finalFiles : files;
  const intermediate = files.filter((file) => !mainFiles.includes(file));
  const input = job.request.input as { source_id?: string; dataset_id?: string } | undefined;
  const newInput = new URLSearchParams();
  if (input?.source_id) newInput.set("source_id", input.source_id);
  if (input?.dataset_id) newInput.set("dataset_id", input.dataset_id);
  const errorResult = job.error?.result as { errors?: Array<{ message?: string }> } | undefined;
  const failure = errorResult?.errors?.[0]?.message || (typeof job.error?.message === "string" ? job.error.message : "");

  return (
    <div className="work-page">
      <Link className="back-link" to="/jobs">← 任务列表</Link>
      <header className="page-heading">
        <div>
          <div className="row"><h2>{job.batch ? `批次 ${job.batch}` : "任务详情"}</h2><TaskStatus status={job.status} /></div>
          <p className="path-text">{input?.dataset_id ? basename(input.dataset_id) : job.job_id}</p>
        </div>
        <div className="row">
          <button type="button" onClick={() => setReload((value) => value + 1)}>刷新</button>
          {canCancel && <button type="button" disabled={Boolean(action)} onClick={() => void runAction("cancel")}>停止任务</button>}
          {canResume && <button type="button" className="primary" disabled={Boolean(action)} onClick={() => void runAction("resume")}>{action === "resume" ? "正在恢复…" : "继续运行"}</button>}
        </div>
      </header>
      {Boolean(loadError) && <p className="notice" role="status">连接暂时中断，以下为上次取得的状态。正在重试。</p>}
      <ErrorBanner error={actionError} />
      <div className="work-layout detail-layout">
        <div className="work-main">
          <section className="panel work-panel">
            <div className="section-line"><h3>运行进度</h3>{current && <span className="help">第 {current.attempt_no} 次执行</span>}</div>
            <div className="status-summary" aria-live="polite">
              {job.status === "running" && (live.length ? `正在处理 ${live.map(nodeLabel).join("、")}` :
                nodes.some((node) => node.kind === "production_node") ? "执行器运行中，等待节点状态。" : "正在运行，详情可在运行日志查看。")}
              {job.status === "queued" && "正在等待 Worker 领取任务。"}
              {job.status === "failed" && (failed.length ? `失败节点 ${failed.map(nodeLabel).join("、")}` : "任务执行失败。")}
              {job.status === "succeeded" && "执行完成，可查看产物。"}
              {job.status === "interrupted" && "执行中断，可按可用断点继续。"}
              {job.status === "cancelling" && "正在等待执行进程停止。"}
              {job.status === "cancelled" && "任务已停止。"}
            </div>
            {failure && <p className="notice failure">{failure}</p>}
            {nodes.length > 0 && <TaskStages nodes={nodes} progress={progress} status={job.status} onNode={(id) => selectLog(id, true)} />}
          </section>
          <section className="panel work-panel" ref={logSection}>
            <div className="section-line">
              <h3>运行日志</h3>
              <button type="button" onClick={() => setShowLogs(!showLogs)}>{showLogs ? "收起日志" : "展开日志"}</button>
            </div>
            {showLogs && (
              <>
                {knownNodes.length > 0 && (
                  <label className="field">
                    <span>日志范围</span>
                    <select value={knownNodes.some((node) => node.node_id === nodeId) ? nodeId : ""} onChange={(event) => selectLog(event.target.value)}>
                      <option value="">任务总日志</option>
                      {knownNodes.map((node) => <option key={node.node_id} value={node.node_id}>{nodeLabel(node)}</option>)}
                    </select>
                  </label>
                )}
                {job.current_attempt > 0 ? (
                  <RunLog key={`${job.current_attempt}:${nodeId}`} endpoint={endpoint} nodeId={nodeId} active={active} />
                ) : <p className="muted">任务开始后会显示日志。</p>}
              </>
            )}
          </section>
        </div>
        <aside className="work-side">
          <section className="panel work-panel">
            <h3>产物</h3>
            {mainFiles.length ? <ResultFiles files={mainFiles} /> : <p className="muted">任务成功后显示已发布产物。</p>}
            {intermediate.length > 0 && (
              <details className="options-fold">
                <summary>中间产物 · {intermediate.length}</summary>
                <ResultFiles files={intermediate} />
              </details>
            )}
          </section>
          <section className="panel work-panel">
            <h3>执行记录</h3>
            <dl className="summary-list">
              <div><dt>创建</dt><dd>{formatDateTime(job.created_at)}</dd></div>
              {current?.started_at && <div><dt>本次耗时</dt><dd>{durationBetween(current.started_at, current.ended_at)}</dd></div>}
            </dl>
            {attempts.length > 0 && <ul className="attempt-list">
              {[...attempts].reverse().map((item) => (
                <li key={item.attempt_no}><span>第 {item.attempt_no} 次</span><TaskStatus status={item.status} /></li>
              ))}
            </ul>}
            <details className="options-fold">
              <summary>任务配置与诊断</summary>
              <div className="row">
                <button type="button" onClick={() => void api.get<Record<string, unknown>>(`${endpoint}/snapshot`).then(setSnapshot).catch(setActionError)}>加载快照</button>
                {!active && <button type="button" disabled={Boolean(action)} onClick={() => void runAction("rerun")}>按原计划重跑</button>}
                {!active && newInput.size > 0 && <Link to={`/jobs/new?${newInput}`}>以此输入新建</Link>}
              </div>
              <pre className="logbox">{JSON.stringify(snapshot || { job_id: jobId, request: job.request, error: job.error }, null, 2)}</pre>
            </details>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default function JobDetail() {
  const { jobId } = useParams();
  return jobId ? <JobRun key={jobId} jobId={jobId} /> : <Link to="/jobs">返回任务列表</Link>;
}
