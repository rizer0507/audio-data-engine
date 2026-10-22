import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { api, copyText, durationBetween, formatDateTime, publicPath } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import TaskStatus from "../components/TaskStatus";
import type { Job, JobLogsResponse } from "../types";

const TERMINAL = new Set(["succeeded", "failed", "cancelled"]);

export default function JobDetail() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [job, setJob] = useState<Job | null>(null);
  const [logs, setLogs] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [connError, setConnError] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const [pollMs, setPollMs] = useState(2000);
  const [nodeId, setNodeId] = useState(searchParams.get("node_id") || "");
  const [snapshot, setSnapshot] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);

  const logRef = useRef<HTMLPreElement>(null);
  const cursorRef = useRef("");
  const attemptRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const visibleRef = useRef(!document.hidden);

  async function fetchJob(): Promise<Job | null> {
    if (!jobId) return null;
    const data = await api.get<Job>(`/api/v1/jobs/${jobId}`);
    setJob(data);
    return data;
  }

  async function fetchLogs(reset = false) {
    if (!jobId) return;
    if (reset) {
      cursorRef.current = "";
      setLogs("");
    }
    const qs = new URLSearchParams();
    if (cursorRef.current) qs.set("cursor", cursorRef.current);
    if (nodeId) qs.set("node_id", nodeId);
    const log = await api.get<JobLogsResponse>(`/api/v1/jobs/${jobId}/logs?${qs.toString()}`);
    if (log.chunks?.length) {
      setLogs((prev) => prev + log.chunks.map((c) => c.text).join(""));
    }
    if (log.next_cursor) cursorRef.current = log.next_cursor;
  }

  async function refresh(opts?: { resetLogs?: boolean }) {
    setError(null);
    setConnError("");
    try {
      const data = await fetchJob();
      if (!data) return;
      if (attemptRef.current !== null && attemptRef.current !== data.current_attempt) {
        // attempt 切换：重建 cursor，不拼接假连续日志
        cursorRef.current = "";
        setLogs("");
      }
      attemptRef.current = data.current_attempt;
      await fetchLogs(Boolean(opts?.resetLogs));
    } catch (e) {
      setConnError("连接中断或请求失败；任务状态以服务端为准，未在本地篡改。");
      setError(e);
    }
  }

  useEffect(() => {
    cursorRef.current = "";
    attemptRef.current = null;
    setLogs("");
    setJob(null);
    void refresh({ resetLogs: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  useEffect(() => {
    cursorRef.current = "";
    setLogs("");
    void fetchLogs(true).catch((e) => setError(e));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeId]);

  useEffect(() => {
    const onVis = () => {
      visibleRef.current = !document.hidden;
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, []);

  useEffect(() => {
    if (!job) return;
    if (TERMINAL.has(job.status)) {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }
    const tick = () => {
      const ms = visibleRef.current ? pollMs : Math.max(pollMs * 3, 6000);
      void refresh().catch(() => undefined);
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = window.setInterval(() => {
        void refresh().catch(() => undefined);
      }, ms);
    };
    tick();
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.status, jobId, pollMs]);

  useEffect(() => {
    if (autoScroll && logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  async function onCancel() {
    if (!jobId) return;
    setBusy(true);
    try {
      await api.post(`/api/v1/jobs/${jobId}/cancel`);
      await refresh();
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  async function onRetry(mode: "resume" | "rerun") {
    if (!jobId) return;
    setBusy(true);
    try {
      const data = await api.post<{ job?: Job; mode: string }>(`/api/v1/jobs/${jobId}/retry`, { mode });
      if (mode === "rerun" && data.job?.job_id) {
        navigate(`/jobs/${data.job.job_id}`);
        return;
      }
      await refresh({ resetLogs: true });
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  async function loadSnapshot() {
    if (!jobId) return;
    try {
      const data = await api.get<Record<string, unknown>>(`/api/v1/jobs/${jobId}/snapshot`);
      setSnapshot(data);
    } catch (e) {
      setError(e);
    }
  }

  if (!job && !error) {
    return (
      <div>
        <h2>任务详情</h2>
        <p className="muted">加载中…</p>
      </div>
    );
  }

  if (!job) {
    return (
      <div>
        <h2>任务详情</h2>
        <ErrorBanner error={error} />
      </div>
    );
  }

  const caps = job.capabilities || {};
  const cancelSupported = caps.supports_cancel !== false;
  const resumeSupported = caps.supports_resume !== false;
  const canCancel = !TERMINAL.has(job.status) && job.status !== "cancelling" && cancelSupported;
  const canResume =
    ["failed", "cancelled", "interrupted"].includes(job.status) && resumeSupported;

  const progress = job.progress || {};
  const req = job.request || {};

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>任务 {job.job_id}</h2>
        <Link to="/jobs">返回列表</Link>
      </div>
      <ErrorBanner error={error} />
      {connError && <div className="warn">{connError}</div>}

      <div className="panel">
        <div className="row">
          <TaskStatus status={job.status} />
          <span className="muted">attempt={job.current_attempt}</span>
          <span className="muted">创建 {formatDateTime(job.created_at)}</span>
          <span className="muted">更新 {formatDateTime(job.updated_at)}</span>
          <span className="muted">
            运行时长{" "}
            {durationBetween(
              job.attempts?.find((a) => a.attempt_no === job.current_attempt)?.started_at as
                | string
                | undefined,
              job.attempts?.find((a) => a.attempt_no === job.current_attempt)?.ended_at as
                | string
                | undefined,
            )}
          </span>
        </div>

        <div className="row" style={{ marginTop: "0.75rem" }}>
          <button type="button" disabled={!canCancel || busy} onClick={() => void onCancel()}>
            {job.status === "cancelling" ? "取消中…" : "取消"}
          </button>
          {!cancelSupported && <span className="help">插件声明不支持可靠取消</span>}
          <button type="button" disabled={!canResume || busy} onClick={() => void onRetry("resume")}>
            恢复（原冻结计划）
          </button>
          {!resumeSupported && <span className="help">插件不支持恢复或断点不兼容时将由后端拒绝</span>}
          <button type="button" disabled={busy} onClick={() => void onRetry("rerun")}>
            重新运行（新任务/默认原计划）
          </button>
          <label className="row">
            <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} />
            日志自动滚动
          </label>
          <label className="row">
            轮询
            <select value={pollMs} onChange={(e) => setPollMs(Number(e.target.value))}>
              <option value={2000}>2s</option>
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
            </select>
          </label>
          <button type="button" onClick={() => void refresh()}>
            立即刷新
          </button>
          <button
            type="button"
            onClick={() => {
              void copyText(logs || JSON.stringify(job.error || {}, null, 2));
            }}
          >
            复制日志/诊断
          </button>
        </div>

        {job.status === "cancelling" && (
          <p className="warn">已请求取消，等待后端确认停止后才会变为 cancelled（不会停止外部 vLLM）。</p>
        )}

        {job.error && (
          <pre className="error" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(job.error, null, 2)}
          </pre>
        )}

        <h4>进度（真实计数；无可靠 total 时不画伪百分比）</h4>
        <pre className="logbox">{JSON.stringify(progress, null, 2)}</pre>

        <h4>输入 / 插件 / 模型摘要</h4>
        <pre className="logbox">
          {JSON.stringify(
            {
              task_type: job.task_type,
              plan_id: job.plan_id,
              plan_digest: job.plan_digest,
              request: req,
              capabilities: caps,
            },
            null,
            2,
          )}
        </pre>

        <div className="row">
          <h4 style={{ margin: 0 }}>增量日志</h4>
          <input
            style={{ maxWidth: 200 }}
            placeholder="node_id 过滤"
            value={nodeId}
            onChange={(e) => {
              setNodeId(e.target.value);
              const p = new URLSearchParams(searchParams);
              if (e.target.value) p.set("node_id", e.target.value);
              else p.delete("node_id");
              setSearchParams(p, { replace: true });
            }}
          />
        </div>
        <pre className="logbox" ref={logRef}>
          {logs || "（暂无日志）"}
        </pre>

        <h4>产物</h4>
        {(job.artifacts || []).length === 0 ? (
          <p className="muted">尚无已登记产物</p>
        ) : (
          <ul>
            {(job.artifacts as Array<Record<string, unknown>>).map((a, i) => {
              const id = String(a.artifact_id || a.path || i);
              const path = String(a.path || a.artifact_id || "");
              return (
                <li key={id}>
                  {JSON.stringify(a)}{" "}
                  {path && (
                    <a href={publicPath(`/api/v1/artifacts/${encodeURIComponent(path)}/download`)} target="_blank" rel="noreferrer">
                      下载
                    </a>
                  )}
                  {typeof a.dataset_id === "string" && (
                    <Link to={`/datasets/${a.dataset_id}`}> 打开数据集</Link>
                  )}
                </li>
              );
            })}
          </ul>
        )}

        <h4>Attempts</h4>
        <pre className="logbox">{JSON.stringify(job.attempts || [], null, 2)}</pre>

        <div className="row">
          <button type="button" onClick={() => void loadSnapshot()}>
            加载脱敏快照
          </button>
        </div>
        {snapshot && <pre className="logbox">{JSON.stringify(snapshot, null, 2)}</pre>}
      </div>
    </div>
  );
}
