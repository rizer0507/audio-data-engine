import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ApiRequestError, api, formatDateTime } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import type { Capabilities, Source, SourceCheckResult } from "../types";

const emptyForm = {
  batch: "",
  name: "",
  path: "",
  format: "wav",
  sample_rate: "",
  bit_depth: "",
  channels: "",
  note: "",
};

export default function Sources() {
  const [items, setItems] = useState<Source[]>([]);
  const [revision, setRevision] = useState<number | null>(null);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [form, setForm] = useState(emptyForm);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [lastCreatedId, setLastCreatedId] = useState<string | null>(null);
  const [checkById, setCheckById] = useState<Record<string, SourceCheckResult>>({});
  const [checking, setChecking] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [c, data] = await Promise.all([
        api.get<Capabilities>("/api/v1/capabilities"),
        api.get<{ items: Source[]; revision?: number }>("/api/v1/sources"),
      ]);
      setCaps(c);
      setItems(data.items || []);
      setRevision(data.revision ?? null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  function fillFromSource(s: Source) {
    setEditingId(s.source_id);
    setForm({
      batch: s.batch || "",
      name: s.name || s.source_name || "",
      path: s.path || "",
      format: s.format || "wav",
      sample_rate: s.sample_rate != null ? String(s.sample_rate) : "",
      bit_depth: s.bit_depth != null ? String(s.bit_depth) : "",
      channels: s.channels != null ? String(s.channels) : "",
      note: s.note || "",
    });
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const body = {
      batch: form.batch,
      name: form.name || undefined,
      path: form.path,
      format: form.format,
      sample_rate: form.sample_rate ? Number(form.sample_rate) : null,
      bit_depth: form.bit_depth ? Number(form.bit_depth) : null,
      channels: form.channels ? Number(form.channels) : null,
      note: form.note,
    };
    try {
      if (editingId) {
        await api.patch(`/api/v1/sources/${editingId}`, body, {
          "If-Match": String(revision ?? ""),
        });
      } else {
        const created = await api.post<Source & { reused?: boolean }>("/api/v1/sources", body);
        setLastCreatedId(created.source_id);
      }
      setForm(emptyForm);
      setEditingId(null);
      await load();
    } catch (err) {
      if (err instanceof ApiRequestError && err.code === "REVISION_CONFLICT") {
        const current = err.details.current_revision;
        setError(err);
        await load();
        setError(
          new ApiRequestError(409, {
            error: {
              code: "REVISION_CONFLICT",
              message: `配置已被他人更新（当前 revision=${current}）。已重新加载列表，表单输入仍保留供比较。`,
              details: err.details,
              request_id: err.requestId,
            },
          }),
        );
        return;
      }
      setError(err);
    }
  }

  async function onCheck(id: string) {
    setChecking(id);
    setError(null);
    try {
      const data = await api.post<SourceCheckResult>(`/api/v1/sources/${id}/check`);
      setCheckById((prev) => ({ ...prev, [id]: data }));
    } catch (err) {
      setError(err);
    } finally {
      setChecking(null);
    }
  }

  function statusLabel(st: string | undefined) {
    if (!st) return "未检查";
    if (st === "sample_ok") return "抽样可用";
    if (st === "check_failed") return "检查失败";
    if (st === "sample_empty") return "抽样为空";
    return st;
  }

  const pcmRequired = caps?.pcm_required_fields || ["sample_rate", "bit_depth", "channels"];

  return (
    <div>
      <h2>数据源</h2>
      <p className="muted">
        填写 batch 与服务器绝对路径；source_id 由后端自动分配。路径须落在 allowed_source_roots 内。revision=
        {revision ?? "—"}
      </p>
      <ErrorBanner error={error} />
      {lastCreatedId && (
        <p className="help">最近登记内部 ID（只读）：{lastCreatedId}</p>
      )}

      <div className="panel">
        <h3>{editingId ? `修改 ${editingId}` : "登记数据源"}</h3>
        <form className="form-grid" onSubmit={onSubmit}>
          <label>
            batch（引擎 source_name）
            <input
              value={form.batch}
              onChange={(e) => setForm({ ...form, batch: e.target.value })}
              required
              placeholder="mt3000"
            />
            <span className="help">字母/数字/_/-；作为引擎 --source-name，不是内部 ID</span>
          </label>
          <label>
            显示名称（可选）
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
          </label>
          <label>
            服务器路径
            <input
              value={form.path}
              onChange={(e) => setForm({ ...form, path: e.target.value })}
              required
              placeholder="D:/data/batch_x"
            />
            <span className="help">必须落在 configs/web.yaml 的 allowed_source_roots 内</span>
          </label>
          <label>
            格式
            <select value={form.format} onChange={(e) => setForm({ ...form, format: e.target.value })}>
              <option value="wav">wav</option>
              <option value="pcm">pcm</option>
              <option value="mp3">mp3</option>
              <option value="flac">flac</option>
            </select>
          </label>
          {form.format === "pcm" && (
            <>
              {pcmRequired.includes("sample_rate") && (
                <label>
                  sample_rate *
                  <input
                    value={form.sample_rate}
                    onChange={(e) => setForm({ ...form, sample_rate: e.target.value })}
                    required
                  />
                </label>
              )}
              {pcmRequired.includes("bit_depth") && (
                <label>
                  bit_depth *
                  <input
                    value={form.bit_depth}
                    onChange={(e) => setForm({ ...form, bit_depth: e.target.value })}
                    required
                  />
                </label>
              )}
              {pcmRequired.includes("channels") && (
                <label>
                  channels *
                  <input
                    value={form.channels}
                    onChange={(e) => setForm({ ...form, channels: e.target.value })}
                    required
                  />
                </label>
              )}
            </>
          )}
          <label>
            备注
            <input value={form.note} onChange={(e) => setForm({ ...form, note: e.target.value })} />
          </label>
          <div className="row">
            <button className="primary" type="submit">
              {editingId ? "保存修改" : "登记"}
            </button>
            {editingId && (
              <button
                type="button"
                onClick={() => {
                  setEditingId(null);
                  setForm(emptyForm);
                }}
              >
                取消编辑
              </button>
            )}
          </div>
        </form>
      </div>

      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>已登记</h3>
          <button type="button" onClick={() => void load()}>
            刷新
          </button>
        </div>
        {loading && <p className="muted">加载中…</p>}
        {!loading && items.length === 0 && (
          <p className="muted">暂无数据源。请先登记服务器上真实存在、进程可读的路径。</p>
        )}
        {!loading && items.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>batch</th>
                <th>名称</th>
                <th>路径</th>
                <th>格式/参数</th>
                <th>最近检查</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((s) => {
                const chk = checkById[s.source_id];
                return (
                  <tr key={s.source_id}>
                    <td>
                      {s.batch || "—"}
                      <div className="help">id={s.source_id}</div>
                    </td>
                    <td>{s.name || s.source_name}</td>
                    <td>{s.path}</td>
                    <td>
                      {s.format}
                      {s.format === "pcm"
                        ? ` / ${s.sample_rate || "?"}Hz ${s.bit_depth || "?"}bit ch=${s.channels || "?"}`
                        : ""}
                      <div className="help">
                        file_count={s.file_count == null ? "未全量统计" : s.file_count}
                      </div>
                    </td>
                    <td>
                      {chk ? (
                        <>
                          <div>{statusLabel(chk.status)}</div>
                          <div className="help">{formatDateTime(chk.checked_at)}</div>
                          <div className="help">
                            抽样 {chk.sample_file_count} 个；full_scan={String(chk.full_scan)}
                          </div>
                          {chk.error && <div className="warn">{chk.error}</div>}
                        </>
                      ) : (
                        <span className="muted">未检查</span>
                      )}
                    </td>
                    <td className="row">
                      <button type="button" disabled={checking === s.source_id} onClick={() => void onCheck(s.source_id)}>
                        {checking === s.source_id ? "检查中…" : "检查"}
                      </button>
                      <button type="button" onClick={() => fillFromSource(s)}>
                        修改
                      </button>
                      <Link to={`/jobs?source_id=${encodeURIComponent(s.source_id)}`}>新建任务</Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {Object.keys(checkById).length > 0 && (
          <details style={{ marginTop: "1rem" }}>
            <summary>最近检查原始结果</summary>
            <pre className="logbox">{JSON.stringify(checkById, null, 2)}</pre>
          </details>
        )}
      </div>
    </div>
  );
}
