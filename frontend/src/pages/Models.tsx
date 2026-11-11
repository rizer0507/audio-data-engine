import { useEffect, useState } from "react";
import { ApiRequestError, api, formatDateTime } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import type { Capabilities, ModelCheckResult, ModelConnection } from "../types";

const emptyForm = {
  name: "",
  provider: "qwen",
  base_url: "http://127.0.0.1:8002/v1",
  model: "",
  model_revision: "deployment_001",
  api_key_env: "",
  timeout_seconds: 120,
  concurrency: 4,
};

export default function Models() {
  const [items, setItems] = useState<ModelConnection[]>([]);
  const [providers, setProviders] = useState<string[]>(["glm", "qwen"]);
  const [revision, setRevision] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [checkOut, setCheckOut] = useState<ModelCheckResult | null>(null);
  const [form, setForm] = useState(emptyForm);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [lastCreatedId, setLastCreatedId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [caps, data] = await Promise.all([
        api.get<Capabilities>("/api/v1/capabilities"),
        api.get<{ items: ModelConnection[]; revision?: number }>("/api/v1/models"),
      ]);
      setProviders(caps.providers || ["glm", "qwen"]);
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

  function fillEdit(m: ModelConnection) {
    setEditingId(m.connection_id);
    setForm({
      name: m.name,
      provider: m.provider,
      base_url: m.base_url,
      model: m.model,
      model_revision: m.model_revision,
      api_key_env: m.api_key_env || "",
      timeout_seconds: m.timeout_seconds,
      concurrency: m.concurrency,
    });
  }

  async function onSave(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      if (editingId) {
        await api.patch(`/api/v1/models/${editingId}`, form, {
          "If-Match": String(revision ?? ""),
        });
      } else {
        const created = await api.post<ModelConnection & { reused?: boolean }>("/api/v1/models", form);
        setLastCreatedId(created.connection_id);
      }
      setForm(emptyForm);
      setEditingId(null);
      await load();
    } catch (err) {
      if (err instanceof ApiRequestError && err.code === "REVISION_CONFLICT") {
        await load();
      }
      setError(err);
    }
  }

  async function onCheck(id: string) {
    setBusy(true);
    setError(null);
    setCheckOut(null);
    try {
      const data = await api.post<ModelCheckResult>(`/api/v1/models/${id}/check`, { level: "port" });
      setCheckOut(data);
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>模型连接</h2>
      <p className="muted">
        请先在服务器启动 vLLM。连接可跨批次复用，端口由 Web 后端探测。
      </p>
      <ErrorBanner error={error} />
      {lastCreatedId && <p className="help">最近登记内部 ID（只读）：{lastCreatedId}</p>}

      <div className="panel">
        <h3>{editingId ? `修改 ${editingId}` : "新增连接"}</h3>
        <form className="form-grid" onSubmit={onSave}>
          <label>
            显示名称（可选）
            <input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="缺省为 provider 主机:端口"
            />
          </label>
          <label>
            provider
            <select value={form.provider} onChange={(e) => setForm({ ...form, provider: e.target.value })}>
              {providers.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
          <label>
            base_url
            <input
              value={form.base_url}
              onChange={(e) => setForm({ ...form, base_url: e.target.value })}
              required
              placeholder="http://127.0.0.1:PORT/v1"
            />
            <span className="help">完整 HTTP(S) 地址，含实际端口；同机用 127.0.0.1，不要用 0.0.0.0</span>
          </label>
          <label>
            model（served-model-name）
            <input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} required />
          </label>
          <label>
            model_revision
            <input
              value={form.model_revision}
              onChange={(e) => setForm({ ...form, model_revision: e.target.value })}
            />
          </label>
          <label>
            api_key_env（环境变量名）
            <input
              value={form.api_key_env}
              onChange={(e) => setForm({ ...form, api_key_env: e.target.value })}
              placeholder="不填密钥正文"
            />
          </label>
          <label>
            timeout_seconds
            <input
              type="number"
              value={form.timeout_seconds}
              onChange={(e) => setForm({ ...form, timeout_seconds: Number(e.target.value) })}
            />
          </label>
          <label>
            concurrency
            <input
              type="number"
              value={form.concurrency}
              onChange={(e) => setForm({ ...form, concurrency: Number(e.target.value) })}
            />
          </label>
          <div className="row">
            <button className="primary" type="submit">
              {editingId ? "保存修改" : "保存"}
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
          <h3>已配置</h3>
          <button type="button" onClick={() => void load()}>
            刷新
          </button>
        </div>
        {loading && <p className="muted">加载中…</p>}
        {!loading && items.length === 0 && <p className="muted">暂无连接。请先登记并自行启动对应 vLLM。</p>}
        {!loading && items.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>名称</th>
                <th>provider</th>
                <th>base_url</th>
                <th>model / revision</th>
                <th>密钥</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {items.map((m) => (
                <tr key={m.connection_id}>
                  <td>
                    {m.name}
                    <div className="help">id={m.connection_id}</div>
                  </td>
                  <td>{m.provider}</td>
                  <td>{m.base_url}</td>
                  <td>
                    {m.model}
                    <div className="help">{m.model_revision}</div>
                  </td>
                  <td>
                    {m.api_key_env || "—"}
                    <div className="help">{m.api_key_configured ? "引用可解析" : "未配置/不可解析"}</div>
                  </td>
                  <td className="row">
                    <button type="button" disabled={busy} onClick={() => void onCheck(m.connection_id)}>
                      检查端口
                    </button>
                    <button type="button" onClick={() => fillEdit(m)}>
                      修改
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {checkOut && (
          <div style={{ marginTop: "1rem" }}>
            <div className={checkOut.ok ? "ok" : "warn"}>
              {checkOut.ok ? "端口可连接" : "端口不可连接"} · {checkOut.base_url} ·{" "}
              {formatDateTime(checkOut.checked_at)}
              {checkOut.latency_ms != null ? ` · ${checkOut.latency_ms}ms` : ""}
            </div>
            {checkOut.message && <div className="help">{checkOut.message}</div>}
            <pre className="logbox">{JSON.stringify(checkOut, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
