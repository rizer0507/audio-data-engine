import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ApiRequestError, api } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import PluginDetailDialog from "../components/PluginDetailDialog";
import type { PluginDescriptor } from "../types";

function statusLabel(status: string): string {
  if (status === "enabled") return "在用";
  if (status === "archived") return "归档";
  if (status === "disabled") return "未启用";
  return status;
}

export default function Plugins() {
  const [params, setParams] = useSearchParams();
  const kind = (params.get("kind") === "tag" ? "tag" : "pipeline") as "pipeline" | "tag";
  const statusFilter = params.get("status") || "";
  const [items, setItems] = useState<PluginDescriptor[]>([]);
  const [generation, setGeneration] = useState<number | null>(null);
  const [dialog, setDialog] = useState<{
    kind: "pipeline" | "tag";
    id: string;
    version: string;
  } | null>(null);
  const [dialogTick, setDialogTick] = useState(0);
  const [q, setQ] = useState(params.get("q") || "");
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [statusBusy, setStatusBusy] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const qs = new URLSearchParams({ kind });
      if (statusFilter) qs.set("status", statusFilter);
      const data = await api.get<{ items: PluginDescriptor[]; generation: number; revision?: number }>(
        `/api/v1/plugins?${qs.toString()}`,
      );
      setItems(data.items || []);
      setGeneration(data.generation ?? data.revision ?? null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, [kind, statusFilter]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return items;
    return items.filter(
      (i) =>
        i.id.toLowerCase().includes(s) ||
        i.name.toLowerCase().includes(s) ||
        i.adapter.toLowerCase().includes(s) ||
        (i.capabilities?.purpose || "").toLowerCase().includes(s),
    );
  }, [items, q]);

  async function refresh() {
    setRefreshing(true);
    setError(null);
    try {
      const res = await api.post<{ generation?: number }>("/api/v1/plugins/refresh");
      if (res.generation != null) setGeneration(res.generation);
      await load();
    } catch (e) {
      setError(e);
    } finally {
      setRefreshing(false);
    }
  }

  function openDetail(item: PluginDescriptor) {
    setDialog({ kind: item.kind, id: item.id, version: item.version });
  }

  async function setStatus(item: PluginDescriptor, next: "enabled" | "archived") {
    const key = `${item.kind}:${item.id}`;
    setStatusBusy(key);
    setError(null);
    try {
      const res = await api.patch<{
        descriptor: PluginDescriptor;
        revision: number;
        generation: number;
      }>(
        `/api/v1/plugins/${item.kind}/${item.id}/status`,
        { status: next, expected_revision: generation },
        generation != null ? { "If-Match": String(generation) } : undefined,
      );
      setGeneration(res.generation ?? res.revision);
      setItems((prev) =>
        prev.map((p) =>
          p.kind === item.kind && p.id === item.id && p.version === item.version
            ? { ...p, ...res.descriptor, summary: p.summary, editable: p.editable, edit_schema: p.edit_schema }
            : p,
        ),
      );
      if (dialog && dialog.kind === item.kind && dialog.id === item.id) {
        setDialogTick((t) => t + 1);
      }
      if (statusFilter && statusFilter !== next) {
        await load();
      }
    } catch (e) {
      if (e instanceof ApiRequestError && e.code === "REVISION_CONFLICT") {
        await load();
      }
      setError(e);
    } finally {
      setStatusBusy(null);
    }
  }

  function statusActions(item: PluginDescriptor) {
    const key = `${item.kind}:${item.id}`;
    const busy = statusBusy === key;
    if (item.status === "enabled") {
      return (
        <button
          type="button"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            void setStatus(item, "archived");
          }}
        >
          {busy ? "处理中…" : "归档"}
        </button>
      );
    }
    if (item.status === "archived") {
      return (
        <button
          type="button"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            void setStatus(item, "enabled");
          }}
        >
          {busy ? "处理中…" : "启用"}
        </button>
      );
    }
    return (
      <span className="row">
        <button
          type="button"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            void setStatus(item, "enabled");
          }}
        >
          启用
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            void setStatus(item, "archived");
          }}
        >
          归档
        </button>
      </span>
    );
  }

  function setKind(next: "pipeline" | "tag") {
    const p = new URLSearchParams(params);
    p.set("kind", next);
    setParams(p);
    setDialog(null);
  }

  return (
    <div>
      <h2>插件库</h2>
      <p className="muted">
        单击名称或详情打开解释弹窗（只改说明，不改规则）；启用/归档写回原配置。查看与编辑不会启动 ASR/DNSMOS。generation=
        {generation ?? "—"}
      </p>
      <ErrorBanner error={error} />

      <div className="panel">
        <div className="row tabs">
          <button type="button" className={kind === "pipeline" ? "active" : ""} onClick={() => setKind("pipeline")}>
            流水线
          </button>
          <button type="button" className={kind === "tag" ? "active" : ""} onClick={() => setKind("tag")}>
            Tag
          </button>
          <input
            style={{ maxWidth: 240 }}
            placeholder="搜索 id/名称/adapter"
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              const p = new URLSearchParams(params);
              if (e.target.value) p.set("q", e.target.value);
              else p.delete("q");
              setParams(p, { replace: true });
            }}
          />
          <select
            value={statusFilter}
            onChange={(e) => {
              const p = new URLSearchParams(params);
              if (e.target.value) p.set("status", e.target.value);
              else p.delete("status");
              setParams(p);
            }}
            style={{ maxWidth: 180 }}
          >
            <option value="">全部</option>
            <option value="enabled">在用（enabled）</option>
            <option value="archived">归档（archived）</option>
          </select>
          <button type="button" disabled={refreshing} onClick={() => void refresh()}>
            {refreshing ? "刷新中…" : "刷新目录"}
          </button>
        </div>

        {loading && <p className="muted">加载中…</p>}
        {!loading && filtered.length === 0 && (
          <p className="muted">无插件。请检查 configs/plugins.yaml / tag 库后刷新目录。</p>
        )}
        {!loading && filtered.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>kind</th>
                <th>ID</th>
                <th>名称</th>
                <th>版本</th>
                <th>状态</th>
                <th>purpose</th>
                <th>adapter</th>
                <th>可执行</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <tr key={`${p.kind}:${p.id}@${p.version}`}>
                  <td>{p.kind}</td>
                  <td>{p.id}</td>
                  <td>
                    <button type="button" className="linkish" onClick={() => openDetail(p)}>
                      {p.name}
                    </button>
                  </td>
                  <td>{p.version}</td>
                  <td>{statusLabel(p.status)}</td>
                  <td>{p.capabilities?.purpose}</td>
                  <td>{p.adapter}</td>
                  <td>{p.runnable ? "是" : (p.not_runnable_reasons || []).join("; ") || "否"}</td>
                  <td className="row">
                    {statusActions(p)}
                    <button type="button" onClick={() => openDetail(p)}>
                      详情
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <PluginDetailDialog
        open={dialog != null}
        kind={dialog?.kind || kind}
        pluginId={dialog?.id || ""}
        version={dialog?.version || "1"}
        listRevision={generation}
        reloadToken={dialogTick}
        onClose={() => setDialog(null)}
        onSaved={(descriptor, revision) => {
          setGeneration(revision);
          setItems((prev) => {
            const next = prev.map((p) => {
              if (p.kind === descriptor.kind && p.id === descriptor.id) {
                return descriptor;
              }
              return p;
            });
            // Version bump may change key — ensure present
            if (!next.some((p) => p.kind === descriptor.kind && p.id === descriptor.id && p.version === descriptor.version)) {
              return [...next.filter((p) => !(p.kind === descriptor.kind && p.id === descriptor.id)), descriptor];
            }
            return next;
          });
          setDialog({ kind: descriptor.kind, id: descriptor.id, version: descriptor.version });
        }}
      />
    </div>
  );
}
