import { useEffect, useId, useRef, useState } from "react";
import { ApiRequestError, api } from "../api";
import type { PluginDescriptor } from "../types";

function formatUpdatedAt(iso: string | null | undefined): string {
  if (!iso || !String(iso).trim()) return "尚未记录";
  const d = new Date(String(iso));
  if (Number.isNaN(d.getTime())) return "尚未记录";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export default function PluginDetailDialog({
  open,
  kind,
  pluginId,
  version,
  listRevision,
  reloadToken = 0,
  onClose,
  onSaved,
}: {
  open: boolean;
  kind: "pipeline" | "tag";
  pluginId: string;
  version: string;
  listRevision: number | null;
  reloadToken?: number;
  onClose: () => void;
  onSaved: (descriptor: PluginDescriptor, revision: number) => void;
}) {
  const titleId = useId();
  const closeBtnRef = useRef<HTMLButtonElement>(null);
  const reqSeq = useRef(0);
  const [detail, setDetail] = useState<PluginDescriptor | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [editing, setEditing] = useState(false);
  const [draftText, setDraftText] = useState("");
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saveMsg, setSaveMsg] = useState<string | null>(null);

  function confirmDiscard(): boolean {
    if (!dirty) return true;
    return window.confirm("有未保存的修改，确定放弃吗？");
  }

  async function load(targetKind: string, id: string, ver: string) {
    const seq = ++reqSeq.current;
    setLoading(true);
    setError(null);
    setSaveMsg(null);
    try {
      const data = await api.get<PluginDescriptor>(
        `/api/v1/plugins/${targetKind}/${id}?version=${encodeURIComponent(ver)}`,
      );
      if (seq !== reqSeq.current) return;
      setDetail(data);
      setDraftText(String(data.description || ""));
      setEditing(false);
      setDirty(false);
    } catch (e) {
      if (seq !== reqSeq.current) return;
      setError(e);
      setDetail(null);
    } finally {
      if (seq === reqSeq.current) setLoading(false);
    }
  }

  useEffect(() => {
    if (!open) return;
    void load(kind, pluginId, version);
  }, [open, kind, pluginId, version, reloadToken]);

  useEffect(() => {
    if (!open) return;
    closeBtnRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        requestClose();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, dirty]);

  function requestClose() {
    if (!confirmDiscard()) return;
    onClose();
  }

  async function runSave() {
    if (!detail) return;
    const text = draftText.trim();
    if (!text) {
      setSaveMsg("解释不能为空");
      return;
    }
    setBusy(true);
    setError(null);
    setSaveMsg(null);
    try {
      const res = await api.patch<{
        descriptor: PluginDescriptor;
        revision: number;
        generation: number;
      }>(
        `/api/v1/plugins/${detail.kind}/${detail.id}/definition?version=${encodeURIComponent(detail.version)}`,
        {
          expected_revision: listRevision ?? detail.revision ?? detail.catalog_generation,
          expected_definition_digest: detail.definition_digest,
          draft: { description: text },
        },
        listRevision != null ? { "If-Match": String(listRevision) } : undefined,
      );
      setDetail(res.descriptor);
      setDraftText(String(res.descriptor.description || ""));
      setEditing(false);
      setDirty(false);
      setSaveMsg(null);
      onSaved(res.descriptor, res.generation ?? res.revision);
    } catch (e) {
      if (e instanceof ApiRequestError && e.code.includes("CONFLICT")) {
        setSaveMsg("保存冲突：请关闭后重新打开再试");
      } else {
        setSaveMsg("保存失败，请稍后重试");
      }
      setError(e);
    } finally {
      setBusy(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) requestClose();
      }}
    >
      <div
        className="modal-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="modal-header row" style={{ justifyContent: "space-between" }}>
          <h3 id={titleId} style={{ margin: 0 }}>
            {detail?.name || pluginId}
          </h3>
          <button type="button" ref={closeBtnRef} onClick={requestClose} aria-label="关闭">
            关闭
          </button>
        </div>

        {loading && <p className="muted">加载中…</p>}
        {error != null && !editing && (
          <div className="error" style={{ marginBottom: "0.75rem" }}>
            {error instanceof ApiRequestError ? error.diagnosticText() : String(error)}
            <div className="row" style={{ marginTop: "0.5rem" }}>
              <button type="button" onClick={() => void load(kind, pluginId, version)}>
                重试
              </button>
            </div>
          </div>
        )}

        {!loading && detail && (
          <div className="modal-body">
            {!editing ? (
              <>
                <div className="plugin-summary" style={{ whiteSpace: "pre-wrap" }}>
                  {detail.description?.trim() || "（暂无解释）"}
                </div>
                <p className="muted" style={{ marginTop: "1rem" }}>
                  最近修改：{formatUpdatedAt(detail.description_updated_at)}
                </p>
              </>
            ) : (
              <div className="form-grid">
                <label>
                  解释
                  <textarea
                    rows={8}
                    value={draftText}
                    onChange={(e) => {
                      setDraftText(e.target.value);
                      setDirty(true);
                      setSaveMsg(null);
                    }}
                  />
                </label>
                <p className="muted">最近修改：{formatUpdatedAt(detail.description_updated_at)}</p>
                {saveMsg && <p className="error">{saveMsg}</p>}
                {error != null && (
                  <div className="error">
                    {error instanceof ApiRequestError ? error.diagnosticText() : String(error)}
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        <div className="modal-footer row" style={{ justifyContent: "flex-end" }}>
          <div className="row">
            {!editing && detail && (
              <button
                type="button"
                className="primary"
                onClick={() => {
                  setEditing(true);
                  setDraftText(String(detail.description || ""));
                  setDirty(false);
                  setSaveMsg(null);
                  setError(null);
                }}
              >
                编辑
              </button>
            )}
            {editing && (
              <>
                <button
                  type="button"
                  className="primary"
                  disabled={busy}
                  onClick={() => void runSave()}
                >
                  {busy ? "保存中…" : "保存"}
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    if (!confirmDiscard()) return;
                    setEditing(false);
                    setDraftText(String(detail?.description || ""));
                    setDirty(false);
                    setSaveMsg(null);
                    setError(null);
                  }}
                >
                  取消
                </button>
              </>
            )}
            {!editing && (
              <button type="button" onClick={requestClose}>
                关闭
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
