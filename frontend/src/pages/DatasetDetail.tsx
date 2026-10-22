import { useEffect, useState } from "react";
import { Link, useLocation, useSearchParams } from "react-router-dom";
import { ApiRequestError, api, audioUrl, newIdempotencyKey } from "../api";
import AudioPlayer from "../components/AudioPlayer";
import ErrorBanner from "../components/ErrorBanner";
import SampleTable from "../components/SampleTable";
import type { SampleRow } from "../types";

export default function DatasetDetail() {
  const location = useLocation();
  const [searchParams, setSearchParams] = useSearchParams();
  const datasetId = decodeURIComponent(location.pathname.replace(/^\/datasets\//, ""));

  const offset = Number(searchParams.get("offset") || "0") || 0;
  const tag = searchParams.get("tag") || "";
  const tagExclude = searchParams.get("tag_exclude") || "";
  const tagAny = searchParams.get("tag_any") || "";
  const ternary = searchParams.get("ternary") || "";
  const routeStatus = searchParams.get("route_status") || "";
  const reviewStatus = searchParams.get("review_status") || "";
  const qualityMin = searchParams.get("quality_min") || "";
  const qualityMax = searchParams.get("quality_max") || "";

  const [meta, setMeta] = useState<Record<string, unknown> | null>(null);
  const [samples, setSamples] = useState<SampleRow[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [frozenFilters, setFrozenFilters] = useState<Record<string, unknown>>({});
  const [selected, setSelected] = useState<Record<string, unknown> | null>(null);
  const [explain, setExplain] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [audioAsset, setAudioAsset] = useState("raw");
  const [exportMsg, setExportMsg] = useState("");
  const [actionMsg, setActionMsg] = useState("");

  const [tagId, setTagId] = useState("");
  const [tagAction, setTagAction] = useState("manual_add");
  const [tagReason, setTagReason] = useState("");
  const [tagRevision, setTagRevision] = useState("");
  const [workRoot, setWorkRoot] = useState("");
  const [batchId, setBatchId] = useState("");

  const [goldKind, setGoldKind] = useState("speech");
  const [goldText, setGoldText] = useState("");
  const [speechScope, setSpeechScope] = useState("target");
  const [humanSemantic, setHumanSemantic] = useState("neutral");
  const [humanNoise, setHumanNoise] = useState("clean");
  const [humanCrosstalk, setHumanCrosstalk] = useState("false");
  const [reviewPass, setReviewPass] = useState("first");
  const [reviewRevision, setReviewRevision] = useState("web_r1");
  const [reviewReason, setReviewReason] = useState("");

  useEffect(() => {
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const m = await api.get<Record<string, unknown>>(
          `/api/v1/datasets/${encodeURIComponent(datasetId)}`,
        );
        setMeta(m);
        const binding = (m.tagging_binding || {}) as Record<string, unknown>;
        if (binding.work_root) setWorkRoot(String(binding.work_root));
        if (binding.batch_id) setBatchId(String(binding.batch_id));
        const qs = new URLSearchParams({
          limit: "50",
          offset: String(offset),
        });
        if (tag) qs.set("tag", tag);
        if (tagExclude) qs.set("tag_exclude", tagExclude);
        if (tagAny) qs.set("tag_any", tagAny);
        if (ternary) qs.set("ternary", ternary);
        if (routeStatus) qs.set("route_status", routeStatus);
        if (reviewStatus) qs.set("review_status", reviewStatus);
        if (qualityMin) qs.set("quality_min", qualityMin);
        if (qualityMax) qs.set("quality_max", qualityMax);
        const s = await api.get<{
          items: SampleRow[];
          total?: number | null;
          filters?: Record<string, unknown>;
        }>(`/api/v1/datasets/${encodeURIComponent(datasetId)}/samples?${qs.toString()}`);
        setSamples(s.items || []);
        setTotal(s.total ?? null);
        setFrozenFilters(s.filters || {});
      } catch (e) {
        setError(e);
      } finally {
        setLoading(false);
      }
    })();
  }, [
    datasetId,
    offset,
    tag,
    tagExclude,
    tagAny,
    ternary,
    routeStatus,
    reviewStatus,
    qualityMin,
    qualityMax,
  ]);

  function setFilter(key: string, value: string) {
    const p = new URLSearchParams(searchParams);
    if (value) p.set(key, value);
    else p.delete(key);
    p.set("offset", "0");
    setSearchParams(p);
  }

  async function selectSample(id: string) {
    setError(null);
    setExplain(null);
    setActionMsg("");
    try {
      const data = await api.get<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/samples/${encodeURIComponent(id)}`,
      );
      setSelected(data);
      if (data.tag_revision != null) setTagRevision(String(data.tag_revision));
      const labels = (data.labels || {}) as Record<string, unknown>;
      if (labels.annotation_revision) setReviewRevision(String(labels.annotation_revision));
      if (labels.gold_text != null) setGoldText(String(labels.gold_text));
      if (labels.gold_kind) setGoldKind(String(labels.gold_kind));
    } catch (e) {
      setError(e);
    }
  }

  async function loadExplain(id: string) {
    setError(null);
    try {
      const data = await api.get<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/samples/${encodeURIComponent(id)}/explain`,
      );
      setExplain(data);
    } catch (e) {
      setError(e);
    }
  }

  async function submitTagAction(sampleId: string) {
    setActionMsg("");
    setError(null);
    try {
      const res = await api.post<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/samples/${encodeURIComponent(sampleId)}/tag-actions`,
        {
          action: tagAction,
          tag_id: tagId,
          reason: tagReason || "web_manual",
          expected_revision: tagRevision || undefined,
          idempotency_id: newIdempotencyKey(),
          work_root: workRoot || undefined,
          batch_id: batchId || undefined,
        },
      );
      setActionMsg(
        res.ok
          ? `人工标签已写入工作区 revision=${String(res.revision ?? "")}；已发布 Manifest 未覆盖。`
          : JSON.stringify(res),
      );
      if (res.revision != null) setTagRevision(String(res.revision));
      await selectSample(sampleId);
    } catch (e) {
      if (e instanceof ApiRequestError && e.code === "REVISION_CONFLICT") {
        setActionMsg("revision 冲突：请刷新样本后重试，编辑内容已保留。");
      }
      setError(e);
    }
  }

  async function submitReview(sampleId: string) {
    setActionMsg("");
    setError(null);
    try {
      const res = await api.post<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/samples/${encodeURIComponent(sampleId)}/review`,
        {
          review_pass: reviewPass,
          decision: "accept",
          gold_kind: goldKind,
          gold_text: goldKind === "non_speech" ? "" : goldText,
          speech_scope: speechScope,
          human_semantic: humanSemantic,
          human_noise: humanNoise,
          human_crosstalk: humanCrosstalk,
          reason: reviewReason || "web_review",
          expected_revision: reviewRevision,
          idempotency_id: newIdempotencyKey(),
          audio_event_tags: goldKind === "non_speech" ? ["environment_noise"] : [],
        },
      );
      setActionMsg(
        `审核写入工作副本 ${String(res.work_dataset_id || "")}；` +
          `published 不变=${String(res.immutable_published_unchanged)}；` +
          `formal_gold=${String(res.has_formal_gold)}；state=${String(res.annotation_state)}`,
      );
      if (res.annotation_revision) setReviewRevision(String(res.annotation_revision));
    } catch (e) {
      if (e instanceof ApiRequestError && (e.code === "REVISION_CONFLICT" || e.code === "REVIEW_REJECTED")) {
        setActionMsg(`审核被拒绝/冲突：${e.message}（正式 Gold 门禁不可绕过）`);
      }
      setError(e);
    }
  }

  async function tryExport() {
    setExportMsg("");
    setError(null);
    try {
      const res = await api.post<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/exports`,
        {
          export_plugin_id: "warehouse_export_annotation",
          filters: frozenFilters,
          parameters: {
            use_work_copy: Boolean(meta?.has_work_copy),
          },
          idempotency_key: newIdempotencyKey(),
        },
      );
      if (res.job_id) {
        setExportMsg(
          `已创建导出任务 ${String(res.job_id)}；冻结筛选=${JSON.stringify(res.frozen_filters || frozenFilters)}；` +
            `数据集=${String(res.frozen_dataset_id || datasetId)}`,
        );
      } else {
        setExportMsg(JSON.stringify(res));
      }
    } catch (e) {
      setError(e);
    }
  }

  async function runGoldGate() {
    setActionMsg("");
    try {
      const res = await api.post<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/gold-gate`,
        { require_formal_gold: true },
      );
      setActionMsg(
        `Gold 门禁 formal_ok=${String(res.formal_gate_ok)}；` +
          `分母=${JSON.stringify(res.denominator)}；errors=${JSON.stringify(res.errors || [])}`,
      );
    } catch (e) {
      setError(e);
    }
  }

  const sampleId = String(selected?.sample_id || selected?.id || "");
  const pageSize = 50;
  const stateKind = String(meta?.state_kind || "published");

  return (
    <div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>数据集详情</h2>
        <Link to="/datasets">返回列表</Link>
      </div>
      <p className="muted">{datasetId}</p>
      <ErrorBanner error={error} />

      <div className="panel">
        <h3>元数据 / 版本状态</h3>
        {loading && !meta ? <p className="muted">加载中…</p> : null}
        <p>
          状态标记：
          <strong>{stateKind === "working" ? "人工工作副本" : "已发布/只读来源"}</strong>
          {meta?.has_work_copy ? (
            <>
              {" "}
              · 工作副本：
              <Link to={`/datasets/${encodeURIComponent(String(meta.work_dataset_id))}`}>
                {String(meta.work_dataset_id)}
              </Link>
            </>
          ) : null}
        </p>
        <pre className="logbox">{JSON.stringify(meta || {}, null, 2)}</pre>
      </div>

      <div className="panel">
        <h3>样本（服务端分页 + 筛选）</h3>
        <div className="form-grid" style={{ marginBottom: "0.75rem" }}>
          <label>
            tag 包含
            <input value={tag} onChange={(e) => setFilter("tag", e.target.value)} placeholder="tag_id" />
          </label>
          <label>
            tag 排除
            <input
              value={tagExclude}
              onChange={(e) => setFilter("tag_exclude", e.target.value)}
              placeholder="tag_id"
            />
          </label>
          <label>
            tag any（逗号）
            <input value={tagAny} onChange={(e) => setFilter("tag_any", e.target.value)} />
          </label>
          <label>
            三态
            <select value={ternary} onChange={(e) => setFilter("ternary", e.target.value)}>
              <option value="">全部</option>
              <option value="matched">matched</option>
              <option value="not_matched">not_matched</option>
              <option value="unknown">unknown</option>
            </select>
          </label>
          <label>
            路次状态
            <input
              value={routeStatus}
              onChange={(e) => setFilter("route_status", e.target.value)}
              placeholder="operator=completed"
            />
          </label>
          <label>
            审核状态
            <input
              value={reviewStatus}
              onChange={(e) => setFilter("review_status", e.target.value)}
              placeholder="annotation_state"
            />
          </label>
          <label>
            质量下限 dnsmos_sig
            <input
              value={qualityMin}
              onChange={(e) => setFilter("quality_min", e.target.value)}
              placeholder="未计算不显示为 0"
            />
          </label>
          <label>
            质量上限
            <input value={qualityMax} onChange={(e) => setFilter("quality_max", e.target.value)} />
          </label>
        </div>
        <p className="help">
          total={total ?? "—"} · offset={offset} · 冻结筛选={JSON.stringify(frozenFilters)}
        </p>

        <SampleTable
          items={samples}
          selectedId={sampleId || null}
          onSelect={(id) => void selectSample(id)}
        />
        <div className="row">
          <button
            type="button"
            disabled={offset === 0}
            onClick={() => {
              const p = new URLSearchParams(searchParams);
              p.set("offset", String(Math.max(0, offset - pageSize)));
              setSearchParams(p);
            }}
          >
            上一页
          </button>
          <button
            type="button"
            disabled={total != null ? offset + pageSize >= total : samples.length < pageSize}
            onClick={() => {
              const p = new URLSearchParams(searchParams);
              p.set("offset", String(offset + pageSize));
              setSearchParams(p);
            }}
          >
            下一页
          </button>
        </div>
      </div>

      {selected && (
        <div className="panel">
          <h3>样本 {sampleId}</h3>
          <div className="row" style={{ marginBottom: "0.5rem" }}>
            <label className="row">
              音频 asset
              <select value={audioAsset} onChange={(e) => setAudioAsset(e.target.value)}>
                <option value="raw">raw</option>
                <option value="resampled_16k">resampled_16k</option>
              </select>
            </label>
            <button type="button" onClick={() => void loadExplain(sampleId)}>
              规则解释（只读，不推理）
            </button>
            <Link to={`/jobs?dataset_id=${encodeURIComponent(datasetId)}`}>新建打标任务</Link>
          </div>
          <AudioPlayer
            src={audioUrl(datasetId, sampleId, audioAsset)}
            label="经后端资产接口试听（HTTP Range / 浏览器拖动）"
          />
          {actionMsg && <p className="help">{actionMsg}</p>}
          {explain && (
            <>
              <h4>解释</h4>
              {(explain.rules_may_differ_from_run || explain.snapshot_id) && (
                <p className="warn">{String(explain.rules_differ_note || "当前库规则可能已变更")}</p>
              )}
              <pre className="logbox">{JSON.stringify(explain, null, 2)}</pre>
            </>
          )}

          <h4>人工标签（工作区，不覆盖已发布）</h4>
          <div className="form-grid">
            <label>
              work_root
              <input value={workRoot} onChange={(e) => setWorkRoot(e.target.value)} placeholder="tagging 工作区" />
            </label>
            <label>
              batch_id
              <input value={batchId} onChange={(e) => setBatchId(e.target.value)} />
            </label>
            <label>
              action
              <select value={tagAction} onChange={(e) => setTagAction(e.target.value)}>
                <option value="manual_add">manual_add</option>
                <option value="manual_block">manual_block</option>
                <option value="restore_auto">restore_auto</option>
                <option value="retract_manual">retract_manual</option>
              </select>
            </label>
            <label>
              tag_id
              <input value={tagId} onChange={(e) => setTagId(e.target.value)} />
            </label>
            <label>
              expected_revision
              <input value={tagRevision} onChange={(e) => setTagRevision(e.target.value)} />
            </label>
            <label>
              reason
              <input value={tagReason} onChange={(e) => setTagReason(e.target.value)} />
            </label>
          </div>
          <button type="button" className="primary" onClick={() => void submitTagAction(sampleId)}>
            提交标签操作
          </button>

          <h4 style={{ marginTop: "1rem" }}>审核 / Gold（写入工作副本）</h4>
          <p className="muted">
            遵守 annotation_v3：speech 需金标文本与人审字段；一次操作不能伪装双人审核；正式 Gold
            不通过时会返回具体缺项。
          </p>
          <div className="form-grid">
            <label>
              review_pass
              <select value={reviewPass} onChange={(e) => setReviewPass(e.target.value)}>
                <option value="first">first</option>
                <option value="second">second</option>
                <option value="adjudication">adjudication</option>
              </select>
            </label>
            <label>
              gold_kind
              <select value={goldKind} onChange={(e) => setGoldKind(e.target.value)}>
                <option value="speech">speech</option>
                <option value="non_speech">non_speech</option>
                <option value="unintelligible">unintelligible</option>
              </select>
            </label>
            <label>
              gold_text
              <input
                value={goldText}
                onChange={(e) => setGoldText(e.target.value)}
                disabled={goldKind === "non_speech"}
                placeholder={goldKind === "non_speech" ? "确认为空串" : "金标文本"}
              />
            </label>
            <label>
              speech_scope
              <select value={speechScope} onChange={(e) => setSpeechScope(e.target.value)}>
                <option value="target">target</option>
                <option value="none">none</option>
                <option value="background_only">background_only</option>
                <option value="mixed">mixed</option>
                <option value="unknown">unknown</option>
              </select>
            </label>
            <label>
              human_semantic
              <select value={humanSemantic} onChange={(e) => setHumanSemantic(e.target.value)}>
                <option value="neutral">neutral</option>
                <option value="positive">positive</option>
                <option value="negative">negative</option>
                <option value="mixed">mixed</option>
                <option value="unknown">unknown</option>
                <option value="not_applicable">not_applicable</option>
              </select>
            </label>
            <label>
              human_noise
              <select value={humanNoise} onChange={(e) => setHumanNoise(e.target.value)}>
                <option value="clean">clean</option>
                <option value="moderate">moderate</option>
                <option value="noisy">noisy</option>
                <option value="unknown">unknown</option>
              </select>
            </label>
            <label>
              human_crosstalk
              <select value={humanCrosstalk} onChange={(e) => setHumanCrosstalk(e.target.value)}>
                <option value="false">false</option>
                <option value="true">true</option>
                <option value="unknown">unknown</option>
              </select>
            </label>
            <label>
              expected_revision
              <input value={reviewRevision} onChange={(e) => setReviewRevision(e.target.value)} />
            </label>
            <label>
              reason
              <input value={reviewReason} onChange={(e) => setReviewReason(e.target.value)} />
            </label>
          </div>
          <div className="row">
            <button type="button" className="primary" onClick={() => void submitReview(sampleId)}>
              提交审核
            </button>
            <button type="button" onClick={() => void runGoldGate()}>
              检查正式 Gold 门禁
            </button>
          </div>

          <h4>样本字段</h4>
          <pre className="logbox">{JSON.stringify(selected, null, 2)}</pre>
        </div>
      )}

      <div className="panel">
        <h3>导出</h3>
        <p className="muted">
          经统一插件任务链执行；当前列表筛选会冻结进导出请求，不随继续浏览而变。完成后在任务详情下载已登记
          artifact。
        </p>
        <div className="row">
          <button type="button" onClick={() => void tryExport()}>
            按当前筛选创建导出任务
          </button>
        </div>
        {exportMsg && <p className="help">{exportMsg}</p>}
      </div>
    </div>
  );
}
