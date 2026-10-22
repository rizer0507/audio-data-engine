import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ApiRequestError, api, formatDateTime, newIdempotencyKey } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import SchemaForm, { defaultsFromSchema } from "../components/SchemaForm";
import TaskStatus from "../components/TaskStatus";
import type {
  DatasetSummary,
  ExecutionPlan,
  Job,
  ModelBinding,
  ModelConnection,
  PluginDescriptor,
  Source,
  TaskRequest,
} from "../types";

export default function Jobs() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const statusFilter = params.get("status") || "";
  const [jobs, setJobs] = useState<Job[]>([]);
  const [pipelines, setPipelines] = useState<PluginDescriptor[]>([]);
  const [tags, setTags] = useState<PluginDescriptor[]>([]);
  const [models, setModels] = useState<ModelConnection[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [evidenceModes, setEvidenceModes] = useState<string[]>([
    "existing",
    "compute_acoustic",
    "plan",
    "preview",
  ]);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [planning, setPlanning] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const [taskType, setTaskType] = useState<"pipeline" | "tagging">("pipeline");
  const [pluginId, setPluginId] = useState("");
  const [pluginVersion, setPluginVersion] = useState("1");
  const [selectedPlugin, setSelectedPlugin] = useState<PluginDescriptor | null>(null);
  const [sourceId, setSourceId] = useState(params.get("source_id") || "");
  const [datasetId, setDatasetId] = useState(params.get("dataset_id") || "");
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [bindings, setBindings] = useState<ModelBinding[]>([]);
  const [evidenceMode, setEvidenceMode] = useState("existing");
  const [plan, setPlan] = useState<ExecutionPlan | null>(null);
  const [planStale, setPlanStale] = useState(false);
  const [idemKey, setIdemKey] = useState(() => newIdempotencyKey());
  const [catalogGen, setCatalogGen] = useState<number | null>(null);

  async function loadMeta() {
    setLoading(true);
    setError(null);
    try {
      const jobQs = statusFilter ? `?status=${encodeURIComponent(statusFilter)}` : "";
      const [caps, p, t, m, s, d, j] = await Promise.all([
        api.get<{ evidence_modes?: string[] }>("/api/v1/capabilities"),
        api.get<{ items: PluginDescriptor[]; generation?: number }>("/api/v1/plugins?kind=pipeline"),
        api.get<{ items: PluginDescriptor[] }>("/api/v1/plugins?kind=tag&status=enabled"),
        api.get<{ items: ModelConnection[] }>("/api/v1/models"),
        api.get<{ items: Source[] }>("/api/v1/sources"),
        api.get<{ items: DatasetSummary[] }>("/api/v1/datasets"),
        api.get<{ items: Job[] }>(`/api/v1/jobs${jobQs}`),
      ]);
      if (caps.evidence_modes?.length) setEvidenceModes(caps.evidence_modes);
      setPipelines((p.items || []).filter((x) => x.status === "enabled"));
      setCatalogGen(p.generation ?? null);
      setTags((t.items || []).filter((x) => x.status === "enabled" && x.runnable !== false));
      setModels(m.items || []);
      setSources(s.items || []);
      setDatasets(d.items || []);
      setJobs(j.items || []);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadMeta();
  }, [statusFilter]);

  useEffect(() => {
    setPlan(null);
    setPlanStale(false);
  }, [taskType, pluginId, sourceId, datasetId, parameters, selectedTags, bindings, evidenceMode]);

  useEffect(() => {
    if (!pluginId || taskType !== "pipeline") {
      setSelectedPlugin(null);
      return;
    }
    const found = pipelines.find((p) => p.id === pluginId);
    if (!found) return;
    setPluginVersion(found.version);
    void api
      .get<PluginDescriptor>(`/api/v1/plugins/pipeline/${found.id}?version=${encodeURIComponent(found.version)}`)
      .then((d) => {
        setSelectedPlugin(d);
        setParameters(defaultsFromSchema(d.parameters_schema || { properties: {} }));
        const routeIds = (d.detail?.asr_route_ids as string[] | undefined) || [];
        const modelRoles = (d.detail?.model_roles as string[] | undefined) || [];
        const routes = (routeIds.length ? routeIds : modelRoles).map((route_id) => ({
          route_id,
          connection_id: "",
        }));
        setBindings(routes);
      })
      .catch((e) => setError(e));
  }, [pluginId, pipelines, taskType]);

  const needsSource =
    taskType === "pipeline" &&
    Boolean(selectedPlugin?.input_kinds?.includes("source")) &&
    !selectedPlugin?.input_kinds?.includes("dataset");
  const needsDataset =
    taskType === "tagging" || Boolean(selectedPlugin?.input_kinds?.includes("dataset"));
  const canSelectTags = taskType === "tagging" || Boolean(selectedPlugin?.capabilities.supports_tag_selection);

  const requestBody: TaskRequest = useMemo(() => {
    const input =
      taskType === "tagging" || (selectedPlugin?.input_kinds?.includes("dataset") && !needsSource)
        ? { kind: "dataset" as const, dataset_id: datasetId }
        : { kind: "source" as const, source_id: sourceId };
    // If both source and dataset supported, prefer source when sourceId set
    const resolvedInput =
      selectedPlugin?.input_kinds?.includes("source") &&
      selectedPlugin?.input_kinds?.includes("dataset")
        ? sourceId
          ? { kind: "source" as const, source_id: sourceId }
          : { kind: "dataset" as const, dataset_id: datasetId }
        : input;

    return {
      task_type: taskType,
      root_plugin:
        taskType === "pipeline"
          ? { kind: "pipeline", id: pluginId, version: pluginVersion }
          : null,
      input: resolvedInput,
      parameters,
      model_bindings: bindings.filter((b) => b.route_id && b.connection_id),
      tags: selectedTags.map((id) => {
        const t = tags.find((x) => x.id === id);
        return { kind: "tag", id, version: t?.version || "1" };
      }),
      execution: { evidence_mode: evidenceMode },
    };
  }, [
    taskType,
    pluginId,
    pluginVersion,
    selectedPlugin,
    datasetId,
    sourceId,
    parameters,
    bindings,
    selectedTags,
    tags,
    evidenceMode,
    needsSource,
  ]);

  const planExpired = useMemo(() => {
    if (!plan?.expires_at) return false;
    return new Date(plan.expires_at).getTime() < Date.now();
  }, [plan]);

  const canSubmit = Boolean(plan?.plan_id && plan?.plan_digest) && !planStale && !planExpired;

  async function onPlan() {
    setPlanning(true);
    setError(null);
    setPlan(null);
    setPlanStale(false);
    try {
      const data = await api.post<ExecutionPlan>("/api/v1/plugins/plan", requestBody);
      setPlan(data);
      if (data.catalog_generation != null) setCatalogGen(data.catalog_generation);
    } catch (e) {
      setError(e);
    } finally {
      setPlanning(false);
    }
  }

  async function onSubmit() {
    if (!plan?.plan_id || !plan?.plan_digest || !canSubmit) return;
    setSubmitting(true);
    setError(null);
    try {
      const job = await api.post<Job>(
        "/api/v1/jobs",
        { plan_id: plan.plan_id, plan_digest: plan.plan_digest },
        { "Idempotency-Key": idemKey },
      );
      setIdemKey(newIdempotencyKey());
      setPlan(null);
      navigate(`/jobs/${job.job_id}`);
    } catch (e) {
      if (e instanceof ApiRequestError) {
        if (e.code === "CATALOG_DRIFT" || e.code === "PLAN_EXPIRED") {
          setPlanStale(true);
          setPlan(null);
          await loadMeta();
          setError(e);
          return;
        }
      }
      // 失败保留表单与同一幂等键
      setError(e);
    } finally {
      setSubmitting(false);
    }
  }

  function markStale() {
    if (plan) setPlanStale(true);
  }

  return (
    <div>
      <h2>任务</h2>
      <p className="muted">
        表单由插件目录与 JSON Schema 生成；规划不执行推理。修改字段后旧计划失效。catalog generation=
        {catalogGen ?? "—"}
      </p>
      <ErrorBanner error={error} />

      <div className="panel">
        <h3>新建任务</h3>
        <div className="form-grid">
          <label>
            任务类型
            <select
              value={taskType}
              onChange={(e) => {
                setTaskType(e.target.value as "pipeline" | "tagging");
                setPluginId("");
                markStale();
              }}
            >
              <option value="pipeline">运行流水线</option>
              <option value="tagging">对已有数据集执行选定 tags</option>
            </select>
          </label>

          {taskType === "pipeline" && (
            <label>
              流水线插件（当前可用版本）
              <select
                value={pluginId}
                onChange={(e) => {
                  setPluginId(e.target.value);
                  markStale();
                }}
              >
                <option value="">选择…</option>
                {pipelines.map((p) => (
                  <option key={p.id} value={p.id} disabled={!p.runnable}>
                    {p.name} ({p.id}@{p.version}){p.runnable ? "" : " [不可执行]"}
                  </option>
                ))}
              </select>
            </label>
          )}

          {(needsSource ||
            (selectedPlugin?.input_kinds?.includes("source") &&
              selectedPlugin?.input_kinds?.includes("dataset"))) && (
            <label>
              数据源（batch / 路径）
              <select
                value={sourceId}
                onChange={(e) => {
                  setSourceId(e.target.value);
                  markStale();
                }}
              >
                <option value="">选择…</option>
                {sources.map((s) => (
                  <option key={s.source_id} value={s.source_id}>
                    {s.batch || s.source_name || s.source_id} — {s.path}
                  </option>
                ))}
              </select>
              {sources.length === 0 && <span className="help">请先在「数据源」登记</span>}
              <span className="help">选择后自动绑定内部 source_id，无需手填</span>
            </label>
          )}

          {(needsDataset ||
            (selectedPlugin?.input_kinds?.includes("source") &&
              selectedPlugin?.input_kinds?.includes("dataset"))) && (
            <label>
              dataset_id
              <select
                value={datasetId}
                onChange={(e) => {
                  setDatasetId(e.target.value);
                  markStale();
                }}
              >
                <option value="">从已发现数据集选择…</option>
                {datasets.map((d) => (
                  <option key={d.dataset_id} value={d.dataset_id}>
                    {d.dataset_id}
                  </option>
                ))}
              </select>
              <input
                style={{ marginTop: "0.35rem" }}
                value={datasetId}
                onChange={(e) => {
                  setDatasetId(e.target.value);
                  markStale();
                }}
                placeholder="或粘贴已登记相对/绝对 parquet 路径"
              />
              <span className="help">不能手填后端输出任意路径；须为已登记数据集引用</span>
            </label>
          )}

          <label>
            证据模式（execution）
            <select
              value={evidenceMode}
              onChange={(e) => {
                setEvidenceMode(e.target.value);
                markStale();
              }}
            >
              {evidenceModes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
        </div>

        {selectedPlugin && (
          <>
            <h4>参数（由 parameters_schema 生成，无插件专用分支）</h4>
            <SchemaForm
              schema={selectedPlugin.parameters_schema || { type: "object", properties: {} }}
              value={parameters}
              onChange={(v) => {
                setParameters(v);
                markStale();
              }}
            />
          </>
        )}

        <h4>模型路次绑定（动态 route）</h4>
        {bindings.length === 0 ? (
          <p className="muted">当前插件未声明 route；可手动添加或留空（由后端校验）</p>
        ) : (
          bindings.map((b, idx) => (
            <div className="row" key={`${b.route_id}-${idx}`} style={{ marginBottom: "0.4rem" }}>
              <input
                style={{ maxWidth: 160 }}
                value={b.route_id}
                onChange={(e) => {
                  const next = [...bindings];
                  next[idx] = { ...b, route_id: e.target.value };
                  setBindings(next);
                  markStale();
                }}
                placeholder="route_id"
              />
              <select
                value={b.connection_id}
                onChange={(e) => {
                  const next = [...bindings];
                  next[idx] = { ...b, connection_id: e.target.value };
                  setBindings(next);
                  markStale();
                }}
              >
                <option value="">选择连接…</option>
                {models.map((m) => (
                  <option key={m.connection_id} value={m.connection_id}>
                    {m.name} · {m.provider} · {m.base_url}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => {
                  setBindings(bindings.filter((_, i) => i !== idx));
                  markStale();
                }}
              >
                删除
              </button>
            </div>
          ))
        )}
        <button
          type="button"
          onClick={() => {
            setBindings([...bindings, { route_id: `route_${bindings.length + 1}`, connection_id: "" }]);
            markStale();
          }}
        >
          添加 route
        </button>

        {canSelectTags && (
          <>
            <h4>显式选择 Tag（含版本；默认不选全库）· 已选 {selectedTags.length}</h4>
            {tags.length === 0 ? (
              <p className="muted">无可用 tag。请刷新插件目录。</p>
            ) : (
              <div className="row" style={{ maxHeight: 180, overflow: "auto" }}>
                {tags.map((t) => (
                  <label key={t.id} className="row" style={{ marginRight: "0.75rem" }}>
                    <input
                      type="checkbox"
                      checked={selectedTags.includes(t.id)}
                      disabled={t.status !== "enabled"}
                      onChange={(e) => {
                        setSelectedTags((prev) =>
                          e.target.checked ? [...prev, t.id] : prev.filter((x) => x !== t.id),
                        );
                        markStale();
                      }}
                    />
                    {t.id}@{t.version}
                  </label>
                ))}
              </div>
            )}
          </>
        )}

        <div className="row" style={{ marginTop: "1rem" }}>
          <button type="button" disabled={planning} onClick={() => void onPlan()}>
            {planning ? "规划中…" : "校验并规划（尚未执行推理）"}
          </button>
          <button className="primary" type="button" disabled={!canSubmit || submitting} onClick={() => void onSubmit()}>
            {submitting ? "提交中…" : "提交任务"}
          </button>
          {(planStale || planExpired) && (
            <span className="warn">{planExpired ? "计划已过期，请重新规划" : "字段已改，旧计划失效"}</span>
          )}
          {!plan && !planStale && <span className="muted">修改字段后需重新规划</span>}
          <span className="help">幂等键 {idemKey.slice(0, 8)}…（失败重试复用，成功后更换）</span>
        </div>

        {plan && !planStale && (
          <pre className="logbox" style={{ marginTop: "1rem" }}>
            {JSON.stringify(
              {
                plan_id: plan.plan_id,
                plan_digest: plan.plan_digest,
                expires_at: plan.expires_at,
                catalog_generation: plan.catalog_generation,
                nodes: plan.nodes,
                resolved_models: plan.resolved_models,
                selected_tags: plan.selected_tags,
                evidence_union: plan.evidence_union,
                warnings: plan.warnings,
                supports_cancel: plan.supports_cancel,
                supports_resume: plan.supports_resume,
                note: "规划完成，尚未执行推理",
              },
              null,
              2,
            )}
          </pre>
        )}
      </div>

      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>任务列表</h3>
          <div className="row">
            <select
              value={statusFilter}
              onChange={(e) => {
                const p = new URLSearchParams(params);
                if (e.target.value) p.set("status", e.target.value);
                else p.delete("status");
                setParams(p);
              }}
            >
              <option value="">全部状态</option>
              {["queued", "running", "succeeded", "failed", "cancelling", "cancelled", "interrupted"].map(
                (s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ),
              )}
            </select>
            <button type="button" onClick={() => void loadMeta()}>
              刷新
            </button>
          </div>
        </div>
        {loading && <p className="muted">加载中…</p>}
        {!loading && jobs.length === 0 && <p className="muted">暂无任务</p>}
        {!loading && jobs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>类型</th>
                <th>根插件/tag 数</th>
                <th>状态</th>
                <th>进度</th>
                <th>创建</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((j) => {
                const req = j.request || {};
                const root = req.root_plugin as { id?: string; version?: string } | null;
                const tagCount = Array.isArray(req.tags) ? req.tags.length : 0;
                const progress = j.progress || {};
                return (
                  <tr key={j.job_id}>
                    <td>{j.job_id}</td>
                    <td>{j.task_type}</td>
                    <td>
                      {root?.id ? `${root.id}@${root.version || "?"}` : "—"}
                      <div className="help">tags={tagCount}</div>
                    </td>
                    <td>
                      <TaskStatus status={j.status} />
                    </td>
                    <td>
                      <span className="help">{JSON.stringify(progress)}</span>
                    </td>
                    <td>{formatDateTime(j.created_at)}</td>
                    <td>
                      <Link to={`/jobs/${j.job_id}`}>详情</Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
