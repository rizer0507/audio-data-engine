import { useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { ApiRequestError, api, formatDateTime, newIdempotencyKey } from "../api";
import BatchDatasetPicker, { useDatasetCatalog } from "../components/BatchDatasetPicker";
import ErrorBanner from "../components/ErrorBanner";
import SchemaForm, { defaultsFromSchema } from "../components/SchemaForm";
import TaskStages from "../components/TaskStages";
import type { ExecutionPlan, Job, ListResponse, ModelConnection, PluginDescriptor, Source, TaskRequest } from "../types";
import { basename, datasetLabel, descriptorNodes } from "../workflows";

type Mode = "production" | "single" | "tagging";
type Resources = {
  pipelines: PluginDescriptor[];
  tags: PluginDescriptor[];
  models: ModelConnection[];
  sources: Source[];
  evidenceModes: string[];
};

function routesFor(plugin: PluginDescriptor | undefined) {
  const dag = plugin?.detail?.dag_nodes as Array<{ id: string; family?: string }> | undefined;
  if (dag) return dag.filter((node) => node.id.startsWith("asr_")).map((node) => ({
    id: node.id.slice(4), family: node.family,
  }));
  const roles = plugin?.detail?.model_roles as string[] | undefined;
  if (roles?.length) return roles.map((id) => ({ id, family: undefined }));
  const steps = plugin?.detail?.steps as Array<{ operator?: string }> | undefined;
  const external = steps?.map((step) => /^asr\.(qwen|glm)_batch$/.exec(step.operator || "")?.[1]).filter(Boolean);
  return (external || []).map((family) => ({ id: family!, family }));
}

export default function NewJob() {
  const [query] = useSearchParams();
  const navigate = useNavigate();
  const initialDataset = query.get("dataset_id") || "";
  const initialSource = query.get("source_id") || "";
  const [mode, setMode] = useState<Mode>(initialDataset ? "tagging" : "production");
  const [resources, setResources] = useState<Resources>();
  const [reload, setReload] = useState(0);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [pluginId, setPluginId] = useState("");
  const [inputKind, setInputKind] = useState<"source" | "dataset">(initialDataset ? "dataset" : "source");
  const [sourceId, setSourceId] = useState(initialSource);
  const [datasetId, setDatasetId] = useState(initialDataset);
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [connections, setConnections] = useState<Record<string, string>>({});
  const [tags, setTags] = useState<string[]>([]);
  const [tagQuery, setTagQuery] = useState("");
  const [evidenceMode, setEvidenceMode] = useState("existing");
  const [manualEvidence, setManualEvidence] = useState(false);
  const [validAdvanced, setValidAdvanced] = useState(true);
  const [validEvidence, setValidEvidence] = useState(true);
  const [plan, setPlan] = useState<{ data: ExecutionPlan; requestKey: string; idem: string }>();
  const [expired, setExpired] = useState(false);
  const [configurationChanged, setConfigurationChanged] = useState(false);
  const [action, setAction] = useState<"planning" | "submitting">();
  const [error, setError] = useState<unknown>(null);
  const catalog = useDatasetCatalog(initialDataset);

  useEffect(() => {
    const abort = new AbortController();
    setLoadError(null);
    void Promise.all([
      api.get<ListResponse<PluginDescriptor>>("/api/v1/plugins?kind=pipeline&status=enabled", abort.signal),
      api.get<ListResponse<PluginDescriptor>>("/api/v1/plugins?kind=tag&status=enabled", abort.signal),
      api.get<ListResponse<ModelConnection>>("/api/v1/models", abort.signal),
      api.get<ListResponse<Source>>("/api/v1/sources", abort.signal),
      api.get<{ evidence_modes: string[] }>("/api/v1/capabilities", abort.signal),
    ]).then(([pipelines, allTags, models, sources, capabilities]) => {
      if (abort.signal.aborted) return;
      const available = pipelines.items.filter((item) => item.runnable && item.capabilities.purpose !== "evaluation");
      setResources({
        pipelines: available, tags: allTags.items.filter((tag) => tag.runnable),
        models: models.items, sources: sources.items,
        evidenceModes: capabilities.evidence_modes || ["existing"],
      });
    }).catch((failure) => {
      if (!abort.signal.aborted) setLoadError(failure);
    });
    return () => abort.abort();
  }, [reload]);

  useEffect(() => {
    setExpired(false);
    if (!plan?.data.expires_at) return;
    const delay = Date.parse(plan.data.expires_at) - Date.now();
    if (!Number.isFinite(delay)) return;
    const timer = window.setTimeout(() => setExpired(true), Math.max(0, delay));
    return () => window.clearTimeout(timer);
  }, [plan]);

  const production = resources?.pipelines.find((item) => item.adapter === "production_pipeline");
  const single = resources?.pipelines.filter((item) => item.adapter !== "production_pipeline") || [];
  const plugin = mode === "production" ? production : mode === "single" ? single.find((item) => item.id === pluginId) : undefined;
  const kind = mode === "tagging" ? "dataset" : plugin?.input_kinds.length === 1 ? plugin.input_kinds[0] : inputKind;
  const selectedSource = resources?.sources.find((source) => source.source_id === sourceId);
  const selectedDataset = catalog.items.find((item) => item.dataset_id === datasetId) ??
    (catalog.initialItem?.dataset_id === datasetId ? catalog.initialItem : undefined);
  const routes = routesFor(plugin).filter((route) =>
    !Array.isArray(parameters.asr_route_ids) || !parameters.asr_route_ids.length || parameters.asr_route_ids.includes(route.id),
  );
  const allowTags = mode === "tagging" || plugin?.capabilities.supports_tag_selection;
  const visibleTags = (resources?.tags || []).filter((tag) =>
    `${tag.name} ${tag.id}`.toLowerCase().includes(tagQuery.trim().toLowerCase()),
  );
  const modelsFor = (family?: string) => (resources?.models || []).filter((model) => !family || family === model.provider);
  function connectionFor(id: string, family?: string) {
    const choices = modelsFor(family);
    if (id in connections) return choices.some((model) => model.connection_id === connections[id]) ? connections[id] : "";
    return family && choices.length === 1 ? choices[0].connection_id : "";
  }
  const properties = (plugin?.parameters_schema.properties || {}) as Record<string, Record<string, unknown>>;
  const hasEvidenceInputs = Boolean(properties.asr_manifests && properties.dnsmos_manifest);
  const advanced = Object.fromEntries(Object.entries(properties).filter(([key]) =>
    key !== "label_xlsx" && !(hasEvidenceInputs && ["asr_manifests", "dnsmos_manifest"].includes(key)),
  ));
  const selectedAsr = Array.isArray(parameters.asr_manifests) ? parameters.asr_manifests as string[] : [];
  const batchEvidence = selectedDataset ? catalog.items : [];
  const asrChoices = batchEvidence.filter((item) => item.stage === "asr" && item.asr_route);
  const asrRoutes = Array.from(new Set(asrChoices.map((item) => item.asr_route!)));
  const evidenceChoices = batchEvidence.filter((item) => item.stage === "dnsmos");
  const input = kind === "source" ? { kind: "source" as const, source_id: sourceId } : {
    kind: "dataset" as const, dataset_id: datasetId.trim(), batch: selectedDataset?.batch || undefined,
  };
  const request: TaskRequest = {
    task_type: mode === "tagging" ? "tagging" : "pipeline",
    root_plugin: plugin ? { kind: "pipeline", id: plugin.id, version: plugin.version } : null,
    input,
    parameters: {
      ...defaultsFromSchema(plugin?.parameters_schema || {}),
      ...parameters,
      ...(properties.batch && !parameters.batch && selectedDataset?.batch ? { batch: selectedDataset.batch } : {}),
    },
    model_bindings: routes.map((route) => ({ route_id: route.id, connection_id: connectionFor(route.id, route.family) }))
      .filter((binding) => binding.connection_id),
    tags: allowTags ? (resources?.tags || []).filter((tag) => tags.includes(tag.id))
      .map((tag) => ({ kind: "tag", id: tag.id, version: tag.version })) : [],
    execution: mode === "tagging" ? { evidence_mode: evidenceMode } : {},
  };
  const requestKey = JSON.stringify(request);
  const validForm = validAdvanced && validEvidence;
  const currentPlan = plan?.requestKey === requestKey && !expired && validForm ? plan : undefined;
  useEffect(() => {
    if (plan && (plan.requestKey !== requestKey || !validForm)) {
      setPlan(undefined);
      setConfigurationChanged(true);
    }
  }, [plan, requestKey, validForm]);
  const problem = !resources ? "等待资源加载" :
    !validForm ? "请修正配置中的 JSON" :
    mode !== "tagging" && !plugin ? "请选择可用流水线" :
    kind === "source" && !selectedSource ? "请选择数据源" :
    kind === "dataset" && !datasetId.trim() ? "请选择输入清单" :
    hasEvidenceInputs && (!selectedAsr.length || !parameters.dnsmos_manifest) ? "请选择 ASR 和 DNSMOS 清单" :
    mode === "tagging" && !request.tags.length ? "至少选择一个标签" :
    mode === "production" && routes.some((route) => ["qwen", "glm"].includes(route.family || "") && !connectionFor(route.id, route.family))
      ? "请为所选 ASR 路次绑定模型" : "";
  const plannedNodes = currentPlan?.data.nodes || descriptorNodes(plugin, parameters);

  function selectMode(next: Mode, selected?: PluginDescriptor) {
    const choice = next === "production" ? production : selected;
    setMode(next);
    setPluginId(selected?.id || "");
    setInputKind(choice?.input_kinds.includes("dataset") && (datasetId || !choice.input_kinds.includes("source")) ? "dataset" : "source");
    setParameters(defaultsFromSchema(choice?.parameters_schema || {}));
    setConnections({});
    setTags([]);
    setManualEvidence(false);
    setValidAdvanced(true);
    setValidEvidence(true);
    setError(null);
    setPlan(undefined);
    setConfigurationChanged(false);
  }

  function setParameter(key: string, value: unknown) {
    setParameters((current) => ({ ...current, [key]: value }));
  }

  function selectDataset(value: string) {
    setDatasetId(value);
    if (hasEvidenceInputs) setParameters((current) => ({
      ...current, asr_manifests: undefined, dnsmos_manifest: undefined,
    }));
  }

  async function createPlan() {
    if (problem || action) return;
    setError(null);
    setPlan(undefined);
    setConfigurationChanged(false);
    setAction("planning");
    try {
      const data = await api.post<ExecutionPlan>("/api/v1/plugins/plan", request);
      setPlan({ data, requestKey, idem: newIdempotencyKey() });
    } catch (failure) {
      setError(failure);
    } finally {
      setAction(undefined);
    }
  }

  async function submit() {
    if (!currentPlan || action) return;
    if (currentPlan.data.expires_at && Date.parse(currentPlan.data.expires_at) <= Date.now()) {
      setExpired(true);
      return;
    }
    setAction("submitting");
    setError(null);
    try {
      const { plan_id, plan_digest } = currentPlan.data;
      const job = await api.post<Job>("/api/v1/jobs", { plan_id, plan_digest }, { "Idempotency-Key": currentPlan.idem });
      navigate(`/jobs/${job.job_id}`);
    } catch (failure) {
      setError(failure);
      if (failure instanceof ApiRequestError && ["CATALOG_DRIFT", "PLAN_EXPIRED", "PLUGIN_ARCHIVED"].includes(failure.code)) {
        setPlan(undefined);
        setReload((value) => value + 1);
      }
    } finally {
      setAction(undefined);
    }
  }

  return (
    <div className="work-page">
      <Link className="back-link" to="/jobs">← 任务列表</Link>
      <header className="page-heading">
        <div><h2>新建任务</h2><p>选择处理范围，确认运行计划后提交。</p></div>
      </header>
      <ErrorBanner error={loadError} />
      {Boolean(loadError) && <button type="button" onClick={() => setReload((value) => value + 1)}>重试加载资源</button>}
      {!resources && !loadError && <p className="muted">正在加载数据源和流水线…</p>}
      {resources && (
        <div className="work-layout">
          <div className="work-main">
            <fieldset className="form-lock" disabled={Boolean(action)}>
              <section className="panel work-panel">
                <h3>1. 选择处理范围</h3>
                <div className="choice-row" aria-label="任务方式">
                  {([
                    ["production", "工序一 · 完整流程"],
                    ["single", "单步流水线"],
                    ["tagging", "已有清单打标"],
                  ] as const).map(([id, label]) => (
                    <button key={id} type="button" aria-pressed={mode === id} onClick={() => mode !== id && selectMode(id)}>{label}</button>
                  ))}
                </div>
                {mode === "single" && (
                  <label className="field">
                    <span>流水线</span>
                    <select value={pluginId} onChange={(event) => selectMode("single", single.find((item) => item.id === event.target.value))}>
                      <option value="">选择流水线</option>
                      {single.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                    </select>
                  </label>
                )}
                {mode === "production" && !production && <p className="muted">工序一当前不可用。请检查插件状态。</p>}
                {plugin?.input_kinds.includes("source") && plugin.input_kinds.includes("dataset") && (
                  <label className="field">
                    <span>输入方式</span>
                    <select value={kind} onChange={(event) => setInputKind(event.target.value as "source" | "dataset")}>
                      <option value="source">已登记数据源</option><option value="dataset">已有清单</option>
                    </select>
                  </label>
                )}
                {(plugin || mode === "tagging") && (kind === "source" ? (
                  <label className="field">
                    <span>数据源批次</span>
                    <select value={sourceId} onChange={(event) => setSourceId(event.target.value)}>
                      <option value="">选择批次</option>
                      {resources.sources.map((source) => (
                        <option key={source.source_id} value={source.source_id}>{source.batch || source.source_name}{source.name ? ` · ${source.name}` : ""}</option>
                      ))}
                    </select>
                    {selectedSource && <span className="help path-text" title={selectedSource.path}>{selectedSource.path}</span>}
                    {!resources.sources.length && <Link to="/sources">先登记数据源</Link>}
                  </label>
                ) : <BatchDatasetPicker catalog={catalog} value={datasetId} onChange={selectDataset} />)}
              </section>

              {(plugin || mode === "tagging") && (
                <section className="panel work-panel">
                  <h3>2. 配置本次运行</h3>
                  {routes.length > 0 && (
                    <div className="form-stack">
                      <div className="section-line"><strong>模型连接</strong><Link to="/models">管理连接</Link></div>
                      <div className="form-columns">
                        {routes.map((route) => (
                          <label className="field" key={route.id}>
                            <span>{route.id}{route.family && ` · ${route.family}`}</span>
                            <select value={connectionFor(route.id, route.family)} onChange={(event) =>
                              setConnections((current) => ({ ...current, [route.id]: event.target.value }))
                            }>
                              <option value="">{mode === "production" ? "选择模型" : "使用服务器默认连接"}</option>
                              {modelsFor(route.family).map((model) => (
                                <option key={model.connection_id} value={model.connection_id}>{model.name} · {model.base_url}</option>
                              ))}
                            </select>
                          </label>
                        ))}
                      </div>
                    </div>
                  )}
                  {properties.label_xlsx && (
                    <label className="field">
                      <span>话术分类 XLSX <span className="help">可选</span></span>
                      <input value={String(parameters.label_xlsx || "")} onChange={(event) => setParameter("label_xlsx", event.target.value)} placeholder="服务器上的表格绝对路径" />
                      <span className="help">填写后会在切段结束后筛除「接洽（身份核实）」片段。</span>
                    </label>
                  )}
                  {hasEvidenceInputs && (
                    <div className="form-stack">
                      <div className="section-line"><strong>合并证据</strong>
                        <button type="button" className="linkish" onClick={() => {
                          setManualEvidence(!manualEvidence);
                          setValidEvidence(true);
                          setParameters((current) => ({ ...current, asr_manifests: undefined, dnsmos_manifest: undefined }));
                        }}>{manualEvidence ? "从批次选择" : "手动填写路径"}</button>
                      </div>
                      {manualEvidence ? (
                        <SchemaForm schema={{ properties: { asr_manifests: properties.asr_manifests, dnsmos_manifest: properties.dnsmos_manifest } }}
                          value={{ asr_manifests: parameters.asr_manifests, dnsmos_manifest: parameters.dnsmos_manifest }}
                          onValidityChange={setValidEvidence}
                          onChange={(values) => setParameters((current) => ({ ...current, ...values }))} />
                      ) : (
                        <div className="form-columns">
                          {asrRoutes.map((route) => (
                            <label className="field" key={route}>
                              <span>{route} 转写清单</span>
                              <select value={selectedAsr.find((item) => item.startsWith(`${route}=`)) || ""} onChange={(event) => setParameter(
                                "asr_manifests", [...selectedAsr.filter((item) => !item.startsWith(`${route}=`)), event.target.value].filter(Boolean),
                              )}>
                                <option value="">选择清单</option>
                                {asrChoices.filter((item) => item.asr_route === route).map((item) => (
                                  <option key={item.dataset_id} value={`${route}=${item.path}`}>{datasetLabel(item)}</option>
                                ))}
                              </select>
                            </label>
                          ))}
                          <label className="field">
                            <span>DNSMOS 清单</span>
                            <select value={String(parameters.dnsmos_manifest || "")} onChange={(event) => setParameter("dnsmos_manifest", event.target.value)}>
                              <option value="">{evidenceChoices.length ? "选择清单" : "当前批次暂无评分清单"}</option>
                              {evidenceChoices.map((item) => <option key={item.dataset_id} value={item.path}>{datasetLabel(item)}</option>)}
                            </select>
                          </label>
                          {!asrRoutes.length && <p className="help">{selectedDataset ? "当前批次暂无已识别的 ASR 清单。" : "请先从批次中选择输入清单。"}</p>}
                        </div>
                      )}
                    </div>
                  )}
                  {mode === "tagging" && (
                    <label className="field">
                      <span>证据模式</span>
                      <select value={evidenceMode} onChange={(event) => setEvidenceMode(event.target.value)}>
                        {resources.evidenceModes.map((value) => <option key={value} value={value}>
                          {({ existing: "使用已有证据", compute_acoustic: "补算声学证据", plan: "仅规划", preview: "预览" } as Record<string, string>)[value] || value}
                        </option>)}
                      </select>
                    </label>
                  )}
                  {allowTags && (
                    <details className="options-fold" open={mode === "tagging" || undefined}>
                      <summary>{mode === "tagging" ? "选择标签" : "自定义标签"} · {tags.length ? `已选 ${tags.length}` : mode === "tagging" ? "未选择" : "使用流水线默认规则"}</summary>
                      <label className="field">
                        <span className="sr-only">搜索标签</span>
                        <input value={tagQuery} placeholder="搜索标签" onChange={(event) => setTagQuery(event.target.value)} />
                      </label>
                      <div className="tag-options">
                        {visibleTags.map((tag) => (
                          <label className="check-row" key={tag.id}>
                            <input type="checkbox" checked={tags.includes(tag.id)} onChange={(event) =>
                              setTags((current) => event.target.checked ? [...current, tag.id] : current.filter((id) => id !== tag.id))
                            } />
                            <span>{tag.name} <span className="help">v{tag.version}</span></span>
                          </label>
                        ))}
                        {!visibleTags.length && <p className="muted">没有匹配的标签。</p>}
                      </div>
                    </details>
                  )}
                  {Object.keys(advanced).length > 0 && (
                    <details className="options-fold">
                      <summary>高级参数</summary>
                      <SchemaForm key={plugin?.id} schema={{ ...plugin?.parameters_schema, properties: advanced }}
                        value={Object.fromEntries(Object.entries(parameters).filter(([key]) => key in advanced))}
                        onValidityChange={setValidAdvanced}
                        onChange={(values) => setParameters((current) => ({
                          ...Object.fromEntries(Object.entries(current).filter(([key]) => !(key in advanced))), ...values,
                        }))} />
                    </details>
                  )}
                  {!routes.length && !hasEvidenceInputs && !allowTags && !Object.keys(properties).length && (
                    <p className="muted">此流水线使用默认配置。</p>
                  )}
                </section>
              )}
            </fieldset>
          </div>

          <aside className="panel work-panel plan-panel">
            <div className="section-line"><h3>3. 确认运行</h3>{currentPlan && <span className="badge succeeded">计划就绪</span>}</div>
            <p className="muted">{plugin?.name || (mode === "tagging" ? "对已有清单执行标签规则" : "选择流水线后查看步骤")}</p>
            {plannedNodes.length > 0 && <TaskStages nodes={plannedNodes} />}
            {currentPlan && (
              <dl className="summary-list">
                <div><dt>输入</dt><dd title={selectedSource?.path || datasetId}>{kind === "source" ? selectedSource?.batch || selectedSource?.source_name : basename(datasetId)}</dd></div>
                {currentPlan.data.resolved_models?.map((model, index) => (
                  <div key={`${model.connection_id}:${index}`}><dt>{model.model || model.name}</dt><dd>{model.base_url}</dd></div>
                ))}
                {request.tags.length > 0 && <div><dt>标签</dt><dd>{request.tags.length} 项</dd></div>}
                {currentPlan.data.expires_at && <div><dt>计划有效至</dt><dd>{formatDateTime(currentPlan.data.expires_at)}</dd></div>}
              </dl>
            )}
            {Boolean(currentPlan?.data.warnings?.length) && (
              <details className="options-fold"><summary>计划说明</summary>
                <ul>{currentPlan?.data.warnings?.map((warning, index) => <li key={index}>{String(warning)}</li>)}</ul>
              </details>
            )}
            <ErrorBanner error={error} />
            {(configurationChanged || (plan && !currentPlan)) && <p className="warn" role="status">{expired ? "计划已过期，请重新生成。" : "配置已变化，请重新生成计划。"}</p>}
            <div className="plan-action">
              {currentPlan ? (
                <button className="primary" type="button" disabled={Boolean(action)} onClick={() => void submit()}>{action === "submitting" ? "提交中…" : "提交任务"}</button>
              ) : (
                <button className="primary" type="button" disabled={Boolean(problem || action)} onClick={() => void createPlan()}>{action === "planning" ? "生成中…" : "生成运行计划"}</button>
              )}
              <span className="help">{currentPlan ? "提交后进入队列" : problem || "生成计划仅校验配置"}</span>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
