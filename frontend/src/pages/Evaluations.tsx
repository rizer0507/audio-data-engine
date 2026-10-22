import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ApiRequestError, api, formatDateTime, newIdempotencyKey } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import TaskStatus from "../components/TaskStatus";
import type {
  DatasetSummary,
  ExecutionPlan,
  Job,
  ModelBinding,
  ModelConnection,
  PluginDescriptor,
} from "../types";
import SchemaForm, { defaultsFromSchema } from "../components/SchemaForm";

export default function Evaluations() {
  const navigate = useNavigate();
  const [plugins, setPlugins] = useState<PluginDescriptor[]>([]);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [models, setModels] = useState<ModelConnection[]>([]);
  const [pluginId, setPluginId] = useState("");
  const [selected, setSelected] = useState<PluginDescriptor | null>(null);
  const [datasetId, setDatasetId] = useState("");
  const [parameters, setParameters] = useState<Record<string, unknown>>({});
  const [bindings, setBindings] = useState<ModelBinding[]>([]);
  const [plan, setPlan] = useState<ExecutionPlan | null>(null);
  const [planStale, setPlanStale] = useState(false);
  const [items, setItems] = useState<Job[]>([]);
  const [gate, setGate] = useState<Record<string, unknown> | null>(null);
  const [detail, setDetail] = useState<Record<string, unknown> | null>(null);
  const [compareId, setCompareId] = useState("");
  const [compareResult, setCompareResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [idem, setIdem] = useState(() => newIdempotencyKey());
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const [p, d, m, ev] = await Promise.all([
        api.get<{ items: PluginDescriptor[] }>("/api/v1/plugins?kind=pipeline&purpose=evaluation"),
        api.get<{ items: DatasetSummary[] }>("/api/v1/datasets"),
        api.get<{ items: ModelConnection[] }>("/api/v1/models"),
        api.get<{ items: Job[] }>("/api/v1/evaluations"),
      ]);
      setPlugins(
        (p.items || []).filter(
          (x) => x.capabilities?.purpose === "evaluation" && x.status === "enabled",
        ),
      );
      setDatasets(d.items || []);
      setModels(m.items || []);
      setItems((ev.items || []) as Job[]);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    setPlan(null);
    setPlanStale(false);
    setGate(null);
  }, [pluginId, datasetId, parameters, bindings]);

  useEffect(() => {
    if (!pluginId) {
      setSelected(null);
      return;
    }
    const found = plugins.find((p) => p.id === pluginId);
    if (!found) return;
    void api
      .get<PluginDescriptor>(
        `/api/v1/plugins/pipeline/${found.id}?version=${encodeURIComponent(found.version)}`,
      )
      .then((d) => {
        setSelected(d);
        setParameters(defaultsFromSchema(d.parameters_schema || { properties: {} }));
        const routes = ((d.detail?.asr_route_ids as string[]) || []).map((route_id) => ({
          route_id,
          connection_id: "",
        }));
        setBindings(routes);
      })
      .catch((e) => setError(e));
  }, [pluginId, plugins]);

  const requestBody = useMemo(() => {
    if (!selected) return null;
    return {
      task_type: "pipeline" as const,
      root_plugin: { kind: "pipeline", id: selected.id, version: selected.version },
      input: { kind: "dataset" as const, dataset_id: datasetId },
      parameters,
      model_bindings: bindings.filter((b) => b.route_id && b.connection_id),
      tags: [],
      execution: {},
    };
  }, [selected, datasetId, parameters, bindings]);

  async function checkGate() {
    if (!datasetId) return;
    setError(null);
    try {
      const g = await api.post<Record<string, unknown>>(
        `/api/v1/datasets/${encodeURIComponent(datasetId)}/gold-gate`,
        { require_formal_gold: true },
      );
      setGate(g);
    } catch (e) {
      setError(e);
    }
  }

  async function onPlan() {
    if (!requestBody) return;
    setError(null);
    setPlan(null);
    setPlanStale(false);
    try {
      await checkGate();
      const data = await api.post<ExecutionPlan>("/api/v1/plugins/plan", requestBody);
      setPlan(data);
    } catch (e) {
      setError(e);
    }
  }

  async function onSubmit() {
    if (!plan?.plan_id || !plan?.plan_digest || planStale) return;
    try {
      const job = await api.post<{ job_id: string }>(
        "/api/v1/evaluations",
        { plan_id: plan.plan_id, plan_digest: plan.plan_digest },
        { "Idempotency-Key": idem },
      );
      setIdem(newIdempotencyKey());
      navigate(`/jobs/${job.job_id}`);
    } catch (e) {
      if (e instanceof ApiRequestError) {
        if (e.code === "CATALOG_DRIFT" || e.code === "PLAN_EXPIRED") {
          setPlanStale(true);
          setPlan(null);
        }
        if (e.code === "FORMAL_GOLD_GATE") {
          setGate({ ...(e.details || {}), formal_gate_ok: false, blocked: true });
        }
      }
      setError(e);
    }
  }

  async function openDetail(id: string) {
    setError(null);
    try {
      const d = await api.get<Record<string, unknown>>(`/api/v1/evaluations/${encodeURIComponent(id)}`);
      setDetail(d);
    } catch (e) {
      setError(e);
    }
  }

  async function runCompare(leftId: string) {
    if (!compareId) return;
    setError(null);
    try {
      const r = await api.post<Record<string, unknown>>(
        `/api/v1/evaluations/${encodeURIComponent(leftId)}/compare`,
        { other_evaluation_id: compareId },
      );
      setCompareResult(r);
    } catch (e) {
      setError(e);
    }
  }

  return (
    <div>
      <h2>评测</h2>
      <p className="muted">
        经统一 plugins/plan + evaluations/jobs 提交。无合法正式 Gold 时后端拒绝正式评测；前端不生成假指标。
      </p>
      <ErrorBanner error={error} />

      <div className="panel">
        <h3>新建评测</h3>
        {loading && <p className="muted">加载中…</p>}
        {!loading && plugins.length === 0 && (
          <p className="muted">无 purpose=evaluation 的流水线插件。请在插件库登记后刷新目录。</p>
        )}
        <div className="form-grid">
          <label>
            评测插件
            <select value={pluginId} onChange={(e) => setPluginId(e.target.value)}>
              <option value="">选择…</option>
              {plugins.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.id}@{p.version})
                </option>
              ))}
            </select>
          </label>
          <label>
            评测数据集版本
            <select value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
              <option value="">选择…</option>
              {datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.dataset_id}
                </option>
              ))}
            </select>
            <input
              style={{ marginTop: "0.35rem" }}
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              placeholder="或粘贴 dataset_id"
            />
          </label>
        </div>

        {selected && (
          <>
            <h4>参数</h4>
            <SchemaForm
              schema={selected.parameters_schema || { properties: {} }}
              value={parameters}
              onChange={setParameters}
            />
          </>
        )}

        {bindings.length > 0 && (
          <>
            <h4>模型绑定</h4>
            {bindings.map((b, idx) => (
              <div className="row" key={b.route_id}>
                <span>{b.route_id}</span>
                <select
                  value={b.connection_id}
                  onChange={(e) => {
                    const next = [...bindings];
                    next[idx] = { ...b, connection_id: e.target.value };
                    setBindings(next);
                  }}
                >
                  <option value="">选择连接…</option>
                  {models.map((m) => (
                    <option key={m.connection_id} value={m.connection_id}>
                      {m.name} ({m.provider})
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </>
        )}

        <div className="row" style={{ marginTop: "0.75rem" }}>
          <button type="button" onClick={() => void checkGate()} disabled={!datasetId}>
            检查 Gold 门禁
          </button>
          <button type="button" onClick={() => void onPlan()}>
            规划（不推理）
          </button>
          <button
            className="primary"
            type="button"
            disabled={!plan || planStale || gate?.formal_gate_ok === false}
            onClick={() => void onSubmit()}
          >
            提交评测
          </button>
          {planStale && <span className="warn">计划已失效，请重新规划</span>}
        </div>
        {gate && (
          <pre className="logbox" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(
              {
                formal_gate_ok: gate.formal_gate_ok,
                eval_trust: gate.eval_trust,
                denominator: gate.denominator,
                errors: gate.errors,
              },
              null,
              2,
            )}
          </pre>
        )}
        {gate?.formal_gate_ok === false && (
          <p className="warn">正式 Gold 门禁未通过：不会提交正式评测，也不会生成虚构准确率。</p>
        )}
        {plan && !planStale && (
          <pre className="logbox" style={{ marginTop: "0.75rem" }}>
            {JSON.stringify(
              {
                plan_id: plan.plan_id,
                plan_digest: plan.plan_digest,
                warnings: plan.warnings,
                resolved_models: plan.resolved_models,
              },
              null,
              2,
            )}
          </pre>
        )}
      </div>

      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>最近评测任务</h3>
          <button type="button" onClick={() => void load()}>
            刷新
          </button>
        </div>
        {items.length === 0 ? (
          <p className="muted">暂无结果。不会用随机指标填充。</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>job_id</th>
                <th>状态</th>
                <th>创建</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {items.map((j) => (
                <tr key={j.job_id}>
                  <td>{j.job_id}</td>
                  <td>
                    <TaskStatus status={j.status} />
                  </td>
                  <td>{formatDateTime(j.created_at)}</td>
                  <td className="row">
                    <button type="button" onClick={() => void openDetail(j.job_id)}>
                      指标/门禁
                    </button>
                    <Link to={`/jobs/${j.job_id}`}>详情/日志</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {detail && (
          <>
            <h4>评测详情摘要</h4>
            <pre className="logbox">{JSON.stringify(detail, null, 2)}</pre>
            <div className="row">
              <input
                style={{ maxWidth: 280 }}
                value={compareId}
                onChange={(e) => setCompareId(e.target.value)}
                placeholder="对比另一 evaluation_id"
              />
              <button
                type="button"
                onClick={() => void runCompare(String(detail.evaluation_id || detail.job_id))}
              >
                同口径对比
              </button>
            </div>
            {compareResult && (
              <pre className="logbox">{JSON.stringify(compareResult, null, 2)}</pre>
            )}
          </>
        )}
      </div>
    </div>
  );
}
