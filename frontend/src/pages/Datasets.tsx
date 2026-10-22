import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import ErrorBanner from "../components/ErrorBanner";
import type { DatasetSummary } from "../types";

export default function Datasets() {
  const [params] = useSearchParams();
  const [items, setItems] = useState<DatasetSummary[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const q = (params.get("q") || "").trim().toLowerCase();

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const d = await api.get<{ items: DatasetSummary[]; total?: number | null }>("/api/v1/datasets");
      setItems(d.items || []);
      setTotal(d.total ?? (d.items || []).length);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  const filtered = q
    ? items.filter((d) => d.dataset_id.toLowerCase().includes(q) || (d.path || "").toLowerCase().includes(q))
    : items;

  return (
    <div>
      <h2>数据集</h2>
      <p className="muted">
        浏览已注册 Manifest / 本服务发现的 parquet。缺元数据时会标明无法完整追溯。共 {total ?? "—"} 项。
      </p>
      <ErrorBanner error={error} />
      <div className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>列表</h3>
          <button type="button" onClick={() => void load()}>
            刷新
          </button>
        </div>
        {loading && <p className="muted">加载中…</p>}
        {!loading && filtered.length === 0 && (
          <p className="muted">未发现数据集。处理任务成功后会出现在 datasets/ 下；请勿用虚构行填充。</p>
        )}
        {!loading && filtered.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>dataset_id</th>
                <th>path</th>
                <th>状态</th>
                <th>登记</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => (
                <tr key={d.dataset_id}>
                  <td>{d.dataset_id}</td>
                  <td>{d.path}</td>
                  <td>
                    {d.state_kind === "working" ? "工作副本" : "已发布"}
                    {d.has_work_copy ? " · 有 work" : ""}
                  </td>
                  <td>{d.registered === false ? "未完整登记" : "已发现"}</td>
                  <td>
                    <Link to={`/datasets/${d.dataset_id}`}>打开</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
