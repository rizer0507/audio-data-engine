import type { SampleRow } from "../types";

function fmtDuration(v?: number | null): string {
  if (v === undefined || v === null || Number.isNaN(Number(v))) return "未计算/缺失";
  return `${Number(v).toFixed(2)}s`;
}

export default function SampleTable({
  items,
  onSelect,
  selectedId,
}: {
  items: SampleRow[];
  onSelect?: (id: string) => void;
  selectedId?: string | null;
}) {
  if (!items.length) {
    return <p className="muted">无样本。请确认数据集路径存在且可读，或调整筛选条件。</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>样本 ID</th>
          <th>时长</th>
          <th>有效 tags</th>
          <th>三态/质量</th>
          <th>路次/审核</th>
        </tr>
      </thead>
      <tbody>
        {items.map((s) => (
          <tr
            key={s.sample_id}
            onClick={() => onSelect?.(s.sample_id)}
            style={{
              cursor: onSelect ? "pointer" : "default",
              background: selectedId === s.sample_id ? "#eff8ff" : undefined,
            }}
          >
            <td>{s.sample_id}</td>
            <td>{fmtDuration(s.duration)}</td>
            <td>{(s.tags || []).join(", ") || "—"}</td>
            <td>
              {s.ternary && Object.keys(s.ternary).length
                ? Object.entries(s.ternary)
                    .slice(0, 3)
                    .map(([k, v]) => `${k}=${v}`)
                    .join("; ")
                : "—"}
              {s.quality && s.quality.dnsmos_sig != null
                ? ` · sig=${String(s.quality.dnsmos_sig)}`
                : s.quality && s.quality.dnsmos_status
                  ? ` · ${String(s.quality.dnsmos_status)}`
                  : ""}
            </td>
            <td>
              {s.status && Object.keys(s.status).length
                ? Object.entries(s.status)
                    .map(([k, v]) => `${k}=${v}`)
                    .join("; ")
                : "—"}
              {s.annotation_state ? ` · ann=${s.annotation_state}` : ""}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
