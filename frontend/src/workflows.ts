import type { DatasetSummary, PlanNode, PluginDescriptor } from "./types";

export const JOB_STATUS: Record<string, string> = {
  queued: "排队中", running: "运行中", succeeded: "已完成", failed: "失败",
  cancelling: "正在停止", cancelled: "已停止", interrupted: "已中断", pending: "待运行",
};

export const ACTIVE_JOBS = new Set(["queued", "running", "cancelling"]);

const STEP_LABELS: Record<string, string> = {
  ingest: "音频入库", vad: "语音切段", filter_identity_verify: "话术筛除",
  dnsmos: "DNSMOS 质量评分", evidence_join: "证据合并", tagging: "自由打标",
  tag_batch: "按规则打标", multi_tag: "自由打标",
  fsmn_vad_segment: "语音切段", normalize_transcripts: "转写清洗",
  qwen_asr: "Qwen 转写", glm_asr: "GLM 转写", sensevoice_asr: "SenseVoice 转写",
  pcm_convert: "音频转换", resample: "重采样", normalize_report: "音频规范化报告",
  probe: "音频检查", audio_pass: "标记可用音频", select_pass: "保留可用音频",
};

export const DATASET_STAGE: Record<string, string> = {
  ingest: "入库清单", segments: "VAD 片段", segments_all: "VAD 全量备份",
  asr: "ASR 转写", dnsmos: "DNSMOS 评分", evidence: "合并证据", tags: "打标结果",
};

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() || path;
}

export function datasetLabel(dataset: DatasetSummary): string {
  const stage = DATASET_STAGE[dataset.stage || ""];
  const hint = [stage, dataset.asr_route].filter(Boolean).join(" · ");
  const version = dataset.job_id ? ` · ${dataset.job_id.slice(-6)}` : "";
  return `${hint ? `${hint} · ` : ""}${basename(dataset.path)}${version}`;
}

export function nodeLabel(node: PlanNode): string {
  if (node.model_route || node.node_id.startsWith("asr_")) {
    return `${node.model_route || node.node_id.slice(4)} 转写`;
  }
  return STEP_LABELS[node.node_id] || node.label || node.node_id;
}

export function descriptorNodes(
  plugin: PluginDescriptor | undefined, parameters: Record<string, unknown>,
): PlanNode[] {
  const dag = plugin?.detail?.dag_nodes as Array<{
    id: string; depends_on?: string[];
  }> | undefined;
  if (!dag) return [];
  const selected = Array.isArray(parameters.asr_route_ids) ? parameters.asr_route_ids : [];
  const filter = Boolean(parameters.label_xlsx);
  const nodes = dag
    .filter((node) => !node.id.startsWith("asr_") || !selected.length || selected.includes(node.id.slice(4)))
    .map((node) => ({
      node_id: node.id,
      label: node.id,
      depends_on: (node.depends_on || []).map((id) => filter && id === "vad" ? "filter_identity_verify" : id),
    }));
  if (filter) nodes.splice(nodes.findIndex((node) => node.node_id === "vad") + 1, 0, {
    node_id: "filter_identity_verify", label: "话术筛除", depends_on: ["vad"],
  });
  return nodes;
}

export function stageGroups(nodes: PlanNode[]): PlanNode[][] {
  const ids = new Set(nodes.map((node) => node.node_id));
  const done = new Set<string>();
  let remaining = nodes;
  const groups: PlanNode[][] = [];
  while (remaining.length) {
    const ready = remaining.filter((node) =>
      node.depends_on.every((dependency) => !ids.has(dependency) || done.has(dependency)),
    );
    if (!ready.length) return nodes.map((node) => [node]);
    groups.push(ready);
    for (const node of ready) done.add(node.node_id);
    remaining = remaining.filter((node) => !done.has(node.node_id));
  }
  return groups;
}

export function stageLabel(group: PlanNode[]): string {
  if (group.length === 1) return nodeLabel(group[0]);
  if (group.some((node) => node.node_id.startsWith("asr_")) && group.some((node) => node.node_id === "dnsmos")) {
    return "转写与音质评分";
  }
  return "并行处理";
}
