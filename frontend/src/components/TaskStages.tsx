import type { NodeProgress, PlanNode } from "../types";
import { JOB_STATUS, nodeLabel, stageGroups, stageLabel } from "../workflows";
import TaskStatus from "./TaskStatus";

function SampleProgress({ node }: { node?: NodeProgress }) {
  if (node?.processed == null || !node.total) return null;
  return (
    <div className="sample-progress">
      <progress aria-label="已记录的输入处理数" value={node.processed} max={node.total} />
      <span>{node.processed.toLocaleString()} / {node.total.toLocaleString()}</span>
    </div>
  );
}

export default function TaskStages({ nodes, progress, status, onNode }: {
  nodes: PlanNode[];
  progress?: Record<string, NodeProgress>;
  status?: string;
  onNode?: (nodeId: string) => void;
}) {
  const planned = status === undefined;
  const state = (node: PlanNode) => {
    const reported = progress?.[node.node_id]?.status;
    if (reported === "running" && status === "interrupted") return "interrupted";
    if (reported === "running" && status === "cancelled") return "cancelled";
    return reported || (status === "succeeded" ? "succeeded" : "pending");
  };
  const done = nodes.filter((node) => state(node) === "succeeded").length;
  return (
    <div className="task-stages">
      {!planned && (progress || status === "succeeded" || status === "queued") && (
        <div className="node-completion">
          <span>节点完成 <strong>{done} / {nodes.length}</strong></span>
          <progress aria-label="已完成节点数" value={done} max={nodes.length} />
        </div>
      )}
      <ol className="stage-list">
        {stageGroups(nodes).map((group, index) => {
          const states = group.map(state);
          const stage = ["failed", "interrupted", "running", "cancelled"].find((value) => states.includes(value))
            ?? (states.every((value) => value === "succeeded") ? "succeeded" : "pending");
          return (
            <li className={`stage ${planned ? "planned" : stage}`} key={group[0].node_id}>
              <span className="stage-marker" aria-hidden="true">{stage === "succeeded" ? "✓" : index + 1}</span>
              <div className="stage-content">
                <div className="stage-heading">
                  {group.length === 1 && onNode && progress?.[group[0].node_id]?.started_at ? (
                    <button className="linkish stage-title" type="button" onClick={() => onNode(group[0].node_id)}>{stageLabel(group)}</button>
                  ) : <strong>{stageLabel(group)}</strong>}
                  {group.length > 1 && <span className="help">并行</span>}
                  {!planned && (progress || status === "succeeded" || status === "queued") && <TaskStatus status={stage} />}
                </div>
                {group.length === 1 && <SampleProgress node={progress?.[group[0].node_id]} />}
                {group.length > 1 && (
                  <ul className="stage-branches">
                    {group.map((node) => (
                      <li key={node.node_id}>
                        {onNode && progress?.[node.node_id]?.started_at ? (
                          <button className="linkish" type="button" onClick={() => onNode(node.node_id)}>{nodeLabel(node)}</button>
                        ) : <span>{nodeLabel(node)}</span>}
                        {!planned && progress && <span className={`branch-state ${state(node)}`}>{JOB_STATUS[state(node)] || state(node)}</span>}
                        <SampleProgress node={progress?.[node.node_id]} />
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
