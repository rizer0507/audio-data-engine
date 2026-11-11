import { JOB_STATUS } from "../workflows";

export default function TaskStatus({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{JOB_STATUS[status] || status}</span>;
}
