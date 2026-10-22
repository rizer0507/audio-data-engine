export default function TaskStatus({ status }: { status: string }) {
  return <span className={`badge ${status}`}>{status}</span>;
}
