import type { ApiError } from "./types";

export class ApiRequestError extends Error {
  code: string;
  details: Record<string, unknown>;
  requestId: string | null;
  status: number;

  constructor(status: number, body: ApiError) {
    super(body.error?.message || `HTTP ${status}`);
    this.name = "ApiRequestError";
    this.status = status;
    this.code = body.error?.code || "HTTP_ERROR";
    this.details = body.error?.details || {};
    this.requestId = body.error?.request_id || null;
  }

  fieldErrors(): Array<{ path: string; message: string }> {
    const out: Array<{ path: string; message: string }> = [];
    const errors = this.details.errors;
    if (Array.isArray(errors)) {
      for (const e of errors) {
        if (e && typeof e === "object") {
          const item = e as Record<string, unknown>;
          const loc = Array.isArray(item.loc) ? item.loc.join(".") : String(item.field_path || item.loc || "");
          out.push({ path: loc, message: String(item.msg || item.message || JSON.stringify(e)) });
        }
      }
    }
    if (typeof this.details.field_path === "string") {
      out.push({ path: this.details.field_path, message: this.message });
    }
    return out;
  }

  diagnosticText(): string {
    return [
      `code=${this.code}`,
      `status=${this.status}`,
      `message=${this.message}`,
      this.requestId ? `request_id=${this.requestId}` : null,
      Object.keys(this.details).length ? `details=${JSON.stringify(this.details)}` : null,
    ]
      .filter(Boolean)
      .join("\n");
  }
}

/** Vite `base`，例如 `/asr_data`。与 Nginx 前缀一致。 */
export function publicPath(path: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, "");
  if (!path.startsWith("/") || !base) return path;
  if (path === base || path.startsWith(`${base}/`)) return path;
  return `${base}${path}`;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {});
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(publicPath(path), { ...init, headers });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: { code: "NON_JSON", message: text.slice(0, 500) } };
  }
  if (!res.ok) {
    throw new ApiRequestError(res.status, data as ApiError);
  }
  return data as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown, headers?: Record<string, string>) =>
    request<T>(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
      headers,
    }),
  patch: <T>(path: string, body: unknown, headers?: Record<string, string>) =>
    request<T>(path, {
      method: "PATCH",
      body: JSON.stringify(body),
      headers,
    }),
};

export function formatApiError(err: unknown): string {
  if (err instanceof ApiRequestError) {
    const fields = err.fieldErrors();
    const fieldPart = fields.length
      ? "\n" + fields.map((f) => `  · ${f.path || "(root)"}: ${f.message}`).join("\n")
      : "";
    const rid = err.requestId ? ` request_id=${err.requestId}` : "";
    return `[${err.code}] ${err.message}${rid}${fieldPart}`;
  }
  if (err instanceof Error) return err.message;
  return String(err);
}

export function newIdempotencyKey(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `idem_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export function durationBetween(start?: string | null, end?: string | null): string {
  if (!start) return "—";
  const a = new Date(start).getTime();
  const b = end ? new Date(end).getTime() : Date.now();
  if (Number.isNaN(a) || Number.isNaN(b)) return "—";
  const sec = Math.max(0, Math.round((b - a) / 1000));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m < 60) return `${m}m ${s}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

export function audioUrl(datasetId: string, sampleId: string, asset = "raw"): string {
  return publicPath(
    `/api/v1/datasets/${encodeURIComponent(datasetId)}/samples/${encodeURIComponent(sampleId)}/audio?asset=${encodeURIComponent(asset)}`,
  );
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
