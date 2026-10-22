export type ApiError = {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
    request_id?: string | null;
  };
};

export type Capabilities = {
  providers: string[];
  evidence_modes: string[];
  task_types: string[];
  input_kinds: string[];
  job_statuses: string[];
  pcm_required_fields: string[];
  operator_id: string;
};

export type Source = {
  source_id: string;
  batch?: string | null;
  source_name: string;
  name?: string;
  path: string;
  format: string;
  sample_rate?: number | null;
  bit_depth?: number | null;
  channels?: number | null;
  note?: string;
  origin?: string;
  ingested_at?: string;
  file_count?: number | null;
  last_check?: SourceCheckResult | null;
  revision?: number;
  reused?: boolean;
};

export type SourceCheckResult = {
  source_id: string;
  path: string;
  exists: boolean;
  readable: boolean;
  sample_file_count: number;
  sampled_files: string[];
  full_scan: boolean;
  status: string;
  checked_at: string;
  error?: string;
  note?: string;
};

export type ModelConnection = {
  connection_id: string;
  name: string;
  provider: string;
  base_url: string;
  model: string;
  model_revision: string;
  api_key_env: string;
  api_key_configured?: boolean;
  timeout_seconds: number;
  concurrency: number;
};

export type ModelCheckResult = {
  connection_id: string;
  level: string;
  base_url: string;
  configured_model: string;
  provider: string;
  ok: boolean;
  checked_at: string;
  http_status?: number;
  returned_model_ids?: string[];
  latency_ms?: number;
  code?: string;
  message?: string;
  identity_scope?: string;
  transcript?: string;
  [key: string]: unknown;
};

export type PluginSummary = {
  kind?: string;
  name?: string;
  purpose?: string;
};

export type PluginDescriptor = {
  kind: "pipeline" | "tag";
  id: string;
  version: string;
  name: string;
  description: string;
  description_updated_at?: string | null;
  adapter: string;
  status: string;
  runnable: boolean;
  not_runnable_reasons?: string[];
  input_kinds: string[];
  output_kinds: string[];
  parameters_schema: Record<string, unknown>;
  capabilities: {
    supports_resume: boolean;
    supports_cancel: boolean;
    supports_tag_selection: boolean;
    supports_isolated_output?: boolean;
    purpose: string;
    notes?: string[];
  };
  definition_path?: string;
  catalog_generation?: number;
  revision?: number;
  detail?: Record<string, unknown>;
  diagnostics?: unknown[];
  definition_digest?: string;
  summary?: PluginSummary;
  editable?: Record<string, unknown>;
  edit_schema?: Record<string, unknown>;
  readonly_fields?: string[];
};

export type PluginRef = {
  kind: string;
  id: string;
  version: string;
};

export type ModelBinding = {
  route_id: string;
  connection_id: string;
};

export type TaskRequest = {
  task_type: "pipeline" | "tagging";
  root_plugin: PluginRef | null;
  input: { kind: "source"; source_id: string } | { kind: "dataset"; dataset_id: string };
  parameters: Record<string, unknown>;
  model_bindings: ModelBinding[];
  tags: PluginRef[];
  execution: Record<string, unknown>;
};

export type ExecutionPlan = {
  plan_id: string;
  plan_digest: string;
  expires_at?: string;
  catalog_generation?: number;
  nodes?: unknown[];
  resolved_models?: unknown[];
  selected_tags?: unknown[];
  evidence_union?: unknown;
  warnings?: unknown[];
  supports_cancel?: boolean;
  supports_resume?: boolean;
  [key: string]: unknown;
};

export type Job = {
  job_id: string;
  task_type: string;
  status: string;
  plan_id: string;
  plan_digest: string;
  current_attempt: number;
  request: Record<string, unknown>;
  progress?: Record<string, unknown> | null;
  artifacts?: unknown[];
  error?: Record<string, unknown> | null;
  snapshot_path?: string | null;
  created_at: string;
  updated_at: string;
  attempts?: Array<Record<string, unknown>>;
  capabilities?: {
    supports_cancel?: boolean;
    supports_resume?: boolean;
  };
};

export type JobLogsResponse = {
  chunks: Array<{ text: string; offset?: number }>;
  next_cursor: string;
  truncated?: boolean;
};

export type DatasetSummary = {
  dataset_id: string;
  path: string;
  format?: string;
  registered?: boolean;
  note?: string;
  state_kind?: string;
  has_work_copy?: boolean;
  work_dataset_id?: string | null;
  tagging_binding?: Record<string, unknown> | null;
};

export type SampleRow = {
  sample_id: string;
  duration?: number | null;
  tags?: string[];
  status?: Record<string, string>;
  ternary?: Record<string, string>;
  quality?: Record<string, unknown>;
  annotation_state?: string | null;
  gold_kind?: string | null;
  transcript_preview?: { route: string; text: string } | null;
  lineage_parents?: Record<string, unknown>;
};

export type ListResponse<T> = {
  items: T[];
  next_cursor?: string | number | null;
  total?: number | null;
  generation?: number;
  revision?: number;
};
