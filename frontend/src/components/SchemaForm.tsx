import { useMemo, useState } from "react";

type JsonSchema = {
  type?: string | string[];
  title?: string;
  description?: string;
  default?: unknown;
  enum?: unknown[];
  minimum?: number;
  maximum?: number;
  minItems?: number;
  maxItems?: number;
  required?: string[];
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  additionalProperties?: boolean | JsonSchema;
};

function primaryType(schema: JsonSchema): string {
  if (Array.isArray(schema.type)) return schema.type.find((t) => t !== "null") || "string";
  return schema.type || (schema.properties ? "object" : schema.enum ? "string" : "string");
}

function isComplex(schema: JsonSchema): boolean {
  const t = primaryType(schema);
  if (t === "object") {
    const props = schema.properties || {};
    return Object.values(props).some((p) => {
      const pt = primaryType(p);
      return pt === "object" || (pt === "array" && p.items && primaryType(p.items) === "object");
    });
  }
  if (t === "array" && schema.items && primaryType(schema.items) === "object") return true;
  return false;
}

function Field({
  name,
  schema,
  value,
  required,
  onChange,
}: {
  name: string;
  schema: JsonSchema;
  value: unknown;
  required?: boolean;
  onChange: (v: unknown) => void;
}) {
  const title = schema.title || name;
  const t = primaryType(schema);
  const help = schema.description;

  if (schema.enum) {
    return (
      <label>
        {title}
        {required ? " *" : ""}
        <select
          value={value === undefined || value === null ? "" : String(value)}
          onChange={(e) => {
            const raw = e.target.value;
            const matched = schema.enum!.find((opt) => String(opt) === raw);
            onChange(matched);
          }}
        >
          <option value="">选择…</option>
          {schema.enum.map((opt) => (
            <option key={String(opt)} value={String(opt)}>
              {String(opt)}
            </option>
          ))}
        </select>
        {help && <span className="help">{help}</span>}
      </label>
    );
  }

  if (t === "boolean") {
    return (
      <label className="row">
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
        />
        {title}
        {required ? " *" : ""}
        {help && <span className="help">{help}</span>}
      </label>
    );
  }

  if (t === "number" || t === "integer") {
    return (
      <label>
        {title}
        {required ? " *" : ""}
        <input
          type="number"
          step={t === "integer" ? 1 : "any"}
          min={schema.minimum}
          max={schema.maximum}
          value={value === undefined || value === null ? "" : Number(value)}
          onChange={(e) =>
            onChange(e.target.value === "" ? undefined : Number(e.target.value))
          }
        />
        {help && <span className="help">{help}</span>}
      </label>
    );
  }

  if (t === "array") {
    const itemType = schema.items ? primaryType(schema.items) : "string";
    if (itemType === "string" || itemType === "number" || itemType === "integer") {
      const asText = Array.isArray(value) ? value.join(", ") : "";
      return (
        <label>
          {title}
          {required ? " *" : ""}（逗号分隔）
          <input
            value={asText}
            onChange={(e) => {
              const parts = e.target.value
                .split(",")
                .map((x) => x.trim())
                .filter(Boolean);
              onChange(
                itemType === "string"
                  ? parts
                  : parts.map((p) => Number(p)).filter((n) => !Number.isNaN(n)),
              );
            }}
          />
          {help && <span className="help">{help}</span>}
        </label>
      );
    }
  }

  if (t === "object" && schema.properties && !isComplex(schema)) {
    const obj = (value && typeof value === "object" ? value : {}) as Record<string, unknown>;
    const req = new Set(schema.required || []);
    return (
      <fieldset className="form-grid" style={{ border: "1px solid var(--line)", borderRadius: 8, padding: "0.75rem" }}>
        <legend>{title}</legend>
        {help && <span className="help">{help}</span>}
        {Object.entries(schema.properties).map(([k, child]) => (
          <Field
            key={k}
            name={k}
            schema={child}
            required={req.has(k)}
            value={obj[k] ?? child.default}
            onChange={(v) => onChange({ ...obj, [k]: v })}
          />
        ))}
      </fieldset>
    );
  }

  return (
    <label>
      {title}
      {required ? " *" : ""}
      <input
        value={value === undefined || value === null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value)}
      />
      {help && <span className="help">{help}</span>}
    </label>
  );
}

export default function SchemaForm({
  schema,
  value,
  onChange,
}: {
  schema: Record<string, unknown>;
  value: Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
}) {
  const s = schema as JsonSchema;
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [jsonError, setJsonError] = useState("");

  const complex = useMemo(() => isComplex(s), [s]);
  const props = s.properties || {};
  const keys = Object.keys(props);
  const required = new Set(s.required || []);

  if (!keys.length && (s.additionalProperties === false || !s.additionalProperties)) {
    return <p className="muted">无可配置参数</p>;
  }

  if (complex || jsonMode || (!keys.length && s.additionalProperties)) {
    const text = jsonMode || complex ? jsonText || JSON.stringify(value, null, 2) : JSON.stringify(value, null, 2);
    return (
      <div className="form-grid">
        <div className="row">
          <span className="help">
            {complex
              ? "该 schema 含嵌套结构，使用受约束的 JSON 编辑区；仍由后端 schema 校验。"
              : "JSON 参数编辑区"}
          </span>
          {keys.length > 0 && !complex && (
            <button
              type="button"
              onClick={() => {
                setJsonMode(false);
                setJsonError("");
              }}
            >
              切回表单
            </button>
          )}
        </div>
        <textarea
          rows={10}
          value={jsonMode || complex ? text : JSON.stringify(value, null, 2)}
          onChange={(e) => {
            setJsonMode(true);
            setJsonText(e.target.value);
            try {
              const parsed = JSON.parse(e.target.value) as Record<string, unknown>;
              setJsonError("");
              onChange(parsed);
            } catch (err) {
              setJsonError(err instanceof Error ? err.message : "JSON 无效");
            }
          }}
        />
        {jsonError && <div className="error">{jsonError}</div>}
      </div>
    );
  }

  return (
    <div className="form-grid">
      <div className="row">
        <button type="button" onClick={() => { setJsonMode(true); setJsonText(JSON.stringify(value, null, 2)); }}>
          高级 JSON
        </button>
      </div>
      {keys.map((key) => (
        <Field
          key={key}
          name={key}
          schema={props[key]}
          required={required.has(key)}
          value={value[key] ?? props[key].default}
          onChange={(v) => onChange({ ...value, [key]: v })}
        />
      ))}
    </div>
  );
}

/** Apply schema defaults into an object (shallow + one nested object level). */
export function defaultsFromSchema(schema: Record<string, unknown>): Record<string, unknown> {
  const s = schema as JsonSchema;
  const out: Record<string, unknown> = {};
  const props = s.properties || {};
  for (const [k, child] of Object.entries(props)) {
    if (child.default !== undefined) {
      out[k] = child.default;
    } else if (primaryType(child) === "object" && child.properties) {
      const nested = defaultsFromSchema(child as unknown as Record<string, unknown>);
      if (Object.keys(nested).length) out[k] = nested;
    } else if (primaryType(child) === "boolean") {
      out[k] = false;
    }
  }
  return out;
}
