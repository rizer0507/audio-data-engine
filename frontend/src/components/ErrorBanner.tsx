import { useState } from "react";
import { ApiRequestError, copyText, formatApiError } from "../api";

export default function ErrorBanner({ error }: { error: unknown }) {
  const [copied, setCopied] = useState(false);
  if (!error) return null;
  const text = formatApiError(error);
  const diagnostic = error instanceof ApiRequestError ? error.diagnosticText() : text;

  return (
    <div className="error" role="alert">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <pre style={{ margin: 0, whiteSpace: "pre-wrap", font: "inherit" }}>{text}</pre>
        <button
          type="button"
          onClick={() => {
            void copyText(diagnostic).then((ok) => {
              setCopied(ok);
              window.setTimeout(() => setCopied(false), 1500);
            });
          }}
        >
          {copied ? "已复制" : "复制诊断"}
        </button>
      </div>
    </div>
  );
}
