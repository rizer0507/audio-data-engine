import { useState } from "react";

export default function AudioPlayer({
  src,
  label,
}: {
  src?: string | null;
  label?: string;
}) {
  const [err, setErr] = useState("");
  if (!src) {
    return (
      <p className="muted">
        {label || "无可播放音频资产"}（需后端 /audio 接口解析已登记路径；不能用浏览器本地路径读服务器文件）
      </p>
    );
  }
  return (
    <div>
      {label && <div className="help">{label}</div>}
      <audio
        controls
        preload="metadata"
        src={src}
        style={{ width: "100%" }}
        onError={() => setErr("音频加载失败：可能是路径未登记、PCM 缺参数或格式浏览器不支持")}
      />
      {err && <div className="error" style={{ marginTop: "0.5rem" }}>{err}</div>}
    </div>
  );
}
