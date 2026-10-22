import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  // Nginx 在原路径前加 /asr_data/。资源、路由都从这里发出，刷新深链才能回到本应用。
  base: "/asr_data/",
  server: {
    port: 5173,
    proxy: {
      "/asr_data/api": {
        target: "http://127.0.0.1:5599",
        rewrite: (path) => path.replace(/^\/asr_data/, ""),
      },
      "/asr_data/health": {
        target: "http://127.0.0.1:5599",
        rewrite: (path) => path.replace(/^\/asr_data/, ""),
      },
    },
  },
});
