import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";

const links = [
  ["数据源", "/sources"],
  ["模型连接", "/models"],
  ["插件库", "/plugins"],
  ["任务", "/jobs"],
  ["数据集", "/datasets"],
  ["评测", "/evaluations"],
] as const;

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="layout">
      <nav className="nav">
        <h1>Audio Data Engine</h1>
        {links.map(([label, to]) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) => (isActive ? "active" : undefined)}
          >
            {label}
          </NavLink>
        ))}
      </nav>
      <main className="main">{children}</main>
    </div>
  );
}
