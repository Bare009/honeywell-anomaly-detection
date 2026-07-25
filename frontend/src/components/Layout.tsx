import { NavLink } from "react-router-dom";
import type { ReactNode } from "react";
import {
  Activity,
  AlertTriangle,
  GitBranch,
  Home,
  LineChart,
  Radar,
  ShieldCheck,
  Users,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Overview", icon: Home, end: true },
  { to: "/alerts", label: "Ranked Alerts", icon: AlertTriangle, end: false },
  { to: "/entities", label: "Entity Explorer", icon: Users, end: false },
  { to: "/storyline", label: "Storyline", icon: GitBranch, end: false },
  { to: "/performance", label: "Model Performance", icon: LineChart, end: false },
  { to: "/drift", label: "Drift Monitor", icon: Radar, end: false },
  { to: "/system", label: "System Health", icon: Activity, end: false },
];

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-200">
      <aside className="flex w-60 flex-col border-r border-slate-800 bg-slate-900/50">
        <div className="flex items-center gap-2 px-4 py-4 text-slate-100">
          <ShieldCheck className="h-6 w-6 text-sky-400" />
          <div className="leading-tight">
            <div className="text-sm font-semibold">Anomaly SOC</div>
            <div className="text-xs text-slate-500">Behavioral detection</div>
          </div>
        </div>
        <nav className="flex-1 space-y-1 px-2">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                  isActive
                    ? "bg-sky-500/15 text-sky-300"
                    : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200"
                }`
              }
            >
              <Icon className="h-4 w-4" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="px-4 py-3 text-xs text-slate-600">Synthetic data · seed 42</div>
      </aside>
      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-6 py-6">{children}</div>
      </main>
    </div>
  );
}
