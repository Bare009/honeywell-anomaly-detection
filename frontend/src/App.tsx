import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Alerts from "./pages/Alerts";
import EntityExplorer from "./pages/EntityExplorer";
import Storyline from "./pages/Storyline";
import ModelPerformance from "./pages/ModelPerformance";
import Drift from "./pages/Drift";
import SystemHealth from "./pages/SystemHealth";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/entities" element={<EntityExplorer />} />
        <Route path="/entities/:entityId" element={<EntityExplorer />} />
        <Route path="/storyline" element={<Storyline />} />
        <Route path="/performance" element={<ModelPerformance />} />
        <Route path="/drift" element={<Drift />} />
        <Route path="/system" element={<SystemHealth />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
