import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Sources from "./pages/Sources";
import Models from "./pages/Models";
import Plugins from "./pages/Plugins";
import Jobs from "./pages/Jobs";
import JobDetail from "./pages/JobDetail";
import Datasets from "./pages/Datasets";
import DatasetDetail from "./pages/DatasetDetail";
import Evaluations from "./pages/Evaluations";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/sources" replace />} />
        <Route path="/sources" element={<Sources />} />
        <Route path="/models" element={<Models />} />
        <Route path="/plugins" element={<Plugins />} />
        <Route path="/jobs" element={<Jobs />} />
        <Route path="/jobs/:jobId" element={<JobDetail />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/datasets/*" element={<DatasetDetail />} />
        <Route path="/evaluations" element={<Evaluations />} />
      </Routes>
    </Layout>
  );
}
