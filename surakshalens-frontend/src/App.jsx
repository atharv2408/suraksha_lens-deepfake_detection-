import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Analyzer from "./pages/Analyzer";
import Complaint from "./pages/Complaint"; // ✅ ADD THIS

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <div className="pt-20 px-4">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/analyze" element={<Analyzer />} />
          <Route path="/complaint" element={<Complaint />} /> {/* ✅ ADD ROUTE */}
        </Routes>
      </div>
    </BrowserRouter>
  );
}
