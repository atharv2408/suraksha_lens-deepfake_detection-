import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();

  const linkClasses = (path) =>
    `text-lg ${
      location.pathname === path
        ? "text-neonPurple font-semibold"
        : "text-gray-300 hover:text-white"
    }`;

  return (
    <nav className="fixed w-full top-0 left-0 backdrop-blur bg-cardDark/40 border-b border-neonPurple/20 shadow-neon z-50">
      <div className="max-w-6xl mx-auto flex justify-between items-center py-4 px-6">
        <h1 className="text-2xl font-bold text-neonPurple drop-shadow-neon">
          SurakshaLens AI
        </h1>

        <div className="space-x-6">
          <Link to="/" className={linkClasses("/")}>Home</Link>
          <Link to="/analyze" className={linkClasses("/analyze")}>Analyzer</Link>
          <Link to="/complaint" className={linkClasses("/complaint")}>Complaint</Link>
        </div>
      </div>
    </nav>
  );
}
