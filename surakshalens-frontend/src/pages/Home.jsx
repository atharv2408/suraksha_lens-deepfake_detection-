// src/pages/Home.jsx

import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="max-w-6xl mx-auto pt-10 pb-20 text-gray-100">

      {/* HERO SECTION */}
      <section className="text-center mt-10">
        <h1 className="text-4xl md:text-6xl font-bold text-neonPurple drop-shadow-neon mb-4">
          Protect Your Digital Identity
        </h1>

        <p className="text-gray-300 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">
          SurakshaLens AI helps you detect deepfakes, generate cybercrime complaints,
          and stay safe online using state-of-the-art AI tools built for real people.
        </p>

        <div className="mt-8 flex justify-center gap-4">
          <Link
            to="/analyze"
            className="px-6 py-3 rounded-full bg-neonPurple text-white font-semibold shadow-neon hover:bg-purple-700"
          >
            Analyze Image
          </Link>

          <Link
            to="/complaint"
            className="px-6 py-3 rounded-full bg-cardDark border border-neonPurple/50 hover:bg-cardDark/70"
          >
            Generate Complaint
          </Link>
        </div>
      </section>

      {/* FEATURE GRID */}
      <section className="mt-20 grid md:grid-cols-3 gap-6">
        {/* Feature 1 */}
        <div className="rounded-2xl bg-cardDark/60 border border-neonPurple/30 p-6 shadow-neon hover:border-neonPurple transition">
          <h3 className="text-xl font-semibold text-neonPurple mb-2">Deepfake Analyzer</h3>
          <p className="text-gray-300 text-sm leading-relaxed">
            Upload an image and get instant deepfake probability, risk analysis,
            and detailed explanation using advanced AI detection models.
          </p>
          <Link to="/analyze" className="text-neonPurple mt-3 block text-sm hover:underline">
            Try Analyzer →
          </Link>
        </div>

        {/* Feature 2 */}
        <div className="rounded-2xl bg-cardDark/60 border border-neonPurple/30 p-6 shadow-neon hover:border-neonPurple transition">
          <h3 className="text-xl font-semibold text-neonPurple mb-2">Complaint Generator</h3>
          <p className="text-gray-300 text-sm leading-relaxed">
            File cybercrime complaints in minutes. Generate legally formatted drafts
            to submit to portals or police with supporting evidence.
          </p>
          <Link to="/complaint" className="text-neonPurple mt-3 block text-sm hover:underline">
            Generate Complaint →
          </Link>
        </div>

        {/* Feature 3 */}
        <div className="rounded-2xl bg-cardDark/60 border border-neonPurple/30 p-6 shadow-neon hover:border-neonPurple transition">
          <h3 className="text-xl font-semibold text-neonPurple mb-2">AI Legal Assistant</h3>
          <p className="text-gray-300 text-sm leading-relaxed">
            Understand your rights, IT Act sections, and legal options for digital crimes.
            A smart AI assistant (coming soon).
          </p>
          <span className="text-gray-500 mt-3 block text-sm italic">
            Coming soon…
          </span>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="text-center mt-20 text-xs text-gray-500">
        © {new Date().getFullYear()} SurakshaLens AI — Digital Safety for All.
      </footer>

    </div>
  );
}
