// src/pages/Complaint.jsx
import { useState } from "react";
import { generateComplaint } from "../utils/api";

export default function Complaint() {
  const [form, setForm] = useState({
    incidentDescription: "",
    platform: "",
    victimAge: "",
    city: "",
    knownOffender: false,
    additionalDetails: "",
  });

  const [evidenceFiles, setEvidenceFiles] = useState([]);
  const [complaintText, setComplaintText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleEvidenceChange = (e) => {
    const files = Array.from(e.target.files || []);
    setEvidenceFiles(files);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setComplaintText("");

    let additional = form.additionalDetails || "";

    if (evidenceFiles.length > 0) {
      const names = evidenceFiles.map((f) => f.name).join(", ");
      const evidenceNote = `\n\nEvidence attached (not included in this text): ${names}`;
      additional += evidenceNote;
    }

    const payload = {
      incident_description: form.incidentDescription,
      platform: form.platform,
      victim_age: form.victimAge ? Number(form.victimAge) : null,
      city: form.city || null,
      known_offender: form.knownOffender,
      additional_details: additional || null,
    };

    try {
      const res = await generateComplaint(payload);
      setComplaintText(res.complaint_text || "");
    } catch (err) {
      console.error(err);
      setError(
        err?.response?.data?.detail ||
          "Unable to generate complaint at the moment."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    if (!complaintText) return;
    try {
      await navigator.clipboard.writeText(complaintText);
      alert("Complaint text copied to clipboard.");
    } catch {
      alert("Could not copy text.");
    }
  };

  const handleDownload = () => {
    if (!complaintText) return;
    const blob = new Blob([complaintText], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cyber_complaint.txt"; // you can change to .docx later
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto mt-4 grid md:grid-cols-[1.1fr,0.9fr] gap-6">
      {/* LEFT: Form */}
      <section>
        <h2 className="text-2xl font-semibold text-neonPurple mb-2">
          Complaint Generator
        </h2>
        <p className="text-sm text-gray-300 mb-4">
          Fill in the details of what happened. SurakshaLens will generate a
          structured cybercrime complaint draft that you can submit to official
          portals or local cyber cells.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Incident description */}
          <div>
            <label className="block text-sm text-gray-200 mb-1">
              Incident description *
            </label>
            <textarea
              name="incidentDescription"
              value={form.incidentDescription}
              onChange={handleChange}
              required
              className="w-full min-h-[100px] rounded-xl bg-cardDark border border-neonPurple/30 px-3 py-2 text-sm text-gray-100 outline-none focus:border-neonPurple"
              placeholder="Explain what happened, how the image/video was misused, where it was posted, and any impact it had."
            />
          </div>

          {/* Platform + city */}
          <div className="grid md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-200 mb-1">
                Platform *
              </label>
              <input
                name="platform"
                value={form.platform}
                onChange={handleChange}
                required
                className="w-full rounded-xl bg-cardDark border border-neonPurple/30 px-3 py-2 text-sm text-gray-100 outline-none focus:border-neonPurple"
                placeholder="e.g., Instagram, WhatsApp, Telegram"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-200 mb-1">
                City (optional)
              </label>
              <input
                name="city"
                value={form.city}
                onChange={handleChange}
                className="w-full rounded-xl bg-cardDark border border-gray-700 px-3 py-2 text-sm text-gray-100 outline-none focus:border-neonPurple"
                placeholder="e.g., Mumbai, Pune"
              />
            </div>
          </div>

          {/* Age + known offender */}
          <div className="grid md:grid-cols-2 gap-3">
            <div>
              <label className="block text-sm text-gray-200 mb-1">
                Your age (optional)
              </label>
              <input
                type="number"
                name="victimAge"
                value={form.victimAge}
                onChange={handleChange}
                className="w-full rounded-xl bg-cardDark border border-gray-700 px-3 py-2 text-sm text-gray-100 outline-none focus:border-neonPurple"
                placeholder="e.g., 20"
                min="1"
              />
            </div>
            <div className="flex items-center mt-5 gap-2">
              <input
                id="knownOffender"
                type="checkbox"
                name="knownOffender"
                checked={form.knownOffender}
                onChange={handleChange}
                className="w-4 h-4 rounded border-gray-600 bg-cardDark text-neonPurple"
              />
              <label htmlFor="knownOffender" className="text-xs text-gray-200">
                I know or strongly suspect who is responsible.
              </label>
            </div>
          </div>

          {/* Additional details */}
          <div>
            <label className="block text-sm text-gray-200 mb-1">
              Additional details (optional)
            </label>
            <textarea
              name="additionalDetails"
              value={form.additionalDetails}
              onChange={handleChange}
              className="w-full min-h-[80px] rounded-xl bg-cardDark border border-gray-700 px-3 py-2 text-sm text-gray-100 outline-none focus:border-neonPurple"
              placeholder="Links, usernames, previous incidents, whether you confronted them, etc."
            />
          </div>

          {/* Evidence upload */}
          <div>
            <label className="block text-sm text-gray-200 mb-1">
              Evidence screenshots (optional)
            </label>
            <label className="block rounded-xl border border-dashed border-neonPurple/40 bg-cardDark/70 px-3 py-4 text-xs text-gray-300 cursor-pointer hover:border-neonPurple">
              <span className="block mb-1">
                Click to select screenshots or images that support your
                complaint.
              </span>
              <span className="text-[11px] text-gray-500">
                These are not uploaded to the server in this version. The
                generated complaint will include their file names so you can
                attach them when submitting.
              </span>
              <input
                type="file"
                multiple
                accept="image/*"
                className="hidden"
                onChange={handleEvidenceChange}
              />
            </label>
            {evidenceFiles.length > 0 && (
              <ul className="mt-1 text-xs text-gray-400 list-disc list-inside">
                {evidenceFiles.map((f, idx) => (
                  <li key={idx}>{f.name}</li>
                ))}
              </ul>
            )}
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-2 px-6 py-2.5 rounded-full bg-neonPurple text-white text-sm font-semibold shadow-neon disabled:opacity-50 disabled:shadow-none"
          >
            {loading ? "Generating complaint…" : "Generate complaint draft"}
          </button>

          {error && (
            <p className="mt-2 text-xs text-red-400 bg-red-900/30 border border-red-500/50 rounded-lg px-3 py-2">
              {error}
            </p>
          )}
        </form>
      </section>

      {/* RIGHT: Legal-style preview */}
      <section className="mt-6 md:mt-0">
        <h2 className="text-lg font-semibold text-neonPurple mb-2">
          Complaint preview
        </h2>
        <p className="text-xs text-gray-400 mb-2">
          This is a plain-text draft. You can copy it into the National
          Cybercrime Reporting Portal or print it for offline submission.
        </p>

        <div className="rounded-2xl border border-neonPurple/40 shadow-neon bg-cardDark/80 p-3">
          <div className="bg-white text-black font-serif rounded-xl p-5 max-h-[360px] overflow-y-auto text-sm leading-relaxed">
            {complaintText ? (
              <pre className="whitespace-pre-wrap">{complaintText}</pre>
            ) : (
              <p className="text-gray-700">
                Your formatted complaint letter will appear here after you fill
                in the form and click{" "}
                <span className="font-semibold">“Generate complaint draft”</span>.
              </p>
            )}
          </div>
          <div className="mt-3 flex gap-2 justify-end">
            <button
              onClick={handleCopy}
              disabled={!complaintText}
              className="px-3 py-1.5 rounded-full text-xs bg-cardDark border border-gray-600 text-gray-100 disabled:opacity-40"
            >
              Copy text
            </button>
            <button
              onClick={handleDownload}
              disabled={!complaintText}
              className="px-3 py-1.5 rounded-full text-xs bg-neonPurple text-white disabled:opacity-40"
            >
              Download (.txt)
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
