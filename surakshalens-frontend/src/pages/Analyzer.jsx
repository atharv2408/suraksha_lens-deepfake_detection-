import { useState } from "react";
import UploadCard from "../components/UploadCard";
import ResultCard from "../components/ResultCard";
import Loader from "../components/Loader";
import { analyzeImage } from "../utils/api";

export default function Analyzer() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const onSelectFile = (f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError("");
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err) {
      console.error(err);
      setError(
        err?.response?.data?.detail ||
          "Something went wrong while analyzing the image."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-5xl mx-auto grid md:grid-cols-[1.1fr,0.9fr] gap-8 mt-4">
      {/* LEFT: upload + preview */}
      <div>
        <h2 className="text-2xl font-semibold text-neonPurple mb-2">
          Deepfake Analyzer
        </h2>
        <p className="text-sm text-gray-300 mb-4">
          Upload an image that you suspect might be AI-generated or manipulated.
          We’ll run it through the SurakshaLens model and estimate its deepfake
          risk.
        </p>

        <UploadCard onSelect={onSelectFile} />

        {previewUrl && (
          <div className="mt-4">
            <p className="text-sm text-gray-400 mb-1">Preview</p>
            <div className="rounded-xl border border-neonPurple/30 bg-cardDark/80 overflow-hidden">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full max-h-80 object-contain"
              />
            </div>
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="mt-5 px-6 py-3 rounded-lg bg-neonPurple text-white font-semibold shadow-neon disabled:opacity-40 disabled:shadow-none"
        >
          {loading ? "Analyzing…" : "Analyze Image"}
        </button>

        {error && (
          <p className="mt-3 text-sm text-red-400 bg-red-900/30 border border-red-500/40 rounded-lg px-3 py-2">
            {error}
          </p>
        )}

        {loading && <Loader />}
      </div>

      {/* RIGHT: result dashboard */}
      <div>
        <ResultCard result={result} />
      </div>
    </div>
  );
}
