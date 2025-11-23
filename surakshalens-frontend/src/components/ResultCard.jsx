function getRiskColor(level) {
  if (level === "high") return "text-red-400";
  if (level === "medium") return "text-yellow-300";
  return "text-green-400";
}

function getRiskBg(level) {
  if (level === "high") return "bg-red-900/40 border-red-500/60";
  if (level === "medium") return "bg-yellow-900/30 border-yellow-400/60";
  return "bg-green-900/30 border-green-500/60";
}

function getRecommendations(level) {
  if (level === "high") {
    return [
      "Do not share or forward this image.",
      "Collect evidence (screenshots, links, usernames).",
      "Consider filing a cybercrime complaint as soon as possible.",
    ];
  }
  if (level === "medium") {
    return [
      "Be cautious before trusting or forwarding this image.",
      "Verify with trusted sources or the person involved.",
      "If it targets you personally, keep a record and monitor further misuse.",
    ];
  }
  return [
    "The image appears low-risk, but it’s still good to stay cautious.",
    "Avoid sharing sensitive content publicly.",
  ];
}

export default function ResultCard({ result }) {
  if (!result) {
    return (
      <div className="p-6 rounded-2xl bg-cardDark/80 border border-gray-700/60">
        <h2 className="text-lg font-semibold text-gray-200 mb-2">
          Risk dashboard
        </h2>
        <p className="text-sm text-gray-400">
          Once you upload and analyze an image, this panel will show a visual
          risk meter, probability, and recommended next steps.
        </p>
      </div>
    );
  }

  const pct = Math.round(result.fake_probability * 100);
  const riskLevel = result.risk_level || "low";
  const recs = getRecommendations(riskLevel);

  return (
    <div className="p-6 rounded-2xl bg-cardDark/90 border border-neonPurple/40 shadow-neon">
      <h2 className="text-lg font-semibold text-neonPurple mb-4">
        Deepfake risk dashboard
      </h2>

      {/* top: gauge + stats */}
      <div className="flex items-center gap-5">
        {/* Neon circular gauge */}
        <div className="relative w-32 h-32">
          <div
            className="absolute inset-0 rounded-full"
            style={{
              background: `conic-gradient(#22c55e 0% ${Math.min(
                100,
                pct
              )}%, #27272f ${Math.min(100, pct)}% 100%)`,
            }}
          />
          <div className="absolute inset-[10px] rounded-full bg-cardDark flex items-center justify-center border border-neonPurple/60">
            <span className="text-2xl font-semibold">{pct}%</span>
          </div>
        </div>

        {/* textual stats */}
        <div className="flex-1 space-y-2">
          <div
            className={`inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold border ${getRiskBg(
              riskLevel
            )}`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                riskLevel === "high"
                  ? "bg-red-400"
                  : riskLevel === "medium"
                  ? "bg-yellow-300"
                  : "bg-green-400"
              }`}
            />
            <span className={getRiskColor(riskLevel)}>
              RISK: {riskLevel.toUpperCase()}
            </span>
          </div>

          <p className="text-sm text-gray-300">
            <span className="font-semibold">Estimated probability:</span>{" "}
            {pct}% that this image shares patterns with known deepfakes.
          </p>
          <p className="text-xs text-gray-500">
            Model:{" "}
            <span className="text-gray-300 font-mono">
              {result.model_used || "UnknownModel"}
            </span>
          </p>
        </div>
      </div>

      {/* explanation */}
      <div className="mt-5">
        <p className="text-sm text-gray-300">{result.explanation}</p>
      </div>

      {/* recommendations */}
      <div className="mt-5">
        <p className="text-xs uppercase tracking-wide text-gray-400 mb-1">
          Recommended next steps
        </p>
        <ul className="text-sm text-gray-300 space-y-1.5 list-disc list-inside">
          {recs.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </div>

      <p className="mt-4 text-[11px] text-gray-500">
        This is an AI-based estimation, not legal or forensic proof. If this
        content is harming you or someone you know, consider reporting it via
        official cybercrime channels.
      </p>
    </div>
  );
}
