import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export function App() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onSubmit = async (event) => {
    event.preventDefault();
    if (!file) return;

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail ?? "Request failed");
      }

      const payload = await response.json();
      setResult(payload);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1>AI Product Discovery MVP</h1>
      <form onSubmit={onSubmit}>
        <input
          type="file"
          accept=".csv"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <button type="submit" disabled={!file || loading}>
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h2>Top Opportunity</h2>
          <p>{result.top_opportunities[0]?.theme_label ?? "N/A"}</p>

          <h2>Recommended Action</h2>
          <p>{result.recommended_action}</p>

          <h2>Evidence</h2>
          <ul>
            {result.evidence.map((item, idx) => (
              <li key={`${item.slice(0, 24)}-${idx}`}>{item}</li>
            ))}
          </ul>

          <div>
            <a href={`${API_URL}/download/prd`} target="_blank" rel="noreferrer">
              <button type="button">Download PRD</button>
            </a>
            <a href={`${API_URL}/download/jira`} target="_blank" rel="noreferrer">
              <button type="button">Download Jira Tickets</button>
            </a>
          </div>
        </section>
      )}
    </main>
  );
}
