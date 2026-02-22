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
      const response = await fetch(`${API_URL}/discover`, {
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
          {loading ? "Analyzing..." : "Upload and Analyze"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section>
          <h2>Opportunities ({result.total_clusters})</h2>
          <ul>
            {result.opportunities.map((item) => (
              <li key={item.cluster_id}>
                <strong>{item.theme}</strong>
                <p>Mentions: {item.size}</p>
                <small>{item.representative_feedback}</small>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}
