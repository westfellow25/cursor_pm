#!/usr/bin/env bash
set -e

echo "══════════════════════════════════════════════"
echo "  Pulse — Product Intelligence Platform setup "
echo "══════════════════════════════════════════════"

echo ""
echo "→ Installing Python dependencies..."
pip install --upgrade pip
pip install -e .

echo ""
echo "→ Installing frontend dependencies..."
cd frontend && npm install && cd ..

echo ""
echo "→ Building frontend..."
cd frontend && npm run build && cd ..

echo ""
echo "→ Creating .env from example..."
if [ ! -f .env ]; then
  cp .env.example .env
  # Generate a random SECRET_KEY
  SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
  sed -i "s|SECRET_KEY=change-me-to-a-random-string|SECRET_KEY=$SECRET|" .env
  echo "  ✓ .env created with random SECRET_KEY"
  echo "  ⚠ Remember to add ANTHROPIC_API_KEY (or OPENAI_API_KEY) for AI features"
fi

echo ""
echo "→ Pre-downloading local embedding model (MiniLM, ~90 MB)..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || echo "  (model download skipped, will fetch on first use)"

echo ""
echo "→ Seeding demo data (2500 feedback items, 6 months)..."
python -m scripts.seed || echo "  (already seeded)"

echo ""
echo "══════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  Start the server:"
echo "    uvicorn pulse.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Open the forwarded port 8000 in your browser."
echo ""
echo "  Login: demo@acme.com / demo1234"
echo "══════════════════════════════════════════════"
