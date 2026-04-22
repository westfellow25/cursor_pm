import { useState, useEffect, useRef } from 'react'
import {
  MessageSquare,
  Plug,
  Target,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Lightbulb,
  Upload,
  Play,
  FileText,
  ListChecks,
  Copy,
  Download,
  Check,
  ChevronDown,
  Zap,
  Sparkles,
  ArrowRight,
} from 'lucide-react'
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api/client'

const severityColors = {
  critical: 'border-red-400 bg-red-50 text-red-900',
  warning: 'border-amber-400 bg-amber-50 text-amber-900',
  info: 'border-blue-400 bg-blue-50 text-blue-900',
  positive: 'border-emerald-400 bg-emerald-50 text-emerald-900',
}

function clusterAccent(score) {
  if (score >= 7) return { border: 'border-red-300', bg: 'from-red-50 to-white', score: 'text-red-700', chip: 'bg-red-100 text-red-800' }
  if (score >= 4) return { border: 'border-amber-300', bg: 'from-amber-50 to-white', score: 'text-amber-700', chip: 'bg-amber-100 text-amber-800' }
  return { border: 'border-emerald-300', bg: 'from-emerald-50 to-white', score: 'text-emerald-700', chip: 'bg-emerald-100 text-emerald-800' }
}

function StatCard({ icon: Icon, label, value, sub, color = 'pulse' }) {
  const colors = {
    pulse: 'bg-pulse-50 text-pulse-600',
    green: 'bg-emerald-50 text-emerald-600',
    amber: 'bg-amber-50 text-amber-600',
    red: 'bg-red-50 text-red-600',
  }
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${colors[color]}`}>
          <Icon className="w-4 h-4" />
        </div>
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className="text-xl font-bold">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-0.5">{sub}</p>}
    </div>
  )
}

function OpportunityCard({ opportunity }) {
  const accent = clusterAccent(opportunity.opportunity_score)
  const sentiment = opportunity.sentiment_avg ?? 0
  const sentimentLabel = sentiment < -0.2 ? 'Negative' : sentiment > 0.2 ? 'Positive' : 'Mixed'
  const topQuote = opportunity.evidence?.[0] || opportunity.summary || ''
  return (
    <div className={`rounded-xl border ${accent.border} bg-gradient-to-b ${accent.bg} p-5 flex flex-col h-full`}>
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="flex-1 min-w-0">
          <div className="text-xs uppercase tracking-wide text-gray-500 mb-1">#{opportunity.rank} Opportunity</div>
          <h3 className="font-semibold text-base leading-tight">{opportunity.label}</h3>
        </div>
        <div className="text-right">
          <div className={`text-2xl font-bold ${accent.score}`}>{opportunity.opportunity_score.toFixed(1)}</div>
          <div className="text-[10px] text-gray-400 uppercase">of 10</div>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 text-xs mb-3">
        <span className={`px-2 py-0.5 rounded-full ${accent.chip}`}>{opportunity.size} items</span>
        <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700">{sentimentLabel} · {sentiment.toFixed(2)}</span>
        {opportunity.top_keywords?.slice(0, 2).map(k => (
          <span key={k} className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-600">{k}</span>
        ))}
      </div>

      {topQuote && (
        <blockquote className="text-sm text-gray-700 italic border-l-2 border-gray-300 pl-3 mb-3 line-clamp-3">
          "{topQuote}"
        </blockquote>
      )}

      {opportunity.recommendation && (
        <div className="mt-auto pt-3 border-t border-gray-200">
          <div className="flex items-center gap-1 text-xs font-medium text-gray-600 mb-1">
            <Sparkles className="w-3 h-3" /> AI recommendation
          </div>
          <p className="text-sm text-gray-800 line-clamp-4">{opportunity.recommendation}</p>
        </div>
      )}
    </div>
  )
}

function ArtifactPreview({ title, icon: Icon, markdown, filename }) {
  const [open, setOpen] = useState(false)
  const [copied, setCopied] = useState(false)

  const copy = async () => {
    if (!markdown) return
    try {
      await navigator.clipboard.writeText(markdown)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* ignore */ }
  }

  const download = () => {
    if (!markdown) return
    const blob = new Blob([markdown], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!markdown) {
    return (
      <div className="rounded-xl border border-gray-200 bg-white p-4 text-sm text-gray-400">
        <div className="flex items-center gap-2"><Icon className="w-4 h-4" /> {title} — not generated yet.</div>
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-gray-500" />
          <span className="font-medium text-sm">{title}</span>
          <span className="text-xs text-gray-400">{markdown.length.toLocaleString()} chars</span>
        </div>
        <div className="flex items-center gap-1">
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); copy() }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); copy() } }}
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded border border-gray-200 hover:bg-white cursor-pointer select-none"
          >
            {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            {copied ? 'Copied' : 'Copy'}
          </span>
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => { e.stopPropagation(); download() }}
            onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); download() } }}
            className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded border border-gray-200 hover:bg-white cursor-pointer select-none"
          >
            <Download className="w-3 h-3" />
            Download
          </span>
          <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`} />
        </div>
      </button>
      {open && (
        <pre className="px-5 py-4 text-xs whitespace-pre-wrap bg-gray-50 border-t border-gray-200 max-h-[400px] overflow-auto font-mono text-gray-800">
{markdown}
        </pre>
      )}
    </div>
  )
}

function AnalyzeFlow({ onComplete }) {
  const [file, setFile] = useState(null)
  const [step, setStep] = useState(null)  // 'uploading' | 'analyzing' | 'done' | 'error'
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const dropRef = useRef(null)
  useEffect(() => {
    const el = dropRef.current
    if (!el) return
    const prevent = (e) => { e.preventDefault() }
    const onDrop = (e) => {
      e.preventDefault()
      const f = e.dataTransfer?.files?.[0]
      if (f && f.name.toLowerCase().endsWith('.csv')) setFile(f)
    }
    el.addEventListener('dragover', prevent)
    el.addEventListener('drop', onDrop)
    return () => {
      el.removeEventListener('dragover', prevent)
      el.removeEventListener('drop', onDrop)
    }
  }, [])

  const start = async () => {
    if (!file) return
    setError(null)
    setStep('uploading')
    try {
      await api.uploadCSV(file)
      setStep('analyzing')
      await api.runAnalysis()
      setStep('done')
      await new Promise(r => setTimeout(r, 400))
      onComplete?.()
      setFile(null)
      setStep(null)
    } catch (err) {
      setError(err.message)
      setStep('error')
    }
  }

  const steps = [
    { id: 'uploading', label: 'Ingesting feedback', desc: 'CSV parsed, sentiment + category + embeddings enriched' },
    { id: 'analyzing', label: 'Clustering & writing artifacts', desc: 'k-means, scoring, AI recommendations, PRD, Jira' },
    { id: 'done', label: 'Report ready', desc: 'Loading the fresh report below' },
  ]
  const stepIdx = step === 'uploading' ? 0 : step === 'analyzing' ? 1 : step === 'done' ? 2 : -1

  return (
    <div className="rounded-xl border border-pulse-200 bg-gradient-to-br from-pulse-50 to-white p-5">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h3 className="font-semibold">Analyze a new CSV</h3>
          <p className="text-sm text-gray-600">Drop or pick a CSV of customer feedback and Pulse will regenerate the whole report.</p>
        </div>
        <span className="text-xs text-gray-500">Supported column: <code className="bg-white px-1.5 py-0.5 rounded border">text</code></span>
      </div>

      {step === null && (
        <div
          ref={dropRef}
          className={`mt-4 rounded-lg border-2 border-dashed ${file ? 'border-emerald-400 bg-emerald-50/40' : 'border-gray-300 bg-white'} p-5 text-center cursor-pointer transition-colors`}
          onClick={() => inputRef.current?.click()}
        >
          <Upload className="w-6 h-6 text-gray-400 mx-auto mb-1" />
          <div className="text-sm text-gray-700">
            {file ? <><strong>{file.name}</strong> · {Math.round(file.size / 1024)} KB</> : 'Drop a CSV here, or click to choose'}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          {file && (
            <button
              onClick={(e) => { e.stopPropagation(); start() }}
              className="mt-3 inline-flex items-center gap-2 px-4 py-2 bg-pulse-600 hover:bg-pulse-700 text-white rounded-lg font-medium text-sm"
            >
              <Play className="w-4 h-4" /> Analyze {file.name}
            </button>
          )}
        </div>
      )}

      {step !== null && (
        <ol className="mt-4 space-y-2">
          {steps.map((s, i) => {
            const done = stepIdx > i || step === 'done'
            const active = stepIdx === i && step !== 'done'
            return (
              <li key={s.id} className="flex items-start gap-3">
                <div className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold flex-shrink-0 mt-0.5 ${
                  done ? 'bg-emerald-500 text-white' :
                  active ? 'bg-pulse-600 text-white' :
                  'bg-gray-200 text-gray-500'
                }`}>
                  {done ? '✓' : active ? <span className="inline-block w-2 h-2 bg-white rounded-full animate-pulse" /> : i + 1}
                </div>
                <div>
                  <div className={`text-sm font-medium ${active ? 'text-pulse-700' : done ? 'text-gray-700' : 'text-gray-400'}`}>{s.label}</div>
                  <div className="text-xs text-gray-500">{s.desc}</div>
                </div>
              </li>
            )
          })}
        </ol>
      )}

      {step === 'error' && (
        <div className="mt-4 p-3 rounded bg-red-50 border border-red-200 text-sm text-red-700">
          {error || 'Something went wrong. Check server logs and try again.'}
        </div>
      )}
    </div>
  )
}

export default function ReportPage() {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [emptyState, setEmptyState] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const r = await api.latestReport()
      setReport(r)
      setEmptyState(false)
      setError(null)
    } catch (err) {
      if (err.message && err.message.toLowerCase().includes('no completed analysis')) {
        setEmptyState(true)
      } else {
        setError(err.message)
      }
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  if (loading && !report) {
    return (
      <div className="p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600 mx-auto" />
        <p className="mt-3 text-sm text-gray-500">Loading your report…</p>
      </div>
    )
  }

  if (emptyState) {
    return (
      <div className="p-8 max-w-3xl mx-auto space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Welcome to Pulse</h1>
          <p className="text-gray-600 mt-1">Upload a CSV of customer feedback and Pulse will cluster the themes, score opportunities, and auto-generate a PRD + Jira tickets.</p>
        </div>
        <AnalyzeFlow onComplete={load} />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-8 max-w-3xl mx-auto">
        <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-sm text-red-700">{error}</div>
      </div>
    )
  }

  if (!report) return null

  const { stats, top_opportunities, evidence_quotes, trends, insights, llm_provider } = report

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* Hero */}
      <header className="space-y-3">
        <div className="inline-flex items-center gap-1.5 text-xs font-medium text-pulse-700 bg-pulse-50 px-2 py-1 rounded-full">
          <Sparkles className="w-3 h-3" />
          {llm_provider === 'anthropic' ? 'Summarised by Claude' : llm_provider === 'openai' ? 'Summarised by GPT' : 'Heuristic summary'}
        </div>
        <h1 className="text-2xl md:text-3xl font-bold leading-tight">{report.headline}</h1>
        <p className="text-gray-700 leading-relaxed max-w-3xl">{report.executive_summary}</p>
      </header>

      {/* Analyze new CSV */}
      <AnalyzeFlow onComplete={load} />

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard icon={MessageSquare} label="Total feedback" value={stats.total_feedback.toLocaleString()}
          sub={`${stats.feedback_this_week} this week`} />
        <StatCard icon={Target} label="Active themes" value={stats.active_clusters} color="amber" />
        <StatCard icon={stats.sentiment_trend === 'improving' ? TrendingUp : stats.sentiment_trend === 'declining' ? TrendingDown : TrendingUp}
          label="Avg sentiment"
          value={stats.avg_sentiment !== null && stats.avg_sentiment !== undefined ? stats.avg_sentiment.toFixed(2) : '—'}
          sub={stats.sentiment_trend}
          color={stats.sentiment_trend === 'declining' ? 'red' : stats.sentiment_trend === 'improving' ? 'green' : 'pulse'} />
        <StatCard icon={Lightbulb} label="Unread insights" value={stats.unread_insights} color="amber" />
      </div>

      {/* Top 3 Opportunities */}
      {top_opportunities?.length > 0 && (
        <section>
          <div className="flex items-baseline justify-between mb-3">
            <h2 className="text-lg font-semibold">Top opportunities</h2>
            <span className="text-xs text-gray-500">ranked by opportunity score</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {top_opportunities.map(o => <OpportunityCard key={o.id} opportunity={o} />)}
          </div>
        </section>
      )}

      {/* Evidence */}
      {evidence_quotes?.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-2">What customers literally said</h2>
          <p className="text-xs text-gray-500 mb-3">Five most representative quotes from the top cluster.</p>
          <ul className="space-y-2">
            {evidence_quotes.map((q, i) => (
              <li key={i} className="rounded-lg bg-white border border-gray-200 p-3 text-sm text-gray-800 italic">
                "{q}"
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Artifacts */}
      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-lg font-semibold">Ready-to-ship artifacts</h2>
          <span className="text-xs text-gray-500">Paste into Notion, Linear, Jira</span>
        </div>
        <div className="space-y-2">
          <ArtifactPreview
            title="Executive summary"
            icon={Sparkles}
            markdown={report.executive_summary_markdown}
            filename="executive_summary.md"
          />
          <ArtifactPreview
            title="Product requirements document (PRD)"
            icon={FileText}
            markdown={report.prd_markdown}
            filename="PRD.md"
          />
          <ArtifactPreview
            title="Jira tickets breakdown"
            icon={ListChecks}
            markdown={report.jira_markdown}
            filename="jira_tickets.md"
          />
        </div>
      </section>

      {/* Trends */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h3 className="font-semibold text-sm mb-2">Feedback volume</h3>
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={trends.volume}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="period" fontSize={11} stroke="#888" />
              <YAxis fontSize={11} stroke="#888" />
              <Tooltip />
              <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="#dbeafe" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="rounded-xl border border-gray-200 bg-white p-4">
          <h3 className="font-semibold text-sm mb-2">Sentiment trend</h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={trends.sentiment}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
              <XAxis dataKey="period" fontSize={11} stroke="#888" />
              <YAxis domain={[-1, 1]} fontSize={11} stroke="#888" />
              <Tooltip />
              <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Insights */}
      {insights?.length > 0 && (
        <section>
          <h2 className="text-lg font-semibold mb-3">Auto-detected insights</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {insights.map(i => (
              <div key={i.id} className={`border-l-4 rounded-r-lg p-3 ${severityColors[i.severity] || severityColors.info}`}>
                <p className="font-medium text-sm">{i.title}</p>
                <p className="text-xs text-gray-700 mt-1 line-clamp-3">{i.description}</p>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
