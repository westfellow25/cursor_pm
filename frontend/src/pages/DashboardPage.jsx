import { useState, useEffect } from 'react'
import {
  MessageSquare,
  Plug,
  Target,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Lightbulb,
  ArrowRight,
  Upload,
  Play,
} from 'lucide-react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { Link } from 'react-router-dom'
import { api } from '../api/client'

function StatCard({ icon: Icon, label, value, sub, color = 'pulse' }) {
  const colors = {
    pulse: 'bg-pulse-50 text-pulse-600',
    green: 'bg-emerald-50 text-emerald-600',
    amber: 'bg-amber-50 text-amber-600',
    red: 'bg-red-50 text-red-600',
  }
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="flex items-center gap-3 mb-3">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${colors[color]}`}>
          <Icon className="w-[18px] h-[18px]" />
        </div>
        <span className="text-sm text-gray-500">{label}</span>
      </div>
      <p className="text-2xl font-bold">{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </div>
  )
}

function InsightCard({ insight }) {
  const severityColors = {
    critical: 'border-red-400 bg-red-50',
    warning: 'border-amber-400 bg-amber-50',
    info: 'border-blue-400 bg-blue-50',
    positive: 'border-emerald-400 bg-emerald-50',
  }
  return (
    <div className={`border-l-4 rounded-r-lg p-4 ${severityColors[insight.severity] || severityColors.info}`}>
      <p className="font-medium text-sm">{insight.title}</p>
      <p className="text-xs text-gray-600 mt-1 line-clamp-2">{insight.description}</p>
    </div>
  )
}

function ClusterRow({ cluster, rank }) {
  const scoreColor = cluster.opportunity_score >= 7 ? 'text-red-600' : cluster.opportunity_score >= 4 ? 'text-amber-600' : 'text-emerald-600'
  return (
    <div className="flex items-center gap-4 py-3 border-b border-gray-100 last:border-0">
      <span className="w-6 h-6 rounded-full bg-gray-100 text-xs font-bold flex items-center justify-center text-gray-500">
        {rank}
      </span>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{cluster.label}</p>
        <p className="text-xs text-gray-400">{cluster.size} items &middot; {cluster.top_keywords?.slice(0, 3).join(', ')}</p>
      </div>
      <span className={`text-lg font-bold ${scoreColor}`}>{cluster.opportunity_score}</span>
    </div>
  )
}

export default function DashboardPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)

  const load = () => {
    setLoading(true)
    api.dashboard()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(load, [])

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await api.uploadCSV(file)
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleAnalyze = async () => {
    setAnalyzing(true)
    try {
      await api.runAnalysis()
      load()
    } catch (err) {
      setError(err.message)
    } finally {
      setAnalyzing(false)
    }
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" /></div>
  }

  const stats = data?.stats
  const volumeData = data?.trends?.volume || []
  const topClusters = data?.top_clusters || []
  const insights = data?.recent_insights || []

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-500 text-sm mt-1">Your product intelligence at a glance</p>
        </div>
        <div className="flex items-center gap-3">
          <label className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg cursor-pointer transition-colors
            ${uploading ? 'bg-gray-100 text-gray-400' : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'}`}>
            <Upload className="w-4 h-4" />
            {uploading ? 'Uploading...' : 'Upload CSV'}
            <input type="file" accept=".csv" className="hidden" onChange={handleUpload} disabled={uploading} />
          </label>
          <button onClick={handleAnalyze} disabled={analyzing}
            className="flex items-center gap-2 px-4 py-2 bg-pulse-600 text-white text-sm font-medium rounded-lg hover:bg-pulse-700 transition-colors disabled:opacity-50">
            <Play className="w-4 h-4" />
            {analyzing ? 'Analyzing...' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {error && <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">{error}</div>}

      {/* Stats */}
      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard icon={MessageSquare} label="Total Feedback" value={stats?.total_feedback?.toLocaleString() || 0} sub={`${stats?.feedback_this_week || 0} this week`} />
        <StatCard icon={Plug} label="Active Sources" value={stats?.total_sources || 0} color="green" />
        <StatCard icon={Target} label="Active Clusters" value={stats?.active_clusters || 0} color="amber" />
        <StatCard
          icon={stats?.sentiment_trend === 'declining' ? TrendingDown : TrendingUp}
          label="Avg Sentiment"
          value={stats?.avg_sentiment != null ? stats.avg_sentiment.toFixed(2) : 'N/A'}
          sub={stats?.sentiment_trend}
          color={stats?.sentiment_trend === 'declining' ? 'red' : 'green'}
        />
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Volume trend chart */}
        <div className="col-span-2 bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Feedback Volume</h3>
          {volumeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={volumeData}>
                <defs>
                  <linearGradient id="volumeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#5c7cfa" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#5c7cfa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} tickFormatter={(v) => v.slice(5, 10)} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#5c7cfa" fill="url(#volumeGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-gray-400 text-sm">
              Upload feedback and run analysis to see trends
            </div>
          )}
        </div>

        {/* Recent insights */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-semibold text-gray-700">Recent Insights</h3>
            <Link to="/insights" className="text-xs text-pulse-600 hover:underline flex items-center gap-1">
              View all <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          {insights.length > 0 ? (
            <div className="space-y-3">
              {insights.slice(0, 4).map((insight) => (
                <InsightCard key={insight.id} insight={insight} />
              ))}
            </div>
          ) : (
            <div className="h-48 flex items-center justify-center text-gray-400 text-sm text-center px-4">
              <div>
                <Lightbulb className="w-8 h-8 mx-auto mb-2 opacity-30" />
                Insights will appear after your first analysis
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Top opportunities */}
      <div className="mt-6 bg-white rounded-xl border border-gray-200 p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold text-gray-700">Top Opportunities</h3>
          <span className="text-xs text-gray-400">Ranked by opportunity score</span>
        </div>
        {topClusters.length > 0 ? (
          <div>
            {topClusters.map((cluster, i) => (
              <ClusterRow key={cluster.id} cluster={cluster} rank={i + 1} />
            ))}
          </div>
        ) : (
          <div className="py-8 text-center text-gray-400 text-sm">
            <Target className="w-8 h-8 mx-auto mb-2 opacity-30" />
            Run an analysis to discover opportunities
          </div>
        )}
      </div>
    </div>
  )
}
