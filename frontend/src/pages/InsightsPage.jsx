import { useState, useEffect } from 'react'
import {
  Lightbulb,
  AlertTriangle,
  TrendingUp,
  Users,
  Target,
  ShieldAlert,
  Check,
  Bell,
} from 'lucide-react'
import { api } from '../api/client'

const TYPE_CONFIG = {
  opportunity_found: { icon: Target, color: 'text-blue-600 bg-blue-50', label: 'Opportunity' },
  trend_spike: { icon: TrendingUp, color: 'text-orange-600 bg-orange-50', label: 'Trend' },
  sentiment_shift: { icon: TrendingUp, color: 'text-purple-600 bg-purple-50', label: 'Sentiment' },
  churn_signal: { icon: ShieldAlert, color: 'text-red-600 bg-red-50', label: 'Churn Risk' },
  emerging_topic: { icon: Lightbulb, color: 'text-teal-600 bg-teal-50', label: 'Emerging' },
  segment_divergence: { icon: Users, color: 'text-indigo-600 bg-indigo-50', label: 'Segment' },
  volume_spike: { icon: AlertTriangle, color: 'text-amber-600 bg-amber-50', label: 'Volume' },
  regression_detected: { icon: AlertTriangle, color: 'text-red-600 bg-red-50', label: 'Regression' },
  benchmark_outlier: { icon: Target, color: 'text-cyan-600 bg-cyan-50', label: 'Benchmark' },
}

const SEVERITY_COLORS = {
  critical: 'border-l-red-500',
  warning: 'border-l-amber-500',
  info: 'border-l-blue-500',
  positive: 'border-l-emerald-500',
}

export default function InsightsPage() {
  const [insights, setInsights] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    const params = {}
    if (filter) params.type = filter
    api.listInsights(params).then(setInsights).finally(() => setLoading(false))
  }, [filter])

  const markRead = async (id) => {
    await api.markInsightRead(id)
    setInsights(prev => prev.map(i => i.id === id ? { ...i, is_read: true } : i))
  }

  const unreadCount = insights.filter(i => !i.is_read).length

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Insights</h1>
          <p className="text-gray-500 text-sm mt-1">AI-generated observations from your feedback data</p>
        </div>
        {unreadCount > 0 && (
          <span className="flex items-center gap-1.5 px-3 py-1.5 bg-pulse-50 text-pulse-700 rounded-full text-sm font-medium">
            <Bell className="w-3.5 h-3.5" /> {unreadCount} unread
          </span>
        )}
      </div>

      {/* Filter chips */}
      <div className="flex items-center gap-2 mb-6 flex-wrap">
        <button onClick={() => setFilter('')}
          className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${!filter ? 'bg-gray-900 text-white' : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
          All
        </button>
        {Object.entries(TYPE_CONFIG).map(([type, config]) => (
          <button key={type} onClick={() => setFilter(type)}
            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${filter === type ? 'bg-gray-900 text-white' : 'bg-white border border-gray-200 text-gray-600 hover:bg-gray-50'}`}>
            {config.label}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" /></div>
      ) : insights.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <Lightbulb className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="text-gray-500 font-medium">No insights yet</p>
          <p className="text-gray-400 text-sm mt-1">Run an analysis to generate insights from your feedback</p>
        </div>
      ) : (
        <div className="space-y-3">
          {insights.map((insight) => {
            const config = TYPE_CONFIG[insight.type] || TYPE_CONFIG.opportunity_found
            const Icon = config.icon
            return (
              <div key={insight.id}
                className={`bg-white rounded-xl border border-gray-200 border-l-4 ${SEVERITY_COLORS[insight.severity]} p-5 transition-all ${!insight.is_read ? 'ring-1 ring-pulse-200' : ''}`}>
                <div className="flex items-start gap-4">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${config.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${config.color}`}>{config.label}</span>
                      <span className="text-xs text-gray-400">{new Date(insight.created_at).toLocaleDateString()}</span>
                      {!insight.is_read && <span className="w-2 h-2 rounded-full bg-pulse-500" />}
                    </div>
                    <h3 className="font-semibold text-gray-900">{insight.title}</h3>
                    <p className="text-sm text-gray-600 mt-1">{insight.description}</p>
                    {insight.data?.keywords && (
                      <div className="flex gap-1.5 mt-2">
                        {insight.data.keywords.slice(0, 5).map(kw => (
                          <span key={kw} className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded">{kw}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  {!insight.is_read && (
                    <button onClick={() => markRead(insight.id)}
                      className="p-2 text-gray-400 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors flex-shrink-0">
                      <Check className="w-4 h-4" />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
