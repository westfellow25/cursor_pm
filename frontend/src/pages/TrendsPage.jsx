import { useState, useEffect } from 'react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts'
import { TrendingUp, Calendar } from 'lucide-react'
import { api } from '../api/client'

const CATEGORY_COLORS = {
  performance: '#f97316',
  bug: '#ef4444',
  ux: '#a855f7',
  'feature-request': '#3b82f6',
  integration: '#06b6d4',
  praise: '#10b981',
  security: '#dc2626',
  onboarding: '#eab308',
  mobile: '#6366f1',
  data: '#14b8a6',
  general: '#6b7280',
  support: '#8b5cf6',
  pricing: '#f59e0b',
}

export default function TrendsPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [weeks, setWeeks] = useState(12)

  useEffect(() => {
    setLoading(true)
    api.trends(weeks).then(setData).finally(() => setLoading(false))
  }, [weeks])

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" /></div>
  }

  const volumeData = (data?.volume || []).map(d => ({
    ...d,
    period: d.period?.slice(5, 10) || '',
  }))

  const sentimentData = (data?.sentiment || []).map(d => ({
    ...d,
    period: d.period?.slice(5, 10) || '',
  }))

  // Build category stacked data
  const allCats = new Set()
  const categoryData = (data?.categories || []).map(d => {
    const entry = { period: d.period?.slice(5, 10) || '' }
    Object.entries(d).forEach(([k, v]) => {
      if (k !== 'period' && typeof v === 'number') {
        entry[k] = v
        allCats.add(k)
      }
    })
    return entry
  })

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Trends</h1>
          <p className="text-gray-500 text-sm mt-1">How your feedback landscape evolves over time</p>
        </div>
        <div className="flex items-center gap-2">
          <Calendar className="w-4 h-4 text-gray-400" />
          <select value={weeks} onChange={(e) => setWeeks(Number(e.target.value))}
            className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg outline-none focus:ring-2 focus:ring-pulse-500">
            <option value={4}>4 weeks</option>
            <option value={12}>12 weeks</option>
            <option value={26}>26 weeks</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Volume trend */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Feedback Volume</h3>
          {volumeData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={volumeData}>
                <defs>
                  <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#5c7cfa" stopOpacity={0.2} />
                    <stop offset="95%" stopColor="#5c7cfa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Area type="monotone" dataKey="count" stroke="#5c7cfa" fill="url(#volGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-gray-400 text-sm">No data available</div>
          )}
        </div>

        {/* Sentiment trend */}
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Sentiment Trend</h3>
          {sentimentData.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="value" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} name="Sentiment" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[250px] flex items-center justify-center text-gray-400 text-sm">No data available</div>
          )}
        </div>
      </div>

      {/* Category breakdown */}
      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Category Distribution Over Time</h3>
        {categoryData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="period" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {[...allCats].slice(0, 8).map(cat => (
                <Bar key={cat} dataKey={cat} stackId="a" fill={CATEGORY_COLORS[cat] || '#6b7280'} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[300px] flex items-center justify-center text-gray-400 text-sm">
            <div className="text-center">
              <TrendingUp className="w-8 h-8 mx-auto mb-2 opacity-30" />
              Upload feedback and run analysis to see trends
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
