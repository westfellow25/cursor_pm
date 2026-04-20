import { useState, useEffect } from 'react'
import {
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  ArrowUpDown,
  Upload,
} from 'lucide-react'
import { api } from '../api/client'

const CATEGORIES = ['', 'performance', 'bug', 'ux', 'feature-request', 'integration', 'onboarding', 'security', 'mobile', 'pricing', 'praise', 'support', 'data', 'general']
const CHANNELS = ['', 'web', 'mobile', 'api', 'email', 'chat', 'slack', 'intercom', 'csv', 'review']

function SentimentBadge({ value }) {
  if (value == null) return <span className="text-xs text-gray-400">—</span>
  const color = value > 0.2 ? 'bg-emerald-100 text-emerald-700' : value < -0.2 ? 'bg-red-100 text-red-700' : 'bg-gray-100 text-gray-600'
  return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${color}`}>{value.toFixed(2)}</span>
}

function CategoryBadge({ value }) {
  if (!value) return null
  const colors = {
    performance: 'bg-orange-100 text-orange-700',
    bug: 'bg-red-100 text-red-700',
    ux: 'bg-purple-100 text-purple-700',
    'feature-request': 'bg-blue-100 text-blue-700',
    integration: 'bg-cyan-100 text-cyan-700',
    praise: 'bg-emerald-100 text-emerald-700',
    security: 'bg-red-100 text-red-700',
    mobile: 'bg-indigo-100 text-indigo-700',
    onboarding: 'bg-yellow-100 text-yellow-700',
    data: 'bg-teal-100 text-teal-700',
  }
  return <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${colors[value] || 'bg-gray-100 text-gray-600'}`}>{value}</span>
}

export default function FeedbackPage() {
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 50 })
  const [loading, setLoading] = useState(true)
  const [filters, setFilters] = useState({ page: 1, page_size: 30, category: '', channel: '', search: '', sort_by: 'created_at', sort_dir: 'desc' })
  const [uploading, setUploading] = useState(false)
  const [stats, setStats] = useState(null)

  const load = () => {
    setLoading(true)
    const params = { ...filters }
    Object.keys(params).forEach(k => { if (!params[k]) delete params[k] })
    api.listFeedback(params).then(setData).finally(() => setLoading(false))
  }

  useEffect(load, [filters])
  useEffect(() => { api.feedbackStats().then(setStats) }, [])

  const setFilter = (k, v) => setFilters(prev => ({ ...prev, [k]: v, page: 1 }))
  const totalPages = Math.ceil(data.total / filters.page_size)

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await api.uploadCSV(file)
      load()
      api.feedbackStats().then(setStats)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Feedback Explorer</h1>
          <p className="text-gray-500 text-sm mt-1">{data.total.toLocaleString()} items across all sources</p>
        </div>
        <label className={`flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg cursor-pointer transition-colors
          ${uploading ? 'bg-gray-100 text-gray-400' : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'}`}>
          <Upload className="w-4 h-4" />
          {uploading ? 'Uploading...' : 'Upload CSV'}
          <input type="file" accept=".csv" className="hidden" onChange={handleUpload} disabled={uploading} />
        </label>
      </div>

      {/* Stats row */}
      {stats && (
        <div className="grid grid-cols-5 gap-3 mb-6">
          {Object.entries(stats.categories || {}).slice(0, 5).map(([cat, count]) => (
            <div key={cat} className="bg-white rounded-lg border border-gray-200 p-3 text-center">
              <p className="text-xs text-gray-500 capitalize">{cat || 'uncategorized'}</p>
              <p className="text-lg font-bold">{count}</p>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="bg-white rounded-xl border border-gray-200 p-4 mb-4 flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input placeholder="Search feedback..." value={filters.search}
            onChange={(e) => setFilter('search', e.target.value)}
            className="w-full pl-9 pr-3 py-2 text-sm border border-gray-200 rounded-lg focus:ring-2 focus:ring-pulse-500 focus:border-transparent outline-none" />
        </div>
        <select value={filters.category} onChange={(e) => setFilter('category', e.target.value)}
          className="px-3 py-2 text-sm border border-gray-200 rounded-lg outline-none focus:ring-2 focus:ring-pulse-500">
          <option value="">All categories</option>
          {CATEGORIES.filter(Boolean).map(c => <option key={c} value={c}>{c}</option>)}
        </select>
        <select value={filters.channel} onChange={(e) => setFilter('channel', e.target.value)}
          className="px-3 py-2 text-sm border border-gray-200 rounded-lg outline-none focus:ring-2 focus:ring-pulse-500">
          <option value="">All channels</option>
          {CHANNELS.filter(Boolean).map(c => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 border-b border-gray-200">
            <tr>
              <th className="text-left px-4 py-3 font-medium text-gray-500 w-[45%]">Feedback</th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">Category</th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">Sentiment</th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">Channel</th>
              <th className="text-left px-4 py-3 font-medium text-gray-500">Date</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={5} className="px-4 py-12 text-center text-gray-400">Loading...</td></tr>
            ) : data.items.length === 0 ? (
              <tr><td colSpan={5} className="px-4 py-12 text-center text-gray-400">No feedback items found</td></tr>
            ) : data.items.map((item) => (
              <tr key={item.id} className="border-b border-gray-50 hover:bg-gray-50/50 transition-colors">
                <td className="px-4 py-3">
                  <p className="line-clamp-2">{item.text}</p>
                  {item.author && <p className="text-xs text-gray-400 mt-1">by {item.author}</p>}
                </td>
                <td className="px-4 py-3"><CategoryBadge value={item.category} /></td>
                <td className="px-4 py-3"><SentimentBadge value={item.sentiment} /></td>
                <td className="px-4 py-3"><span className="text-xs text-gray-500">{item.channel || '—'}</span></td>
                <td className="px-4 py-3"><span className="text-xs text-gray-500">{new Date(item.created_at).toLocaleDateString()}</span></td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200">
            <span className="text-xs text-gray-500">
              Showing {((filters.page - 1) * filters.page_size) + 1}–{Math.min(filters.page * filters.page_size, data.total)} of {data.total}
            </span>
            <div className="flex items-center gap-1">
              <button onClick={() => setFilters(p => ({ ...p, page: p.page - 1 }))} disabled={filters.page <= 1}
                className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30"><ChevronLeft className="w-4 h-4" /></button>
              <span className="text-xs px-2">Page {filters.page} of {totalPages}</span>
              <button onClick={() => setFilters(p => ({ ...p, page: p.page + 1 }))} disabled={filters.page >= totalPages}
                className="p-1.5 rounded hover:bg-gray-100 disabled:opacity-30"><ChevronRight className="w-4 h-4" /></button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
