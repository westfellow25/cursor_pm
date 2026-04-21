import { useState, useEffect } from 'react'
import { FileText, Download, Eye, X, ChevronDown, FileCode, BarChart3, FileCheck } from 'lucide-react'
import { api } from '../api/client'

const TYPE_CONFIG = {
  prd: { icon: FileText, label: 'Product Requirements', color: 'bg-blue-50 text-blue-600' },
  jira_tickets: { icon: FileCode, label: 'Jira Tickets', color: 'bg-purple-50 text-purple-600' },
  executive_summary: { icon: BarChart3, label: 'Executive Summary', color: 'bg-emerald-50 text-emerald-600' },
  impact_report: { icon: FileCheck, label: 'Impact Report', color: 'bg-amber-50 text-amber-600' },
  roadmap: { icon: FileText, label: 'Roadmap', color: 'bg-cyan-50 text-cyan-600' },
}

export default function ArtifactsPage() {
  const [artifacts, setArtifacts] = useState([])
  const [loading, setLoading] = useState(true)
  const [viewing, setViewing] = useState(null)
  const [typeFilter, setTypeFilter] = useState('')

  useEffect(() => {
    const params = {}
    if (typeFilter) params.type = typeFilter
    api.listArtifacts(params).then(setArtifacts).finally(() => setLoading(false))
  }, [typeFilter])

  const download = (artifact) => {
    const blob = new Blob([artifact.content], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${artifact.title.replace(/\s+/g, '_')}.md`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" /></div>
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Artifacts</h1>
          <p className="text-gray-500 text-sm mt-1">Auto-generated product documents from your analysis</p>
        </div>
        <select value={typeFilter} onChange={(e) => setTypeFilter(e.target.value)}
          className="px-3 py-1.5 text-sm border border-gray-200 rounded-lg outline-none focus:ring-2 focus:ring-pulse-500">
          <option value="">All types</option>
          {Object.entries(TYPE_CONFIG).map(([type, conf]) => (
            <option key={type} value={type}>{conf.label}</option>
          ))}
        </select>
      </div>

      {/* Viewer modal */}
      {viewing && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-8">
          <div className="bg-white rounded-2xl w-full max-w-4xl max-h-full flex flex-col overflow-hidden">
            <div className="flex items-center justify-between p-5 border-b border-gray-200">
              <h2 className="font-semibold">{viewing.title}</h2>
              <div className="flex items-center gap-2">
                <button onClick={() => download(viewing)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm bg-pulse-600 text-white rounded-lg hover:bg-pulse-700">
                  <Download className="w-3.5 h-3.5" /> Download
                </button>
                <button onClick={() => setViewing(null)} className="p-1.5 hover:bg-gray-100 rounded-lg"><X className="w-5 h-5" /></button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-6">
              <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono leading-relaxed">{viewing.content}</pre>
            </div>
          </div>
        </div>
      )}

      {artifacts.length === 0 ? (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <FileText className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="text-gray-500 font-medium">No artifacts generated yet</p>
          <p className="text-gray-400 text-sm mt-1">Run an analysis to auto-generate PRDs, Jira tickets, and executive summaries</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {artifacts.map(artifact => {
            const config = TYPE_CONFIG[artifact.type] || TYPE_CONFIG.prd
            const Icon = config.icon
            return (
              <div key={artifact.id} className="bg-white rounded-xl border border-gray-200 p-5 hover:shadow-md transition-shadow">
                <div className="flex items-start gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 ${config.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${config.color}`}>{config.label}</span>
                    <h3 className="font-semibold mt-1 text-sm">{artifact.title}</h3>
                    <p className="text-xs text-gray-400 mt-1">{new Date(artifact.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 mt-4">
                  <button onClick={() => setViewing(artifact)}
                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-sm font-medium border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                    <Eye className="w-3.5 h-3.5" /> View
                  </button>
                  <button onClick={() => download(artifact)}
                    className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 text-sm font-medium bg-pulse-600 text-white rounded-lg hover:bg-pulse-700 transition-colors">
                    <Download className="w-3.5 h-3.5" /> Download
                  </button>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
