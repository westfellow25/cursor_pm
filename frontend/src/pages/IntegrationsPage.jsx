import { useState, useEffect } from 'react'
import {
  Plug,
  Plus,
  Trash2,
  CheckCircle2,
  AlertCircle,
  PauseCircle,
  MessageSquare,
  Hash,
  Globe,
  FileSpreadsheet,
  X,
} from 'lucide-react'
import { api } from '../api/client'

const CONNECTOR_ICONS = {
  csv: FileSpreadsheet,
  intercom: MessageSquare,
  slack: Hash,
  api: Globe,
  zendesk: MessageSquare,
}

const STATUS_CONFIG = {
  active: { icon: CheckCircle2, color: 'text-emerald-500', label: 'Active' },
  paused: { icon: PauseCircle, color: 'text-amber-500', label: 'Paused' },
  error: { icon: AlertCircle, color: 'text-red-500', label: 'Error' },
  disconnected: { icon: AlertCircle, color: 'text-gray-400', label: 'Disconnected' },
}

export default function IntegrationsPage() {
  const [sources, setSources] = useState([])
  const [available, setAvailable] = useState([])
  const [loading, setLoading] = useState(true)
  const [showAdd, setShowAdd] = useState(false)
  const [newSource, setNewSource] = useState({ type: '', name: '' })

  useEffect(() => {
    Promise.all([api.listSources(), api.availableConnectors()])
      .then(([s, a]) => { setSources(s); setAvailable(a) })
      .finally(() => setLoading(false))
  }, [])

  const addSource = async () => {
    if (!newSource.type || !newSource.name) return
    const source = await api.createSource(newSource)
    setSources([source, ...sources])
    setShowAdd(false)
    setNewSource({ type: '', name: '' })
  }

  const deleteSource = async (id) => {
    await api.deleteSource(id)
    setSources(sources.filter(s => s.id !== id))
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" /></div>
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold">Integrations</h1>
          <p className="text-gray-500 text-sm mt-1">Connect your feedback sources for continuous intelligence</p>
        </div>
        <button onClick={() => setShowAdd(true)}
          className="flex items-center gap-2 px-4 py-2 bg-pulse-600 text-white text-sm font-medium rounded-lg hover:bg-pulse-700 transition-colors">
          <Plus className="w-4 h-4" /> Add Source
        </button>
      </div>

      {/* Add source modal */}
      {showAdd && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl w-full max-w-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">Add Feedback Source</h2>
              <button onClick={() => setShowAdd(false)} className="p-1 hover:bg-gray-100 rounded"><X className="w-5 h-5" /></button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Source type</label>
                <div className="grid grid-cols-2 gap-2">
                  {available.map(conn => {
                    const Icon = CONNECTOR_ICONS[conn.type] || Plug
                    return (
                      <button key={conn.type} onClick={() => setNewSource({ ...newSource, type: conn.type, name: conn.name })}
                        className={`p-3 border rounded-lg text-left transition-all ${newSource.type === conn.type ? 'border-pulse-500 bg-pulse-50 ring-1 ring-pulse-300' : 'border-gray-200 hover:border-gray-300'}`}>
                        <div className="flex items-center gap-2 mb-1">
                          <Icon className="w-4 h-4" />
                          <span className="text-sm font-medium">{conn.name}</span>
                        </div>
                        <p className="text-xs text-gray-500 line-clamp-2">{conn.description}</p>
                      </button>
                    )
                  })}
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Display name</label>
                <input value={newSource.name} onChange={(e) => setNewSource({ ...newSource, name: e.target.value })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pulse-500 focus:border-transparent outline-none text-sm" />
              </div>
              <button onClick={addSource} disabled={!newSource.type || !newSource.name}
                className="w-full py-2.5 bg-pulse-600 text-white rounded-lg font-medium hover:bg-pulse-700 transition-colors disabled:opacity-50 text-sm">
                Connect Source
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Connected sources */}
      {sources.length > 0 ? (
        <div className="space-y-3">
          {sources.map(source => {
            const Icon = CONNECTOR_ICONS[source.type] || Plug
            const statusConf = STATUS_CONFIG[source.status] || STATUS_CONFIG.active
            const StatusIcon = statusConf.icon
            return (
              <div key={source.id} className="bg-white rounded-xl border border-gray-200 p-5 flex items-center gap-4">
                <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center">
                  <Icon className="w-6 h-6 text-gray-600" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold">{source.name}</h3>
                    <span className={`flex items-center gap-1 text-xs ${statusConf.color}`}>
                      <StatusIcon className="w-3 h-3" /> {statusConf.label}
                    </span>
                  </div>
                  <p className="text-sm text-gray-500 mt-0.5">
                    {source.items_synced.toLocaleString()} items synced
                    {source.last_sync_at && <> &middot; Last sync: {new Date(source.last_sync_at).toLocaleDateString()}</>}
                  </p>
                </div>
                <span className="text-xs text-gray-400 uppercase font-medium px-2 py-1 bg-gray-50 rounded">{source.type}</span>
                <button onClick={() => deleteSource(source.id)}
                  className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
          <Plug className="w-12 h-12 mx-auto mb-3 text-gray-300" />
          <p className="text-gray-500 font-medium">No sources connected</p>
          <p className="text-gray-400 text-sm mt-1">Add your first feedback source to get started</p>
        </div>
      )}

      {/* Available connectors */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold mb-4">Available Connectors</h2>
        <div className="grid grid-cols-2 gap-4">
          {available.map(conn => {
            const Icon = CONNECTOR_ICONS[conn.type] || Plug
            const isConnected = sources.some(s => s.type === conn.type)
            return (
              <div key={conn.type} className={`bg-white rounded-xl border p-5 ${isConnected ? 'border-emerald-200 bg-emerald-50/30' : 'border-gray-200'}`}>
                <div className="flex items-center gap-3 mb-2">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${isConnected ? 'bg-emerald-100' : 'bg-gray-100'}`}>
                    <Icon className={`w-5 h-5 ${isConnected ? 'text-emerald-600' : 'text-gray-600'}`} />
                  </div>
                  <div>
                    <h3 className="font-medium text-sm">{conn.name}</h3>
                    {isConnected && <span className="text-xs text-emerald-600">Connected</span>}
                  </div>
                </div>
                <p className="text-xs text-gray-500">{conn.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
