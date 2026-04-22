import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  MessageSquare,
  Lightbulb,
  TrendingUp,
  Plug,
  FileText,
  LogOut,
  Zap,
  Brain,
} from 'lucide-react'
import { api } from '../api/client'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Report' },
  { to: '/feedback', icon: MessageSquare, label: 'Feedback' },
  { to: '/insights', icon: Lightbulb, label: 'Insights' },
  { to: '/trends', icon: TrendingUp, label: 'Trends' },
  { to: '/artifacts', icon: FileText, label: 'Artifacts' },
  { to: '/integrations', icon: Plug, label: 'Integrations' },
]

const PROVIDER_LABELS = {
  anthropic: { name: 'Claude', color: 'bg-amber-500/20 text-amber-300' },
  openai: { name: 'GPT', color: 'bg-emerald-500/20 text-emerald-300' },
  none: { name: 'Heuristic', color: 'bg-gray-600 text-gray-300' },
}

export default function Layout({ children, user, onLogout }) {
  const [llm, setLlm] = useState(null)

  useEffect(() => {
    api.systemStatus().then(s => setLlm(s.llm)).catch(() => {})
  }, [])

  const providerInfo = llm ? PROVIDER_LABELS[llm.provider] || PROVIDER_LABELS.none : null

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 bg-gray-900 text-white flex flex-col fixed inset-y-0">
        <div className="p-5 flex items-center gap-2.5 border-b border-gray-800">
          <div className="w-8 h-8 bg-pulse-500 rounded-lg flex items-center justify-center">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-lg font-bold leading-tight">Pulse</h1>
            <p className="text-[11px] text-gray-400 leading-tight">Product Intelligence</p>
          </div>
        </div>

        <nav className="flex-1 py-4 px-3 space-y-1">
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-pulse-600/20 text-pulse-300 font-medium'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`
              }
            >
              <Icon className="w-[18px] h-[18px]" />
              {label}
            </NavLink>
          ))}
        </nav>

        {providerInfo && (
          <div className="mx-3 mb-2 px-3 py-2 rounded-lg bg-gray-800/70 border border-gray-800">
            <div className="flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 text-gray-400" />
              <span className="text-[11px] text-gray-400 uppercase tracking-wide">AI Engine</span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${providerInfo.color}`}>
                {providerInfo.name}
              </span>
              <span className="text-[10px] text-gray-500 truncate ml-2" title={llm.model}>
                {llm.model}
              </span>
            </div>
          </div>
        )}

        <div className="p-4 border-t border-gray-800">
          <div className="flex items-center justify-between">
            <div className="min-w-0">
              <p className="text-sm font-medium truncate">{user?.name}</p>
              <p className="text-xs text-gray-500 truncate">{user?.email}</p>
            </div>
            <button onClick={onLogout} className="text-gray-500 hover:text-white p-1.5 rounded transition-colors">
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 ml-64 p-8 overflow-auto">
        {children}
      </main>
    </div>
  )
}
