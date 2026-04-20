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
} from 'lucide-react'

const NAV = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/feedback', icon: MessageSquare, label: 'Feedback' },
  { to: '/insights', icon: Lightbulb, label: 'Insights' },
  { to: '/trends', icon: TrendingUp, label: 'Trends' },
  { to: '/artifacts', icon: FileText, label: 'Artifacts' },
  { to: '/integrations', icon: Plug, label: 'Integrations' },
]

export default function Layout({ children, user, onLogout }) {
  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
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

      {/* Main content */}
      <main className="flex-1 ml-64 p-8 overflow-auto">
        {children}
      </main>
    </div>
  )
}
