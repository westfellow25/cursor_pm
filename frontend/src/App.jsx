import { useState, useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ReportPage from './pages/ReportPage'
import HowItWorksPage from './pages/HowItWorksPage'
import DashboardPage from './pages/DashboardPage'
import FeedbackPage from './pages/FeedbackPage'
import InsightsPage from './pages/InsightsPage'
import TrendsPage from './pages/TrendsPage'
import IntegrationsPage from './pages/IntegrationsPage'
import ArtifactsPage from './pages/ArtifactsPage'
import LoginPage from './pages/LoginPage'
import { api } from './api/client'

export default function App() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('pulse_token')
    if (token) {
      api.me()
        .then(setUser)
        .catch(() => localStorage.removeItem('pulse_token'))
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-pulse-600" />
      </div>
    )
  }

  if (!user) {
    return <LoginPage onLogin={setUser} />
  }

  return (
    <Layout user={user} onLogout={() => { localStorage.removeItem('pulse_token'); setUser(null) }}>
      <Routes>
        <Route path="/" element={<ReportPage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/feedback" element={<FeedbackPage />} />
        <Route path="/insights" element={<InsightsPage />} />
        <Route path="/trends" element={<TrendsPage />} />
        <Route path="/integrations" element={<IntegrationsPage />} />
        <Route path="/artifacts" element={<ArtifactsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  )
}
