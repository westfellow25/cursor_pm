const BASE = '/api/v1'

function getToken() {
  return localStorage.getItem('pulse_token')
}

async function request(path, options = {}) {
  const token = getToken()
  const headers = { ...options.headers }
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  const res = await fetch(`${BASE}${path}`, { ...options, headers })

  if (res.status === 401) {
    localStorage.removeItem('pulse_token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }

  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.detail || `Request failed (${res.status})`)
  }

  return res.json()
}

export const api = {
  // Auth
  signup: (data) => request('/auth/signup', { method: 'POST', body: JSON.stringify(data) }),
  login: (data) => request('/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  me: () => request('/auth/me'),

  // Dashboard
  dashboard: () => request('/dashboard'),

  // Feedback
  listFeedback: (params = {}) => {
    const qs = new URLSearchParams(params).toString()
    return request(`/feedback?${qs}`)
  },
  feedbackStats: () => request('/feedback/stats'),
  uploadCSV: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/feedback/upload', { method: 'POST', body: form })
  },
  submitFeedback: (items) => request('/feedback', { method: 'POST', body: JSON.stringify(items) }),

  // Analysis
  runAnalysis: () => request('/analysis/run', { method: 'POST' }),
  latestAnalysis: () => request('/analysis/latest'),
  analysisRuns: () => request('/analysis/runs'),
  clusterDeepDive: (id) => request(`/analysis/cluster/${id}/deep-dive`),

  // Insights
  listInsights: (params = {}) => {
    const qs = new URLSearchParams(params).toString()
    return request(`/insights?${qs}`)
  },
  markInsightRead: (id) => request(`/insights/${id}/read`, { method: 'POST' }),

  // Artifacts
  listArtifacts: (params = {}) => {
    const qs = new URLSearchParams(params).toString()
    return request(`/artifacts?${qs}`)
  },
  getArtifact: (id) => request(`/artifacts/${id}`),

  // Integrations
  availableConnectors: () => request('/integrations/available'),
  listSources: () => request('/integrations'),
  createSource: (data) => request('/integrations', { method: 'POST', body: JSON.stringify(data) }),
  deleteSource: (id) => request(`/integrations/${id}`, { method: 'DELETE' }),

  // Trends
  trends: (weeks = 12) => request(`/trends?weeks=${weeks}`),

  // System
  systemStatus: () => request('/system/status'),
}
