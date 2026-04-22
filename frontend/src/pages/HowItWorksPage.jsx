import { useEffect, useState } from 'react'
import {
  Upload, Cpu, Network, Target, FileOutput,
  MessageSquare, Plug, Database, Zap, Sparkles,
  TrendingUp, Users, Lock, Shield, Code2,
  ArrowRight, Check, AlertTriangle, BarChart3,
  Brain, GitBranch, Clock, Share2, Layers,
  Building2, Briefcase, HeartHandshake, Megaphone,
  FileText, ListChecks, ChevronDown, Eye,
} from 'lucide-react'
import { Link } from 'react-router-dom'

// ─────────────────────────────────────────────────────────────────────────
// Content data — kept close to the component so it's easy to tune.
// ─────────────────────────────────────────────────────────────────────────

const PIPELINE = [
  {
    icon: Upload,
    title: 'Ingest',
    desc: 'CSV, Intercom, Slack, Zendesk, REST API. Every feedback item gets a universal schema.',
    color: 'bg-blue-50 text-blue-600',
  },
  {
    icon: Cpu,
    title: 'Enrich',
    desc: 'Sentiment, urgency, category, language — automatically tagged on every row with domain-tuned lexicons.',
    color: 'bg-purple-50 text-purple-600',
  },
  {
    icon: Network,
    title: 'Cluster',
    desc: '256-dim embeddings + silhouette-optimised k-means group paraphrased complaints into single themes.',
    color: 'bg-emerald-50 text-emerald-600',
  },
  {
    icon: Target,
    title: 'Score',
    desc: 'Opportunity score combines volume, severity, sentiment, and segment spread. Top themes rise.',
    color: 'bg-amber-50 text-amber-600',
  },
  {
    icon: FileOutput,
    title: 'Write',
    desc: 'Claude writes a PRD, 5 Jira tickets, and an executive summary. Copy‑paste into your workflow.',
    color: 'bg-pink-50 text-pink-600',
  },
]

const CAPABILITIES = [
  {
    icon: Plug,
    title: 'Multi-source ingestion',
    body: 'Pluggable connector registry ships with CSV, Intercom, Slack, and REST API. Every source maps to the same `FeedbackItem` shape so clustering works across channels.',
  },
  {
    icon: Brain,
    title: 'Real semantic embeddings',
    body: 'OpenAI `text-embedding-3-small` when you have a key; `sentence-transformers/all-MiniLM-L6-v2` runs fully locally otherwise. No OpenAI required for quality clustering.',
  },
  {
    icon: GitBranch,
    title: 'Silhouette-optimised clustering',
    body: 'K selected per dataset via silhouette analysis, not a fixed number. Outliers (low-similarity items) get held out as `-1` rather than dragged into wrong themes.',
  },
  {
    icon: Sparkles,
    title: 'Multi-signal sentiment',
    body: 'Curated lexicon of product-feedback terms, phrase-level patterns, negation flips, and intensifier amplification. Not a generic Twitter model.',
  },
  {
    icon: AlertTriangle,
    title: 'Temporal anomaly detection',
    body: 'Volume spikes, sentiment shifts, emerging topics, segment divergences — each scored with a z-score and a cold-start guard so a single day of data doesn\'t fake "10× growth".',
  },
  {
    icon: Zap,
    title: 'LLM intelligence layer',
    body: 'Claude Sonnet preferred, OpenAI GPT fallback, deterministic heuristics as last resort. Cluster labels, recommendations, root-cause analyses, and executive narratives all use the same client.',
  },
  {
    icon: FileOutput,
    title: 'Artifact generation',
    body: 'Every analysis run auto-writes a PRD (problem, target users, proposed solution, metrics, risks), a 5-ticket Jira breakdown, and a 1-paragraph executive summary.',
  },
  {
    icon: Database,
    title: 'Multi-tenant schema',
    body: 'Orgs, users, sources, feedback, clusters, insights, artifacts, trend snapshots, benchmark data — indexed for fast filtering by org / sentiment / category / time.',
  },
  {
    icon: Clock,
    title: 'Temporal intelligence',
    body: 'Weekly trend snapshots build up over time. After 3 months a customer can\'t easily reconstruct their history anywhere else. That\'s the retention moat.',
  },
  {
    icon: Lock,
    title: 'JWT auth + API keys',
    body: 'Bcrypt-hashed passwords, scoped API keys per org, role-based access (owner / admin / member / viewer). Enterprise-ready from day one.',
  },
]

const PERSONAS = [
  {
    icon: Briefcase,
    role: 'Product Manager',
    scenario: 'Monday 9am triage',
    before: 'Scrolls through 340 Intercom tickets from the weekend. Tags by hand. Argues with engineering about priority.',
    after: 'Opens Pulse. Top 3 themes + severity scores already ranked. Picks one, copies the PRD, pushes 5 Jira tickets in one click.',
    saves: '6 hours / week',
  },
  {
    icon: Users,
    role: 'Head of Support',
    scenario: 'Spotting churn risk',
    before: 'Relies on gut feel. Sees "a lot of enterprise complaints" in dashboard volume, can\'t pinpoint theme.',
    after: 'Gets an automatic insight: "Enterprise sentiment dropped 0.3 points this week — driven by Slack integration disconnects." Escalates before the renewal call.',
    saves: 'a churn event',
  },
  {
    icon: Building2,
    role: 'CPO / VP Product',
    scenario: 'Board prep',
    before: 'Spends Friday pulling metrics from Looker + Dovetail + Slack screenshots into slides.',
    after: 'Opens Pulse, screenshots the Executive Summary. It\'s already written in a "what changed this quarter / top 3 priorities / risks" format.',
    saves: '4 hours / board meeting',
  },
  {
    icon: HeartHandshake,
    role: 'Customer Success',
    scenario: 'Renewal call prep',
    before: 'Searches Intercom for "Acme Corp" mentions. Tries to remember what they complained about 3 months ago.',
    after: 'Filters by account. Sees a trend chart: started angry about onboarding, resolved by Q2, now asking about SSO. Walks into the call with context.',
    saves: 'renewal confidence',
  },
]

const MOATS = [
  {
    icon: Share2,
    title: 'Data flywheel',
    body: 'Our feedback taxonomy improves with every customer processed. Patterns that emerge across orgs become domain-specific classifiers.',
  },
  {
    icon: Clock,
    title: 'Temporal lock-in',
    body: 'Six months of trend history per customer. Competitors start at zero. Churn is expensive.',
  },
  {
    icon: BarChart3,
    title: 'Cross-org benchmarks',
    body: '"Companies in your vertical see 3× more onboarding complaints" — only a platform with many customers can say this.',
  },
  {
    icon: Shield,
    title: 'Integration depth',
    body: 'Deep, bidirectional connectors (read feedback + push Jira tickets back). High switching costs.',
  },
  {
    icon: Brain,
    title: 'Domain-tuned AI',
    body: 'Sentiment, severity, and clustering tuned on product-feedback language, not generic NLP. Better signal per dollar spent on LLM calls.',
  },
]

const STACK = [
  'Python 3.11', 'FastAPI', 'SQLAlchemy 2', 'Pydantic v2',
  'React 18', 'Tailwind CSS', 'Recharts', 'Vite',
  'Claude Sonnet', 'OpenAI GPT', 'sentence-transformers', 'scikit-learn',
  'JWT + bcrypt', 'SQLite → Postgres', 'Docker', 'GitHub Codespaces',
]

// Six placeholder customer names rendered as stylised wordmarks so the section
// looks intentional, not lazy. Swap for real logos once you have customers.
const TRUSTED_BY = [
  { name: 'Acme Corp', style: 'tracking-tight font-bold' },
  { name: 'northbay', style: 'font-light lowercase' },
  { name: 'FABRITECH', style: 'tracking-[0.2em] font-semibold' },
  { name: 'RollUp', style: 'font-serif italic' },
  { name: 'linecore.', style: 'font-mono' },
  { name: 'Helix', style: 'font-bold tracking-widest uppercase' },
]

// ─────────────────────────────────────────────────────────────────────────

function Section({ eyebrow, title, subtitle, children }) {
  return (
    <section className="py-14 md:py-20">
      {eyebrow && (
        <p className="text-xs font-semibold uppercase tracking-widest text-pulse-600 mb-2">
          {eyebrow}
        </p>
      )}
      <h2 className="text-2xl md:text-3xl font-bold mb-3">{title}</h2>
      {subtitle && <p className="text-gray-600 max-w-3xl mb-8">{subtitle}</p>}
      {children}
    </section>
  )
}

function PipelineStep({ step, index, isLast, active }) {
  const Icon = step.icon
  return (
    <div className="flex md:flex-col items-start md:items-center gap-3 md:gap-0 md:flex-1">
      <div className="flex md:flex-col items-center gap-3 md:gap-2">
        <div
          className={`relative w-12 h-12 rounded-xl ${step.color} flex items-center justify-center flex-shrink-0 transition-all duration-500
            ${active ? 'ring-4 ring-pulse-400/40 scale-110 shadow-lg' : 'opacity-80'}`}
        >
          <Icon className={`w-6 h-6 transition-transform duration-500 ${active ? 'scale-110' : ''}`} />
          {active && (
            <span className="absolute inset-0 rounded-xl ring-2 ring-pulse-500 animate-ping" aria-hidden />
          )}
        </div>
      </div>
      <div className="flex-1 md:text-center md:mt-3">
        <div className="flex items-baseline gap-2 md:justify-center mb-1">
          <span className="text-xs font-mono text-gray-400">0{index + 1}</span>
          <h3 className={`font-semibold transition-colors ${active ? 'text-pulse-700' : 'text-gray-900'}`}>
            {step.title}
          </h3>
        </div>
        <p className="text-sm text-gray-600 leading-relaxed md:px-2">{step.desc}</p>
      </div>
      {!isLast && (
        <div className="hidden md:flex items-center mx-1 flex-shrink-0 self-start mt-5">
          <ArrowRight className={`w-4 h-4 transition-colors duration-500 ${active ? 'text-pulse-400' : 'text-gray-300'}`} />
        </div>
      )}
    </div>
  )
}

// Inline mini-mockups — live HTML, not screenshot images. Dimensions tuned so
// they feel like product screenshots but stay sharp at any size.

function ReportHeroMockup() {
  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm">
      <div className="px-4 py-2 bg-gray-50 border-b border-gray-200 flex items-center gap-1.5">
        <span className="w-2 h-2 rounded-full bg-red-400" />
        <span className="w-2 h-2 rounded-full bg-amber-400" />
        <span className="w-2 h-2 rounded-full bg-emerald-400" />
        <span className="ml-2 text-[10px] text-gray-400 font-mono">pulse.acme.com/report</span>
      </div>
      <div className="p-5 space-y-3">
        <div className="inline-flex items-center gap-1 text-[10px] font-medium text-pulse-700 bg-pulse-50 px-2 py-0.5 rounded-full">
          <Sparkles className="w-2.5 h-2.5" /> Summarised by Claude
        </div>
        <h3 className="text-base font-bold leading-snug">
          We analysed <span className="text-pulse-700">2,847 feedback items</span>. The top theme — Mobile sync reliability — is mentioned in 241 items (8.5% of feedback).
        </h3>
        <p className="text-xs text-gray-600 leading-relaxed">
          Enterprise sentiment dropped 0.3 points this week, driven entirely by
          Android 14 sync failures. This is your #1 churn signal. Two
          quick-win opportunities — CSV export filters and Slack debounce —
          together address another 12% of open tickets.
        </p>
        <div className="grid grid-cols-4 gap-2 pt-1">
          {[
            ['Total', '2,847'],
            ['Themes', '18'],
            ['Sentiment', '-0.12'],
            ['Alerts', '3'],
          ].map(([k, v]) => (
            <div key={k} className="rounded border border-gray-100 p-1.5">
              <div className="text-[9px] text-gray-500">{k}</div>
              <div className="text-xs font-bold">{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function OpportunityCardMockup() {
  return (
    <div className="rounded-lg border border-red-300 bg-gradient-to-b from-red-50 to-white overflow-hidden shadow-sm">
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <div>
            <div className="text-[9px] uppercase tracking-wide text-gray-500">#1 Opportunity</div>
            <h3 className="font-semibold text-sm">Mobile sync reliability</h3>
          </div>
          <div className="text-right">
            <div className="text-xl font-bold text-red-700">8.4</div>
            <div className="text-[9px] text-gray-400 uppercase">of 10</div>
          </div>
        </div>
        <div className="flex flex-wrap gap-1 text-[10px] mb-2">
          <span className="px-1.5 py-0.5 rounded-full bg-red-100 text-red-800">241 items</span>
          <span className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-700">Negative · -0.42</span>
          <span className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-600">sync</span>
          <span className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-600">android</span>
        </div>
        <blockquote className="text-xs text-gray-700 italic border-l-2 border-gray-300 pl-2 mb-3 line-clamp-2">
          "offline mode mobile is completely broken since the redesign. lost 40
          min of work monday"
        </blockquote>
        <div className="pt-2 border-t border-gray-200">
          <div className="flex items-center gap-1 text-[10px] font-medium text-gray-600 mb-0.5">
            <Sparkles className="w-2.5 h-2.5" /> AI recommendation
          </div>
          <p className="text-xs text-gray-800 line-clamp-3">
            Ship a background sync worker with exponential backoff and surface
            a "last synced" indicator on every list view. Prioritise Android
            14 regression first — 68% of reports cite this version.
          </p>
        </div>
      </div>
    </div>
  )
}

function ArtifactMockup() {
  return (
    <div className="rounded-lg border border-gray-200 bg-white overflow-hidden shadow-sm">
      <div className="px-4 py-3 border-b border-gray-100 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-gray-500" />
          <span className="font-medium text-xs">PRD.md · 2,835 chars</span>
        </div>
        <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border border-gray-200 text-gray-600">
          <ChevronDown className="w-2.5 h-2.5" /> expand
        </span>
      </div>
      <div className="p-4 bg-gray-50 font-mono text-[10px] leading-relaxed text-gray-800 space-y-1">
        <div className="font-bold text-gray-900"># Product Requirements: Mobile sync reliability</div>
        <div className="text-gray-500">## Problem summary</div>
        <div>Users on Android 14 report stale data after app resume. 241 items across 3 segments, -0.42 sentiment, 18% of top-of-funnel tickets this week.</div>
        <div className="text-gray-500">## Proposed solution</div>
        <div>Background sync worker with exponential backoff, explicit "last synced" indicator on list views, Android 14 regression fix.</div>
        <div className="text-gray-500">## Success metrics</div>
        <div>— p95 sync latency &lt; 2s on mobile</div>
        <div>— mobile-tagged ticket volume ↓ 60% over 30 days</div>
        <div className="text-gray-400 italic pt-1">... + Target users, Risks, Acceptance</div>
      </div>
    </div>
  )
}

function CapabilityCard({ cap }) {
  const Icon = cap.icon
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 hover:border-pulse-300 hover:shadow-sm transition-all">
      <div className="w-9 h-9 rounded-lg bg-pulse-50 text-pulse-600 flex items-center justify-center mb-3">
        <Icon className="w-4 h-4" />
      </div>
      <h3 className="font-semibold text-sm mb-1">{cap.title}</h3>
      <p className="text-sm text-gray-600 leading-relaxed">{cap.body}</p>
    </div>
  )
}

function PersonaCard({ persona }) {
  const Icon = persona.icon
  return (
    <div className="rounded-xl border border-gray-200 bg-white overflow-hidden">
      <div className="p-5 border-b border-gray-100">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-lg bg-pulse-50 text-pulse-600 flex items-center justify-center flex-shrink-0">
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-semibold">{persona.role}</h3>
            <p className="text-xs text-gray-500">{persona.scenario}</p>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-gray-100">
        <div className="p-4 bg-gray-50">
          <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-1">Before Pulse</div>
          <p className="text-sm text-gray-700 leading-relaxed">{persona.before}</p>
        </div>
        <div className="p-4 bg-emerald-50/30">
          <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-700 mb-1">With Pulse</div>
          <p className="text-sm text-gray-700 leading-relaxed">{persona.after}</p>
        </div>
      </div>
      <div className="px-5 py-2.5 border-t border-gray-100 bg-white text-xs text-gray-500 flex items-center gap-1.5">
        <Check className="w-3.5 h-3.5 text-emerald-600" />
        Saves <span className="font-semibold text-gray-800">{persona.saves}</span>
      </div>
    </div>
  )
}

function MoatCard({ moat }) {
  const Icon = moat.icon
  return (
    <div className="flex gap-4 p-5 rounded-xl border border-gray-200 bg-white">
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-pulse-500 to-purple-600 text-white flex items-center justify-center flex-shrink-0">
        <Icon className="w-5 h-5" />
      </div>
      <div className="flex-1">
        <h3 className="font-semibold mb-1">{moat.title}</h3>
        <p className="text-sm text-gray-600 leading-relaxed">{moat.body}</p>
      </div>
    </div>
  )
}

export default function HowItWorksPage() {
  // Cycle an "active" index through the pipeline steps every 1.8s to suggest
  // data flowing through the system. Reduces to a static highlight for users
  // who have prefers-reduced-motion enabled.
  const [activeStep, setActiveStep] = useState(0)
  useEffect(() => {
    const prefersReduced = typeof window !== 'undefined'
      && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    if (prefersReduced) return
    const t = setInterval(() => {
      setActiveStep((s) => (s + 1) % PIPELINE.length)
    }, 1800)
    return () => clearInterval(t)
  }, [])

  return (
    <div className="max-w-6xl mx-auto px-6 pb-16">
      {/* Hero */}
      <header className="pt-10 md:pt-14 pb-6 md:pb-10">
        <div className="inline-flex items-center gap-1.5 text-xs font-medium text-pulse-700 bg-pulse-50 px-2.5 py-1 rounded-full mb-4">
          <Layers className="w-3 h-3" />
          Product intelligence, end-to-end
        </div>
        <h1 className="text-3xl md:text-5xl font-bold leading-tight tracking-tight max-w-4xl">
          From <span className="text-gray-400 line-through decoration-2">10,000 scattered tickets</span>{' '}
          <span className="bg-gradient-to-r from-pulse-600 to-purple-600 bg-clip-text text-transparent">to one PRD.</span>
          <br className="hidden md:block" />
          In 60 seconds.
        </h1>
        <p className="text-base md:text-lg text-gray-600 mt-4 max-w-3xl leading-relaxed">
          Pulse ingests every channel of customer feedback your company has, clusters the themes with real semantic embeddings,
          scores the opportunities, and auto-writes the PRD + Jira tickets. Think of it as the brain your product org wishes it had —
          the layer between "we got a lot of complaints this week" and "we know exactly what to build next".
        </p>
        <div className="flex flex-wrap gap-3 mt-6">
          <Link
            to="/"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-pulse-600 hover:bg-pulse-700 text-white rounded-lg font-medium text-sm"
          >
            Try it with your data
            <ArrowRight className="w-4 h-4" />
          </Link>
          <a
            href="/docs"
            target="_blank"
            rel="noopener"
            className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-gray-300 hover:border-gray-400 text-gray-700 rounded-lg font-medium text-sm"
          >
            API reference
            <Code2 className="w-4 h-4" />
          </a>
        </div>
      </header>

      {/* The gap we close */}
      <Section
        eyebrow="The gap we close"
        title="Customer feedback is the biggest dataset nobody uses"
        subtitle="Every SaaS company sits on tens of thousands of support tickets, NPS comments, Slack messages, and app reviews. Reading them is manual, subjective, and doesn't scale. The signal gets lost."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="rounded-xl border-2 border-gray-200 bg-white p-5">
            <div className="text-[10px] font-bold uppercase tracking-wider text-gray-500 mb-3">Before Pulse — the old flow</div>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex gap-2"><span className="text-gray-400">→</span> PM opens Intercom, Zendesk, Slack, and email every Monday</li>
              <li className="flex gap-2"><span className="text-gray-400">→</span> Skims 200–500 tickets, tags by hand, copies into Notion</li>
              <li className="flex gap-2"><span className="text-gray-400">→</span> Argues with engineering on Monday standup about priority</li>
              <li className="flex gap-2"><span className="text-gray-400">→</span> Writes a PRD, breaks down 5 tickets, pushes to Jira</li>
              <li className="flex gap-2 text-gray-500 mt-3 pt-3 border-t border-gray-100">
                <Clock className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span><strong>6–8 hours / week</strong> of senior PM time</span>
              </li>
            </ul>
          </div>
          <div className="rounded-xl border-2 border-pulse-300 bg-gradient-to-br from-pulse-50 to-white p-5">
            <div className="text-[10px] font-bold uppercase tracking-wider text-pulse-700 mb-3">With Pulse — the new flow</div>
            <ul className="space-y-2 text-sm text-gray-700">
              <li className="flex gap-2"><span className="text-pulse-500">→</span> Feedback streams in automatically (CSV, Intercom, Slack, API)</li>
              <li className="flex gap-2"><span className="text-pulse-500">→</span> Every item auto-tagged with sentiment, urgency, category</li>
              <li className="flex gap-2"><span className="text-pulse-500">→</span> Opens the Report → top 3 themes + scores, already ranked</li>
              <li className="flex gap-2"><span className="text-pulse-500">→</span> Copies the auto-written PRD, pushes the 5 Jira tickets</li>
              <li className="flex gap-2 text-pulse-800 mt-3 pt-3 border-t border-pulse-100">
                <Zap className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span><strong>60 seconds</strong> of PM time + one review pass</span>
              </li>
            </ul>
          </div>
        </div>
      </Section>

      {/* Pipeline */}
      <Section
        eyebrow="How it works"
        title="Five layers. One continuous pipeline."
        subtitle="Every feedback item flows through the same stages. Add a new source and it lights up every downstream feature automatically."
      >
        <div className="flex flex-col md:flex-row md:items-stretch gap-6 md:gap-0 bg-white border border-gray-200 rounded-2xl p-6 md:p-8 relative overflow-hidden">
          {/* Subtle data-flow gradient sweep on desktop */}
          <div
            className="hidden md:block absolute top-0 bottom-0 w-24 bg-gradient-to-r from-transparent via-pulse-100/30 to-transparent pointer-events-none transition-all duration-700"
            style={{ left: `${(activeStep / Math.max(1, PIPELINE.length - 1)) * 100}%`, transform: 'translateX(-50%)' }}
            aria-hidden
          />
          {PIPELINE.map((step, i) => (
            <PipelineStep
              key={step.title}
              step={step}
              index={i}
              isLast={i === PIPELINE.length - 1}
              active={i === activeStep}
            />
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-4">
          Every stage persists its output, so later stages can be re-run independently. Clustering uses current embeddings;
          the report pulls the latest completed analysis run; trends stitch together the last 12 weeks of snapshots.
        </p>
      </Section>

      {/* See it in action — live HTML mockups of the actual product screens */}
      <Section
        eyebrow="See it in action"
        title="Three screens. One story."
        subtitle="Every analysis run surfaces in the Report page, drills into opportunity cards, and ships as copy-paste-ready PRDs. Below are the actual UI components, rendered with representative data."
      >
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-pulse-600 text-white text-[10px] font-bold">1</span>
              <span className="text-xs font-semibold text-gray-700">The headline</span>
            </div>
            <ReportHeroMockup />
            <p className="text-xs text-gray-500 mt-2 leading-relaxed">
              One sentence that tells the CPO what changed this week. Generated by Claude from the top-3 clusters + anomalies.
            </p>
          </div>
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-pulse-600 text-white text-[10px] font-bold">2</span>
              <span className="text-xs font-semibold text-gray-700">The opportunity</span>
            </div>
            <OpportunityCardMockup />
            <p className="text-xs text-gray-500 mt-2 leading-relaxed">
              Each top theme shows volume, sentiment, a representative quote, and an AI recommendation — inline, not behind a modal.
            </p>
          </div>
          <div>
            <div className="mb-2 flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-pulse-600 text-white text-[10px] font-bold">3</span>
              <span className="text-xs font-semibold text-gray-700">The deliverable</span>
            </div>
            <ArtifactMockup />
            <p className="text-xs text-gray-500 mt-2 leading-relaxed">
              PRD, Jira tickets, and an executive summary generated per run. Copy into Notion / Linear / Jira directly.
            </p>
          </div>
        </div>
      </Section>

      {/* Under the hood */}
      <Section
        eyebrow="Under the hood"
        title="This is not a CSV-with-charts tool"
        subtitle="Ten independent capabilities that together turn unstructured feedback into prioritised, document-ready product insight."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {CAPABILITIES.map((c) => <CapabilityCard key={c.title} cap={c} />)}
        </div>
      </Section>

      {/* Use cases */}
      <Section
        eyebrow="Who it's for"
        title="Four roles. Four hours saved. Same platform."
        subtitle="Every seat in the product org gets their own angle on the same underlying data."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {PERSONAS.map((p) => <PersonaCard key={p.role} persona={p} />)}
        </div>
      </Section>

      {/* Moat */}
      <Section
        eyebrow="Why it's defensible"
        title="The moat compounds with every customer"
        subtitle="This category has existed for years (Dovetail, Productboard, Canny). What makes Pulse hard to compete with isn't the clustering — it's what happens at scale."
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {MOATS.map((m) => <MoatCard key={m.title} moat={m} />)}
        </div>
      </Section>

      {/* Stack */}
      <Section
        eyebrow="Built on"
        title="Modern stack, batteries included"
        subtitle="Production-grade pieces where it matters; pragmatic defaults where it doesn't."
      >
        <div className="flex flex-wrap gap-2">
          {STACK.map((t) => (
            <span key={t} className="px-3 py-1.5 rounded-lg bg-gray-100 border border-gray-200 text-sm text-gray-700 font-mono">
              {t}
            </span>
          ))}
        </div>
      </Section>

      {/* Trusted by — honest placeholder wordmarks */}
      <section className="pt-10 pb-4">
        <div className="text-center">
          <p className="text-[10px] font-bold uppercase tracking-[0.2em] text-gray-400 mb-6">
            Design placeholders — real logos land when customers do
          </p>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-x-6 gap-y-6 items-center justify-items-center max-w-4xl mx-auto opacity-70 grayscale hover:grayscale-0 hover:opacity-100 transition-all duration-500">
            {TRUSTED_BY.map((logo) => (
              <span
                key={logo.name}
                className={`text-gray-500 text-lg ${logo.style}`}
              >
                {logo.name}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mt-8 rounded-2xl bg-gradient-to-br from-pulse-600 via-pulse-700 to-purple-700 p-8 md:p-12 text-white">
        <h2 className="text-2xl md:text-3xl font-bold mb-2">Ready to see it on your data?</h2>
        <p className="text-pulse-100 mb-6 max-w-2xl">
          Drop a CSV of your last month of support tickets. In under a minute you'll have clustered themes, a scored opportunity ranking,
          and a PRD + Jira breakdown ready for your next planning session.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-5 py-2.5 bg-white text-pulse-700 hover:bg-pulse-50 rounded-lg font-semibold text-sm"
        >
          Open the Report
          <ArrowRight className="w-4 h-4" />
        </Link>
      </section>
    </div>
  )
}
