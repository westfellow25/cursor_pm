import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import translations from './translations'

const LanguageContext = createContext(null)

const DEFAULT_LANG = 'en'
const STORAGE_KEY = 'pulse_lang'

function detectInitialLang() {
  if (typeof window === 'undefined') return DEFAULT_LANG
  const stored = localStorage.getItem(STORAGE_KEY)
  if (stored && translations[stored]) return stored
  // Fall back to browser language prefix if we support it
  const browser = (navigator.language || '').slice(0, 2)
  if (translations[browser]) return browser
  return DEFAULT_LANG
}

export function LanguageProvider({ children }) {
  const [lang, setLangState] = useState(detectInitialLang)

  const setLang = useCallback((next) => {
    if (!translations[next]) return
    setLangState(next)
    try { localStorage.setItem(STORAGE_KEY, next) } catch { /* ignore */ }
    if (typeof document !== 'undefined') {
      document.documentElement.lang = next
    }
  }, [])

  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = lang
    }
  }, [lang])

  const t = useCallback((path, vars) => {
    const keys = path.split('.')
    let cursor = translations[lang]
    for (const k of keys) {
      if (cursor && typeof cursor === 'object' && k in cursor) {
        cursor = cursor[k]
      } else {
        cursor = null
        break
      }
    }
    // Fall back to English if a key is missing in the active language
    if (cursor == null && lang !== 'en') {
      let fallback = translations.en
      for (const k of keys) {
        if (fallback && typeof fallback === 'object' && k in fallback) {
          fallback = fallback[k]
        } else {
          fallback = null
          break
        }
      }
      cursor = fallback
    }
    if (cursor == null) return path
    if (typeof cursor === 'string' && vars) {
      return cursor.replace(/\{(\w+)\}/g, (_, k) => (vars[k] ?? `{${k}}`))
    }
    return cursor
  }, [lang])

  const value = { lang, setLang, t }
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useT() {
  const ctx = useContext(LanguageContext)
  if (!ctx) throw new Error('useT must be used inside <LanguageProvider>')
  return ctx
}
