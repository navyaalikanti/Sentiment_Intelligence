import React, { useMemo, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';

const topLevelItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/analyze', label: 'Analyse Review' },
  { to: '/playground', label: 'NLP Playground' },
  { to: '/model-evaluation', label: 'Model Evaluation' },
];

const amazonMenuItems = [
  { to: '/amazon-reviews', label: 'Data Analysis' },
  { to: '/unusual', label: 'Unusual Reviews' },
];

function navBase(isActive) {
  return [
    'inline-flex items-center rounded-full px-4 py-2 text-[15px] font-medium transition',
    isActive
      ? 'bg-blue-500/15 text-white ring-1 ring-inset ring-blue-500/30'
      : 'text-slate-300 hover:bg-white/5 hover:text-white',
  ].join(' ');
}

function BotIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 3v3" />
      <path d="M9 3h6" />
      <rect x="5" y="7" width="14" height="11" rx="4" />
      <path d="M9 12h.01" />
      <path d="M15 12h.01" />
      <path d="M8 17h8" />
    </svg>
  );
}

export default function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [amazonMenuOpen, setAmazonMenuOpen] = useState(false);
  const mobileNavItems = useMemo(() => [...topLevelItems, ...amazonMenuItems], []);

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100">
      <header className="sticky top-0 z-50 h-[72px] border-b border-white/10 bg-[#050816]/92 shadow-[0_12px_40px_rgba(0,0,0,0.35)] backdrop-blur-xl">
        <div className="mx-auto flex h-full max-w-7xl items-center gap-4 px-4 sm:px-6 lg:px-8">
          <NavLink to="/" className="flex min-w-0 items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-blue-500/20 bg-blue-500/10">
              <BotIcon />
            </div>
            <div className="min-w-0 leading-tight">
              <div className="truncate text-[22px] font-bold tracking-tight text-white">Sentiment Analysis</div>
              <div className="truncate text-[12px] font-medium text-slate-400">AI Review Intelligence</div>
            </div>
          </NavLink>

          <div className="flex-1" aria-hidden="true" />

          <nav className="hidden items-center gap-1 lg:flex">
            {topLevelItems.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.to === '/'} className={({ isActive }) => navBase(isActive)}>
                {item.label}
              </NavLink>
            ))}

            <div className="relative">
              <button
                type="button"
                onClick={() => setAmazonMenuOpen((value) => !value)}
                className={[
                  'inline-flex items-center gap-2 rounded-full px-4 py-2 text-[15px] font-medium transition',
                  amazonMenuOpen
                    ? 'bg-blue-500/15 text-white ring-1 ring-inset ring-blue-500/30'
                    : 'text-slate-300 hover:bg-white/5 hover:text-white',
                ].join(' ')}
                aria-haspopup="menu"
                aria-expanded={amazonMenuOpen}
              >
                Amazon Reviews
                <span className="text-[11px] text-slate-400">▼</span>
              </button>

              {amazonMenuOpen ? (
                <div className="absolute right-0 top-[calc(100%+10px)] w-60 overflow-hidden rounded-[16px] border border-white/10 bg-[#0F172A] shadow-[0_24px_60px_rgba(0,0,0,0.45)]">
                  {amazonMenuItems.map((item) => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      className={({ isActive }) =>
                        [
                          'block px-4 py-3 text-[15px] transition',
                          isActive ? 'bg-blue-500/15 text-white' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                        ].join(' ')
                      }
                      onClick={() => setAmazonMenuOpen(false)}
                    >
                      {item.label}
                    </NavLink>
                  ))}
                </div>
              ) : null}
            </div>
          </nav>

          <button
            type="button"
            className="inline-flex items-center justify-center rounded-2xl border border-white/10 bg-white/5 p-3 text-slate-100 transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-blue-500/40 lg:hidden"
            onClick={() => setMobileMenuOpen((value) => !value)}
            aria-label="Toggle navigation menu"
          >
            <span className="flex h-4 w-4 flex-col justify-between">
              <span className="block h-0.5 w-full rounded-full bg-current" />
              <span className="block h-0.5 w-full rounded-full bg-current" />
              <span className="block h-0.5 w-full rounded-full bg-current" />
            </span>
          </button>
        </div>

        {mobileMenuOpen ? (
          <div className="border-t border-white/10 bg-[#050816]/98 lg:hidden">
            <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-4 sm:px-6">
              {mobileNavItems.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === '/'}
                  onClick={() => setMobileMenuOpen(false)}
                  className={({ isActive }) =>
                    [
                      'rounded-[14px] px-4 py-3 text-[15px] transition',
                      isActive ? 'bg-blue-500/15 text-white ring-1 ring-inset ring-blue-500/30' : 'text-slate-300 hover:bg-white/5 hover:text-white',
                    ].join(' ')
                  }
                >
                  {item.label}
                </NavLink>
              ))}
            </div>
          </div>
        ) : null}
      </header>

      <main className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
        <Outlet />
      </main>
    </div>
  );
}
