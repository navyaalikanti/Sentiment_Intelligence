import React from 'react';

export function Panel({ title, subtitle, actions, children, className = '' }) {
  return (
    <section className={`glass rounded-[16px] p-6 ${className}`}>
      {(title || subtitle || actions) && (
        <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
          <div>
            {title ? <h2 className="text-[18px] font-semibold tracking-tight text-white">{title}</h2> : null}
            {subtitle ? <p className="mt-2 text-[15px] leading-6 text-slate-400">{subtitle}</p> : null}
          </div>
          {actions ? <div>{actions}</div> : null}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatCard({ label, value, helper, tone = 'cyan' }) {
  const tones = {
    cyan: 'border-blue-500/20',
    violet: 'border-blue-500/20',
    emerald: 'border-blue-500/20',
    amber: 'border-blue-500/20',
    rose: 'border-blue-500/20',
    slate: 'border-blue-500/20',
  };

  return (
    <div className={`surface relative overflow-hidden rounded-[16px] border p-5 ${tones[tone] || tones.cyan}`}>
      <div className="absolute inset-x-0 top-0 h-px bg-blue-500/35" />
      <div className="text-[13px] font-medium tracking-wide text-slate-400">{label}</div>
      <div className="mt-4 text-[28px] font-semibold tracking-tight text-white">{value}</div>
      {helper ? <div className="mt-2 max-w-[26ch] text-[15px] leading-6 text-slate-400">{helper}</div> : null}
    </div>
  );
}

export function Badge({ children, tone = 'cyan' }) {
  const tones = {
    cyan: 'border-blue-500/20 bg-blue-500/10 text-blue-100',
    emerald: 'border-blue-500/20 bg-blue-500/10 text-blue-100',
    amber: 'border-blue-500/20 bg-blue-500/10 text-blue-100',
    rose: 'border-blue-500/20 bg-blue-500/10 text-blue-100',
    slate: 'border-white/10 bg-white/5 text-slate-200',
    violet: 'border-blue-500/20 bg-blue-500/10 text-blue-100',
  };

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1 text-[13px] font-medium tracking-normal ${tones[tone] || tones.cyan}`}>
      {children}
    </span>
  );
}

export function EmptyState({ title, description }) {
  return (
    <div className="surface rounded-[16px] border border-dashed border-white/10 p-8 text-center">
      <div className="text-[18px] font-semibold tracking-tight text-white">{title}</div>
      <p className="mx-auto mt-3 max-w-2xl text-[15px] leading-6 text-slate-400">{description}</p>
    </div>
  );
}

export function SectionTitle({ eyebrow, title, description }) {
  return (
    <div className="mb-8">
      {eyebrow ? (
        <div className="inline-flex items-center rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1 text-[12px] font-medium tracking-normal text-blue-100">
          {eyebrow}
        </div>
      ) : null}
      <h1 className="mt-4 text-[36px] font-semibold tracking-tight text-white">{title}</h1>
      {description ? <p className="mt-3 max-w-4xl text-[15px] leading-7 text-slate-300">{description}</p> : null}
    </div>
  );
}

export function InlineField({ label, children }) {
  return (
    <label className="block">
      <span className="mb-3 block text-[13px] font-medium tracking-normal text-slate-400">{label}</span>
      {children}
    </label>
  );
}
