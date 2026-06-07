import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../lib/api';
import { EmptyState } from '../components/UI';

function WorkflowIcon({ type }) {
  const icons = {
    amazon: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M7 7h10l-1 12H8L7 7z" />
        <path d="M9 7a3 3 0 1 1 6 0" />
      </svg>
    ),
    process: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M4 8h16" />
        <path d="M4 16h16" />
        <circle cx="8" cy="8" r="2" />
        <circle cx="16" cy="16" r="2" />
      </svg>
    ),
    sentiment: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M4 19V5" />
        <path d="M8 19V9" />
        <path d="M12 19V7" />
        <path d="M16 19V11" />
        <path d="M20 19V6" />
      </svg>
    ),
    nlp: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M6 7h12" />
        <path d="M6 12h12" />
        <path d="M6 17h12" />
        <circle cx="9" cy="7" r="1" />
        <circle cx="14" cy="12" r="1" />
        <circle cx="10" cy="17" r="1" />
      </svg>
    ),
    compare: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M5 6h6v12H5z" />
        <path d="M13 6h6v8h-6z" />
        <path d="M13 18h6" />
      </svg>
    ),
    insight: (
      <svg viewBox="0 0 24 24" className="workflow-icon h-6 w-6 text-blue-100" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 3v3" />
        <path d="M7 8l-2 2" />
        <path d="M17 8l2 2" />
        <path d="M6 13h12" />
        <path d="M9 17h6" />
      </svg>
    ),
  };

  return icons[type] || icons.process;
}

function PipelineStep({ title, description, icon, onClick, index, isActive, isBeforeActive, onHover, onLeave }) {
  const stepClass = isActive
    ? 'border-blue-400/80 bg-white/[0.12] shadow-[0_28px_80px_rgba(37,99,235,0.36)] ring-1 ring-blue-400/20'
    : isBeforeActive
      ? 'border-blue-400/45 bg-blue-500/[0.075] shadow-[0_18px_54px_rgba(37,99,235,0.2)]'
      : 'border-white/10 bg-white/[0.035]';
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={onHover}
      onFocus={onHover}
      onMouseLeave={onLeave}
      className={`workflow-card group relative flex h-[120px] min-w-0 flex-1 flex-col items-center justify-center overflow-visible rounded-[16px] border px-4 py-4 text-center outline-none animate-fade-up transition duration-[250ms] ${stepClass} hover:-translate-y-2 hover:scale-[1.04] focus-visible:-translate-y-2 focus-visible:scale-[1.04] focus-visible:border-blue-400/80 focus-visible:ring-2 focus-visible:ring-blue-400/25`}
      style={{ animationDelay: `${index * 95}ms` }}
      aria-label={title}
    >
      <div className="workflow-icon-shell flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-blue-500/10">
        <WorkflowIcon type={icon} />
      </div>
      <div className="mt-3 text-[17px] font-semibold tracking-tight text-white">{title}</div>
      <div className="pointer-events-none absolute inset-x-3 bottom-3 rounded-[14px] border border-white/10 bg-[rgba(8,12,26,0.9)] px-3 py-2 text-[12px] leading-4 text-slate-300 opacity-0 shadow-[0_18px_34px_rgba(0,0,0,0.32)] backdrop-blur-xl transition duration-[250ms] translate-y-2 group-hover:translate-y-0 group-hover:opacity-100 group-focus-visible:translate-y-0 group-focus-visible:opacity-100">
        {description}
      </div>
    </button>
  );
}

function PipelineArrow({ index, isActive }) {
  return (
    <div
      className="relative hidden h-[116px] w-16 items-center justify-center lg:flex animate-fade-up"
      style={{ animationDelay: `${index * 95 + 50}ms` }}
    >
      <div className={`pipeline-arrow relative h-[2px] w-full overflow-hidden rounded-full ${isActive ? 'bg-blue-500/25 shadow-[0_0_28px_rgba(96,165,250,0.3)]' : 'bg-blue-500/15 shadow-[0_0_18px_rgba(59,130,246,0.16)]'}`}>
        <span className={`absolute left-0 top-1/2 h-2 w-2 -translate-y-1/2 rounded-full bg-blue-300 ${isActive ? 'shadow-[0_0_22px_rgba(96,165,250,0.95)]' : 'shadow-[0_0_18px_rgba(96,165,250,0.85)]'}`} />
        <span className={`absolute right-0 top-1/2 -translate-y-1/2 text-[16px] leading-none ${isActive ? 'text-white drop-shadow-[0_0_14px_rgba(96,165,250,0.7)]' : 'text-blue-200 drop-shadow-[0_0_10px_rgba(96,165,250,0.5)]'}`}>→</span>
      </div>
    </div>
  );
}

function FeatureCard({ title, description, onClick, index }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="workflow-card group rounded-[16px] border border-white/10 bg-white/[0.04] p-5 text-left animate-fade-up"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[22px] font-semibold tracking-tight text-white sm:text-[24px]">{title}</div>
          <div className="mt-3 text-[16px] leading-7 text-slate-400">{description}</div>
        </div>
        <div className="rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1 text-[12px] font-medium text-blue-100">
          Open
        </div>
      </div>
    </button>
  );
}

function TechBadge({ children, index }) {
  return (
    <span
      className="inline-flex items-center rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1.5 text-[12px] font-medium text-slate-200 transition duration-300 hover:-translate-y-0.5 hover:border-blue-400/30 hover:bg-blue-500/[0.08] hover:text-white animate-fade-up"
      style={{ animationDelay: `${index * 45}ms` }}
    >
      {children}
    </span>
  );
}

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [ratings, setRatings] = useState([]);
  const [sentiments, setSentiments] = useState({ vader: [], roberta: [] });
  const navigate = useNavigate();
  const [activeWorkflowIndex, setActiveWorkflowIndex] = useState(null);

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([
      api.health(),
      api.datasetStatistics(5000),
      api.ratingDistribution(5000),
      api.sentimentDistribution(150),
    ]).then((results) => {
      if (!mounted) return;

      const errors = [];
      const [healthResult, statsResult, ratingResult, sentimentResult] = results;

      if (healthResult.status === 'fulfilled') {
        setHealth(healthResult.value);
      } else {
        console.error('[Dashboard] health request failed', healthResult.reason);
        errors.push(`health: ${healthResult.reason.message || healthResult.reason}`);
      }

      if (statsResult.status === 'fulfilled') {
        setStats(statsResult.value);
      } else {
        console.error('[Dashboard] dataset statistics request failed', statsResult.reason);
        errors.push(`dataset statistics: ${statsResult.reason.message || statsResult.reason}`);
      }

      if (ratingResult.status === 'fulfilled') {
        setRatings(ratingResult.value.data || []);
      } else {
        console.error('[Dashboard] rating distribution request failed', ratingResult.reason);
        errors.push(`rating distribution: ${ratingResult.reason.message || ratingResult.reason}`);
      }

      if (sentimentResult.status === 'fulfilled') {
        setSentiments({
          vader: sentimentResult.value.data?.vader || [],
          roberta: sentimentResult.value.data?.roberta || [],
          agreement_summary: sentimentResult.value.agreement_summary || null,
        });
      } else {
        console.error('[Dashboard] sentiment distribution request failed', sentimentResult.reason);
        errors.push(`sentiment distribution: ${sentimentResult.reason.message || sentimentResult.reason}`);
      }

      setError(errors.length ? errors.join(' | ') : '');
      setLoading(false);
    });

    return () => {
      mounted = false;
    };
  }, []);

  const vaderData = sentiments.vader || [];
  const robertaData = sentiments.roberta || [];
  const agreementSummary = sentiments.agreement_summary || null;

  const workflowSteps = [
    {
      title: 'Amazon Reviews',
      description: 'Raw customer reviews collected from Amazon product datasets.',
      icon: 'amazon',
      to: '/amazon-reviews#dataset-analysis',
      accent: 'bg-blue-500/10',
    },
    {
      title: 'Text Processing',
      description: 'Cleaning, tokenization, and preparation of review text.',
      icon: 'process',
      to: '/amazon-reviews#dataset-analysis',
      accent: 'bg-blue-500/10',
    },
    {
      title: 'Sentiment Analysis',
      description: 'VADER and RoBERTa generate sentiment predictions.',
      icon: 'sentiment',
      to: '/analyze',
      accent: 'bg-blue-500/10',
    },
    {
      title: 'NLP Analysis',
      description: 'Tokenization, POS tagging, and entity recognition.',
      icon: 'nlp',
      to: '/playground',
      accent: 'bg-blue-500/10',
    },
    {
      title: 'Model Comparison',
      description: 'Comparison of VADER and RoBERTa predictions.',
      icon: 'compare',
      to: '/amazon-reviews#comparison',
      accent: 'bg-blue-500/10',
    },
    {
      title: 'Business Insights',
      description: 'Generate actionable insights from customer feedback.',
      icon: 'insight',
      to: '/unusual',
      accent: 'bg-blue-500/10',
    },
  ];

  const features = [
    {
      title: 'Sentiment Intelligence',
      description: 'Explore sentiment scores, model confidence, NLP insights, and review intelligence in a unified AI-powered workspace.',
      to: '/analyze',
    },
    {
      title: 'Analyze Review',
      description: 'Analyze customer reviews using VADER and RoBERTa models with real-time sentiment prediction.',
      to: '/analyze',
    },
    {
      title: 'NLP Playground',
      description: 'Explore tokenization, POS tagging, named entity recognition, and linguistic insights.',
      to: '/playground',
    },
    {
      title: 'Amazon Reviews',
      description: 'Browse dataset analysis, sentiment distribution, and model comparison in one unified Amazon reviews workspace.',
      to: '/amazon-reviews',
    },
    {
      title: 'Machine Learning Evaluation',
      description: 'Train, evaluate, and compare traditional machine learning models against modern NLP sentiment models. ',
      to: '/model-evaluation',
    },
    {
      title: 'Unusual Reviews',
      description: 'Identify reviews where sentiment differs significantly from the user rating.',
      to: '/unusual',
    },
  ];

  const techStack = ['Python', 'Flask', 'React', 'VADER', 'RoBERTa', 'NLTK', 'Pandas', 'Machine Learning'];

  return (
    <div className="space-y-0 overflow-x-hidden">
      {error ? <EmptyState title="API unavailable" description={error} /> : null}

      <section className="relative flex min-h-[100svh] items-center justify-center px-4 py-16">
        <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
          <span className="absolute left-[12%] top-[18%] h-2.5 w-2.5 rounded-full bg-blue-400/70 shadow-[0_0_18px_rgba(96,165,250,0.65)] animate-pulse" />
          <span className="absolute left-[18%] top-[30%] h-1.5 w-1.5 rounded-full bg-blue-300/60 shadow-[0_0_14px_rgba(96,165,250,0.55)] animate-pulse [animation-delay:700ms]" />
          <span className="absolute right-[15%] top-[20%] h-2 w-2 rounded-full bg-blue-300/70 shadow-[0_0_16px_rgba(96,165,250,0.6)] animate-pulse [animation-delay:300ms]" />
          <span className="absolute right-[22%] top-[36%] h-1.5 w-1.5 rounded-full bg-blue-200/60 shadow-[0_0_12px_rgba(147,197,253,0.5)] animate-pulse [animation-delay:1100ms]" />
          <span className="absolute left-[28%] bottom-[22%] h-2 w-2 rounded-full bg-blue-400/50 shadow-[0_0_14px_rgba(59,130,246,0.5)] animate-pulse [animation-delay:500ms]" />
          <span className="absolute right-[28%] bottom-[18%] h-1.5 w-1.5 rounded-full bg-blue-300/55 shadow-[0_0_10px_rgba(96,165,250,0.45)] animate-pulse [animation-delay:900ms]" />
          <span className="absolute left-[50%] top-[12%] h-1 w-1 rounded-full bg-blue-200/70 shadow-[0_0_10px_rgba(96,165,250,0.6)] animate-pulse [animation-delay:1300ms]" />
        </div>

        <div className="mx-auto w-full max-w-[980px] -translate-y-10 text-center sm:-translate-y-12 lg:-translate-y-14">
          <h1 className="mx-auto animate-fade-up text-[clamp(3.2rem,5vw,4.3rem)] font-bold tracking-tight text-white sm:text-[clamp(3.8rem,5.4vw,4.9rem)]">
            <span className="block">Discover the Voice</span>
            <span className="block">Behind Every Review</span>
          </h1>
          <p className="mx-auto mt-8 max-w-[900px] animate-fade-up text-[clamp(1.15rem,2vw,1.35rem)] font-normal leading-8 text-slate-300 sm:text-[clamp(1.25rem,1.8vw,1.4rem)]">
            Artificial Intelligence • Machine Learning • Natural Language Processing
          </p>
        </div>
      </section>

      <section className="flex min-h-[100svh] items-center px-4 py-16">
        <div className="mx-auto w-full max-w-[1500px]">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="text-[clamp(2.25rem,4vw,3.25rem)] font-semibold tracking-tight text-white">
              From Review to Insight
            </h2>
            <p className="mx-auto mt-4 max-w-3xl text-[16px] leading-7 text-slate-300">
              See how customer feedback flows through our AI and NLP pipeline to generate meaningful business intelligence.
            </p>
          </div>

          <div className="mt-14 rounded-[18px] border border-white/10 bg-[#0F172A]/80 p-4 sm:p-5 lg:p-6">
            <div className="grid gap-4 lg:flex lg:items-stretch lg:gap-0" onMouseLeave={() => setActiveWorkflowIndex(null)}>
              {workflowSteps.map((step, index) => (
                <React.Fragment key={step.title}>
                  <div className="relative flex-1">
                    <PipelineStep
                      index={index}
                      title={step.title}
                      description={step.description}
                      icon={step.icon}
                      isActive={activeWorkflowIndex === index}
                      isBeforeActive={activeWorkflowIndex !== null && index <= activeWorkflowIndex}
                      onClick={() => navigate(step.to)}
                      onHover={() => setActiveWorkflowIndex(index)}
                      onLeave={() => setActiveWorkflowIndex(null)}
                    />
                  </div>
                  {index < workflowSteps.length - 1 ? (
                    <PipelineArrow index={index} isActive={activeWorkflowIndex !== null && index < activeWorkflowIndex} />
                  ) : null}
                </React.Fragment>
              ))}
            </div>
            <div className="mt-6 hidden lg:block">
              <div className="relative h-[3px] rounded-full bg-white/5">
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-blue-500 transition-all duration-300"
                  style={{
                    width:
                      activeWorkflowIndex === null
                        ? '0%'
                        : `${((activeWorkflowIndex + 1) / workflowSteps.length) * 100}%`,
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="flex min-h-[100svh] items-center px-4 py-16">
        <div className="mx-auto w-full max-w-[1500px] space-y-12">
          <div className="mx-auto max-w-4xl text-center">
            <h2 className="text-[clamp(2.25rem,4vw,3.25rem)] font-semibold tracking-tight text-white">
              Explore the Platform
            </h2>
            <p className="mx-auto mt-4 max-w-3xl text-[16px] leading-7 text-slate-300">
              Jump directly into the parts of the product that drive the analysis workflow.
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {features.map((feature, index) => (
              <FeatureCard
                key={feature.title}
                index={index}
                title={feature.title}
                description={feature.description}
                onClick={() => navigate(feature.to)}
              />
            ))}
          </div>

          <div className="space-y-4 pt-2">
            <div className="text-center">
              <h2 className="text-[32px] font-semibold tracking-tight text-white">Powered By</h2>
            </div>

            <div className="flex flex-wrap justify-center gap-2.5">
              {techStack.map((item, index) => (
                <TechBadge key={item} index={index}>
                  {item}
                </TechBadge>
              ))}
            </div>
          </div>

          <section className="glass mx-auto max-w-5xl rounded-[16px] border border-white/10 px-5 py-4 sm:px-6 sm:py-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <h2 className="text-[18px] font-semibold tracking-tight text-white">Start Exploring</h2>
                <p className="mt-2 max-w-3xl text-[14px] leading-6 text-slate-400">
                  Choose a module from the workflow or feature cards above to continue into the analytics experience.
                </p>
              </div>
              <div className="inline-flex self-start rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1 text-[12px] font-medium text-blue-100 sm:self-auto">
                Compact closing section
              </div>
            </div>
          </section>
        </div>
      </section>
    </div>
  );
}
