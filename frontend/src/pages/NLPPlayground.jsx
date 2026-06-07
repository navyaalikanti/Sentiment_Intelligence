import React, { useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import { Badge, EmptyState, InlineField, Panel, SectionTitle } from '../components/UI';
import { useReviewWorkflow } from '../context/ReviewWorkflowContext';

export default function NLPPlayground() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const { workflow, setWorkflow } = useReviewWorkflow();
  const lastAutoAnalyzedRef = useRef('');

  useEffect(() => {
    if (!workflow.autoAnalyze || !workflow.text) return;
    if (lastAutoAnalyzedRef.current === workflow.text) return;

    lastAutoAnalyzedRef.current = workflow.text;
    setText(workflow.text);
    setWorkflow((current) => ({
      ...current,
      autoAnalyze: false,
    }));

    void analyzeText(workflow.text);
  }, [workflow.autoAnalyze, workflow.text, setWorkflow]);

  async function analyzeText(value) {
    const input = String(value || '').trim();
    if (!input) return;

    setLoading(true);
    setError('');
    try {
      setResult(await api.nlpAnalysis(input));
    } catch (err) {
      console.error('[NLPPlayground] analysis failed', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const runManualAnalysis = () => {
    void analyzeText(text);
  };

  return (
    <div className="space-y-8">
      <SectionTitle
        eyebrow="NLP Lab"
        title="NLP Playground"
        description="Dedicated linguistic inspection workspace for tokens, POS tags, named entities, and model details."
      />

      <Panel
        title="Experiment"
        subtitle="Type any short sentence or review"
        actions={
          <button
            type="button"
            onClick={runManualAnalysis}
            disabled={loading}
            className="inline-flex items-center justify-center rounded-2xl bg-blue-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? 'Running...' : 'Run NLP'}
          </button>
        }
        className="overflow-hidden"
      >
        <InlineField label="Input text">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={7}
            className="min-h-[210px] w-full rounded-[16px] border border-white/10 bg-[#0A0A0A]/60 px-5 py-4 text-[15px] text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
          />
        </InlineField>
      </Panel>

      {error ? <EmptyState title="NLP analysis failed" description={error} /> : null}

      {result ? (
        <div className="grid gap-6 xl:grid-cols-2">
          <Panel title="Tokens + POS Tags" subtitle="Token-level structure from NLTK">
            <div className="space-y-5">
              <div className="flex flex-wrap gap-2">
                {(result.tokens || []).map((token) => (
                  <Badge key={token} tone="slate">
                    {token}
                  </Badge>
                ))}
              </div>

              <div className="rounded-[16px] border border-white/10 bg-[#0A0A0A]/45 p-4">
                <div className="grid grid-cols-[1fr_120px] gap-3 text-[15px]">
                  {result.pos_tags?.map((item, index) => (
                    <React.Fragment key={`${item.token}-${index}`}>
                      <div className="truncate font-medium text-slate-100">{item.token}</div>
                      <div className="text-slate-400">{item.tag}</div>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            </div>
          </Panel>

          <Panel title="Named Entities + Sentiment" subtitle="Entity extraction plus VADER and RoBERTa">
            <div className="space-y-5">
              <div className="flex flex-wrap gap-2">
                {result.named_entities?.length ? (
                  result.named_entities.map((entity, index) => (
                    <Badge key={`${entity.text}-${index}`} tone="slate">
                      {entity.text} ({entity.label})
                    </Badge>
                  ))
                ) : (
                  <span className="text-sm text-slate-400">No named entities detected.</span>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                  <div className="text-[13px] font-medium text-slate-400">VADER</div>
                  <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{result.vader?.label || '—'}</div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge tone="slate">{`compound ${Number(result.vader?.compound || 0).toFixed(3)}`}</Badge>
                    <Badge tone="slate">{`pos ${Number(result.vader?.pos || 0).toFixed(3)}`}</Badge>
                    <Badge tone="slate">{`neu ${Number(result.vader?.neu || 0).toFixed(3)}`}</Badge>
                    <Badge tone="slate">{`neg ${Number(result.vader?.neg || 0).toFixed(3)}`}</Badge>
                  </div>
                </div>

                <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                  <div className="text-[13px] font-medium text-slate-400">RoBERTa</div>
                  <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{result.roberta?.roberta_label || '—'}</div>
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Badge tone="slate">{`positive ${Number(result.roberta?.roberta_pos || 0).toFixed(3)}`}</Badge>
                    <Badge tone="slate">{`neutral ${Number(result.roberta?.roberta_neu || 0).toFixed(3)}`}</Badge>
                    <Badge tone="slate">{`negative ${Number(result.roberta?.roberta_neg || 0).toFixed(3)}`}</Badge>
                  </div>
                </div>
              </div>
            </div>
          </Panel>
        </div>
      ) : null}
    </div>
  );
}
