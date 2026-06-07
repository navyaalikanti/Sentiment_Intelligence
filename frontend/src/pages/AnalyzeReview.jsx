import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../lib/api';
import { Badge, EmptyState, InlineField, Panel, SectionTitle } from '../components/UI';
import { useReviewWorkflow } from '../context/ReviewWorkflowContext';

function scoreTone(label) {
  if (label === 'Positive') return 'emerald';
  if (label === 'Negative') return 'rose';
  return 'amber';
}

function confidenceValue(scores) {
  const values = Object.values(scores || {}).map((value) => Number(value) || 0);
  if (!values.length) return 0;
  return Math.max(...values) * 100;
}

function modelLabel(result, fallback = '—') {
  return result?.label || result?.roberta_label || result?.ml_label || fallback;
}

function ConfidenceMeter({ title, label, confidence }) {
  const pct = Math.max(0, Math.min(100, Number(confidence) || 0));
  return (
    <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[13px] font-medium text-slate-400">{title}</div>
          <div className="mt-2 text-[22px] font-semibold tracking-tight text-white">{label}</div>
        </div>
        <div className="text-right">
          <div className="text-[13px] font-medium text-slate-400">Confidence</div>
          <div className="mt-2 text-[22px] font-semibold tracking-tight text-white">{pct.toFixed(1)}%</div>
        </div>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-white/5">
        <div className="h-full rounded-full bg-blue-500" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function IndicatorList({ title, items, emptyMessage = 'No strong sentiment-bearing words detected.' }) {
  return (
    <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
      <div className="text-[18px] font-semibold text-white">{title}</div>
      <div className="mt-4 space-y-2">
        {items?.length ? (
          items.map((item) => (
            <div key={item} className="text-[15px] leading-6 text-slate-300">
              <span className="mr-2 text-blue-300">✓</span>
              {item}
            </div>
          ))
        ) : (
          <div className="text-[15px] leading-6 text-slate-400">{emptyMessage}</div>
        )}
      </div>
    </div>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[14px] border border-white/10 bg-[#050816]/95 px-3 py-2 text-sm shadow-2xl">
      <div className="font-medium text-white">{label}</div>
      <div className="mt-1 text-slate-300">{Number(payload[0].value).toFixed(3)}</div>
    </div>
  );
}

function SentimentChart({ title, data }) {
  return (
    <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="text-[18px] font-semibold text-white">{title}</div>
          <div className="mt-1 text-[13px] text-slate-400">Sentiment score distribution</div>
        </div>
      </div>
      <div className="mt-4 h-[250px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
            <XAxis dataKey="label" stroke="#94A3B8" tick={{ fill: '#94A3B8', fontSize: 12 }} />
            <YAxis stroke="#94A3B8" tick={{ fill: '#94A3B8', fontSize: 12 }} />
            <Tooltip content={<ChartTooltip />} />
            <Bar dataKey="value" radius={[12, 12, 0, 0]}>
              {data.map((entry) => (
                <Cell key={entry.label} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function deriveOverallSentiment(labels) {
  const filtered = labels.filter(Boolean);
  if (!filtered.length) return 'Neutral';

  const counts = filtered.reduce((accumulator, label) => {
    accumulator[label] = (accumulator[label] || 0) + 1;
    return accumulator;
  }, {});

  const entries = Object.entries(counts).sort((left, right) => right[1] - left[1]);
  const [topLabel, topCount] = entries[0] || ['Neutral', 0];
  if (topCount > 1) return topLabel;
  if (filtered.includes('Positive') && !filtered.includes('Negative')) return 'Positive';
  if (filtered.includes('Negative') && !filtered.includes('Positive')) return 'Negative';
  return topLabel || 'Neutral';
}

export default function AnalyzeReview() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { shareReviewForPlayground } = useReviewWorkflow();

  const runAnalysis = async () => {
    setLoading(true);
    setError('');
    try {
      const payload = await api.analyzeReview(text);
      setResult(payload);
      shareReviewForPlayground(text);
    } catch (err) {
      console.error('[AnalyzeReview] analysis failed', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const vaderChart = useMemo(() => {
    const vader = result?.vader || {};
    return [
      { label: 'Positive', value: Number(vader.pos || 0), fill: '#2563EB' },
      { label: 'Neutral', value: Number(vader.neu || 0), fill: '#3B82F6' },
      { label: 'Negative', value: Number(vader.neg || 0), fill: '#60A5FA' },
      { label: 'Compound', value: Math.abs(Number(vader.compound || 0)), fill: '#93C5FD' },
    ];
  }, [result]);

  const robertaChart = useMemo(() => {
    const roberta = result?.roberta || {};
    return [
      { label: 'Positive', value: Number(roberta.roberta_pos || 0), fill: '#2563EB' },
      { label: 'Neutral', value: Number(roberta.roberta_neu || 0), fill: '#3B82F6' },
      { label: 'Negative', value: Number(roberta.roberta_neg || 0), fill: '#60A5FA' },
    ];
  }, [result]);

  const logisticChart = useMemo(() => {
    const logistic = result?.logistic_regression || result?.ml || {};
    return [
      { label: 'Positive', value: Number(logistic.pos || 0), fill: '#2563EB' },
      { label: 'Neutral', value: Number(logistic.neu || 0), fill: '#3B82F6' },
      { label: 'Negative', value: Number(logistic.neg || 0), fill: '#60A5FA' },
    ];
  }, [result]);

  const logisticResult = result?.logistic_regression || result?.ml || {};
  const overallSentiment = result
    ? deriveOverallSentiment([
        result.vader?.label,
        modelLabel(logisticResult),
        result.roberta?.roberta_label,
      ])
    : 'Neutral';
  const vaderConfidence = confidenceValue({
    positive: result?.vader?.pos,
    neutral: result?.vader?.neu,
    negative: result?.vader?.neg,
  });
  const logisticConfidence = confidenceValue({
    positive: logisticResult?.pos,
    neutral: logisticResult?.neu,
    negative: logisticResult?.neg,
  });
  const robertaConfidence = confidenceValue({
    positive: result?.roberta?.roberta_pos,
    neutral: result?.roberta?.roberta_neu,
    negative: result?.roberta?.roberta_neg,
  });
  const explanations = result?.explanations || {};

  return (
    <div className="space-y-8">
      <SectionTitle
        eyebrow="Review Intelligence"
        title="Analyze Review"
        description="Focused sentiment analysis with VADER, Logistic Regression, and RoBERTa outputs."
      />

      <Panel
        title="Text Input"
        subtitle="Paste one review and run sentiment analysis"
        actions={
          <button
            type="button"
            onClick={runAnalysis}
            className="inline-flex items-center justify-center rounded-2xl bg-blue-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={loading}
          >
            {loading ? 'Analyzing...' : 'Analyze Review'}
          </button>
        }
        className="overflow-hidden"
      >
        <InlineField label="Review text">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={7}
            className="min-h-[210px] w-full rounded-[16px] border border-white/10 bg-[#0A0A0A]/60 px-5 py-4 text-[15px] text-slate-100 outline-none transition placeholder:text-slate-500 focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
            placeholder="Type a review..."
          />
        </InlineField>
      </Panel>

      {error ? <EmptyState title="Analysis failed" description={error} /> : null}

      {result ? (
        <div className="space-y-6">
          <div className="grid gap-6 xl:grid-cols-3">
            <Panel title="VADER Results" subtitle="Sentiment polarity scores from the rule-based model">
              <div className="grid gap-4">
                <ConfidenceMeter
                  title="VADER"
                  label={result.vader?.label || '—'}
                  confidence={vaderConfidence}
                />
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Positive</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.vader?.pos || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Neutral</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.vader?.neu || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Negative</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.vader?.neg || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Compound</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.vader?.compound || 0).toFixed(3)}</div>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="Logistic Regression Results" subtitle="Traditional TF-IDF sentiment classifier">
              <div className="grid gap-4">
                <ConfidenceMeter
                  title="Logistic Regression"
                  label={modelLabel(logisticResult)}
                  confidence={logisticConfidence}
                />
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Positive</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(logisticResult?.pos || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Neutral</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(logisticResult?.neu || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5 md:col-span-2">
                    <div className="text-[13px] font-medium text-slate-400">Negative</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(logisticResult?.neg || 0).toFixed(3)}</div>
                  </div>
                </div>
              </div>
            </Panel>

            <Panel title="RoBERTa Results" subtitle="Model probabilities from the transformer classifier">
              <div className="grid gap-4">
                <ConfidenceMeter
                  title="RoBERTa"
                  label={result.roberta?.roberta_label || '—'}
                  confidence={robertaConfidence}
                />
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Positive</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.roberta?.roberta_pos || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
                    <div className="text-[13px] font-medium text-slate-400">Neutral</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.roberta?.roberta_neu || 0).toFixed(3)}</div>
                  </div>
                  <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5 md:col-span-2">
                    <div className="text-[13px] font-medium text-slate-400">Negative</div>
                    <div className="mt-3 text-[28px] font-semibold tracking-tight text-white">{Number(result.roberta?.roberta_neg || 0).toFixed(3)}</div>
                  </div>
                </div>
              </div>
            </Panel>
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <SentimentChart title="VADER Sentiment Chart" data={vaderChart} />
            <SentimentChart title="Logistic Regression Chart" data={logisticChart} />
            <SentimentChart title="RoBERTa Sentiment Chart" data={robertaChart} />
          </div>

          <Panel title="Overall Sentiment Conclusion" subtitle="Combined sentiment view from all three models">
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className="text-[13px] font-medium text-slate-400">Conclusion</div>
                <div className="mt-2 text-[28px] font-semibold tracking-tight text-white">{overallSentiment}</div>
              </div>
              <Badge tone={scoreTone(overallSentiment)}>{overallSentiment}</Badge>
            </div>
          </Panel>

          <Panel title="Why this prediction?" subtitle="Explainable sentiment indicators based on the review text">
            <div className="grid gap-4 xl:grid-cols-2">
              <IndicatorList
                title="Positive Indicators"
                items={explanations?.vader?.positive_indicators?.length ? explanations.vader.positive_indicators : explanations?.roberta?.positive_indicators}
              />
              <IndicatorList
                title="Negative Indicators"
                items={explanations?.vader?.negative_indicators?.length ? explanations.vader.negative_indicators : explanations?.roberta?.negative_indicators}
              />
            </div>
            <p className="mt-4 text-[15px] leading-7 text-slate-400">
              {explanations?.summary || 'No strong sentiment-bearing words detected.'}
            </p>
          </Panel>

          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => navigate('/playground')}
              className="inline-flex items-center justify-center rounded-2xl border border-white/10 bg-white/5 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/10"
            >
              Open NLP Playground
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
