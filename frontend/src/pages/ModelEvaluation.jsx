import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../lib/api';
import { Badge, EmptyState, Panel, SectionTitle, StatCard } from '../components/UI';

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(1)}%` : '—';
}

function MetricTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-2xl border border-white/10 bg-[#050816]/95 px-4 py-3 text-sm shadow-2xl">
      <div className="font-semibold text-white">{label}</div>
      {payload.map((item) => (
        <div key={item.name} className="mt-1 text-slate-300">
          {item.name}: {Number(item.value).toFixed(1)}%
        </div>
      ))}
    </div>
  );
}

function ConfusionMatrix({ labels, matrix }) {
  const maxValue = Math.max(...matrix.flat().map((value) => Number(value) || 0), 1);

  return (
    <div className="overflow-hidden rounded-[16px] border border-white/10">
      <div className="grid grid-cols-[140px_repeat(3,minmax(0,1fr))] bg-white/[0.04] text-sm text-slate-300">
        <div className="px-4 py-4 font-medium text-slate-400">Actual \ Predicted</div>
        {labels.map((label) => (
          <div key={label} className="px-4 py-4 text-center font-medium text-white">
            {label}
          </div>
        ))}
      </div>
      {matrix.map((row, rowIndex) => (
        <div key={labels[rowIndex] || rowIndex} className="grid grid-cols-[140px_repeat(3,minmax(0,1fr))] border-t border-white/10">
          <div className="px-4 py-4 font-medium text-white">{labels[rowIndex]}</div>
          {row.map((value, columnIndex) => {
            const intensity = Math.max(0.12, Number(value) / maxValue);
            return (
              <div
                key={`${rowIndex}-${columnIndex}`}
                className="border-l border-white/10 px-4 py-4 text-center text-[15px] font-semibold text-white"
                style={{
                  backgroundColor: `rgba(37, 99, 235, ${0.12 + intensity * 0.42})`,
                }}
              >
                {value}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default function ModelEvaluation() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    api.mlEvaluation()
      .then((payload) => {
        if (!mounted) return;
        setData(payload);
      })
      .catch((err) => {
        if (!mounted) return;
        console.error('[ModelEvaluation] request failed', err);
        setError(err.message);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const metrics = data?.model_metrics || {};
  const evaluationChart = useMemo(
    () =>
      (data?.comparison_chart_data || []).map((entry) => ({
        model: entry.model,
        Accuracy: Number(entry.accuracy || 0) * 100,
        Precision: Number(entry.precision || 0) * 100,
        Recall: Number(entry.recall || 0) * 100,
        F1: Number(entry.f1 || 0) * 100,
      })),
    [data],
  );

  const report = data?.classification_report || {};
  const reportRows = ['Negative', 'Neutral', 'Positive', 'macro avg', 'weighted avg']
    .filter((label) => report[label])
    .map((label) => ({
      label,
      ...report[label],
    }));

  const summary = data?.summary || {};
  const confusion = data?.confusion_matrix || { labels: [], matrix: [] };

  return (
    <div className="space-y-8">
      <SectionTitle
        eyebrow="ML Evaluation"
        title="Model Evaluation"
        description="Custom TF-IDF + Logistic Regression evaluation with side-by-side comparison against VADER and RoBERTa."
      />

      {error ? <EmptyState title="Evaluation unavailable" description={error} /> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Accuracy" value={formatPercent(metrics.logistic_regression?.accuracy)} helper="Logistic Regression on the test split" />
        <StatCard label="Precision" value={formatPercent(metrics.logistic_regression?.precision)} tone="amber" helper="Macro precision" />
        <StatCard label="Recall" value={formatPercent(metrics.logistic_regression?.recall)} tone="violet" helper="Macro recall" />
        <StatCard label="F1 Score" value={formatPercent(metrics.logistic_regression?.f1_score)} tone="emerald" helper="Macro F1 score" />
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Dataset Size" value={summary.dataset_size ?? '—'} helper="Rows used to train and evaluate" />
        <StatCard label="Training Samples" value={summary.training_samples ?? '—'} tone="amber" helper="Stratified train split" />
        <StatCard label="Testing Samples" value={summary.testing_samples ?? '—'} tone="violet" helper="Held-out evaluation split" />
        <StatCard label="TF-IDF Features" value={summary.tfidf_features ?? '—'} tone="emerald" helper="Vocabulary used by the model" />
      </div>

      <Panel title="Performance Comparison" subtitle="Accuracy, precision, recall, and F1 across all three sentiment approaches">
        <div className="h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={evaluationChart} barCategoryGap="18%">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
              <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
              <Tooltip content={<MetricTooltip />} />
              <Legend />
              <Bar dataKey="Accuracy" fill="#2563eb" radius={[10, 10, 0, 0]} />
              <Bar dataKey="Precision" fill="#3b82f6" radius={[10, 10, 0, 0]} />
              <Bar dataKey="Recall" fill="#60a5fa" radius={[10, 10, 0, 0]} />
              <Bar dataKey="F1" fill="#93c5fd" radius={[10, 10, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Confusion Matrix" subtitle="Logistic Regression predictions against the true rating-derived labels">
          {confusion.matrix?.length ? (
            <ConfusionMatrix labels={confusion.labels || []} matrix={confusion.matrix || []} />
          ) : (
            <EmptyState title="Confusion matrix unavailable" description="The backend did not return the matrix for this evaluation run." />
          )}
        </Panel>

        <Panel title="Classification Report" subtitle="Per-class precision, recall, F1, and support">
          <div className="overflow-hidden rounded-[16px] border border-white/10">
            <table className="min-w-full text-left text-sm">
              <thead className="bg-white/[0.04] text-slate-300">
                <tr>
                  <th className="px-5 py-4">Label</th>
                  <th className="px-5 py-4">Precision</th>
                  <th className="px-5 py-4">Recall</th>
                  <th className="px-5 py-4">F1</th>
                  <th className="px-5 py-4">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/10">
                {reportRows.map((row) => (
                  <tr key={row.label} className="bg-white/[0.02]">
                    <td className="px-5 py-4 font-medium text-white">{row.label}</td>
                    <td className="px-5 py-4 text-slate-300">{formatPercent(row.precision)}</td>
                    <td className="px-5 py-4 text-slate-300">{formatPercent(row.recall)}</td>
                    <td className="px-5 py-4 text-slate-300">{formatPercent(row.f1_score)}</td>
                    <td className="px-5 py-4 text-slate-300">{row.support}</td>
                  </tr>
                ))}
                {report.accuracy != null ? (
                  <tr className="bg-white/[0.03]">
                    <td className="px-5 py-4 font-medium text-white">Accuracy</td>
                    <td className="px-5 py-4 text-slate-300">{formatPercent(report.accuracy)}</td>
                    <td className="px-5 py-4 text-slate-300">—</td>
                    <td className="px-5 py-4 text-slate-300">—</td>
                    <td className="px-5 py-4 text-slate-300">{summary.testing_samples ?? '—'}</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      <Panel title="Model Summary" subtitle="High-level facts about the custom model and the evaluation run">
        <div className="flex flex-wrap gap-3">
          <Badge tone="cyan">Logistic Regression</Badge>
          <Badge tone="slate">TF-IDF Vectorizer</Badge>
          <Badge tone="slate">{`Training: ${summary.training_samples ?? '—'}`}</Badge>
          <Badge tone="slate">{`Testing: ${summary.testing_samples ?? '—'}`}</Badge>
          <Badge tone="slate">{`Features: ${summary.tfidf_features ?? '—'}`}</Badge>
        </div>
      </Panel>
    </div>
  );
}
