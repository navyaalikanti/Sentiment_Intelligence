import React, { useEffect, useState } from 'react';
import {
  Bar,
  BarChart,
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../lib/api';
import { EmptyState, Panel, SectionTitle, StatCard } from '../components/UI';

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-2xl border border-white/10 bg-[#0A0A0A]/95 px-4 py-3 text-sm shadow-2xl">
      <div className="font-semibold text-white">{label ?? payload[0].payload.score}</div>
      {payload.map((item) => (
        <div key={item.name} className="mt-1 text-slate-300">
          {item.name}: {Number(item.value).toFixed(3)}
        </div>
      ))}
    </div>
  );
}

function formatMetric(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(1) : '—';
}

function formatPercent(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}%` : '—';
}

function buildPerformanceData(evaluation) {
  return (evaluation?.comparison_chart_data || []).map((entry) => ({
    model: entry.model,
    Accuracy: Number(entry.accuracy || 0) * 100,
    Precision: Number(entry.precision || 0) * 100,
    Recall: Number(entry.recall || 0) * 100,
    F1: Number(entry.f1 || 0) * 100,
  }));
}

export default function ModelComparison({ showHeader = true } = {}) {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    api.modelComparison(150)
      .then((payload) => {
        if (!mounted) return;
        setData(payload);
      })
      .catch((err) => {
        if (!mounted) return;
        console.error('[ModelComparison] model comparison request failed', err);
        setError(err.message);
      });

    return () => {
      mounted = false;
    };
  }, []);

  const byRating = data?.by_rating || [];
  const agreement = data?.agreement || {};
  const pairwiseAgreement = agreement.pairwise || {};
  const evaluation = data?.evaluation || {};
  const performanceData = buildPerformanceData(evaluation);
  const metrics = evaluation.model_metrics || {};
  const agreementRate = Number.isFinite(Number(agreement.agreement_percentage))
    ? Number(agreement.agreement_percentage)
    : agreement.agree != null && agreement.disagree != null && agreement.agree + agreement.disagree > 0
      ? (agreement.agree / (agreement.agree + agreement.disagree)) * 100
      : 0;
  const allThreeAgreement = Number.isFinite(Number(agreement.all_three_agree_percentage))
    ? Number(agreement.all_three_agree_percentage)
    : 0;

  return (
    <div className="space-y-8">
      {showHeader ? (
        <SectionTitle
          eyebrow="Model Check"
          title="Model Comparison"
          description="Compare Logistic Regression, VADER, and RoBERTa across rating buckets and inspect agreement, disagreement, and the pairplot summary."
        />
      ) : null}

      {error ? <EmptyState title="Comparison unavailable" description={error} /> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="VADER vs RoBERTa" value={formatPercent(agreementRate)} helper="Normalized label match rate" />
        <StatCard label="VADER vs ML" value={formatPercent(pairwiseAgreement.vader_ml?.agreement_percentage)} tone="amber" helper="Pairwise agreement rate" />
        <StatCard label="RoBERTa vs ML" value={formatPercent(pairwiseAgreement.roberta_ml?.agreement_percentage)} tone="violet" helper="Pairwise agreement rate" />
        <StatCard label="All Three" value={formatPercent(allThreeAgreement)} tone="emerald" helper="Three-way agreement" />
      </div>

      <Panel title="Agreement Summary" subtitle="Normalized label comparison between VADER, Logistic Regression, and RoBERTa">
        <div className="space-y-4">
          <div className="flex items-end justify-between gap-4">
            <div>
              <div className="text-[13px] font-medium text-slate-400">Agreement rate</div>
              <div className="mt-2 text-[2rem] font-semibold tracking-tight text-white">{agreementRate.toFixed(1)}%</div>
            </div>
            <div className="text-right text-sm text-slate-400">
              <div>{agreement.agree ?? 0} agreement</div>
              <div>{agreement.disagree ?? 0} disagreement</div>
            </div>
          </div>
          <div className="h-3 overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full rounded-full bg-blue-500"
              style={{ width: `${Math.max(0, Math.min(100, agreementRate))}%` }}
            />
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <div className="rounded-[14px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-300">
              VADER vs ML: <span className="font-semibold text-white">{formatPercent(pairwiseAgreement.vader_ml?.agreement_percentage)}</span>
            </div>
            <div className="rounded-[14px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-300">
              RoBERTa vs ML: <span className="font-semibold text-white">{formatPercent(pairwiseAgreement.roberta_ml?.agreement_percentage)}</span>
            </div>
            <div className="rounded-[14px] border border-white/10 bg-white/[0.04] px-4 py-3 text-sm text-slate-300">
              All Three: <span className="font-semibold text-white">{formatPercent(allThreeAgreement)}</span>
            </div>
          </div>
        </div>
      </Panel>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Model Trends by Rating" subtitle="Average sentiment signals for each star bucket">
          <div className="h-[380px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={byRating}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                <XAxis dataKey="score" stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <Tooltip content={<TooltipBox />} />
                <Legend />
                <Line type="monotone" dataKey="vader_compound" stroke="#2563eb" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="roberta_pos" stroke="#3b82f6" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="roberta_neg" stroke="#60a5fa" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="ml_pos" stroke="#93c5fd" strokeWidth={3} dot={false} />
                <Line type="monotone" dataKey="ml_neg" stroke="#1d4ed8" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Sentiment Balance" subtitle="Average positive and negative signals from all three models">
          <div className="h-[380px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={byRating}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                <XAxis dataKey="score" stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                <Tooltip content={<TooltipBox />} />
                <Legend />
                <Area type="monotone" dataKey="vader_pos" stroke="#2563eb" fill="#2563eb22" />
                <Area type="monotone" dataKey="vader_neg" stroke="#3b82f6" fill="#3b82f622" />
                <Area type="monotone" dataKey="roberta_pos" stroke="#60a5fa" fill="#60a5fa22" />
                <Area type="monotone" dataKey="roberta_neg" stroke="#93c5fd" fill="#93c5fd22" />
                <Area type="monotone" dataKey="ml_pos" stroke="#1d4ed8" fill="#1d4ed822" />
                <Area type="monotone" dataKey="ml_neg" stroke="#1e3a8a" fill="#1e3a8a22" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>

      <Panel title="Performance Comparison" subtitle="Evaluation metrics on the held-out test split">
        {performanceData.length ? (
          <div className="space-y-6">
            <div className="h-[360px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={performanceData} barCategoryGap="18%">
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.14)" />
                  <XAxis dataKey="model" stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Accuracy" fill="#2563eb" radius={[10, 10, 0, 0]} />
                  <Bar dataKey="Precision" fill="#3b82f6" radius={[10, 10, 0, 0]} />
                  <Bar dataKey="Recall" fill="#60a5fa" radius={[10, 10, 0, 0]} />
                  <Bar dataKey="F1" fill="#93c5fd" radius={[10, 10, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="overflow-hidden rounded-[16px] border border-white/10">
              <table className="min-w-full text-left text-sm">
                <thead className="bg-white/[0.04] text-slate-300">
                  <tr>
                    <th className="px-5 py-4">Model</th>
                    <th className="px-5 py-4">Accuracy</th>
                    <th className="px-5 py-4">Precision</th>
                    <th className="px-5 py-4">Recall</th>
                    <th className="px-5 py-4">F1 Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {performanceData.map((row) => (
                    <tr key={row.model} className="bg-white/[0.02]">
                      <td className="px-5 py-4 font-medium text-white">{row.model}</td>
                      <td className="px-5 py-4 text-slate-300">{formatMetric(row.Accuracy)}%</td>
                      <td className="px-5 py-4 text-slate-300">{formatMetric(row.Precision)}%</td>
                      <td className="px-5 py-4 text-slate-300">{formatMetric(row.Recall)}%</td>
                      <td className="px-5 py-4 text-slate-300">{formatMetric(row.F1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <EmptyState title="Performance data unavailable" description="The backend did not return evaluation metrics for the custom model." />
        )}
      </Panel>

      <Panel title="Pairplot Snapshot" subtitle="Dense model relationships exported from the backend">
        {data?.pairplot_base64 ? (
          <div className="overflow-hidden rounded-[16px] border border-white/10 bg-black">
            <img
              src={`data:image/png;base64,${data.pairplot_base64}`}
              alt="Pairplot of VADER and RoBERTa scores"
              className="w-full object-contain"
            />
          </div>
        ) : (
          <EmptyState title="Pairplot unavailable" description="The backend did not return a pairplot image for this sample." />
        )}
      </Panel>
    </div>
  );
}
