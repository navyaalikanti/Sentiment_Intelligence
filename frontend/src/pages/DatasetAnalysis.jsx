import React, { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api } from '../lib/api';
import { Badge, EmptyState, InlineField, Panel, SectionTitle, StatCard } from '../components/UI';
import WordCloud from '../components/WordCloud';

function TooltipBox({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-2xl border border-white/10 bg-[#0A0A0A]/95 px-4 py-3 text-sm shadow-2xl">
      <div className="font-semibold text-white">{label ?? payload[0].payload.score}</div>
      <div className="mt-1 text-slate-300">{payload[0].value}</div>
    </div>
  );
}

export default function DatasetAnalysis({ showHeader = true } = {}) {
  const [stats, setStats] = useState(null);
  const [ratings, setRatings] = useState([]);
  const [sentiments, setSentiments] = useState({ vader: [], roberta: [] });
  const [preview, setPreview] = useState([]);
  const [wordClouds, setWordClouds] = useState({ positive: [], negative: [] });
  const [searchQuery, setSearchQuery] = useState('');
  const [ratingFilter, setRatingFilter] = useState('all');
  const [sentimentFilter, setSentimentFilter] = useState('all');
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    Promise.allSettled([
      api.datasetStatistics(5000),
      api.ratingDistribution(5000),
      api.sentimentDistribution(150),
      api.datasetPreview(400, true),
      api.wordClouds(3000),
    ]).then((results) => {
      if (!mounted) return;

      const errors = [];
      const [statsResult, ratingResult, sentimentResult, previewResult, wordCloudResult] = results;

      if (statsResult.status === 'fulfilled') {
        setStats(statsResult.value);
      } else {
        console.error('[DatasetAnalysis] dataset statistics request failed', statsResult.reason);
        errors.push(`dataset statistics: ${statsResult.reason.message || statsResult.reason}`);
      }

      if (ratingResult.status === 'fulfilled') {
        setRatings(ratingResult.value.data || []);
      } else {
        console.error('[DatasetAnalysis] rating distribution request failed', ratingResult.reason);
        errors.push(`rating distribution: ${ratingResult.reason.message || ratingResult.reason}`);
      }

      if (sentimentResult.status === 'fulfilled') {
        setSentiments(sentimentResult.value.data || { vader: [], roberta: [] });
      } else {
        console.error('[DatasetAnalysis] sentiment distribution request failed', sentimentResult.reason);
        errors.push(`sentiment distribution: ${sentimentResult.reason.message || sentimentResult.reason}`);
      }

      if (previewResult.status === 'fulfilled') {
        setPreview(previewResult.value.rows || []);
      } else {
        console.error('[DatasetAnalysis] preview request failed', previewResult.reason);
        errors.push(`dataset preview: ${previewResult.reason.message || previewResult.reason}`);
      }

      if (wordCloudResult.status === 'fulfilled') {
        setWordClouds(wordCloudResult.value.data || { positive: [], negative: [] });
      } else {
        console.error('[DatasetAnalysis] word clouds request failed', wordCloudResult.reason);
        errors.push(`word clouds: ${wordCloudResult.reason.message || wordCloudResult.reason}`);
      }

      setError(errors.length ? errors.join(' | ') : '');
    });

    return () => {
      mounted = false;
    };
  }, []);

  const filteredPreview = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return (preview || []).filter((row) => {
      const summary = String(row.Summary || '').toLowerCase();
      const text = String(row.Text || '').toLowerCase();
      const rowRating = String(row.Score || '');
      const rowSentiment = String(row.sentiment || row.vader_label || '').toLowerCase();

      const matchesQuery = !query || summary.includes(query) || text.includes(query);
      const matchesRating = ratingFilter === 'all' || rowRating === ratingFilter;
      const matchesSentiment =
        sentimentFilter === 'all' ||
        rowSentiment === sentimentFilter.toLowerCase();

      return matchesQuery && matchesRating && matchesSentiment;
    });
  }, [preview, ratingFilter, searchQuery, sentimentFilter]);

  return (
    <div className="space-y-8">
      {showHeader ? (
        <SectionTitle
          eyebrow="Dataset"
          title="Dataset Analysis"
          description="Explore the review corpus, rating spread, sentiment balance, and a clean preview of the raw rows."
        />
      ) : null}

      {error ? <EmptyState title="Could not load dataset" description={error} /> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Rows" value={stats?.rows ?? '—'} helper="Records currently loaded" />
        <StatCard label="Average Length" value={stats?.average_review_length?.toFixed?.(1) ?? '—'} tone="emerald" helper="Words per review" />
        <StatCard label="Median Length" value={stats?.median_review_length?.toFixed?.(1) ?? '—'} tone="violet" helper="More robust than average" />
        <StatCard label="Average Score" value={stats?.average_score?.toFixed?.(2) ?? '—'} tone="amber" helper="Star-rating mean" />
      </div>

      <Panel title="Rating Distribution" subtitle="Frequency of 1 to 5 star reviews">
        <div className="h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={ratings}>
              <XAxis dataKey="score" stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
              <Tooltip content={<TooltipBox />} />
              <Bar dataKey="count" radius={[12, 12, 0, 0]}>
                {ratings.map((entry, index) => (
                  <Cell key={entry.score} fill={['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'][index]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Panel>

      <Panel title="Sentiment Distribution" subtitle="Positive, neutral, and negative counts from the model analysis">
        <div className="grid gap-6 md:grid-cols-2">
          {[
            { title: 'VADER', data: sentiments.vader || [] },
            { title: 'RoBERTa', data: sentiments.roberta || [] },
          ].map((item) => (
            <div key={item.title} className="rounded-[16px] border border-white/10 bg-white/5 p-5">
              <div className="mb-3 flex items-center justify-between">
                <div className="text-[15px] font-semibold text-white">{item.title}</div>
                <Badge tone="slate">{item.data.reduce((sum, entry) => sum + entry.count, 0)} total</Badge>
              </div>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={item.data}
                      dataKey="count"
                      nameKey="label"
                      innerRadius={50}
                      outerRadius={80}
                      paddingAngle={3}
                    >
                      {item.data.map((entry, index) => (
                        <Cell
                          key={`${item.title}-${entry.code || entry.label}`}
                          fill={['#1d4ed8', '#2563eb', '#60a5fa'][index % 3]}
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<TooltipBox />} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </Panel>

      <Panel title="NLP Word Clouds" subtitle="Most frequent words from positive and negative reviews">
        <div className="grid gap-6 xl:grid-cols-2">
          <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-3 text-[18px] font-semibold text-white">Positive Review Word Cloud</div>
            <WordCloud words={wordClouds.positive || []} emptyLabel="No positive word cloud data available." />
          </div>
          <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5">
            <div className="mb-3 text-[18px] font-semibold text-white">Negative Review Word Cloud</div>
            <WordCloud words={wordClouds.negative || []} emptyLabel="No negative word cloud data available." />
          </div>
        </div>
      </Panel>

      <Panel title="Preview Filters" subtitle="Search and filter the review rows without refreshing the page">
        <div className="grid gap-4 xl:grid-cols-3">
          <InlineField label="Search Review Text">
            <input
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="w-full rounded-[16px] border border-white/10 bg-[#0A0A0A]/60 px-4 py-3 text-[15px] text-white outline-none transition placeholder:text-slate-500 focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
              placeholder="Search summary or review text"
            />
          </InlineField>

          <InlineField label="Filter by Rating">
            <select
              value={ratingFilter}
              onChange={(event) => setRatingFilter(event.target.value)}
              className="w-full rounded-[16px] border border-white/10 bg-[#0A0A0A]/60 px-4 py-3 text-[15px] text-white outline-none transition focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
            >
              <option value="all">All</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
            </select>
          </InlineField>

          <InlineField label="Filter by Sentiment">
            <select
              value={sentimentFilter}
              onChange={(event) => setSentimentFilter(event.target.value)}
              className="w-full rounded-[16px] border border-white/10 bg-[#0A0A0A]/60 px-4 py-3 text-[15px] text-white outline-none transition focus:border-blue-500/40 focus:ring-2 focus:ring-blue-500/10"
            >
              <option value="all">All</option>
              <option value="positive">Positive</option>
              <option value="neutral">Neutral</option>
              <option value="negative">Negative</option>
            </select>
          </InlineField>
        </div>
      </Panel>

      <Panel title="Preview Rows" subtitle="Sample of the loaded CSV records">
        <div className="space-y-4">
          <div className="hidden overflow-hidden rounded-[16px] border border-white/10 md:block">
            <div className="max-h-[560px] overflow-auto scrollbar-thin">
              <table className="min-w-full table-fixed text-left text-sm">
                <thead className="sticky top-0 z-10 bg-[#0A0A0A]/95 text-slate-300 backdrop-blur">
                  <tr>
                    <th className="px-5 py-4">Id</th>
                    <th className="px-5 py-4">Score</th>
                    <th className="px-5 py-4">Summary</th>
                    <th className="px-5 py-4">Text</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/10">
                  {filteredPreview.length ? (
                    filteredPreview.map((row, index) => (
                      <tr key={row.Id} className={index % 2 === 0 ? 'bg-white/[0.03] align-top' : 'bg-white/[0.015] align-top'}>
                        <td className="px-5 py-4 font-semibold text-white">{row.Id}</td>
                        <td className="px-5 py-4">
                          <Badge tone={row.Score >= 4 ? 'emerald' : row.Score <= 2 ? 'rose' : 'amber'}>{row.Score}</Badge>
                        </td>
                        <td className="px-5 py-4 text-slate-200">{row.Summary}</td>
                        <td className="px-5 py-4 text-slate-400">
                          {String(row.Text || '').slice(0, 220)}
                          {String(row.Text || '').length > 220 ? '...' : ''}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan="4" className="px-5 py-8 text-center text-sm text-slate-400">
                        No reviews match the current search or filter settings.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="grid gap-4 md:hidden">
            {filteredPreview.length ? (
              filteredPreview.map((row) => (
                <div key={row.Id} className="rounded-[16px] border border-white/10 bg-white/5 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-white">Review {row.Id}</div>
                    <Badge tone={row.Score >= 4 ? 'emerald' : row.Score <= 2 ? 'rose' : 'amber'}>{row.Score}</Badge>
                  </div>
                  <div className="mt-3 text-sm font-medium text-slate-200">{row.Summary}</div>
                  <p className="mt-3 text-sm leading-6 text-slate-400">
                    {String(row.Text || '').slice(0, 180)}
                    {String(row.Text || '').length > 180 ? '...' : ''}
                  </p>
                </div>
              ))
            ) : (
              <div className="rounded-[16px] border border-white/10 bg-white/[0.04] p-5 text-sm text-slate-400">
                No reviews match the current search or filter settings.
              </div>
            )}
          </div>
        </div>
      </Panel>
    </div>
  );
}
