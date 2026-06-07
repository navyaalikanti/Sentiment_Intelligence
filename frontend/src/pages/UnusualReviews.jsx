import React, { useEffect, useState } from 'react';
import { api } from '../lib/api';
import { Badge, EmptyState, Panel, SectionTitle, StatCard } from '../components/UI';

function ReviewCard({ item }) {
  return (
    <div className="rounded-[16px] border border-white/10 bg-white/5 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="cyan">{item.reason}</Badge>
        <Badge tone="slate">Score {item.review.Score}</Badge>
        <Badge tone="slate">Id {item.review.Id}</Badge>
      </div>
      <div className="mt-4 text-[15px] font-semibold leading-6 text-white">{item.review.Summary}</div>
      <p className="mt-3 text-sm leading-7 text-slate-300">{item.review.Text}</p>
      <div className="mt-4 flex flex-wrap gap-2">
        <Badge tone="cyan">{item.vader.label}</Badge>
        <Badge tone="slate">{`compound ${Number(item.vader.compound).toFixed(3)}`}</Badge>
        <Badge tone="slate">{item.roberta.label}</Badge>
      </div>
    </div>
  );
}

export default function UnusualReviews() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let mounted = true;
    api.unusualReviews(250, 8)
      .then((payload) => {
        if (!mounted) return;
        setData(payload);
      })
      .catch((err) => {
        if (!mounted) return;
        console.error('[UnusualReviews] request failed', err);
        setError(err.message);
      });

    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="space-y-8">
      <SectionTitle
        eyebrow="Anomaly Search"
        title="Unusual Reviews"
        description="Find reviews whose text sentiment disagrees with the star rating and inspect the model outputs side by side."
      />

      {error ? <EmptyState title="Unable to load unusual reviews" description={error} /> : null}

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard label="Positive Text, Low Rating" value={data?.counts?.positive_sentiment_low_rating ?? '—'} tone="emerald" />
        <StatCard label="Negative Text, High Rating" value={data?.counts?.negative_sentiment_high_rating ?? '—'} tone="rose" />
        <StatCard label="Total Unusual" value={data?.counts?.total_unusual ?? '—'} tone="amber" />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Panel title="Positive sentiment + low rating" subtitle="Potentially sarcastic or mis-scored reviews">
          <div className="space-y-4">
            {(data?.positive_sentiment_low_rating || []).map((item, index) => (
              <ReviewCard key={`${item.review.Id}-${index}`} item={item} />
            ))}
          </div>
        </Panel>

        <Panel title="Negative sentiment + high rating" subtitle="Reviews that sound unhappy but still received 4 or 5 stars">
          <div className="space-y-4">
            {(data?.negative_sentiment_high_rating || []).map((item, index) => (
              <ReviewCard key={`${item.review.Id}-${index}`} item={item} />
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
