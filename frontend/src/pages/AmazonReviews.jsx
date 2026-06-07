import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { SectionTitle } from '../components/UI';
import DatasetAnalysis from './DatasetAnalysis';
import ModelComparison from './ModelComparison';

export default function AmazonReviews() {
  const location = useLocation();

  useEffect(() => {
    const hash = location.hash?.replace('#', '');
    if (!hash) return;

    const element = document.getElementById(hash);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [location.hash]);

  return (
    <div className="space-y-10">
      <SectionTitle
        eyebrow="Amazon Reviews"
        title="Amazon Reviews"
        description="Dataset analysis and model comparison are combined here so the review corpus, sentiment breakdowns, and model performance live in one place."
      />

      <section id="dataset-analysis" className="scroll-mt-24 space-y-8">
        <DatasetAnalysis showHeader={false} />
      </section>

      <section id="comparison" className="scroll-mt-24 space-y-8">
        <ModelComparison showHeader={false} />
      </section>
    </div>
  );
}
