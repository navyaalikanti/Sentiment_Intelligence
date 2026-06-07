import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

const STORAGE_KEY = 'sentiment-analysis.review-workflow';

const ReviewWorkflowContext = createContext(null);

function readStoredWorkflow() {
  if (typeof window === 'undefined') return { text: '', autoAnalyze: false };

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { text: '', autoAnalyze: false };
    const parsed = JSON.parse(raw);
    return {
      text: typeof parsed.text === 'string' ? parsed.text : '',
      autoAnalyze: Boolean(parsed.autoAnalyze),
    };
  } catch {
    return { text: '', autoAnalyze: false };
  }
}

export function ReviewWorkflowProvider({ children }) {
  const [workflow, setWorkflow] = useState(() => readStoredWorkflow());

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(workflow));
    } catch {
      // Ignore storage failures and keep the in-memory handoff working.
    }
  }, [workflow]);

  const value = useMemo(
    () => ({
      workflow,
      setWorkflow,
      shareReviewForPlayground(text) {
        setWorkflow({
          text: String(text || ''),
          autoAnalyze: true,
        });
      },
      clearSharedReview() {
        setWorkflow({ text: '', autoAnalyze: false });
      },
    }),
    [workflow],
  );

  return <ReviewWorkflowContext.Provider value={value}>{children}</ReviewWorkflowContext.Provider>;
}

export function useReviewWorkflow() {
  const context = useContext(ReviewWorkflowContext);
  if (!context) {
    throw new Error('useReviewWorkflow must be used within ReviewWorkflowProvider');
  }
  return context;
}
