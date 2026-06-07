const API_BASE = import.meta.env.VITE_API_URL || '/api';

async function requestJSON(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
      },
      ...options,
    });
  } catch (error) {
    console.error(`[api] ${path} network error`, error);
    throw error;
  }

  const rawBody = await response.text();
  let payload = {};

  if (rawBody) {
    try {
      payload = JSON.parse(rawBody);
    } catch (error) {
      console.error(`[api] ${path} returned non-JSON`, rawBody);
      throw new Error(`Invalid JSON response from ${path}`);
    }
  }

  if (!response.ok) {
    const message = payload.error || `Request failed: ${response.status}`;
    console.error(`[api] ${path} failed`, { status: response.status, payload });
    throw new Error(message);
  }

  return payload;
}

export const api = {
  health: () => requestJSON('/health'),
  analyzeReview: (review) =>
    requestJSON('/analyze-review', {
      method: 'POST',
      body: JSON.stringify({ review, text: review }),
    }),
  nlpAnalysis: (review) =>
    requestJSON('/nlp-analysis', {
      method: 'POST',
      body: JSON.stringify({ review, text: review }),
    }),
  datasetStatistics: (limit = 5000) => requestJSON(`/dataset-statistics?limit=${limit}`),
  ratingDistribution: (limit = 5000) => requestJSON(`/rating-distribution?limit=${limit}`),
  sentimentDistribution: (limit = 150) => requestJSON(`/sentiment-distribution?limit=${limit}`),
  modelComparison: (limit = 150) => requestJSON(`/model-comparison?limit=${limit}`),
  mlEvaluation: () => requestJSON('/ml-evaluation'),
  unusualReviews: (limit = 250, topN = 8) =>
    requestJSON(`/unusual-reviews?limit=${limit}&top_n=${topN}`),
  datasetPreview: (limit = 10, withSentiment = false) =>
    requestJSON(`/dataset-preview?limit=${limit}&with_sentiment=${withSentiment ? '1' : '0'}`),
  wordClouds: (limit = 5000) => requestJSON(`/word-clouds?limit=${limit}`),
};
