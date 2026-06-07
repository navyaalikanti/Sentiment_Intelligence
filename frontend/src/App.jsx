import React from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import AnalyzeReview from './pages/AnalyzeReview';
import AmazonReviews from './pages/AmazonReviews';
import Dashboard from './pages/Dashboard';
import DatasetAnalysis from './pages/DatasetAnalysis';
import ModelComparison from './pages/ModelComparison';
import ModelEvaluation from './pages/ModelEvaluation';
import NLPPlayground from './pages/NLPPlayground';
import UnusualReviews from './pages/UnusualReviews';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="/analyze" element={<AnalyzeReview />} />
        <Route path="/amazon-reviews" element={<AmazonReviews />} />
        <Route path="/dataset" element={<DatasetAnalysis />} />
        <Route path="/comparison" element={<ModelComparison />} />
        <Route path="/model-evaluation" element={<ModelEvaluation />} />
        <Route path="/unusual" element={<UnusualReviews />} />
        <Route path="/playground" element={<NLPPlayground />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
