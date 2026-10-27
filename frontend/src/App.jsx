import React, { useState } from 'react';
import RetinalStatistics from './components/RetinalStatistics';
import AccuracyVerifier from './components/AccuracyVerifier';
import HistogramChart from './components/HistogramChart';
import BatchEvaluation from './components/BatchEvaluation';
import SampleGallery from './components/SampleGallery';
import { analyzeImageFile, predictRetinopathy, SEVERITY_STAGES } from './services/imageAnalysis';
import './styles/App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('single');
  const [currentFile, setCurrentFile] = useState(null);
  const [imageStats, setImageStats] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [groundTruth, setGroundTruth] = useState(-1);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  const processFile = async (file, forcedGroundTruth = -1) => {
    setLoading(true);
    setCurrentFile(file);
    setGroundTruth(forcedGroundTruth);

    try {
      const stats = await analyzeImageFile(file);
      setImageStats(stats);

      const pred = await predictRetinopathy(file, stats);
      setPrediction(pred);
    } catch (err) {
      console.error("Failed to process image:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleSampleSelect = (file, stage) => {
    setActiveTab('single');
    processFile(file, stage);
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon">👁️</div>
          <div>
            <h1 className="brand-title">Diabetic Retinopathy Diagnostic Interface</h1>
            <p className="brand-subtitle">
              InceptionV3 Deep Learning Screening Engine &amp; Retinal Photometric Statistics Analyzer
            </p>
          </div>
        </div>
        <div className="header-badges">
          <span className="pill-badge">TensorFlow InceptionV3</span>
          <span className="pill-badge">ICDR Clinical Standard</span>
          <span className="pill-badge">React 18</span>
        </div>
      </header>

      {/* Tab Navigation */}
      <nav className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          <span>📸</span>
          <span>Single Image Diagnostic &amp; Statistics</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <span>📂</span>
          <span>Batch Upload &amp; Cohort Accuracy</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'gallery' ? 'active' : ''}`}
          onClick={() => setActiveTab('gallery')}
        >
          <span>🖼️</span>
          <span>Clinical Sample Gallery</span>
        </button>
      </nav>

      {/* TAB 1: Single Image */}
      {activeTab === 'single' && (
        <div>
          <div className="grid-2" style={{ marginBottom: '20px' }}>
            {/* Upload Area */}
            <div className="card">
              <div className="card-title">
                <span>📤</span>
                <span>Upload Retinal Fundus Image</span>
              </div>
              <div
                className={`dropzone ${dragOver ? 'dragover' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById('single-file-input').click()}
              >
                <input
                  id="single-file-input"
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={handleFileInput}
                />
                <div className="dropzone-icon">📷</div>
                <div style={{ fontWeight: 600, marginBottom: '4px' }}>
                  Click to browse or drag &amp; drop fundus photograph
                </div>
                <div style={{ fontSize: '0.8rem', color: '#9ca3af' }}>
                  Supports PNG, JPG, JPEG, TIFF, BMP (High resolution recommended)
                </div>
              </div>
              {currentFile && (
                <div style={{ marginTop: '12px', fontSize: '0.85rem', color: '#9ca3af', display: 'flex', justifyContent: 'space-between' }}>
                  <span>Loaded: <strong>{currentFile.name}</strong></span>
                  <span>{imageStats?.file?.size}</span>
                </div>
              )}
            </div>

            {/* Accuracy Verifier against Ground Truth */}
            <AccuracyVerifier
              predictedClass={prediction?.predicted_class ?? 0}
              groundTruth={groundTruth}
              onGroundTruthChange={setGroundTruth}
            />
          </div>

          {loading && (
            <div style={{ textAlign: 'center', padding: '40px', color: '#06b6d4' }}>
              <div style={{ fontSize: '2rem' }}>⚡</div>
              <div>Computing InceptionV3 predictions and photometric statistics...</div>
            </div>
          )}

          {prediction && imageStats && !loading && (
            <div>
              {/* Diagnostic Banner */}
              <div
                className="diag-banner"
                style={{
                  backgroundColor: `${SEVERITY_STAGES[prediction.predicted_class].color}15`,
                  borderColor: SEVERITY_STAGES[prediction.predicted_class].color,
                }}
              >
                <div>
                  <div className="diag-title" style={{ color: SEVERITY_STAGES[prediction.predicted_class].color }}>
                    <span>{prediction.predicted_class === 0 ? '🟢' : prediction.predicted_class === 1 ? '🟡' : prediction.predicted_class === 2 ? '🟠' : prediction.predicted_class === 3 ? '🔴' : '🚨'}</span>
                    <span>Predicted Stage {prediction.predicted_class}: {prediction.stage_name}</span>
                  </div>
                  <div className="diag-desc">
                    <strong>Clinical Severity:</strong> {prediction.severity} | <strong>Recommendation:</strong> {prediction.clinical_action}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: '#9ca3af', marginTop: '4px' }}>
                    <strong>Key Findings:</strong> {prediction.findings}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '2rem', fontWeight: 800, fontFamily: 'var(--font-mono)', color: SEVERITY_STAGES[prediction.predicted_class].color }}>
                    {Math.round(prediction.confidence * 100)}%
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>Diagnostic Confidence</div>
                </div>
              </div>

              {/* Main Content Layout */}
              <div className="grid-2" style={{ marginBottom: '20px' }}>
                {/* Visual Views */}
                <div className="card">
                  <div className="card-title">
                    <span>🖼️</span>
                    <span>Fundus Visual Inspection</span>
                  </div>
                  <div className="preview-container">
                    <img src={imageStats.dataUrl} alt="Retinal Fundus" />
                  </div>
                  <div style={{ marginTop: '10px', fontSize: '0.8rem', color: '#9ca3af', display: 'flex', justifyContent: 'space-between' }}>
                    <span>FOV Coverage: {imageStats.photometry.coveragePct}%</span>
                    <span>Sharpness: {imageStats.sharpness.score} ({imageStats.sharpness.quality})</span>
                  </div>
                </div>

                {/* Probability Distribution */}
                <div className="card">
                  <div className="card-title">
                    <span>📈</span>
                    <span>ICDR 5-Class Probability Distribution</span>
                  </div>
                  <div style={{ paddingTop: '8px' }}>
                    {prediction.probabilities.map((prob, idx) => {
                      const info = SEVERITY_STAGES[idx];
                      const pct = Math.round(prob * 100);
                      const isPredicted = idx === prediction.predicted_class;
                      return (
                        <div key={idx} className="prob-row">
                          <div className="prob-header">
                            <span style={{ color: isPredicted ? info.color : '#d1d5db', fontWeight: isPredicted ? 700 : 500 }}>
                              Stage {idx} - {info.name} {isPredicted && '⭐ (Predicted)'}
                            </span>
                            <span style={{ fontFamily: 'var(--font-mono)' }}>{pct}%</span>
                          </div>
                          <div className="prob-bar-bg">
                            <div
                              className="prob-bar-fill"
                              style={{ width: `${pct}%`, backgroundColor: info.color }}
                            ></div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Technical Statistics and Histograms */}
              <div className="grid-2">
                <RetinalStatistics stats={imageStats} />
                <HistogramChart histograms={imageStats.histograms} channels={imageStats.channels} />
              </div>
            </div>
          )}

          {!prediction && !loading && (
            <div style={{ marginTop: '20px' }}>
              <SampleGallery onSelectSample={handleSampleSelect} />
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Batch Upload */}
      {activeTab === 'batch' && <BatchEvaluation />}

      {/* TAB 3: Preloaded Gallery */}
      {activeTab === 'gallery' && (
        <SampleGallery onSelectSample={handleSampleSelect} />
      )}
    </div>
  );
}
