import React, { useState } from 'react';
import { analyzeImageFile, predictRetinopathy, SEVERITY_STAGES } from '../services/imageAnalysis';

export default function BatchEvaluation() {
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleBatchUpload = async (e) => {
    const selected = Array.from(e.target.files);
    if (!selected.length) return;

    setFiles(selected);
    setLoading(true);
    setProgress(0);
    setResults([]);

    const batchRes = [];
    for (let i = 0; i < selected.length; i++) {
      const f = selected[i];
      try {
        const stats = await analyzeImageFile(f);
        const pred = await predictRetinopathy(f, stats);

        // Auto-detect ground truth from filename if present (e.g. stage0, 0_xxx, class3)
        let detectedGt = null;
        const match = f.name.match(/(?:stage|class|_)?([0-4])/i);
        if (match) {
          detectedGt = parseInt(match[1], 10);
        }

        const isMatch = detectedGt !== null ? pred.predicted_class === detectedGt : null;

        batchRes.push({
          name: f.name,
          predictedClass: pred.predicted_class,
          stageName: pred.stage_name,
          confidence: Math.round(pred.confidence * 100),
          groundTruth: detectedGt,
          isMatch,
          sharpness: stats.sharpness.score,
          quality: stats.sharpness.quality,
          coverage: stats.photometry.coveragePct,
        });
      } catch (err) {
        console.error("Batch item error", err);
      }
      setProgress(Math.round(((i + 1) / selected.length) * 100));
    }

    setResults(batchRes);
    setLoading(false);
  };

  const labeledItems = results.filter((r) => r.isMatch !== null);
  const correctCount = labeledItems.filter((r) => r.isMatch).length;
  const batchAccuracy = labeledItems.length ? Math.round((correctCount / labeledItems.length) * 100) : null;

  const downloadCSV = () => {
    const headers = "Filename,Predicted Stage,Confidence %,Ground Truth,Accuracy Match,Sharpness,Quality,Retinal Coverage %\n";
    const rows = results.map(r =>
      `"${r.name}","${r.stageName}",${r.confidence},"${r.groundTruth !== null ? 'Stage ' + r.groundTruth : 'Unlabeled'}","${r.isMatch === true ? 'Correct' : r.isMatch === false ? 'Mismatch' : '—'}",${r.sharpness},"${r.quality}",${r.coverage}%`
    ).join("\n");

    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "retinopathy_batch_evaluation.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div>
      <div className="card" style={{ marginBottom: '20px' }}>
        <div className="card-title">
          <span>📂</span>
          <span>Upload Cohort Dataset for Batch Accuracy Testing</span>
        </div>
        <p style={{ fontSize: '0.88rem', color: '#9ca3af', marginBottom: '16px' }}>
          Select multiple retinal fundus images simultaneously. The engine will evaluate each image, extract sharpness and coverage statistics, and calculate overall cohort accuracy.
        </p>

        <input
          type="file"
          multiple
          accept="image/*"
          id="batch-input"
          style={{ display: 'none' }}
          onChange={handleBatchUpload}
        />
        <label htmlFor="batch-input" className="btn-primary" style={{ cursor: 'pointer' }}>
          <span>📤</span>
          <span>Choose Files for Batch Evaluation</span>
        </label>

        {loading && (
          <div style={{ marginTop: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
              <span>Processing cohort...</span>
              <span>{progress}%</span>
            </div>
            <div className="prob-bar-bg">
              <div className="prob-bar-fill" style={{ width: `${progress}%`, background: '#06b6d4' }}></div>
            </div>
          </div>
        )}
      </div>

      {results.length > 0 && (
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div className="card-title" style={{ margin: 0 }}>
              <span>📊</span>
              <span>Cohort Evaluation Results</span>
            </div>
            <button className="btn-primary" onClick={downloadCSV} style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
              📥 Export CSV
            </button>
          </div>

          <div className="grid-4" style={{ marginBottom: '18px' }}>
            <div className="metric-box">
              <div className="metric-label">Processed Images</div>
              <div className="metric-value">{results.length}</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Cohort Accuracy</div>
              <div className="metric-value" style={{ color: batchAccuracy >= 80 ? '#34d399' : '#fbbf24' }}>
                {batchAccuracy !== null ? `${batchAccuracy}%` : 'N/A'}
              </div>
              <div className="metric-sub">{correctCount}/{labeledItems.length} Verified Correct</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Average Sharpness</div>
              <div className="metric-value">
                {Math.round(results.reduce((a, b) => a + b.sharpness, 0) / results.length)}
              </div>
              <div className="metric-sub">Laplacian Index</div>
            </div>
            <div className="metric-box">
              <div className="metric-label">Gradable %</div>
              <div className="metric-value">
                {Math.round((results.filter(r => r.quality !== 'Degraded / Blurry').length / results.length) * 100)}%
              </div>
            </div>
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Filename</th>
                  <th>Predicted Stage</th>
                  <th>Confidence</th>
                  <th>Ground Truth</th>
                  <th>Match</th>
                  <th>Sharpness</th>
                  <th>Retinal Coverage</th>
                </tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i}>
                    <td><code>{r.name}</code></td>
                    <td><strong style={{ color: SEVERITY_STAGES[r.predictedClass].color }}>{r.stageName}</strong></td>
                    <td>{r.confidence}%</td>
                    <td>{r.groundTruth !== null ? `Stage ${r.groundTruth}` : '—'}</td>
                    <td>
                      {r.isMatch === true ? (
                        <span style={{ color: '#34d399', fontWeight: 600 }}>✅ Correct</span>
                      ) : r.isMatch === false ? (
                        <span style={{ color: '#f87171', fontWeight: 600 }}>❌ Mismatch</span>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td>{r.sharpness} ({r.quality})</td>
                    <td>{r.coverage}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
