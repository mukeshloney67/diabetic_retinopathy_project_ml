import React from 'react';
import { evaluateAccuracy, SEVERITY_STAGES } from '../services/imageAnalysis';

export default function AccuracyVerifier({ predictedClass, groundTruth, onGroundTruthChange }) {
  const evaluation = evaluateAccuracy(predictedClass, groundTruth);

  return (
    <div className="card" style={{ height: '100%' }}>
      <div className="card-title">
        <span>🎯</span>
        <span>Ground-Truth Diagnostic Accuracy Check</span>
      </div>

      <div style={{ marginBottom: '14px' }}>
        <label style={{ display: 'block', fontSize: '0.85rem', color: '#9ca3af', marginBottom: '6px' }}>
          Select Known Clinical Diagnosis (Ground Truth):
        </label>
        <select
          value={groundTruth}
          onChange={(e) => onGroundTruthChange(Number(e.target.value))}
          style={{
            width: '100%',
            padding: '10px 12px',
            borderRadius: '8px',
            backgroundColor: '#1f2937',
            color: '#f9fafb',
            border: '1px solid #374151',
            fontSize: '0.9rem',
            outline: 'none',
          }}
        >
          <option value={-1}>-- Unlabeled / Evaluation Only --</option>
          <option value={0}>Stage 0 - No DR (Healthy Retina)</option>
          <option value={1}>Stage 1 - Mild NPDR (Microaneurysms)</option>
          <option value={2}>Stage 2 - Moderate NPDR (Hard Exudates/Hemorrhages)</option>
          <option value={3}>Stage 3 - Severe NPDR (4-2-1 Criteria)</option>
          <option value={4}>Stage 4 - Proliferative DR (Neovascularization)</option>
        </select>
      </div>

      {evaluation ? (
        <div
          className="acc-panel"
          style={{
            backgroundColor: `${evaluation.badgeColor}15`,
            borderColor: evaluation.badgeColor,
            color: '#fff',
          }}
        >
          <div>
            <div style={{ fontWeight: 700, fontSize: '1.05rem', color: evaluation.badgeColor }}>
              {evaluation.statusText}
            </div>
            <div style={{ fontSize: '0.82rem', color: '#d1d5db', marginTop: '4px' }}>
              Predicted: <strong>Stage {evaluation.predicted} ({SEVERITY_STAGES[evaluation.predicted].name})</strong> | Ground Truth: <strong>Stage {evaluation.groundTruth}</strong>
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '1.3rem', fontWeight: 800, color: evaluation.badgeColor }}>
              {evaluation.quadraticScore}%
            </div>
            <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>Quadratic Score</div>
          </div>
        </div>
      ) : (
        <div style={{ padding: '16px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', textAlign: 'center', color: '#9ca3af', fontSize: '0.85rem' }}>
          Select a verified diagnosis above to test exact match accuracy and quadratic penalty scores.
        </div>
      )}
    </div>
  );
}
