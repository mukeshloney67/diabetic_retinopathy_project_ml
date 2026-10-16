import React from 'react';

export default function RetinalStatistics({ stats }) {
  if (!stats) return null;

  const { dimensions, sharpness, photometry, gradability } = stats;

  return (
    <div className="card">
      <div className="card-title">
        <span>📊</span>
        <span>Photometric & Retinal Quality Metrics</span>
      </div>

      <div className="grid-4" style={{ marginBottom: '16px' }}>
        <div className="metric-box">
          <div className="metric-label">Sharpness Index</div>
          <div className="metric-value">{sharpness.score}</div>
          <div className="metric-sub" style={{ color: sharpness.badge === 'success' ? '#34d399' : sharpness.badge === 'warning' ? '#fbbf24' : '#f87171' }}>
            ● {sharpness.quality}
          </div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Retinal Coverage</div>
          <div className="metric-value">{photometry.coveragePct}%</div>
          <div className="metric-sub">Field of View</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">Mean Luminance</div>
          <div className="metric-value">{photometry.meanBrightness}</div>
          <div className="metric-sub">0 - 255 scale</div>
        </div>

        <div className="metric-box">
          <div className="metric-label">RMS Contrast</div>
          <div className="metric-value">{photometry.contrastRMS}</div>
          <div className="metric-sub">Standard Deviation</div>
        </div>
      </div>

      <table className="data-table">
        <tbody>
          <tr>
            <td><strong>Image Dimensions</strong></td>
            <td>{dimensions.width} × {dimensions.height} px ({dimensions.aspect}:1 aspect ratio, {dimensions.megapixels} MP)</td>
          </tr>
          <tr>
            <td><strong>Illumination Uniformity</strong></td>
            <td>{photometry.uniformityRatio} (10th/90th percentile luminance ratio)</td>
          </tr>
          <tr>
            <td><strong>Flash Glare Exposure</strong></td>
            <td>{photometry.overexposedPct}% overexposed pixels (&gt;240)</td>
          </tr>
          <tr>
            <td><strong>Shadow / Underexposure</strong></td>
            <td>{photometry.underexposedPct}% underexposed pixels (&lt;25)</td>
          </tr>
          <tr>
            <td><strong>Clinical Screening Status</strong></td>
            <td>
              <span className="pill-badge" style={{
                background: gradability.isGradable ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                color: gradability.isGradable ? '#34d399' : '#f87171',
                borderColor: gradability.isGradable ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.3)'
              }}>
                {gradability.status}
              </span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
