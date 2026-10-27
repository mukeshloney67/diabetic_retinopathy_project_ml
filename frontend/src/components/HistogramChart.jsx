import React from 'react';

export default function HistogramChart({ histograms, channels }) {
  if (!histograms) return null;

  // Build SVG polygon points for Red, Green, and Blue histograms
  const renderChannel = (hist, color, fill) => {
    const maxVal = Math.max(...hist, 1);
    const width = 500;
    const height = 120;

    let points = `0,${height} `;
    for (let i = 0; i < 256; i++) {
      const x = (i / 255) * width;
      const y = height - (hist[i] / maxVal) * (height - 10);
      points += `${x.toFixed(1)},${y.toFixed(1)} `;
    }
    points += `${width},${height}`;

    return (
      <polygon
        points={points}
        fill={fill}
        stroke={color}
        strokeWidth="1.5"
        opacity="0.75"
      />
    );
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>🎨</span>
        <span>RGB Spectral Histograms (Vascular Profiling)</span>
      </div>

      <div style={{ width: '100%', height: '140px', background: '#0f172a', borderRadius: '8px', padding: '10px 14px', position: 'relative' }}>
        <svg viewBox="0 0 500 120" style={{ width: '100%', height: '100%', overflow: 'visible' }}>
          {renderChannel(histograms.red, '#ef4444', 'rgba(239, 68, 68, 0.25)')}
          {renderChannel(histograms.green, '#22c55e', 'rgba(34, 197, 94, 0.25)')}
          {renderChannel(histograms.blue, '#3b82f6', 'rgba(59, 130, 246, 0.25)')}
        </svg>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-around', marginTop: '12px', fontSize: '0.82rem' }}>
        <span style={{ color: '#f87171' }}>● Red Mean: {channels?.red?.mean || '—'}</span>
        <span style={{ color: '#4ade80' }}>● Green Mean: {channels?.green?.mean || '—'} (Lesion contrast)</span>
        <span style={{ color: '#60a5fa' }}>● Blue Mean: {channels?.blue?.mean || '—'}</span>
      </div>
    </div>
  );
}
