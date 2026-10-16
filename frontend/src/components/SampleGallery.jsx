import React from 'react';
import { SEVERITY_STAGES } from '../services/imageAnalysis';

export default function SampleGallery({ onSelectSample }) {
  // Generate sample SVG retinal fundus representations for each stage
  const generateSampleDataUrl = (stage) => {
    const canvas = document.createElement('canvas');
    canvas.width = 300;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');

    // Background
    ctx.fillStyle = '#05070a';
    ctx.fillRect(0, 0, 300, 300);

    // Retinal fundus circle
    ctx.beginPath();
    ctx.arc(150, 150, 130, 0, Math.PI * 2);
    ctx.fillStyle = '#b91c1c';
    ctx.fill();

    // Optic disc
    ctx.beginPath();
    ctx.arc(105, 150, 24, 0, Math.PI * 2);
    ctx.fillStyle = '#fef08a';
    ctx.fill();

    // Lesions based on stage
    if (stage >= 1) {
      // Microaneurysms
      for (let i = 0; i < stage * 10; i++) {
        const x = 120 + Math.random() * 80;
        const y = 110 + Math.random() * 80;
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = '#7f1d1d';
        ctx.fill();
      }
    }

    if (stage >= 2) {
      // Hard exudates (yellow patches)
      for (let i = 0; i < stage * 4; i++) {
        const x = 130 + Math.random() * 60;
        const y = 120 + Math.random() * 60;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#fde047';
        ctx.fill();
      }
    }

    if (stage >= 3) {
      // Cotton wool spots
      for (let i = 0; i < stage * 2; i++) {
        const x = 140 + Math.random() * 50;
        const y = 130 + Math.random() * 50;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.fillStyle = '#e2e8f0';
        ctx.fill();
      }
    }

    if (stage === 4) {
      // Neovascularization
      ctx.strokeStyle = '#450a0a';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(105, 150);
      ctx.lineTo(80, 120);
      ctx.lineTo(60, 100);
      ctx.stroke();
    }

    return canvas.toDataURL('image/png');
  };

  const handleSelect = (stage) => {
    const dataUrl = generateSampleDataUrl(stage);
    fetch(dataUrl)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], `sample_stage_${stage}_${SEVERITY_STAGES[stage].name.replace(/\s+/g, '_')}.png`, { type: 'image/png' });
        onSelectSample(file, stage);
      });
  };

  return (
    <div className="card">
      <div className="card-title">
        <span>🖼️</span>
        <span>Preloaded Clinical Fundus Test Samples</span>
      </div>
      <p style={{ fontSize: '0.88rem', color: '#9ca3af', marginBottom: '16px' }}>
        Don't have retinal images on hand? Click any clinical stage below to instantly run diagnostic inference, test ground-truth accuracy, and inspect photometric statistics!
      </p>

      <div className="grid-4" style={{ gridTemplateColumns: 'repeat(5, 1fr)' }}>
        {[0, 1, 2, 3, 4].map((stage) => {
          const info = SEVERITY_STAGES[stage];
          return (
            <div
              key={stage}
              onClick={() => handleSelect(stage)}
              style={{
                background: '#111827',
                border: '1px solid #374151',
                borderRadius: '8px',
                padding: '12px',
                cursor: 'pointer',
                textAlign: 'center',
                transition: 'border-color 0.2s',
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = info.color}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = '#374151'}
            >
              <div style={{ fontSize: '1.8rem', marginBottom: '6px' }}>
                {stage === 0 ? '🟢' : stage === 1 ? '🟡' : stage === 2 ? '🟠' : stage === 3 ? '🔴' : '🚨'}
              </div>
              <div style={{ fontWeight: 700, fontSize: '0.9rem', color: info.color }}>
                Stage {stage}
              </div>
              <div style={{ fontSize: '0.78rem', color: '#d1d5db', marginTop: '2px' }}>
                {info.name}
              </div>
              <div style={{ fontSize: '0.72rem', color: '#9ca3af', marginTop: '4px' }}>
                {info.severity}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
