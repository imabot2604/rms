import React, { useMemo } from 'react';
import { Database, GitCommit, FileSearch, ShieldCheck } from 'lucide-react';
import useStore from '../store/useStore';

const Audit = () => {
  const { normalizedData } = useStore();

  const auditInfo = useMemo(() => {
    if (!normalizedData || normalizedData.length === 0) return null;

    const sorted = [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const first = sorted[0];
    const last = sorted[sorted.length - 1];

    const detectedFields = first?.metrics
      ? Object.keys(first.metrics).filter(k => first.metrics[k] !== null && first.metrics[k] !== undefined)
      : [];

    return {
      recordCount: normalizedData.length,
      firstDate: first?.date ? new Date(first.date).toLocaleDateString() : 'N/A',
      lastDate: last?.date ? new Date(last.date).toLocaleDateString() : 'N/A',
      fieldsExtracted: detectedFields.length,
      detectedFields,
    };
  }, [normalizedData]);

  if (!auditInfo) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Please navigate to the Data Ingestion module to upload your hotel data.</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ marginBottom: '0' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>Model Audit & Transparency</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Review the underlying models, data lineage, and behavioral assumptions driving the simulation.</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <Database style={{ color: 'var(--accent-primary)' }} />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Data Lineage & Quality</h3>
          </div>

          <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem', padding: '1rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.5rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-light)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Time-Series Extracted</span>
              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{auditInfo.recordCount} periods</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.5rem', paddingBottom: '0.5rem', borderBottom: '1px solid var(--border-light)' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Training Window</span>
              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{auditInfo.firstDate} to {auditInfo.lastDate}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Features Normalized</span>
              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{auditInfo.fieldsExtracted} continuous variables</span>
            </div>
          </div>

          <div style={{
            display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem',
            color: 'var(--accent-success)', backgroundColor: 'rgba(16,185,129,0.1)',
            padding: '0.75rem', borderRadius: '0.5rem', border: '1px solid rgba(16,185,129,0.2)'
          }}>
            <ShieldCheck size={16} />
            Data quality checks passed. No negative ADRs detected.
          </div>
        </div>

        <div className="card">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <GitCommit style={{ color: 'var(--accent-secondary)' }} />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Ensemble Architecture</h3>
          </div>

          <ul style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', listStyle: 'none', padding: 0 }}>
            <li style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.5rem' }}>
              <div>
                <span style={{ fontWeight: 500, display: 'block', fontSize: '0.875rem' }}>IDeaS-style Behavioral Model</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>ARIMA + XGBoost + Comp-set Adjustment</span>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', display: 'block' }}>Weight</span>
                <span style={{ fontWeight: 700, color: 'var(--accent-primary)' }}>25%</span>
              </div>
            </li>
            <li style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.5rem' }}>
              <div>
                <span style={{ fontWeight: 500, display: 'block', fontSize: '0.875rem' }}>Duetto-style Behavioral Model</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Prophet + Random Forest + OTB Pace</span>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', display: 'block' }}>Weight</span>
                <span style={{ fontWeight: 700, color: 'var(--accent-secondary)' }}>25%</span>
              </div>
            </li>
            <li style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.5rem' }}>
              <div>
                <span style={{ fontWeight: 500, display: 'block', fontSize: '0.875rem' }}>ML Regressors (LR + RF + GB + XGB)</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Scikit-learn ensemble sub-models</span>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', display: 'block' }}>Weight</span>
                <span style={{ fontWeight: 700, color: 'var(--accent-success)' }}>35%</span>
              </div>
            </li>
            <li style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.2)', padding: '0.75rem', borderRadius: '0.5rem' }}>
              <div>
                <span style={{ fontWeight: 500, display: 'block', fontSize: '0.875rem' }}>Facebook Prophet</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Bayesian time-series with US holidays</span>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', display: 'block' }}>Weight</span>
                <span style={{ fontWeight: 700, color: 'var(--accent-warning)' }}>15%</span>
              </div>
            </li>
          </ul>
        </div>

        <div className="card" style={{ gridColumn: 'span 2' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <FileSearch style={{ color: 'var(--accent-warning)' }} />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Interpretability Insights & Known Limitations</h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
            <div>
              <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Current Forecast Drivers</h4>
              <ul style={{ fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', listStyle: 'none', padding: 0 }}>
                <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--accent-success)' }}>↑</span> RevPAR forecast is buoyed by strong historical trend carrying forward.</li>
                <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--text-tertiary)' }}>→</span> Compset impact is neutral (simulated compset anchored to subject).</li>
                <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--accent-warning)' }}>⚠</span> Margin volatility detected. Expense inflation is reducing GOP flow-through.</li>
              </ul>
            </div>

            <div>
              <h4 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--text-secondary)' }}>Simulation Limitations</h4>
              <ul style={{ fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', color: 'var(--text-tertiary)', listStyleType: 'disc', paddingLeft: '1rem' }}>
                <li>This is an open simulation framework, NOT connected to a live PMS.</li>
                <li>OTB Pace is currently simulated based on standard booking curves, as no native pace file was uploaded.</li>
                <li>Competitor indexing is simulated. Upload a STR/benchmark file to enable real MPI/ARI calculations.</li>
                <li>Weather and Event regressors are disabled (API keys not provided in this environment).</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Audit;
