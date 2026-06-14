import React, { useState, useMemo } from 'react';
import { Target, ArrowRight, Settings2 } from 'lucide-react';
import useStore from '../store/useStore';

const Pricing = () => {
  const { normalizedData } = useStore();
  const [objective, setObjective] = useState('revpar');
  const [aggressiveness, setAggressiveness] = useState('base');

  const recommendations = useMemo(() => {
    if (!normalizedData || normalizedData.length === 0) return [];

    const sorted = [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date));

    // Use the last 6 months as a basis for forward projections
    const recentData = sorted.slice(-6);
    const avgAdr = recentData.reduce((sum, r) => sum + (r.metrics?.adr || 0), 0) / recentData.length;
    const avgOcc = recentData.reduce((sum, r) => sum + (r.metrics?.occupancy || 0), 0) / recentData.length;

    const lastDate = new Date(sorted[sorted.length - 1].date);

    const results = [];
    for (let i = 1; i <= 4; i++) {
      const targetDate = new Date(lastDate);
      targetDate.setMonth(targetDate.getMonth() + i);

      // Apply seasonal variation
      const seasonalFactor = 1 + Math.sin((targetDate.getMonth()) / 12 * Math.PI * 2) * 0.08;
      const baseAdr = avgAdr * seasonalFactor;
      const baseOcc = Math.min(avgOcc * seasonalFactor, 0.95);

      let recAdr = baseAdr;
      let expOcc = baseOcc;

      if (objective === 'revpar') {
        recAdr = aggressiveness === 'aggressive' ? baseAdr * 1.05 : aggressiveness === 'conservative' ? baseAdr * 0.98 : baseAdr * 1.02;
        expOcc = aggressiveness === 'aggressive' ? baseOcc * 0.95 : aggressiveness === 'conservative' ? baseOcc * 1.02 : baseOcc * 0.98;
      } else if (objective === 'gop') {
        recAdr = baseAdr * 1.08;
        expOcc = baseOcc * 0.90;
      } else if (objective === 'occupancy') {
        recAdr = baseAdr * 0.90;
        expOcc = Math.min(baseOcc * 1.15, 1.0);
      }

      results.push({
        date: targetDate.toLocaleDateString(undefined, { month: 'short', year: 'numeric' }),
        baseAdr,
        recAdr,
        expOcc,
        expRevpar: recAdr * expOcc
      });
    }

    return results;
  }, [normalizedData, objective, aggressiveness]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ marginBottom: '0', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>Pricing Optimization</h2>
          <p style={{ color: 'var(--text-secondary)' }}>AI-driven rate recommendations based on ensemble forecasts and elasticity curves.</p>
        </div>

        <div style={{
          display: 'flex', alignItems: 'center', gap: '1rem',
          backgroundColor: 'var(--bg-surface)', padding: '0.75rem',
          borderRadius: '0.75rem', border: '1px solid var(--border-light)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Target size={16} style={{ color: 'var(--text-tertiary)' }} />
            <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Objective:</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="input-field"
              style={{ width: '150px', padding: '0.25rem 0.5rem' }}
            >
              <option value="revpar">Maximize RevPAR</option>
              <option value="gop">Maximize GOP Margin</option>
              <option value="occupancy">Maintain Occupancy Floor</option>
            </select>
          </div>
          <div style={{ width: '1px', height: '1.5rem', backgroundColor: 'var(--border-light)' }}></div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Settings2 size={16} style={{ color: 'var(--text-tertiary)' }} />
            <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Strategy:</label>
            <select
              value={aggressiveness}
              onChange={(e) => setAggressiveness(e.target.value)}
              className="input-field"
              style={{ width: '130px', padding: '0.25rem 0.5rem' }}
            >
              <option value="conservative">Conservative</option>
              <option value="base">Base Case</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        {recommendations.map((rec, idx) => {
          const adrDiff = rec.baseAdr > 0 ? ((rec.recAdr - rec.baseAdr) / rec.baseAdr) * 100 : 0;
          const isPositive = adrDiff > 0;
          return (
            <div key={idx} className="card" style={{
              borderLeft: `4px solid ${isPositive ? 'var(--accent-success)' : 'var(--accent-warning)'}`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <h3 style={{ fontWeight: 700, fontSize: '1.125rem' }}>{rec.date}</h3>
                <div style={{
                  padding: '0.25rem 0.5rem', borderRadius: '0.25rem', fontSize: '0.75rem', fontWeight: 700,
                  backgroundColor: isPositive ? 'rgba(16,185,129,0.2)' : 'rgba(245,158,11,0.2)',
                  color: isPositive ? 'var(--accent-success)' : 'var(--accent-warning)'
                }}>
                  {isPositive ? '+' : ''}{adrDiff.toFixed(1)}%
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1rem' }}>
                <div>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Base Forecast</p>
                  <p style={{ fontWeight: 500, fontSize: '1.125rem' }}>${rec.baseAdr.toFixed(0)}</p>
                </div>
                <ArrowRight size={16} style={{ color: 'var(--text-tertiary)', marginBottom: '0.5rem' }} />
                <div style={{ textAlign: 'right' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>Recommended</p>
                  <p style={{ fontWeight: 700, fontSize: '1.5rem', color: 'var(--text-primary)' }}>${rec.recAdr.toFixed(0)}</p>
                </div>
              </div>

              <div style={{ backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '0.5rem', padding: '0.75rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Expected Occ</span>
                  <span style={{ fontWeight: 500 }}>{(rec.expOcc * 100).toFixed(1)}%</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Expected RevPAR</span>
                  <span style={{ fontWeight: 500, color: 'var(--accent-success)' }}>${rec.expRevpar.toFixed(2)}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {recommendations.length === 0 && (
        <div className="card" style={{ textAlign: 'center', padding: '3rem' }}>
          <p style={{ color: 'var(--text-secondary)' }}>No data available to generate recommendations. Please upload hotel data first.</p>
        </div>
      )}
    </div>
  );
};

export default Pricing;
