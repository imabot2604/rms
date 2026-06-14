import React, { useMemo } from 'react';
import { Activity, SlidersHorizontal } from 'lucide-react';
import useStore from '../store/useStore';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Scenarios = () => {
  const { normalizedData, scenarioParams, updateScenarioParams } = useStore();

  const handleSliderChange = (e, paramName) => {
    updateScenarioParams({ [paramName]: parseFloat(e.target.value) });
  };

  const chartData = useMemo(() => {
    if (!normalizedData || normalizedData.length === 0) return [];

    const sorted = [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date));

    // Use the last 6 months to compute a baseline for forward projections
    const recentData = sorted.slice(-6);
    const avgAdr = recentData.reduce((sum, r) => sum + (r.metrics?.adr || 0), 0) / recentData.length;
    const avgOcc = recentData.reduce((sum, r) => sum + (r.metrics?.occupancy || 0), 0) / recentData.length;

    const lastDate = new Date(sorted[sorted.length - 1].date);

    const results = [];
    for (let i = 1; i <= 6; i++) {
      const targetDate = new Date(lastDate);
      targetDate.setMonth(targetDate.getMonth() + i);

      const seasonalFactor = 1 + Math.sin((targetDate.getMonth()) / 12 * Math.PI * 2) * 0.1;
      const baseOcc = Math.min(avgOcc * seasonalFactor, 0.95);
      const baseAdr = avgAdr * seasonalFactor;
      const baseRevpar = baseOcc * baseAdr;

      // Apply shocks
      const adrMultiplier = 1 + (scenarioParams.adrShock / 100);
      const demandMultiplier = 1 + (scenarioParams.demandShock / 100);

      const scenarioOcc = Math.min(baseOcc * demandMultiplier, 1.0);
      const scenarioAdr = baseAdr * adrMultiplier;
      const scenarioRevpar = scenarioOcc * scenarioAdr;

      // GOP calculation
      const baseGopMargin = 0.35;
      const expenseMultiplier = 1 + (scenarioParams.expenseInflation / 100);
      const baseGop = baseRevpar * baseGopMargin;
      const scenarioGop = scenarioRevpar - (baseRevpar * (1 - baseGopMargin) * expenseMultiplier);

      results.push({
        date: targetDate.toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
        'Base RevPAR': Math.round(baseRevpar * 100) / 100,
        'Scenario RevPAR': Math.round(scenarioRevpar * 100) / 100,
        'Base GOP': Math.round(baseGop * 100) / 100,
        'Scenario GOP': Math.max(0, Math.round(scenarioGop * 100) / 100)
      });
    }

    return results;
  }, [normalizedData, scenarioParams]);

  return (
    <div style={{ display: 'flex', gap: '1.5rem', height: 'calc(100vh - 140px)' }}>
      {/* Sidebar Controls */}
      <div style={{ width: '320px', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="card" style={{ height: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
            <SlidersHorizontal style={{ color: 'var(--accent-primary)' }} />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Shock Parameters</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Demand Shock</label>
                <span style={{
                  fontSize: '0.875rem', fontWeight: 700,
                  color: scenarioParams.demandShock > 0 ? 'var(--accent-success)' : scenarioParams.demandShock < 0 ? 'var(--accent-danger)' : 'var(--text-primary)'
                }}>
                  {scenarioParams.demandShock > 0 ? '+' : ''}{scenarioParams.demandShock}%
                </span>
              </div>
              <input
                type="range" min="-50" max="50" step="5"
                value={scenarioParams.demandShock}
                onChange={(e) => handleSliderChange(e, 'demandShock')}
                style={{ width: '100%', accentColor: 'var(--accent-primary)' }}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Simulates event uplift or market collapse.</p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Global ADR Adjustment</label>
                <span style={{
                  fontSize: '0.875rem', fontWeight: 700,
                  color: scenarioParams.adrShock > 0 ? 'var(--accent-success)' : scenarioParams.adrShock < 0 ? 'var(--accent-danger)' : 'var(--text-primary)'
                }}>
                  {scenarioParams.adrShock > 0 ? '+' : ''}{scenarioParams.adrShock}%
                </span>
              </div>
              <input
                type="range" min="-30" max="30" step="2"
                value={scenarioParams.adrShock}
                onChange={(e) => handleSliderChange(e, 'adrShock')}
                style={{ width: '100%' }}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Blanket rate increase/decrease across all segments.</p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label style={{ fontSize: '0.875rem', fontWeight: 500, color: 'var(--text-secondary)' }}>Expense Inflation</label>
                <span style={{
                  fontSize: '0.875rem', fontWeight: 700,
                  color: scenarioParams.expenseInflation > 0 ? 'var(--accent-danger)' : 'var(--text-primary)'
                }}>
                  {scenarioParams.expenseInflation > 0 ? '+' : ''}{scenarioParams.expenseInflation}%
                </span>
              </div>
              <input
                type="range" min="0" max="30" step="1"
                value={scenarioParams.expenseInflation}
                onChange={(e) => handleSliderChange(e, 'expenseInflation')}
                style={{ width: '100%' }}
              />
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Impacts GOP and NOI forecasts directly.</p>
            </div>

            <button
              className="btn btn-secondary"
              style={{ marginTop: '1rem', width: '100%' }}
              onClick={() => updateScenarioParams({ demandShock: 0, adrShock: 0, expenseInflation: 0 })}
            >
              Reset Scenarios
            </button>
          </div>
        </div>
      </div>

      {/* Main Chart Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 700 }}>Scenario Recalculation (Next 6 Months)</h3>
            <div style={{
              padding: '0.5rem 0.75rem', backgroundColor: 'rgba(0,0,0,0.2)',
              borderRadius: '9999px', border: '1px solid var(--border-light)',
              display: 'flex', alignItems: 'center', gap: '0.5rem'
            }}>
              <Activity size={14} style={{ color: 'var(--accent-secondary)' }} />
              <span style={{ fontSize: '0.75rem', fontWeight: 500 }}>Real-time Adaptation Active</span>
            </div>
          </div>

          <div style={{ flex: 1, minHeight: 0 }}>
            <ResponsiveContainer>
              <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
                <XAxis dataKey="date" stroke="var(--text-tertiary)" />
                <YAxis stroke="var(--text-tertiary)" tickFormatter={(val) => `$${val}`} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--bg-surface-hover)', borderColor: 'var(--border-light)', borderRadius: '8px' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  formatter={(value) => `$${value.toFixed(0)}`}
                />
                <Legend wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="Base RevPAR" fill="var(--text-tertiary)" opacity={0.5} radius={[4, 4, 0, 0]} />
                <Bar dataKey="Scenario RevPAR" fill="var(--accent-primary)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Scenario GOP" fill="var(--accent-success)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Scenarios;
