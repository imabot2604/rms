import React, { useMemo } from 'react';
import useStore from '../store/useStore';
import KpiCard from '../components/ui/KpiCard';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const Dashboard = () => {
  const { normalizedData, isDataLoaded } = useStore();

  const aggregatedMetrics = useMemo(() => {
    if (!isDataLoaded || !normalizedData || normalizedData.length === 0) return null;

    const sorted = [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date));
    const latest = sorted[sorted.length - 1]?.metrics;
    const first = sorted[0]?.metrics;

    if (!latest || !first) return null;

    const calcTrend = (now, old) => {
      if (now == null || old == null || old === 0) return 0;
      return ((now - old) / Math.abs(old)) * 100;
    };

    return {
      occ: {
        value: latest.occupancy != null ? latest.occupancy * 100 : null,
        trend: calcTrend(latest.occupancy, first.occupancy)
      },
      adr: { value: latest.adr, trend: calcTrend(latest.adr, first.adr) },
      revpar: { value: latest.revpar, trend: calcTrend(latest.revpar, first.revpar) },
      revenue: { value: latest.roomRevenue, trend: calcTrend(latest.roomRevenue, first.roomRevenue) }
    };
  }, [normalizedData, isDataLoaded]);

  const chartData = useMemo(() => {
    if (!isDataLoaded || !normalizedData) return [];
    return [...normalizedData]
      .sort((a, b) => new Date(a.date) - new Date(b.date))
      .map(row => ({
        date: new Date(row.date).toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
        RevPAR: row.metrics?.revpar || 0,
        ADR: row.metrics?.adr || 0
      }));
  }, [normalizedData, isDataLoaded]);

  if (!isDataLoaded || !aggregatedMetrics) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem 1.5rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '0.5rem' }}>Welcome to RMS Simulation Studio</h2>
        <p style={{ color: 'var(--text-secondary)' }}>Please navigate to the Data Ingestion module to upload your hotel data.</p>
      </div>
    );
  }

  const formatNumber = (val) => val != null ? val.toLocaleString('en-US', { maximumFractionDigits: 0 }) : '0';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
        <KpiCard
          title="Occupancy"
          value={aggregatedMetrics.occ.value != null ? aggregatedMetrics.occ.value.toFixed(1) : '-'}
          suffix="%"
          trend={aggregatedMetrics.occ.trend}
        />
        <KpiCard
          title="ADR"
          value={aggregatedMetrics.adr.value != null ? aggregatedMetrics.adr.value.toFixed(2) : '-'}
          prefix="$"
          trend={aggregatedMetrics.adr.trend}
        />
        <KpiCard
          title="RevPAR"
          value={aggregatedMetrics.revpar.value != null ? aggregatedMetrics.revpar.value.toFixed(2) : '-'}
          prefix="$"
          trend={aggregatedMetrics.revpar.trend}
        />
        <KpiCard
          title="Room Revenue"
          value={formatNumber(aggregatedMetrics.revenue.value)}
          prefix="$"
          trend={aggregatedMetrics.revenue.trend}
        />
      </div>

      <div className="card" style={{ marginTop: '0' }}>
        <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '1rem' }}>RevPAR Trend</h3>
        <div style={{ width: '100%', height: 300 }}>
          <ResponsiveContainer>
            <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="colorRevpar" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
              <XAxis dataKey="date" stroke="var(--text-tertiary)" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} />
              <YAxis stroke="var(--text-tertiary)" tick={{ fill: 'var(--text-secondary)', fontSize: 12 }} tickFormatter={(val) => `$${val}`} />
              <Tooltip
                contentStyle={{ backgroundColor: 'var(--bg-surface-hover)', borderColor: 'var(--border-light)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
              <Area type="monotone" dataKey="RevPAR" stroke="var(--accent-primary)" strokeWidth={2} fillOpacity={1} fill="url(#colorRevpar)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1.5rem' }}>
        <div className="card">
          <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '0.5rem' }}>Simulation Status</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Current model ensemble and real-time adaptations.</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem', borderRadius: '0.5rem', backgroundColor: 'rgba(255,255,255,0.05)' }}>
              <span style={{ fontWeight: 500, fontSize: '0.875rem' }}>Active Ensemble</span>
              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', backgroundColor: 'rgba(59,130,246,0.2)', color: 'var(--accent-primary)', borderRadius: '9999px', border: '1px solid rgba(59,130,246,0.3)' }}>IDeaS (25%) + Duetto (25%) + ML (50%)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem', borderRadius: '0.5rem', backgroundColor: 'rgba(255,255,255,0.05)' }}>
              <span style={{ fontWeight: 500, fontSize: '0.875rem' }}>Compset Mode</span>
              <span style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem', backgroundColor: 'rgba(139,92,246,0.2)', color: 'var(--accent-secondary)', borderRadius: '9999px', border: '1px solid rgba(139,92,246,0.3)' }}>Simulated Anchored</span>
            </div>
          </div>
        </div>

        <div className="card">
          <h3 style={{ fontSize: '1.125rem', fontWeight: 700, marginBottom: '0.5rem' }}>Interpretability Insights</h3>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>Key drivers of current forecast.</p>
          <ul style={{ fontSize: '0.875rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', listStyle: 'none', padding: 0 }}>
            <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--accent-success)' }}>↑</span> RevPAR forecast increased due to stronger simulated OTB pace.</li>
            <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--accent-success)' }}>↑</span> Competitor ADR index suggests room for rate push in Q3.</li>
            <li style={{ display: 'flex', gap: '0.5rem' }}><span style={{ color: 'var(--accent-warning)' }}>⚠</span> Margin compression detected in last period (UOE rising).</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
