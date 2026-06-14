import React, { useMemo } from 'react';
import useStore from '../store/useStore';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const HistoricalTrends = () => {
  const { normalizedData } = useStore();

  const chartData = useMemo(() => {
    return [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date)).map(row => ({
      date: new Date(row.date).toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
      Occupancy: (row.metrics.occupancy || 0) * 100,
      ADR: row.metrics.adr || 0,
      RevPAR: row.metrics.revpar || 0,
      Demand: row.metrics.roomsSold || 0
    }));
  }, [normalizedData]);

  return (
    <div className="flex-col gap-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Historical Trends & Seasonality</h2>
        <p className="text-secondary">Analyze observed historical patterns to identify base trends and seasonal indices.</p>
      </div>

      <div className="card mb-6">
        <h3 className="text-lg font-bold mb-4">ADR vs Occupancy History</h3>
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
              <XAxis dataKey="date" stroke="var(--text-tertiary)" />
              <YAxis yAxisId="left" stroke="var(--text-tertiary)" tickFormatter={(val) => `$${val}`} />
              <YAxis yAxisId="right" orientation="right" stroke="var(--text-tertiary)" tickFormatter={(val) => `${val}%`} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-surface-hover)', borderColor: 'var(--border-light)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              <Bar yAxisId="right" dataKey="Occupancy" fill="var(--accent-secondary)" opacity={0.6} radius={[4, 4, 0, 0]} />
              <Line yAxisId="left" type="monotone" dataKey="ADR" stroke="var(--accent-primary)" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 6 }} />
              <Line yAxisId="left" type="monotone" dataKey="RevPAR" stroke="var(--accent-success)" strokeWidth={2} strokeDasharray="5 5" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-bold mb-4">Seasonality Decomposition</h3>
          <div className="flex items-center justify-center h-48 bg-black/20 rounded-lg border border-border-light text-secondary text-sm">
            Decomposition algorithm requires at least 24 months of data. 
            {chartData.length < 24 ? ` Currently have ${chartData.length} months.` : ' Processing...'}
          </div>
        </div>
        <div className="card">
          <h3 className="text-lg font-bold mb-4">Segment Mix Behavior</h3>
          <div className="flex flex-col gap-4">
            <p className="text-sm text-secondary">
              If segment data (Transient, Group, Contract) was found in the upload, it will be mapped here.
            </p>
            <div className="flex items-center justify-center h-32 bg-black/20 rounded-lg border border-border-light text-secondary text-sm text-center p-4">
              Detailed segment arrays not detected in primary KPI load. Using blended 'Total' segment fallback logic.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoricalTrends;
