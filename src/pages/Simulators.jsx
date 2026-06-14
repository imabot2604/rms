import React, { useState, useEffect } from 'react';
import useStore from '../store/useStore';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Simulators = () => {
  const { normalizedData } = useStore();
  const [horizon, setHorizon] = useState(6);
  const [forecastData, setForecastData] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (normalizedData.length === 0) return;
    
    const fetchForecast = async () => {
      setIsLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/api/forecast?horizon=${horizon}`);
        const result = await response.json();
        
        if (result.status === 'success') {
          // Merge historical and forecast for charting
          const historical = [...normalizedData].sort((a, b) => new Date(a.date) - new Date(b.date));
          const histPoints = historical.slice(-6).map(h => ({
            date: new Date(h.date).toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
            'Actual Occupancy': h.metrics.occupancy
          }));
          
          const futurePoints = result.forecast.map(f => ({
            date: new Date(f.date).toLocaleDateString(undefined, { month: 'short', year: '2-digit' }),
            'IDeaS-style Forecast': f.models.IDeaS_Simulator,
            'Duetto-style Forecast': f.models.Duetto_Simulator,
            'Ensemble Forecast': f.models.Ensemble,
            'Prophet Lower': f.models.Prophet_Lower,
            'Prophet Upper': f.models.Prophet_Upper
          }));
          
          setForecastData([...histPoints, ...futurePoints]);
        }
      } catch (err) {
        console.error("Failed to fetch forecast from backend:", err);
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchForecast();
  }, [normalizedData, horizon]);

  return (
    <div className="flex-col gap-6">
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold mb-2">Behavioral Simulators</h2>
          <p className="text-secondary">Explore different RM paradigms applied to your normalized data.</p>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="text-sm font-medium text-secondary">Forecast Horizon (Months):</label>
          <select 
            value={horizon} 
            onChange={(e) => setHorizon(Number(e.target.value))}
            className="input-field"
            style={{ width: '100px' }}
          >
            <option value={3}>3 Months</option>
            <option value={6}>6 Months</option>
            <option value={12}>12 Months</option>
            <option value={24}>24 Months</option>
          </select>
        </div>
      </div>

      <div className="card mb-6">
        <h3 className="text-lg font-bold mb-4">
          Ensemble Occupancy Forecast 
          {isLoading && <span className="ml-4 text-sm font-normal text-tertiary">Running models...</span>}
        </h3>
        <div style={{ width: '100%', height: 400 }}>
          <ResponsiveContainer>
            <LineChart data={forecastData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" vertical={false} />
              <XAxis dataKey="date" stroke="var(--text-tertiary)" />
              <YAxis stroke="var(--text-tertiary)" tickFormatter={(val) => `${(val*100).toFixed(0)}%`} />
              <Tooltip 
                contentStyle={{ backgroundColor: 'var(--bg-surface-hover)', borderColor: 'var(--border-light)', borderRadius: '8px' }}
                itemStyle={{ color: 'var(--text-primary)' }}
                formatter={(value) => `${(value*100).toFixed(1)}%`}
              />
              <Legend wrapperStyle={{ paddingTop: '20px' }} />
              
              <Line type="monotone" dataKey="Actual Occupancy" stroke="var(--text-primary)" strokeWidth={3} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="IDeaS-style Forecast" stroke="var(--accent-primary)" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              <Line type="monotone" dataKey="Duetto-style Forecast" stroke="var(--accent-secondary)" strokeWidth={2} strokeDasharray="5 5" dot={false} />
              <Line type="monotone" dataKey="Ensemble Forecast" stroke="var(--accent-success)" strokeWidth={4} dot={{ r: 4 }} activeDot={{ r: 6 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="card">
          <h3 className="text-lg font-bold mb-2">IDeaS-Style Architecture</h3>
          <p className="text-sm text-secondary mb-4">Emphasizes unconstrained demand estimation and statistical time-series forecasting (ARIMA proxies).</p>
          <ul className="text-sm flex flex-col gap-2">
            <li className="flex justify-between border-b border-border-light pb-1">
              <span className="text-tertiary">Primary Driver</span>
              <span className="font-medium">Time-Series Trends</span>
            </li>
            <li className="flex justify-between border-b border-border-light pb-1">
              <span className="text-tertiary">Constrained Logic</span>
              <span className="font-medium">Demand Multipliers</span>
            </li>
          </ul>
        </div>
        
        <div className="card">
          <h3 className="text-lg font-bold mb-2">Duetto-Style Architecture</h3>
          <p className="text-sm text-secondary mb-4">Emphasizes OTB pace, booking velocity, and open pricing elasticity.</p>
          <ul className="text-sm flex flex-col gap-2">
            <li className="flex justify-between border-b border-border-light pb-1">
              <span className="text-tertiary">Primary Driver</span>
              <span className="font-medium">OTB Pace Index</span>
            </li>
            <li className="flex justify-between border-b border-border-light pb-1">
              <span className="text-tertiary">Pricing Logic</span>
              <span className="font-medium">Dynamic Responsiveness</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Simulators;
