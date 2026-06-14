import React from 'react';
import useStore from '../../store/useStore';

const Header = () => {
  const { isDataLoaded, activeModule } = useStore();

  const moduleTitles = {
    dashboard: 'Executive Dashboard',
    ingestion: 'Data Ingestion Engine',
    historical: 'Historical Trends & Seasonality',
    forecasting: 'Simulation & Forecasting',
    pricing: 'Pricing Optimization',
    scenarios: 'Scenario Simulator',
    audit: 'Model Audit & Interpretability',
  };

  return (
    <header className="header">
      <div>
        <h1>{moduleTitles[activeModule]}</h1>
        <p className="text-secondary">
          {isDataLoaded 
            ? 'Operating with uploaded data.' 
            : 'Upload a monthly hotel P&L or operating report to begin.'}
        </p>
      </div>
      
      <div className="flex gap-4 items-center">
        <div className={`px-3 py-1 rounded-full text-xs font-medium ${isDataLoaded ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'}`}
             style={{ 
               backgroundColor: isDataLoaded ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
               color: isDataLoaded ? 'var(--accent-success)' : 'var(--accent-warning)',
               padding: '4px 12px',
               borderRadius: '999px',
               border: `1px solid ${isDataLoaded ? 'rgba(16, 185, 129, 0.2)' : 'rgba(245, 158, 11, 0.2)'}`
             }}>
          {isDataLoaded ? '● System Active' : '○ Waiting for Data'}
        </div>
      </div>
    </header>
  );
};

export default Header;
