import React from 'react';
import { Home, Upload, TrendingUp, BarChart2, DollarSign, Activity, Settings, FileText, Anchor } from 'lucide-react';
import useStore from '../../store/useStore';

const Sidebar = () => {
  const { activeModule, setActiveModule, isDataLoaded } = useStore();

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: <Home /> },
    { id: 'ingestion', label: 'Data Ingestion', icon: <Upload /> },
    { id: 'historical', label: 'Historical Trends', icon: <TrendingUp />, requiresData: true },
    { id: 'forecasting', label: 'Simulation & Forecast', icon: <BarChart2 />, requiresData: true },
    { id: 'pricing', label: 'Pricing Optimization', icon: <DollarSign />, requiresData: true },
    { id: 'scenarios', label: 'Scenario Simulator', icon: <Activity />, requiresData: true },
    { id: 'audit', label: 'Model Audit', icon: <FileText />, requiresData: true },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="icon-container">
          <Anchor size={20} color="white" />
        </div>
        RMS Studio
      </div>

      <nav className="nav-menu">
        {navItems.map((item) => {
          const isDisabled = item.requiresData && !isDataLoaded;
          return (
            <button
              key={item.id}
              className={`nav-item ${activeModule === item.id ? 'active' : ''}`}
              onClick={() => !isDisabled && setActiveModule(item.id)}
              disabled={isDisabled}
              style={{ opacity: isDisabled ? 0.5 : 1, cursor: isDisabled ? 'not-allowed' : 'pointer' }}
              title={isDisabled ? 'Upload data first' : ''}
            >
              {item.icon}
              {item.label}
            </button>
          );
        })}
      </nav>

      <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
        <button className="nav-item" style={{ width: '100%' }}>
          <Settings size={20} />
          Settings
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
