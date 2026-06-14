import React from 'react';
import Layout from './components/layout/Layout';
import useStore from './store/useStore';
import DataIngestion from './pages/DataIngestion';
import Dashboard from './pages/Dashboard';
import HistoricalTrends from './pages/HistoricalTrends';
import Simulators from './pages/Simulators';
import Pricing from './pages/Pricing';
import Scenarios from './pages/Scenarios';
import Audit from './pages/Audit';
import './App.css';







function App() {
  const activeModule = useStore((state) => state.activeModule);

  const renderModule = () => {
    switch (activeModule) {
      case 'dashboard': return <Dashboard />;
      case 'ingestion': return <DataIngestion />;
      case 'historical': return <HistoricalTrends />;
      case 'forecasting': return <Simulators />;
      case 'pricing': return <Pricing />;
      case 'scenarios': return <Scenarios />;
      case 'audit': return <Audit />;
      default: return <Dashboard />;
    }
  };

  return (
    <Layout>
      {renderModule()}
    </Layout>
  );
}

export default App;
