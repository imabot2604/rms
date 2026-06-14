/**
 * Mock implementation of advanced forecasting algorithms for the client-side simulation.
 */

// Basic Moving Average for Baseline
const calculateMovingAverage = (data, key, window = 3) => {
  const values = data.map(d => d.metrics[key]).filter(v => v !== null && !isNaN(v));
  if (values.length === 0) return 0;
  if (values.length < window) window = values.length;
  
  const recent = values.slice(-window);
  return recent.reduce((sum, val) => sum + val, 0) / window;
};

// Simulate an IDeaS-style Forecast (Time-series + Unconstrained Demand)
export const runIdeasSimulator = (historicalData, horizonMonths = 12) => {
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  const baseOcc = calculateMovingAverage(historicalData, 'occupancy', 6) || 0.70;
  const baseAdr = calculateMovingAverage(historicalData, 'adr', 6) || 150;

  const forecast = [];
  for (let i = 1; i <= horizonMonths; i++) {
    const targetDate = new Date(lastDate);
    targetDate.setMonth(targetDate.getMonth() + i);
    
    // Unconstrained demand proxy (adds some noise)
    const demandMultiplier = 1 + (Math.sin(targetDate.getMonth() / 12 * Math.PI * 2) * 0.15); 
    const constrainedOcc = Math.min(baseOcc * demandMultiplier, 1.0);
    
    // Pricing power linked to demand
    const adrMultiplier = 1 + (demandMultiplier - 1) * 0.5;
    
    forecast.push({
      date: targetDate.toISOString(),
      model: 'IDeaS-style',
      metrics: {
        occupancy: constrainedOcc,
        adr: baseAdr * adrMultiplier,
        revpar: constrainedOcc * (baseAdr * adrMultiplier),
      }
    });
  }
  return forecast;
};

// Simulate a Duetto-style Forecast (Trend + Pace/OTB emphasis)
export const runDuettoSimulator = (historicalData, horizonMonths = 12) => {
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  const baseOcc = calculateMovingAverage(historicalData, 'occupancy', 3) || 0.70;
  const baseAdr = calculateMovingAverage(historicalData, 'adr', 3) || 150;

  const forecast = [];
  for (let i = 1; i <= horizonMonths; i++) {
    const targetDate = new Date(lastDate);
    targetDate.setMonth(targetDate.getMonth() + i);
    
    // OTB Pace proxy (heavier near-term weight, fades out)
    const paceStrength = Math.max(0, 1 - (i * 0.1)); // Stronger for close months
    const paceNoise = 1 + (Math.random() * 0.1 - 0.05); // +/- 5% noise
    
    const occ = Math.min(baseOcc * paceNoise * (1 + paceStrength * 0.1), 0.95);
    // Dynamic open pricing (reacts to pace)
    const adr = baseAdr * (1 + paceStrength * 0.2); 

    forecast.push({
      date: targetDate.toISOString(),
      model: 'Duetto-style',
      metrics: {
        occupancy: occ,
        adr: adr,
        revpar: occ * adr,
      }
    });
  }
  return forecast;
};

// Ensemble Forecast (Blend of models)
export const runEnsembleForecast = (historicalData, horizonMonths = 12) => {
  const ideasF = runIdeasSimulator(historicalData, horizonMonths);
  const duettoF = runDuettoSimulator(historicalData, horizonMonths);
  
  const forecast = [];
  for (let i = 0; i < horizonMonths; i++) {
    // 70% IDeaS, 30% Duetto logic as per requirements
    const occ = (ideasF[i].metrics.occupancy * 0.7) + (duettoF[i].metrics.occupancy * 0.3);
    const adr = (ideasF[i].metrics.adr * 0.7) + (duettoF[i].metrics.adr * 0.3);
    
    forecast.push({
      date: ideasF[i].date,
      model: 'Ensemble (70/30)',
      metrics: {
        occupancy: occ,
        adr: adr,
        revpar: occ * adr,
      }
    });
  }
  return forecast;
};
