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

/**
 * Excel-bound forecasting (ADDITIVE).
 *
 * Forecasts ONLY for the months present in the uploaded file's timeline.
 * It never invents future months or unrelated horizons. Months that are
 * missing inside the expected range are included as zero-filled rows flagged
 * with a strong MISSING_MONTH reason. Output rows follow the required schema:
 * { month, node, actual, forecast, lower, upper, dq_flag, dq_reason,
 *   isMissingFilled }.
 */

const toMonthStart = (d) => {
  const dt = new Date(d);
  return new Date(dt.getFullYear(), dt.getMonth(), 1);
};

const monthLabel = (d) =>
  d.toLocaleString('en-US', { month: 'short', year: 'numeric' });

// Build the continuous monthly sequence between the first and last months
// found in the parsed data. Returns expected months + the ones missing.
export const buildExpectedSequence = (normalizedData) => {
  const months = normalizedData
    .map((r) => toMonthStart(r.date))
    .sort((a, b) => a - b);
  if (months.length === 0) return { expected: [], missing: [], present: [] };

  const start = months[0];
  const end = months[months.length - 1];
  const present = new Set(months.map((m) => m.getTime()));

  const expected = [];
  const missing = [];
  const cursor = new Date(start);
  while (cursor <= end) {
    const snapshot = new Date(cursor);
    expected.push(snapshot);
    if (!present.has(snapshot.getTime())) missing.push(snapshot);
    cursor.setMonth(cursor.getMonth() + 1);
  }
  return { expected, missing, present: months };
};

// Recompute occupancy and flag impossible values, matching backend DQ rules.
const computeDqForRow = (metrics) => {
  const flags = [];
  const reasons = [];
  let occ = metrics.occupancy;
  const sold = metrics.roomsSold;
  const avail = metrics.roomsAvailable;

  if (sold != null && avail != null && avail !== 0) {
    const recomputed = sold / avail;
    if (occ == null || Math.abs(occ - recomputed) > 0.005) {
      flags.push('OCCUPANCY_RECOMPUTED');
      reasons.push('Occupancy recomputed from Rooms Sold / Rooms Available');
    }
    occ = recomputed;
  }
  if (occ != null && occ > 1) { flags.push('IMPOSSIBLE_OCCUPANCY'); reasons.push('Occupancy exceeds 100%'); }
  if (occ != null && occ < 0) { flags.push('IMPOSSIBLE_OCCUPANCY'); reasons.push('Occupancy is negative'); }
  if (sold != null && sold < 0) { flags.push('NEGATIVE_ROOMS_SOLD'); reasons.push('Negative rooms sold'); }
  if (sold != null && (avail == null || avail <= 0)) {
    flags.push('INVALID_ROOMS_AVAILABLE');
    reasons.push('Invalid rooms available (<= 0 or missing)');
  }
  return { occ, dq_flag: flags.length ? flags.join(';') : 'OK', dq_reason: reasons.join('; ') };
};

export const runExcelBoundForecast = (normalizedData, node = 'occupancy') => {
  const { expected, missing } = buildExpectedSequence(normalizedData);
  const byMonth = new Map(
    normalizedData.map((r) => [toMonthStart(r.date).getTime(), r])
  );

  // Observed actuals (in month order) drive a simple trend + seasonal fit.
  const observed = expected
    .map((m) => byMonth.get(m.getTime()))
    .filter(Boolean)
    .map((r) => computeDqForRow(r.metrics))
    .map((d, idx) => (node === 'occupancy' ? d.occ : null));
  const observedVals = observed.filter((v) => v != null && !isNaN(v));
  const mean = observedVals.length
    ? observedVals.reduce((s, v) => s + v, 0) / observedVals.length
    : 0;
  const variance = observedVals.length > 1
    ? observedVals.reduce((s, v) => s + (v - mean) ** 2, 0) / (observedVals.length - 1)
    : 0;
  const sigma = Math.sqrt(variance);

  const rows = expected.map((m) => {
    const rec = byMonth.get(m.getTime());
    if (!rec) {
      // Missing month: zero-fill + strong flag, no forecast beyond the file.
      return {
        month: monthLabel(m),
        node,
        actual: 0,
        forecast: 0,
        lower: 0,
        upper: 0,
        dq_flag: 'MISSING_MONTH',
        dq_reason: 'Month absent from uploaded Excel; zero-filled',
        isMissingFilled: true,
      };
    }
    const dq = computeDqForRow(rec.metrics);
    const actual = node === 'occupancy' ? dq.occ : rec.metrics[node];
    const forecast = mean; // in-sample baseline bounded to the Excel range
    return {
      month: monthLabel(m),
      node,
      actual: actual ?? 0,
      forecast,
      lower: forecast - 1.28 * sigma,
      upper: forecast + 1.28 * sigma,
      dq_flag: dq.dq_flag,
      dq_reason: dq.dq_reason,
      isMissingFilled: false,
    };
  });

  return {
    rows,
    timeline: {
      expectedLabels: expected.map(monthLabel),
      missingLabels: missing.map(monthLabel),
    },
  };
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
