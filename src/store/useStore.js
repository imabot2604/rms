import { create } from 'zustand';

const useStore = create((set) => ({
  // Data State
  isDataLoaded: false,
  rawData: null, // Raw parsed spreadsheet data
  normalizedData: [], // Canonical hospitality schema (Historical)
  
  // Model & Forecast State
  forecastData: [], // Output of models
  simulatedPaceData: [], // OTB / Booking Pace
  compsetData: [], // Competitor benchmark
  
  // UI & Scenario State
  activeModule: 'dashboard', // dashboard, ingestion, historical, forecasting, pricing, audit
  
  scenarioParams: {
    adrShock: 0, // % change
    demandShock: 0, // % change
    expenseInflation: 0, // % change
  },
  
  // Actions
  setRawData: (data) => set({ rawData: data }),
  setNormalizedData: (data) => set({ normalizedData: data, isDataLoaded: true }),
  setForecastData: (data) => set({ forecastData: data }),
  setSimulatedPaceData: (data) => set({ simulatedPaceData: data }),
  setCompsetData: (data) => set({ compsetData: data }),
  setActiveModule: (module) => set({ activeModule: module }),
  updateScenarioParams: (params) => set((state) => ({ 
    scenarioParams: { ...state.scenarioParams, ...params } 
  })),
  resetData: () => set({ 
    isDataLoaded: false, 
    rawData: null, 
    normalizedData: [], 
    forecastData: [],
    simulatedPaceData: [],
    compsetData: []
  }),
}));

export default useStore;
