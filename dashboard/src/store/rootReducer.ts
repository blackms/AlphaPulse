import { combineReducers } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import alertsReducer from './slices/alertsSlice';
import metricsReducer from './slices/metricsSlice';
import portfolioReducer from './slices/portfolioSlice';
import tradingReducer from './slices/tradingSlice';
import systemReducer from './slices/systemSlice';
import uiReducer from './slices/uiSlice';

const rootReducer = combineReducers({
  auth: authReducer,
  alerts: alertsReducer,
  metrics: metricsReducer,
  portfolio: portfolioReducer,
  trading: tradingReducer,
  system: systemReducer,
  ui: uiReducer,
});

export default rootReducer;