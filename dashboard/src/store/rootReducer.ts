import { combineReducers } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import portfolioReducer from './slices/portfolioSlice';
import tradingReducer from './slices/tradingSlice';
import alertsReducer from './slices/alertsSlice';
import systemReducer from './slices/systemSlice';
import uiReducer from './slices/uiSlice';

const rootReducer = combineReducers({
  auth: authReducer,
  portfolio: portfolioReducer,
  trading: tradingReducer,
  alerts: alertsReducer,
  system: systemReducer,
  ui: uiReducer,
});

export type RootState = ReturnType<typeof rootReducer>;

export default rootReducer;