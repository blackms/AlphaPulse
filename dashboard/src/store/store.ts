import { configureStore } from '@reduxjs/toolkit';
import alertsReducer from './slices/alertsSlice';
import authReducer from './slices/authSlice';
import metricsReducer from './slices/metricsSlice';
import portfolioReducer from './slices/portfolioSlice';
import systemReducer from './slices/systemSlice';
import tradingReducer from './slices/tradingSlice';
import uiReducer from './slices/uiSlice';

// Create the store with all reducers
const store = configureStore({
  reducer: {
    alerts: alertsReducer,
    auth: authReducer,
    metrics: metricsReducer,
    portfolio: portfolioReducer,
    system: systemReducer,
    trading: tradingReducer,
    ui: uiReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these fields
        ignoredActions: ['persist/PERSIST'],
        ignoredActionPaths: ['meta.arg', 'payload.timestamp'],
        ignoredPaths: ['items.dates'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export default store;