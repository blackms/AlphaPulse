import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import alertsReducer from './slices/alertsSlice';
import authReducer from './slices/authSlice';
import metricsReducer from './slices/metricsSlice';
import portfolioReducer from './slices/portfolioSlice';
import systemReducer from './slices/systemSlice';
import tradingReducer from './slices/tradingSlice';
import uiReducer from './slices/uiSlice';

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
        // Ignore these action types
        ignoredActions: ['alerts/addAlert'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['meta.arg', 'payload.timestamp', 'payload.actions'],
        // Ignore these paths in the state
        ignoredPaths: [
          'alerts.alerts',
          'portfolio.positions',
          'trading.signals',
          'trading.strategies',
        ],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;

export default store;