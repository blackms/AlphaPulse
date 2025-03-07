import { configureStore } from '@reduxjs/toolkit';
import rootReducer from './rootReducer';
import { apiMiddleware } from './middleware/apiMiddleware';
import { socketMiddleware } from './middleware/socketMiddleware';

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(apiMiddleware, socketMiddleware),
  devTools: process.env.NODE_ENV !== 'production',
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;