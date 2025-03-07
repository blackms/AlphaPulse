import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type ThemeMode = 'light' | 'dark';

interface UiState {
  theme: ThemeMode;
  sidebarOpen: boolean;
  compactMode: boolean;
  notifications: boolean;
  loading: {
    dashboard: boolean;
    portfolio: boolean;
    trading: boolean;
    alerts: boolean;
    system: boolean;
  };
}

const initialState: UiState = {
  theme: 'light',
  sidebarOpen: true,
  compactMode: false,
  notifications: true,
  loading: {
    dashboard: false,
    portfolio: false,
    trading: false,
    alerts: false,
    system: false,
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<ThemeMode>) => {
      state.theme = action.payload;
    },
    toggleTheme: (state) => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    setCompactMode: (state, action: PayloadAction<boolean>) => {
      state.compactMode = action.payload;
    },
    toggleCompactMode: (state) => {
      state.compactMode = !state.compactMode;
    },
    setNotifications: (state, action: PayloadAction<boolean>) => {
      state.notifications = action.payload;
    },
    setLoading: (state, action: PayloadAction<{
      section: keyof UiState['loading'];
      loading: boolean;
    }>) => {
      state.loading[action.payload.section] = action.payload.loading;
    },
  },
});

export const {
  setTheme,
  toggleTheme,
  toggleSidebar,
  setSidebarOpen,
  setCompactMode,
  toggleCompactMode,
  setNotifications,
  setLoading,
} = uiSlice.actions;

// Selectors
export const selectTheme = (state: RootState) => state.ui.theme;
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectCompactMode = (state: RootState) => state.ui.compactMode;
export const selectNotifications = (state: RootState) => state.ui.notifications;
export const selectIsLoading = (section: keyof UiState['loading']) => 
  (state: RootState) => state.ui.loading[section];

export default uiSlice.reducer;