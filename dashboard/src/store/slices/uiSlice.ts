import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UiState {
  theme: 'light' | 'dark' | 'system';
  sidebarOpen: boolean;
  drawerWidth: number;
  notifications: {
    showBadges: boolean;
    sound: boolean;
  };
  chartSettings: {
    showAnimations: boolean;
    detailedTooltips: boolean;
    defaultTimeframe: string;
  };
}

const initialState: UiState = {
  theme: 'light',
  sidebarOpen: true,
  drawerWidth: 240,
  notifications: {
    showBadges: true,
    sound: false,
  },
  chartSettings: {
    showAnimations: true,
    detailedTooltips: true,
    defaultTimeframe: '1d',
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    setTheme(state, action: PayloadAction<'light' | 'dark' | 'system'>) {
      state.theme = action.payload;
    },
    toggleSidebar(state) {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen(state, action: PayloadAction<boolean>) {
      state.sidebarOpen = action.payload;
    },
    setDrawerWidth(state, action: PayloadAction<number>) {
      state.drawerWidth = action.payload;
    },
    toggleNotificationBadges(state) {
      state.notifications.showBadges = !state.notifications.showBadges;
    },
    toggleNotificationSound(state) {
      state.notifications.sound = !state.notifications.sound;
    },
    updateChartSettings(state, action: PayloadAction<Partial<UiState['chartSettings']>>) {
      state.chartSettings = {
        ...state.chartSettings,
        ...action.payload,
      };
    },
    resetUiSettings(state) {
      return initialState;
    },
  },
});

export const {
  setTheme,
  toggleSidebar,
  setSidebarOpen,
  setDrawerWidth,
  toggleNotificationBadges,
  toggleNotificationSound,
  updateChartSettings,
  resetUiSettings,
} = uiSlice.actions;

export default uiSlice.reducer;