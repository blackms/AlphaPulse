import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../rootReducer';

export enum ThemeMode {
  LIGHT = 'light',
  DARK = 'dark',
  SYSTEM = 'system'
}

export enum DisplayDensity {
  COMFORTABLE = 'comfortable',
  COMPACT = 'compact',
  DEFAULT = 'default'
}

export enum ChartStyle {
  MODERN = 'modern',
  CLASSIC = 'classic',
  MINIMAL = 'minimal'
}

interface UiState {
  darkMode: boolean;
  sidebarOpen: boolean;
  sidebarSize: 'normal' | 'compact';
  themeMode: ThemeMode;
  displayDensity: DisplayDensity;
  chartStyle: ChartStyle;
  animationsEnabled: boolean;
  notifications: {
    show: boolean;
    message: string;
    type: 'success' | 'error' | 'info' | 'warning';
  };
  confirmDialog: {
    open: boolean;
    title: string;
    message: string;
    confirmText: string;
    cancelText: string;
    onConfirm: (() => void) | null;
  };
}

const initialState: UiState = {
  darkMode: false,
  sidebarOpen: true,
  sidebarSize: 'normal',
  themeMode: ThemeMode.LIGHT,
  displayDensity: DisplayDensity.DEFAULT,
  chartStyle: ChartStyle.MODERN,
  animationsEnabled: true,
  notifications: {
    show: false,
    message: '',
    type: 'info'
  },
  confirmDialog: {
    open: false,
    title: '',
    message: '',
    confirmText: 'Confirm',
    cancelText: 'Cancel',
    onConfirm: null
  }
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleDarkMode(state) {
      state.darkMode = !state.darkMode;
    },
    setDarkMode(state, action: PayloadAction<boolean>) {
      state.darkMode = action.payload;
    },
    toggleSidebar(state) {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen(state, action: PayloadAction<boolean>) {
      state.sidebarOpen = action.payload;
    },
    setSidebarSize(state, action: PayloadAction<'normal' | 'compact'>) {
      state.sidebarSize = action.payload;
    },
    setThemeMode(state, action: PayloadAction<ThemeMode>) {
      state.themeMode = action.payload;
    },
    setDisplayDensity(state, action: PayloadAction<DisplayDensity>) {
      state.displayDensity = action.payload;
    },
    setChartStyle(state, action: PayloadAction<ChartStyle>) {
      state.chartStyle = action.payload;
    },
    toggleAnimations(state) {
      state.animationsEnabled = !state.animationsEnabled;
    },
    showNotification(state, action: PayloadAction<{ message: string; type: 'success' | 'error' | 'info' | 'warning' }>) {
      state.notifications = {
        show: true,
        message: action.payload.message,
        type: action.payload.type
      };
    },
    hideNotification(state) {
      state.notifications.show = false;
    },
    showConfirmDialog(state, action: PayloadAction<{
      title: string;
      message: string;
      confirmText?: string;
      cancelText?: string;
      onConfirm?: () => void;
    }>) {
      state.confirmDialog = {
        open: true,
        title: action.payload.title,
        message: action.payload.message,
        confirmText: action.payload.confirmText || 'Confirm',
        cancelText: action.payload.cancelText || 'Cancel',
        onConfirm: action.payload.onConfirm || null
      };
    },
    hideConfirmDialog(state) {
      state.confirmDialog.open = false;
    }
  }
});

export const {
  toggleDarkMode,
  setDarkMode,
  toggleSidebar,
  setSidebarOpen,
  setSidebarSize,
  setThemeMode,
  setDisplayDensity,
  setChartStyle,
  toggleAnimations,
  showNotification,
  hideNotification,
  showConfirmDialog,
  hideConfirmDialog
} = uiSlice.actions;

// Selectors
export const selectDarkMode = (state: RootState) => state.ui.darkMode;
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectSidebarSize = (state: RootState) => state.ui.sidebarSize;
export const selectThemeMode = (state: RootState) => state.ui.themeMode;
export const selectDisplayDensity = (state: RootState) => state.ui.displayDensity;
export const selectChartStyle = (state: RootState) => state.ui.chartStyle;
export const selectAnimationsEnabled = (state: RootState) => state.ui.animationsEnabled;
export const selectNotification = (state: RootState) => state.ui.notifications;
export const selectConfirmDialog = (state: RootState) => state.ui.confirmDialog;

export default uiSlice.reducer;
