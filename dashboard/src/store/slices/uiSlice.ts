import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

// Define the UI state interface
interface UIState {
  sidebarOpen: boolean;
  darkMode: boolean;
  notifications: Notification[];
  currentTheme: string;
  isMobile: boolean;
  modalOpen: {
    [key: string]: boolean;
  };
}

// Define the notification interface
interface Notification {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info' | 'warning';
  autoHide: boolean;
  duration?: number;
}

// Initial state
const initialState: UIState = {
  sidebarOpen: true,
  darkMode: localStorage.getItem('darkMode') === 'true',
  notifications: [],
  currentTheme: localStorage.getItem('theme') || 'default',
  isMobile: window.innerWidth < 768,
  modalOpen: {},
};

// Create the UI slice
const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    toggleDarkMode: (state) => {
      state.darkMode = !state.darkMode;
      localStorage.setItem('darkMode', state.darkMode.toString());
    },
    setDarkMode: (state, action: PayloadAction<boolean>) => {
      state.darkMode = action.payload;
      localStorage.setItem('darkMode', action.payload.toString());
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id'>>) => {
      const id = Date.now().toString();
      state.notifications.push({
        ...action.payload,
        id,
      });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(
        (notification) => notification.id !== action.payload
      );
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setCurrentTheme: (state, action: PayloadAction<string>) => {
      state.currentTheme = action.payload;
      localStorage.setItem('theme', action.payload);
    },
    setIsMobile: (state, action: PayloadAction<boolean>) => {
      state.isMobile = action.payload;
    },
    openModal: (state, action: PayloadAction<string>) => {
      state.modalOpen[action.payload] = true;
    },
    closeModal: (state, action: PayloadAction<string>) => {
      state.modalOpen[action.payload] = false;
    },
  },
});

// Export actions
export const {
  toggleSidebar,
  setSidebarOpen,
  toggleDarkMode,
  setDarkMode,
  addNotification,
  removeNotification,
  clearNotifications,
  setCurrentTheme,
  setIsMobile,
  openModal,
  closeModal,
} = uiSlice.actions;

// Export selectors
export const selectSidebarOpen = (state: RootState) => state.ui.sidebarOpen;
export const selectDarkMode = (state: RootState) => state.ui.darkMode;
export const selectNotifications = (state: RootState) => state.ui.notifications;
export const selectCurrentTheme = (state: RootState) => state.ui.currentTheme;
export const selectIsMobile = (state: RootState) => state.ui.isMobile;
export const selectModalOpen = (state: RootState, modalId: string) => state.ui.modalOpen[modalId] || false;

// Export reducer
export default uiSlice.reducer;