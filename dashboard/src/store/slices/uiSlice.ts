import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type ThemeMode = 'light' | 'dark' | 'system';
export type SidebarSize = 'expanded' | 'collapsed' | 'hidden';
export type DisplayDensity = 'comfortable' | 'compact' | 'spacious';
export type ChartStyle = 'line' | 'candle' | 'area' | 'bar';
export type DateRangePreset = '1D' | '1W' | '1M' | '3M' | '6M' | '1Y' | 'YTD' | 'ALL';

interface UIState {
  // Layout settings
  sidebarSize: SidebarSize;
  isMobileMenuOpen: boolean;
  currentPage: string;
  pageHistory: string[];
  breadcrumbs: string[];
  
  // Appearance settings
  themeMode: ThemeMode;
  displayDensity: DisplayDensity;
  animationsEnabled: boolean;
  chartStyle: ChartStyle;
  
  // Dashboard settings
  activeDashboardId: string | null;
  dashboardLayout: Record<string, any>;
  
  // Filter settings
  dateRange: {
    preset: DateRangePreset;
    start: string | null;
    end: string | null;
  };
  selectedAssets: string[];
  
  // Modal state
  activeModal: string | null;
  modalData: Record<string, any> | null;
  
  // Tour and help
  showTutorial: boolean;
  tourStep: number;
  showHelp: boolean;
}

const initialState: UIState = {
  // Layout settings
  sidebarSize: 'expanded',
  isMobileMenuOpen: false,
  currentPage: 'dashboard',
  pageHistory: ['dashboard'],
  breadcrumbs: ['Home', 'Dashboard'],
  
  // Appearance settings
  themeMode: 'dark',
  displayDensity: 'comfortable',
  animationsEnabled: true,
  chartStyle: 'candle',
  
  // Dashboard settings
  activeDashboardId: 'default',
  dashboardLayout: {
    default: {
      widgets: [
        { id: 'portfolio-overview', position: { x: 0, y: 0, w: 6, h: 4 } },
        { id: 'asset-allocation', position: { x: 6, y: 0, w: 6, h: 4 } },
        { id: 'recent-trades', position: { x: 0, y: 4, w: 8, h: 6 } },
        { id: 'agent-signals', position: { x: 8, y: 4, w: 4, h: 6 } },
        { id: 'system-health', position: { x: 0, y: 10, w: 12, h: 3 } }
      ]
    }
  },
  
  // Filter settings
  dateRange: {
    preset: '1M',
    start: null,
    end: null
  },
  selectedAssets: ['BTC', 'ETH', 'SOL'],
  
  // Modal state
  activeModal: null,
  modalData: null,
  
  // Tour and help
  showTutorial: false,
  tourStep: 0,
  showHelp: false
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // Layout actions
    setSidebarSize: (state, action: PayloadAction<SidebarSize>) => {
      state.sidebarSize = action.payload;
    },
    toggleMobileMenu: (state) => {
      state.isMobileMenuOpen = !state.isMobileMenuOpen;
    },
    setMobileMenuOpen: (state, action: PayloadAction<boolean>) => {
      state.isMobileMenuOpen = action.payload;
    },
    navigateTo: (state, action: PayloadAction<string>) => {
      // Don't add to history if it's the same page
      if (state.currentPage !== action.payload) {
        state.pageHistory.push(action.payload);
        if (state.pageHistory.length > 10) {
          state.pageHistory.shift();
        }
      }
      state.currentPage = action.payload;
    },
    goBack: (state) => {
      if (state.pageHistory.length > 1) {
        state.pageHistory.pop(); // Remove current page
        state.currentPage = state.pageHistory[state.pageHistory.length - 1];
      }
    },
    setBreadcrumbs: (state, action: PayloadAction<string[]>) => {
      state.breadcrumbs = action.payload;
    },
    
    // Appearance actions
    setThemeMode: (state, action: PayloadAction<ThemeMode>) => {
      state.themeMode = action.payload;
    },
    setDisplayDensity: (state, action: PayloadAction<DisplayDensity>) => {
      state.displayDensity = action.payload;
    },
    toggleAnimations: (state) => {
      state.animationsEnabled = !state.animationsEnabled;
    },
    setChartStyle: (state, action: PayloadAction<ChartStyle>) => {
      state.chartStyle = action.payload;
    },
    
    // Dashboard actions
    setActiveDashboard: (state, action: PayloadAction<string>) => {
      state.activeDashboardId = action.payload;
    },
    updateDashboardLayout: (state, action: PayloadAction<{
      dashboardId: string;
      layout: any;
    }>) => {
      state.dashboardLayout[action.payload.dashboardId] = action.payload.layout;
    },
    addDashboardWidget: (state, action: PayloadAction<{
      dashboardId: string;
      widget: { id: string; position: { x: number; y: number; w: number; h: number } }
    }>) => {
      if (!state.dashboardLayout[action.payload.dashboardId]) {
        state.dashboardLayout[action.payload.dashboardId] = { widgets: [] };
      }
      
      state.dashboardLayout[action.payload.dashboardId].widgets.push(action.payload.widget);
    },
    removeDashboardWidget: (state, action: PayloadAction<{
      dashboardId: string;
      widgetId: string;
    }>) => {
      if (state.dashboardLayout[action.payload.dashboardId]) {
        state.dashboardLayout[action.payload.dashboardId].widgets = 
          state.dashboardLayout[action.payload.dashboardId].widgets.filter(
            (widget: any) => widget.id !== action.payload.widgetId
          );
      }
    },
    
    // Filter actions
    setDateRangePreset: (state, action: PayloadAction<DateRangePreset>) => {
      state.dateRange.preset = action.payload;
      // Clear custom dates when using preset
      state.dateRange.start = null;
      state.dateRange.end = null;
    },
    setCustomDateRange: (state, action: PayloadAction<{
      start: string;
      end: string;
    }>) => {
      state.dateRange.preset = 'ALL'; // Set to custom
      state.dateRange.start = action.payload.start;
      state.dateRange.end = action.payload.end;
    },
    selectAsset: (state, action: PayloadAction<string>) => {
      if (!state.selectedAssets.includes(action.payload)) {
        state.selectedAssets.push(action.payload);
      }
    },
    deselectAsset: (state, action: PayloadAction<string>) => {
      state.selectedAssets = state.selectedAssets.filter(
        asset => asset !== action.payload
      );
    },
    setSelectedAssets: (state, action: PayloadAction<string[]>) => {
      state.selectedAssets = action.payload;
    },
    
    // Modal actions
    openModal: (state, action: PayloadAction<{
      modalId: string;
      data?: Record<string, any>;
    }>) => {
      state.activeModal = action.payload.modalId;
      state.modalData = action.payload.data || null;
    },
    closeModal: (state) => {
      state.activeModal = null;
      state.modalData = null;
    },
    updateModalData: (state, action: PayloadAction<Record<string, any>>) => {
      if (state.modalData) {
        state.modalData = {
          ...state.modalData,
          ...action.payload
        };
      }
    },
    
    // Tour and help actions
    startTutorial: (state) => {
      state.showTutorial = true;
      state.tourStep = 0;
    },
    setTutorialStep: (state, action: PayloadAction<number>) => {
      state.tourStep = action.payload;
    },
    endTutorial: (state) => {
      state.showTutorial = false;
      state.tourStep = 0;
    },
    toggleHelp: (state) => {
      state.showHelp = !state.showHelp;
    }
  }
});

export const {
  // Layout actions
  setSidebarSize,
  toggleMobileMenu,
  setMobileMenuOpen,
  navigateTo,
  goBack,
  setBreadcrumbs,
  
  // Appearance actions
  setThemeMode,
  setDisplayDensity,
  toggleAnimations,
  setChartStyle,
  
  // Dashboard actions
  setActiveDashboard,
  updateDashboardLayout,
  addDashboardWidget,
  removeDashboardWidget,
  
  // Filter actions
  setDateRangePreset,
  setCustomDateRange,
  selectAsset,
  deselectAsset,
  setSelectedAssets,
  
  // Modal actions
  openModal,
  closeModal,
  updateModalData,
  
  // Tour and help actions
  startTutorial,
  setTutorialStep,
  endTutorial,
  toggleHelp
} = uiSlice.actions;

// Selectors
export const selectSidebarSize = (state: RootState) => state.ui.sidebarSize;
export const selectIsMobileMenuOpen = (state: RootState) => state.ui.isMobileMenuOpen;
export const selectCurrentPage = (state: RootState) => state.ui.currentPage;
export const selectBreadcrumbs = (state: RootState) => state.ui.breadcrumbs;

export const selectThemeMode = (state: RootState) => state.ui.themeMode;
export const selectDisplayDensity = (state: RootState) => state.ui.displayDensity;
export const selectAnimationsEnabled = (state: RootState) => state.ui.animationsEnabled;
export const selectChartStyle = (state: RootState) => state.ui.chartStyle;

export const selectActiveDashboardId = (state: RootState) => state.ui.activeDashboardId;
export const selectDashboardLayout = (dashboardId: string) => 
  (state: RootState) => state.ui.dashboardLayout[dashboardId];
export const selectAllDashboardLayouts = (state: RootState) => state.ui.dashboardLayout;

export const selectDateRange = (state: RootState) => state.ui.dateRange;
export const selectSelectedAssets = (state: RootState) => state.ui.selectedAssets;

export const selectActiveModal = (state: RootState) => state.ui.activeModal;
export const selectModalData = (state: RootState) => state.ui.modalData;

export const selectShowTutorial = (state: RootState) => state.ui.showTutorial;
export const selectTourStep = (state: RootState) => state.ui.tourStep;
export const selectShowHelp = (state: RootState) => state.ui.showHelp;

export default uiSlice.reducer;