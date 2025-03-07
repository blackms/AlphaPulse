import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

export type UserRole = 'admin' | 'trader' | 'analyst' | 'viewer';

export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  permissions: string[];
  preferences: UserPreferences;
  lastLogin: number | null;
  createdAt: number;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  dashboardLayout: string;
  notifications: boolean;
  alertSounds: boolean;
  defaultTimeframe: string;
  defaultAssets: string[];
  timezone: string;
  language: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  initializing: boolean;
}

// Initial state
const initialState: AuthState = {
  user: {
    id: 'user1',
    username: 'trader1',
    email: 'trader@example.com',
    firstName: 'John',
    lastName: 'Smith',
    role: 'trader',
    permissions: [
      'view:portfolio',
      'view:trading',
      'view:system',
      'manage:portfolio',
      'manage:trading'
    ],
    preferences: {
      theme: 'dark',
      dashboardLayout: 'default',
      notifications: true,
      alertSounds: true,
      defaultTimeframe: '1d',
      defaultAssets: ['BTC', 'ETH', 'SOL'],
      timezone: 'UTC',
      language: 'en'
    },
    lastLogin: Date.now() - 24 * 60 * 60 * 1000, // 1 day ago
    createdAt: Date.now() - 90 * 24 * 60 * 60 * 1000 // 90 days ago
  },
  tokens: {
    accessToken: 'mock-access-token',
    refreshToken: 'mock-refresh-token',
    expiresAt: Date.now() + 60 * 60 * 1000 // 1 hour from now
  },
  isAuthenticated: true,
  isLoading: false,
  error: null,
  initializing: false
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    initAuth: (state) => {
      state.initializing = true;
      state.error = null;
    },
    initAuthSuccess: (state, action: PayloadAction<{
      user: User;
      tokens: AuthTokens;
    }>) => {
      state.user = action.payload.user;
      state.tokens = action.payload.tokens;
      state.isAuthenticated = true;
      state.initializing = false;
    },
    initAuthFailure: (state) => {
      state.user = null;
      state.tokens = null;
      state.isAuthenticated = false;
      state.initializing = false;
    },
    loginStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    loginSuccess: (state, action: PayloadAction<{
      user: User;
      tokens: AuthTokens;
    }>) => {
      state.user = action.payload.user;
      state.tokens = action.payload.tokens;
      state.isAuthenticated = true;
      state.isLoading = false;
      state.user.lastLogin = Date.now();
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    logout: (state) => {
      state.user = null;
      state.tokens = null;
      state.isAuthenticated = false;
    },
    refreshTokenStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    refreshTokenSuccess: (state, action: PayloadAction<AuthTokens>) => {
      state.tokens = action.payload;
      state.isLoading = false;
    },
    refreshTokenFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
      // Don't automatically logout here - that decision should be made by the auth middleware
    },
    updateUserStart: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    updateUserSuccess: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        state.user = {
          ...state.user,
          ...action.payload
        };
      }
      state.isLoading = false;
    },
    updateUserFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.error = action.payload;
    },
    updatePreferences: (state, action: PayloadAction<Partial<UserPreferences>>) => {
      if (state.user) {
        state.user.preferences = {
          ...state.user.preferences,
          ...action.payload
        };
      }
    },
    clearError: (state) => {
      state.error = null;
    }
  }
});

export const {
  initAuth,
  initAuthSuccess,
  initAuthFailure,
  loginStart,
  loginSuccess,
  loginFailure,
  logout,
  refreshTokenStart,
  refreshTokenSuccess,
  refreshTokenFailure,
  updateUserStart,
  updateUserSuccess,
  updateUserFailure,
  updatePreferences,
  clearError
} = authSlice.actions;

// Selectors
export const selectUser = (state: RootState) => state.auth.user;
export const selectIsAuthenticated = (state: RootState) => state.auth.isAuthenticated;
export const selectAuthLoading = (state: RootState) => state.auth.isLoading;
export const selectAuthError = (state: RootState) => state.auth.error;
export const selectUserPreferences = (state: RootState) => state.auth.user?.preferences;
export const selectUserRole = (state: RootState) => state.auth.user?.role;
export const selectUserPermissions = (state: RootState) => state.auth.user?.permissions;
export const selectTokens = (state: RootState) => state.auth.tokens;
export const selectIsInitializing = (state: RootState) => state.auth.initializing;

// Helper selector to check if user has a specific permission
export const hasPermission = (permission: string) => 
  (state: RootState) => {
    const permissions = state.auth.user?.permissions;
    if (!permissions) return false;
    
    // Admin role has all permissions
    if (state.auth.user?.role === 'admin') return true;
    
    return permissions.includes(permission);
  };

export default authSlice.reducer;