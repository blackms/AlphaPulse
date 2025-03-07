import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../store';

// Define the authentication state interface
interface AuthState {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

// Define the user interface
interface User {
  id: string;
  username: string;
  email: string;
  role: string;
}

// Define the login payload interface
interface LoginPayload {
  token: string;
  user: User;
}

// Initial state
const initialState: AuthState = {
  token: localStorage.getItem('token'),
  user: JSON.parse(localStorage.getItem('user') || 'null'),
  isAuthenticated: !!localStorage.getItem('token'),
  isLoading: false,
  error: null,
};

// Create the auth slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginRequest: (state) => {
      state.isLoading = true;
      state.error = null;
    },
    loginSuccess: (state, action: PayloadAction<LoginPayload>) => {
      state.isLoading = false;
      state.isAuthenticated = true;
      state.token = action.payload.token;
      state.user = action.payload.user;
      state.error = null;
      
      // Store token and user in localStorage
      localStorage.setItem('token', action.payload.token);
      localStorage.setItem('user', JSON.stringify(action.payload.user));
    },
    loginFailure: (state, action: PayloadAction<string>) => {
      state.isLoading = false;
      state.isAuthenticated = false;
      state.token = null;
      state.user = null;
      state.error = action.payload;
      
      // Clear localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    },
    logout: (state) => {
      state.isAuthenticated = false;
      state.token = null;
      state.user = null;
      state.error = null;
      
      // Clear localStorage
      localStorage.removeItem('token');
      localStorage.removeItem('user');
    },
    refreshToken: (state, action: PayloadAction<string>) => {
      state.token = action.payload;
      
      // Update token in localStorage
      localStorage.setItem('token', action.payload);
    },
    updateUser: (state, action: PayloadAction<User>) => {
      state.user = action.payload;
      
      // Update user in localStorage
      localStorage.setItem('user', JSON.stringify(action.payload));
    },
    clearError: (state) => {
      state.error = null;
    },
  },
});

// Export actions
export const {
  loginRequest,
  loginSuccess,
  loginFailure,
  logout,
  refreshToken,
  updateUser,
  clearError,
} = authSlice.actions;

// Export selectors
export const selectAuth = (state: RootState) => state.auth;
export const selectIsAuthenticated = (state: RootState) => state.auth.isAuthenticated;
export const selectUser = (state: RootState) => state.auth.user;
export const selectToken = (state: RootState) => state.auth.token;
export const selectAuthError = (state: RootState) => state.auth.error;
export const selectIsLoading = (state: RootState) => state.auth.isLoading;

// Export reducer
export default authSlice.reducer;