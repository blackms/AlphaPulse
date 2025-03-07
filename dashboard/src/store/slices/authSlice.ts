import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface UserData {
  id: string;
  username: string;
  email: string;
  firstName?: string;
  lastName?: string;
  avatar?: string;
  role: 'admin' | 'user' | 'viewer';
}

interface AuthState {
  isAuthenticated: boolean;
  user: UserData | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

const initialState: AuthState = {
  isAuthenticated: localStorage.getItem('isAuthenticated') === 'true',
  user: null,
  token: localStorage.getItem('token') || null,
  loading: false,
  error: null,
};

const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    loginStart(state) {
      state.loading = true;
      state.error = null;
    },
    loginSuccess(state, action: PayloadAction<{ user: UserData; token: string }>) {
      state.isAuthenticated = true;
      state.user = action.payload.user;
      state.token = action.payload.token;
      state.loading = false;
      state.error = null;
      
      // Save to localStorage
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('token', action.payload.token);
    },
    loginFailure(state, action: PayloadAction<string>) {
      state.loading = false;
      state.error = action.payload;
    },
    logout(state) {
      state.isAuthenticated = false;
      state.user = null;
      state.token = null;
      
      // Clear localStorage
      localStorage.removeItem('isAuthenticated');
      localStorage.removeItem('token');
    },
    updateUserProfile(state, action: PayloadAction<Partial<UserData>>) {
      if (state.user) {
        state.user = { ...state.user, ...action.payload };
      }
    },
  },
});

export const { loginStart, loginSuccess, loginFailure, logout, updateUserProfile } = authSlice.actions;

export default authSlice.reducer;