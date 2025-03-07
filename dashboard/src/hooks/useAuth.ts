import { useCallback } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { 
  loginRequest, 
  loginSuccess, 
  loginFailure, 
  logout as logoutAction,
  selectIsAuthenticated,
  selectUser,
  selectToken,
  selectAuthError,
  selectIsLoading
} from '../store/slices/authSlice';
import authService from '../services/auth/authService';
import { LoginResponse, User } from '../types';

/**
 * Hook for authentication functionality
 */
export const useAuth = () => {
  const dispatch = useDispatch();
  const isAuthenticated = useSelector(selectIsAuthenticated);
  const user = useSelector(selectUser);
  const token = useSelector(selectToken);
  const error = useSelector(selectAuthError);
  const isLoading = useSelector(selectIsLoading);
  
  /**
   * Login with username and password
   */
  const login = useCallback(
    async (username: string, password: string) => {
      try {
        dispatch(loginRequest());
        const response = await authService.login(username, password);
        dispatch(loginSuccess(response as LoginResponse));
        return response;
      } catch (error: any) {
        const errorMessage = error.response?.data?.message || 'Login failed';
        dispatch(loginFailure(errorMessage));
        throw error;
      }
    },
    [dispatch]
  );
  
  /**
   * Logout the current user
   */
  const logout = useCallback(() => {
    authService.logout();
    dispatch(logoutAction());
  }, [dispatch]);
  
  /**
   * Check if the user is authenticated
   */
  const checkAuth = useCallback(async () => {
    if (!token) {
      return false;
    }
    
    if (!authService.isTokenValid()) {
      try {
        // Try to refresh the token
        await authService.refreshToken();
        return true;
      } catch (error) {
        // If refresh fails, logout
        logout();
        return false;
      }
    }
    
    return true;
  }, [token, logout]);
  
  /**
   * Get the current user
   */
  const getCurrentUser = useCallback((): User | null => {
    return authService.getCurrentUser();
  }, []);
  
  return {
    isAuthenticated,
    user,
    token,
    error,
    isLoading,
    login,
    logout,
    checkAuth,
    getCurrentUser,
  };
};