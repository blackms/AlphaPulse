import axios from 'axios';
import jwtDecode from 'jwt-decode';
import { User } from '../../types';
import { UserRole } from '../../store/slices/authSlice';

// API URL from environment variables
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

/**
 * Authentication service for handling login, logout, and token management
 */
class AuthService {
  /**
   * Login with username and password
   * @param username - User's username
   * @param password - User's password
   * @returns Promise with login response
   */
  async login(username: string, password: string) {
    try {
      // Use the correct endpoint from the API server output
      // Create form data for token endpoint
      const formData = new URLSearchParams();
      formData.append('username', username);
      formData.append('password', password);
      
      const response = await axios.post(`${API_URL.replace('/api/v1', '')}/token`,
        formData.toString(),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );
      
      if (response.data.access_token) {
        localStorage.setItem('token', response.data.access_token);
        
        // Create a user object from the token data
        const user = {
          id: response.data.user_id || 'user1',
          username: username,
          email: `${username}@example.com`,
          firstName: 'Admin',
          lastName: 'User',
          role: 'admin' as UserRole,
          permissions: ['*'],
          preferences: {
            theme: 'dark' as 'dark' | 'light' | 'system',
            dashboardLayout: 'default',
            notifications: true,
            alertSounds: true,
            defaultTimeframe: '1d',
            defaultAssets: ['BTC', 'ETH', 'SOL'],
            timezone: 'UTC',
            language: 'en'
          },
          lastLogin: Date.now(),
          createdAt: Date.now() - 90 * 24 * 60 * 60 * 1000 // 90 days ago
        };
        
        localStorage.setItem('user', JSON.stringify(user));
        
        return {
          user,
          accessToken: response.data.access_token,
          refreshToken: response.data.refresh_token || '',
          expiresAt: Date.now() + (response.data.expires_in || 3600) * 1000
        };
      }
      
      throw new Error('Invalid response from server');
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }
  
  /**
   * Logout the current user
   */
  logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  }
  
  /**
   * Register a new user
   * @param username - User's username
   * @param email - User's email
   * @param password - User's password
   * @returns Promise with registration response
   */
  async register(username: string, email: string, password: string) {
    return axios.post(`${API_URL}/auth/register`, {
      username,
      email,
      password,
    });
  }
  
  /**
   * Get the current user
   * @returns The current user or null
   */
  getCurrentUser(): User | null {
    const userStr = localStorage.getItem('user');
    if (userStr) {
      return JSON.parse(userStr);
    }
    return null;
  }
  
  /**
   * Get the current token
   * @returns The current token or null
   */
  getToken(): string | null {
    return localStorage.getItem('token');
  }
  
  /**
   * Check if the token is valid
   * @returns True if the token is valid, false otherwise
   */
  isTokenValid(): boolean {
    const token = this.getToken();
    if (!token) {
      return false;
    }
    
    try {
      const decoded: any = jwtDecode(token);
      const currentTime = Date.now() / 1000;
      
      // Check if token is expired
      if (decoded.exp < currentTime) {
        return false;
      }
      
      return true;
    } catch (error) {
      return false;
    }
  }
  
  /**
   * Refresh the token
   * @returns Promise with refresh response
   */
  async refreshToken() {
    const token = this.getToken();
    if (!token) {
      throw new Error('No token found');
    }
    
    const response = await axios.post(
      `${API_URL}/auth/refresh`,
      {},
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    );
    
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
    }
    
    return response.data;
  }
}

export default new AuthService();