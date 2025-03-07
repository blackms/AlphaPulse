import axios from 'axios';
import jwtDecode from 'jwt-decode';
import { User } from '../../types';

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
    const response = await axios.post(`${API_URL}/auth/login`, {
      username,
      password,
    });
    
    if (response.data.token) {
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
    }
    
    return response.data;
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