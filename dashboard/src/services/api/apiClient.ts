import axios, { AxiosInstance, InternalAxiosRequestConfig } from 'axios';
import authService from '../auth/authService';

// Create a base axios instance
const client: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  },
  timeout: 10000 // 10 seconds
});

// Add request interceptor for authentication
client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = authService.getToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
client.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    // Handle authentication errors
    if (error.response && error.response.status === 401) {
      // Redirect to login or refresh token
      authService.logout();
      window.location.href = '/login';
    }
    
    // Handle server errors
    if (error.response && error.response.status >= 500) {
      console.error('Server error:', error.response.data);
    }
    
    return Promise.reject(error);
  }
);

export default client;