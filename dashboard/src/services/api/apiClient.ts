import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import authService from '../auth/authService';

// API URL from environment variables
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

/**
 * Create a configured axios instance
 */
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_URL,
    headers: {
      'Content-Type': 'application/json',
    },
  });
  
  // Request interceptor for adding token to requests
  client.interceptors.request.use(
    (config: AxiosRequestConfig) => {
      const token = authService.getToken();
      if (token && config.headers) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => Promise.reject(error)
  );
  
  // Response interceptor for handling token refresh
  client.interceptors.response.use(
    (response: AxiosResponse) => response,
    async (error) => {
      const originalRequest = error.config;
      
      // If the error is unauthorized and not already retrying
      if (error.response?.status === 401 && !originalRequest._retry) {
        originalRequest._retry = true;
        
        try {
          // Try to refresh the token
          await authService.refreshToken();
          
          // Update authorization header
          const token = authService.getToken();
          if (token) {
            originalRequest.headers.Authorization = `Bearer ${token}`;
          }
          
          // Retry the original request
          return client(originalRequest);
        } catch (refreshError) {
          // If refresh fails, redirect to login
          authService.logout();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      }
      
      return Promise.reject(error);
    }
  );
  
  return client;
};

// Create and export the API client
const apiClient = createApiClient();

export default apiClient;

/**
 * Generic API request function
 * @param method - HTTP method
 * @param url - API endpoint
 * @param data - Request data
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const apiRequest = async <T>(
  method: string,
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<T> => {
  try {
    const response = await apiClient.request<T>({
      method,
      url,
      data,
      ...config,
    });
    
    return response.data;
  } catch (error: any) {
    // Handle error
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      throw new Error(error.response.data.message || 'API request failed');
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('No response from server');
    } else {
      // Something happened in setting up the request that triggered an Error
      throw new Error('Error setting up request');
    }
  }
};

/**
 * GET request
 * @param url - API endpoint
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const get = <T>(url: string, config?: AxiosRequestConfig): Promise<T> => {
  return apiRequest<T>('GET', url, undefined, config);
};

/**
 * POST request
 * @param url - API endpoint
 * @param data - Request data
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const post = <T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> => {
  return apiRequest<T>('POST', url, data, config);
};

/**
 * PUT request
 * @param url - API endpoint
 * @param data - Request data
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const put = <T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> => {
  return apiRequest<T>('PUT', url, data, config);
};

/**
 * PATCH request
 * @param url - API endpoint
 * @param data - Request data
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const patch = <T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<T> => {
  return apiRequest<T>('PATCH', url, data, config);
};

/**
 * DELETE request
 * @param url - API endpoint
 * @param config - Additional axios config
 * @returns Promise with response data
 */
export const del = <T>(url: string, config?: AxiosRequestConfig): Promise<T> => {
  return apiRequest<T>('DELETE', url, undefined, config);
};