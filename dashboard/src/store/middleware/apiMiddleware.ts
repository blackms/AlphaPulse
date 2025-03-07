import { Middleware } from 'redux';
import axios, { AxiosRequestConfig } from 'axios';
import { RootState } from '../rootReducer';

// API middleware for Redux
const apiMiddleware: Middleware = ({ dispatch, getState }) => next => action => {
  // Check if the action is an API call
  if (!action.meta || !action.meta.api) {
    return next(action);
  }
  
  // Extract API details from the action
  const { url, method = 'GET', data, onSuccess, onError } = action.meta.api;
  
  // Get the authentication token from the state
  const token = getState().auth.tokens?.accessToken;
  
  // Create headers with authentication token
  const requestHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };
  
  if (token) {
    requestHeaders['Authorization'] = `Bearer ${token}`;
  }
  
  // Create the request configuration
  const requestConfig: AxiosRequestConfig = {
    url,
    method,
    headers: requestHeaders,
    data
  };
  
  // Dispatch the request action
  dispatch({
    type: `${action.type}_REQUEST`
  });
  
  // Make the API call
  return axios(requestConfig)
    .then(response => {
      // Dispatch the success action
      dispatch({
        type: `${action.type}_SUCCESS`,
        payload: response.data,
        meta: action.meta
      });
      
      // Call the onSuccess callback if provided
      if (onSuccess) {
        dispatch(onSuccess(response.data));
      }
      
      return response.data;
    })
    .catch(error => {
      // Dispatch the error action
      dispatch({
        type: `${action.type}_FAILURE`,
        error: error.response ? error.response.data : error.message,
        meta: action.meta
      });
      
      // Call the onError callback if provided
      if (onError) {
        dispatch(onError(error.response ? error.response.data : error.message));
      }
      
      return Promise.reject(error);
    });
};

export default apiMiddleware;