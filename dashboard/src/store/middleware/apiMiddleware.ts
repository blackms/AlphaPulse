import { Middleware } from '@reduxjs/toolkit';
import { RootState } from '../store';

/**
 * Middleware for handling API requests
 * This middleware intercepts actions with an API request and handles the request lifecycle
 */
export const apiMiddleware: Middleware<{}, RootState> = ({ dispatch, getState }) => (next) => (action) => {
  // If the action doesn't have an API request, pass it to the next middleware
  if (!action.meta || !action.meta.api) {
    return next(action);
  }

  // Extract API request details from the action
  const { url, method, data, onSuccess, onError, headers } = action.meta.api;
  
  // Get the authentication token from the state
  const token = getState().auth.token;
  
  // Create headers with authentication token
  const requestHeaders = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...headers,
  };
  
  // Dispatch the request start action
  dispatch({ type: `${action.type}_REQUEST` });
  
  // Make the API request
  fetch(url, {
    method: method || 'GET',
    headers: requestHeaders,
    body: data ? JSON.stringify(data) : undefined,
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }
      return response.json();
    })
    .then((responseData) => {
      // Dispatch the success action
      dispatch({
        type: `${action.type}_SUCCESS`,
        payload: responseData,
      });
      
      // Call the onSuccess callback if provided
      if (onSuccess) {
        dispatch(onSuccess(responseData));
      }
    })
    .catch((error) => {
      // Dispatch the error action
      dispatch({
        type: `${action.type}_FAILURE`,
        error: error.message,
      });
      
      // Call the onError callback if provided
      if (onError) {
        dispatch(onError(error));
      }
    });
  
  // Return the original action
  return next(action);
};