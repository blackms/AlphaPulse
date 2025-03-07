import { Middleware } from '@reduxjs/toolkit';
import { io, Socket } from 'socket.io-client';
import { RootState } from '../store';

// Action types
const SOCKET_CONNECT = 'socket/connect';
const SOCKET_DISCONNECT = 'socket/disconnect';
const SOCKET_SUBSCRIBE = 'socket/subscribe';
const SOCKET_UNSUBSCRIBE = 'socket/unsubscribe';
const SOCKET_MESSAGE = 'socket/message';

// Socket instance
let socket: Socket | null = null;

/**
 * Middleware for handling WebSocket connections
 * This middleware manages the WebSocket connection lifecycle and message handling
 */
export const socketMiddleware: Middleware<{}, RootState> = ({ dispatch, getState }) => (next) => (action) => {
  const { type, payload } = action;
  
  switch (type) {
    case SOCKET_CONNECT:
      // If already connected, do nothing
      if (socket) {
        break;
      }
      
      // Get the WebSocket URL and token from the payload or state
      const { url, token } = payload || {};
      const authToken = token || getState().auth.token;
      
      // Create a new socket connection
      socket = io(url, {
        auth: {
          token: authToken,
        },
        reconnection: true,
        reconnectionAttempts: 5,
        reconnectionDelay: 1000,
      });
      
      // Set up event listeners
      socket.on('connect', () => {
        dispatch({ type: 'socket/connected' });
      });
      
      socket.on('disconnect', () => {
        dispatch({ type: 'socket/disconnected' });
      });
      
      socket.on('error', (error) => {
        dispatch({ type: 'socket/error', payload: error });
      });
      
      socket.on('message', (message) => {
        dispatch({
          type: SOCKET_MESSAGE,
          payload: message,
        });
      });
      
      // Set up listeners for specific channels
      ['metrics', 'alerts', 'portfolio', 'trades', 'system'].forEach((channel) => {
        socket.on(channel, (data) => {
          dispatch({
            type: `${channel}/update`,
            payload: data,
          });
        });
      });
      
      break;
      
    case SOCKET_DISCONNECT:
      if (socket) {
        socket.disconnect();
        socket = null;
      }
      break;
      
    case SOCKET_SUBSCRIBE:
      if (socket) {
        const { channel, params } = payload;
        socket.emit('subscribe', { channel, params });
      }
      break;
      
    case SOCKET_UNSUBSCRIBE:
      if (socket) {
        const { channel } = payload;
        socket.emit('unsubscribe', { channel });
      }
      break;
      
    default:
      // If the action has a socket emit property, send the message
      if (action.meta && action.meta.socket && socket) {
        const { event, data } = action.meta.socket;
        socket.emit(event, data);
      }
      break;
  }
  
  // Pass the action to the next middleware
  return next(action);
};

// Action creators
export const connectSocket = (url: string, token?: string) => ({
  type: SOCKET_CONNECT,
  payload: { url, token },
});

export const disconnectSocket = () => ({
  type: SOCKET_DISCONNECT,
});

export const subscribeToChannel = (channel: string, params?: any) => ({
  type: SOCKET_SUBSCRIBE,
  payload: { channel, params },
});

export const unsubscribeFromChannel = (channel: string) => ({
  type: SOCKET_UNSUBSCRIBE,
  payload: { channel },
});