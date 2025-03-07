import { Middleware } from 'redux';
import { Socket, io } from 'socket.io-client';
import { RootState } from '../rootReducer';

// Socket middleware for Redux
const socketMiddleware = (): Middleware => {
  let socket: Socket | null = null;

  return ({ dispatch, getState }) => next => action => {
    const { type, payload } = action;

    // Handle socket connection
    if (type === 'socket/connect') {
      // Close existing socket if it exists
      if (socket) {
        socket.close();
      }

      // Get the WebSocket URL and token from the payload or state
      const { url, token } = payload || {};
      const authToken = token || getState().auth.tokens?.accessToken;
      
      // Create a new socket connection
      socket = io(url, {
        transports: ['websocket'],
        auth: {
          token: authToken
        }
      });

      // Set up connection event handlers
      socket.on('connect', () => {
        dispatch({
          type: 'socket/connected',
          payload: { connected: true }
        });
      });

      socket.on('disconnect', (reason) => {
        dispatch({
          type: 'socket/disconnected',
          payload: { reason }
        });
      });

      socket.on('error', (error) => {
        dispatch({
          type: 'socket/error',
          payload: { error }
        });
      });

      // Set up listeners for specific channels
      ['metrics', 'alerts', 'portfolio', 'trades', 'system'].forEach((channel) => {
        if (socket) {
          socket.on(channel, (data) => {
            dispatch({
              type: `${channel}/update`,
              payload: data,
            });
          });
        }
      });

      return;
    }

    // Handle socket disconnection
    if (type === 'socket/disconnect' && socket) {
      socket.close();
      socket = null;
      return;
    }

    // Handle sending messages
    if (type === 'socket/send' && socket) {
      const { channel, data } = payload;
      socket.emit(channel, data);
      return;
    }

    return next(action);
  };
};

export default socketMiddleware;