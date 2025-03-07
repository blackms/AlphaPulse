import { useEffect, useCallback, useState } from 'react';
import socketClient from '../services/socket/socketClient';
import { useAuth } from './useAuth';

/**
 * Hook for WebSocket functionality
 * @param autoConnect - Whether to automatically connect to the WebSocket server
 * @returns WebSocket functionality
 */
export const useSocket = (autoConnect = true) => {
  const [isConnected, setIsConnected] = useState(socketClient.isConnected());
  const { token } = useAuth();
  
  /**
   * Connect to the WebSocket server
   */
  const connect = useCallback(async () => {
    try {
      await socketClient.connect();
      setIsConnected(true);
    } catch (error) {
      console.error('Failed to connect to WebSocket server:', error);
      setIsConnected(false);
    }
  }, []);
  
  /**
   * Disconnect from the WebSocket server
   */
  const disconnect = useCallback(() => {
    socketClient.disconnect();
    setIsConnected(false);
  }, []);
  
  /**
   * Subscribe to a channel
   * @param channel - Channel name
   * @param params - Subscription parameters
   */
  const subscribe = useCallback((channel: string, params?: any) => {
    socketClient.subscribe(channel, params);
  }, []);
  
  /**
   * Unsubscribe from a channel
   * @param channel - Channel name
   */
  const unsubscribe = useCallback((channel: string) => {
    socketClient.unsubscribe(channel);
  }, []);
  
  /**
   * Add a listener for a channel
   * @param channel - Channel name
   * @param listener - Listener function
   */
  const addListener = useCallback((channel: string, listener: (data: any) => void) => {
    socketClient.addListener(channel, listener);
    
    // Return a cleanup function
    return () => {
      socketClient.removeListener(channel, listener);
    };
  }, []);
  
  /**
   * Send a message to the server
   * @param event - Event name
   * @param data - Data to send
   */
  const emit = useCallback((event: string, data: any) => {
    socketClient.emit(event, data);
  }, []);
  
  // Connect to the WebSocket server when the component mounts
  useEffect(() => {
    if (autoConnect && token) {
      connect();
    }
    
    // Disconnect when the component unmounts
    return () => {
      if (autoConnect) {
        disconnect();
      }
    };
  }, [autoConnect, token, connect, disconnect]);
  
  return {
    isConnected,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    addListener,
    emit,
  };
};