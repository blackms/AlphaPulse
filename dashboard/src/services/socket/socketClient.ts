import { io, Socket } from 'socket.io-client';
import authService from '../auth/authService';
import { WebSocketMessage } from '../../types';

// WebSocket URL from environment variables
const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

/**
 * Socket client for handling WebSocket connections
 */
class SocketClient {
  private socket: Socket | null = null;
  private listeners: Map<string, Set<(data: any) => void>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  
  /**
   * Connect to the WebSocket server
   * @param url - WebSocket server URL
   * @returns Promise that resolves when connected
   */
  connect(url: string = WS_URL): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket && this.socket.connected) {
        resolve();
        return;
      }
      
      // Get the authentication token
      const token = authService.getToken();
      
      // Create a new socket connection
      this.socket = io(url, {
        auth: {
          token,
        },
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
      });
      
      // Set up event listeners
      this.socket.on('connect', () => {
        console.log('Socket connected');
        this.reconnectAttempts = 0;
        resolve();
      });
      
      this.socket.on('disconnect', (reason) => {
        console.log(`Socket disconnected: ${reason}`);
      });
      
      this.socket.on('error', (error) => {
        console.error('Socket error:', error);
        reject(error);
      });
      
      this.socket.on('connect_error', (error) => {
        console.error('Socket connection error:', error);
        this.reconnectAttempts++;
        
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          reject(new Error('Max reconnect attempts reached'));
        }
      });
      
      // Set up listeners for specific channels
      ['metrics', 'alerts', 'portfolio', 'trades', 'system'].forEach((channel) => {
        this.socket.on(channel, (data: any) => {
          this.notifyListeners(channel, data);
        });
      });
      
      // Listen for general messages
      this.socket.on('message', (message: WebSocketMessage<any>) => {
        this.notifyListeners('message', message);
      });
    });
  }
  
  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
  
  /**
   * Subscribe to a channel
   * @param channel - Channel name
   * @param params - Subscription parameters
   */
  subscribe(channel: string, params?: any): void {
    if (this.socket && this.socket.connected) {
      this.socket.emit('subscribe', { channel, params });
    } else {
      console.warn('Socket not connected, cannot subscribe');
    }
  }
  
  /**
   * Unsubscribe from a channel
   * @param channel - Channel name
   */
  unsubscribe(channel: string): void {
    if (this.socket && this.socket.connected) {
      this.socket.emit('unsubscribe', { channel });
    } else {
      console.warn('Socket not connected, cannot unsubscribe');
    }
  }
  
  /**
   * Add a listener for a channel
   * @param channel - Channel name
   * @param listener - Listener function
   */
  addListener(channel: string, listener: (data: any) => void): void {
    if (!this.listeners.has(channel)) {
      this.listeners.set(channel, new Set());
    }
    
    this.listeners.get(channel)?.add(listener);
  }
  
  /**
   * Remove a listener from a channel
   * @param channel - Channel name
   * @param listener - Listener function
   */
  removeListener(channel: string, listener: (data: any) => void): void {
    if (this.listeners.has(channel)) {
      this.listeners.get(channel)?.delete(listener);
    }
  }
  
  /**
   * Notify all listeners for a channel
   * @param channel - Channel name
   * @param data - Data to send to listeners
   */
  private notifyListeners(channel: string, data: any): void {
    if (this.listeners.has(channel)) {
      this.listeners.get(channel)?.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in listener for channel ${channel}:`, error);
        }
      });
    }
  }
  
  /**
   * Check if the socket is connected
   * @returns True if connected, false otherwise
   */
  isConnected(): boolean {
    return !!this.socket && this.socket.connected;
  }
  
  /**
   * Send a message to the server
   * @param event - Event name
   * @param data - Data to send
   */
  emit(event: string, data: any): void {
    if (this.socket && this.socket.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn('Socket not connected, cannot emit event');
    }
  }
}

// Create and export the socket client
const socketClient = new SocketClient();

export default socketClient;