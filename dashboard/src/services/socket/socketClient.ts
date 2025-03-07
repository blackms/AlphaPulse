import { io, Socket } from 'socket.io-client';
import authService from '../auth/authService';

type SocketListener = (data: any) => void;

interface SocketListeners {
  [channel: string]: SocketListener[];
}

class SocketClient {
  private socket: Socket | null = null;
  private listeners: SocketListeners = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 2000; // 2 seconds
  private url: string;
  private connected = false;

  constructor() {
    this.url = process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws';
    this.listeners = {
      metrics: [],
      alerts: [],
      portfolio: [],
      trades: [],
      system: [],
      connect: [],
      disconnect: [],
      error: []
    };
  }

  public connect(): void {
    if (this.socket) {
      return; // Already connected
    }

    const token = authService.getToken();
    
    this.socket = io(this.url, {
      transports: ['websocket'],
      auth: {
        token
      },
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay
    });

    // Set up connection event handlers
    this.socket.on('connect', () => {
      console.log('Socket connected');
      this.connected = true;
      this.reconnectAttempts = 0;
      this.notifyListeners('connect', { connected: true });
    });

    this.socket.on('disconnect', (reason) => {
      console.log('Socket disconnected:', reason);
      this.connected = false;
      this.notifyListeners('disconnect', { reason });
    });

    this.socket.on('error', (error) => {
      console.error('Socket error:', error);
      this.notifyListeners('error', { error });
    });

    // Set up listeners for specific channels
    ['metrics', 'alerts', 'portfolio', 'trades', 'system'].forEach((channel) => {
      if (this.socket) {
        this.socket.on(channel, (data: any) => {
          this.notifyListeners(channel, data);
        });
      }
    });
  }

  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
    }
  }

  public isConnected(): boolean {
    return this.connected;
  }

  public on(channel: string, callback: SocketListener): void {
    if (!this.listeners[channel]) {
      this.listeners[channel] = [];
    }
    this.listeners[channel].push(callback);
  }

  public off(channel: string, callback: SocketListener): void {
    if (this.listeners[channel]) {
      this.listeners[channel] = this.listeners[channel].filter(
        (listener) => listener !== callback
      );
    }
  }

  private notifyListeners(channel: string, data: any): void {
    if (this.listeners[channel]) {
      this.listeners[channel].forEach((listener) => {
        listener(data);
      });
    }
  }
}

// Create a singleton instance
const socketClient = new SocketClient();

export default socketClient;