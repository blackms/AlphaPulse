import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import store from './store/store';
import App from './App';
import './assets/styles/global.css';
// Import initializeApp and call it directly for testing
import { initializeApp } from './initApp';
// Import debugging utilities
import { debugApiConnection } from './services/api/debug_api';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

// Initialize the application immediately for testing
// This will fetch data regardless of authentication status
initializeApp();

// Debug API connection
console.log('Using proxy configuration from package.json');
debugApiConnection('');

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>
);