import React from 'react';
import ReactDOM from 'react-dom/client';
import { Provider } from 'react-redux';
import store from './store/store';
import App from './App';
import './assets/styles/global.css';
// Import but don't call initializeApp - it will be called after login
import './initApp';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

// We no longer initialize the application here
// This will be done after login in the LoginPage component

root.render(
  <React.StrictMode>
    <Provider store={store}>
      <App />
    </Provider>
  </React.StrictMode>
);