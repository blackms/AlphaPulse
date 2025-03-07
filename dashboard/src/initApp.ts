import store from './store/store';
import { fetchPortfolioStart } from './store/slices/portfolioSlice';
import { fetchSystemStart } from './store/slices/systemSlice';
import { fetchAlertsStart } from './store/slices/alertsSlice';

/**
 * Initialize the application by fetching initial data
 */
export const initializeApp = () => {
  console.log('Initializing application...');
  
  // Only fetch data and set up polling if the user is authenticated
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  
  if (isAuthenticated) {
    console.log('User is authenticated, fetching initial data...');
    // Fetch initial data
    store.dispatch(fetchPortfolioStart());
    store.dispatch(fetchSystemStart());
    store.dispatch(fetchAlertsStart());
    
    // Set up polling for real-time data
    setupPolling();
    
    console.log('Application initialized with data fetching');
  } else {
    console.log('User is not authenticated, skipping data fetching');
  }
};

/**
 * Initialize data fetching after login
 * This should be called after successful authentication
 */
export const initializeDataFetching = () => {
  console.log('Initializing data fetching after login...');
  
  // Fetch initial data
  store.dispatch(fetchPortfolioStart());
  store.dispatch(fetchSystemStart());
  store.dispatch(fetchAlertsStart());
  
  // Set up polling for real-time data
  setupPolling();
  
  console.log('Data fetching initialized after login');
};

/**
 * Set up polling for real-time data updates
 */
const setupPolling = () => {
  // Portfolio data - every 30 seconds
  setInterval(() => {
    store.dispatch(fetchPortfolioStart());
  }, 30000);
  
  // System status - every minute
  setInterval(() => {
    store.dispatch(fetchSystemStart());
  }, 60000);
  
  // Alerts - every 30 seconds
  setInterval(() => {
    store.dispatch(fetchAlertsStart());
  }, 30000);
};

export default initializeApp;