// This file provides enhanced debugging for API calls

// Save original fetch for logging purposes
const originalFetch = window.fetch;

// Override fetch with a debugging version
window.fetch = async function(...args) {
  console.log('Fetch Request:', args);
  try {
    const response = await originalFetch(...args);
    // Clone the response to avoid consuming it
    const clonedResponse = response.clone();
    // Try to log the response body
    try {
      const text = await clonedResponse.text();
      console.log('Fetch Response:', {
        url: args[0],
        status: response.status,
        statusText: response.statusText,
        headers: [...response.headers.entries()].reduce((acc, [key, value]) => {
          acc[key] = value;
          return acc;
        }, {}),
        body: text.substring(0, 500) + (text.length > 500 ? '...' : '')
      });
    } catch (e) {
      console.log('Fetch Response (body unavailable):', {
        url: args[0],
        status: response.status,
        statusText: response.statusText
      });
    }
    return response;
  } catch (error) {
    console.error('Fetch Error:', error, { request: args });
    throw error;
  }
};

// Also enhance axios for more detailed logging
if (window.axios) {
  const originalAxiosRequest = window.axios.request;
  window.axios.request = function(config) {
    console.log('Axios Request:', config);
    return originalAxiosRequest.apply(this, arguments).then(
      response => {
        console.log('Axios Response:', {
          url: config.url,
          method: config.method,
          status: response.status,
          data: response.data
        });
        return response;
      },
      error => {
        console.error('Axios Error:', {
          url: config.url,
          method: config.method,
          error: error.message,
          response: error.response ? {
            status: error.response.status,
            data: error.response.data
          } : 'No response'
        });
        throw error;
      }
    );
  };
}

console.log('API Debug Tools Initialized');

export const debugApiConnection = (baseUrl) => {
  console.log('Testing API connection to:', baseUrl);
  console.log('Using proxy configuration from package.json');
  
  // Test basic connectivity
  fetch('/', { method: 'GET' })
    .then(response => {
      console.log(`API Root Connectivity: ${response.status}`);
    })
    .catch(error => {
      console.error(`API Root Connection Error: ${error.message}`);
    });
    
  // Test token endpoint
  const tokenParams = new URLSearchParams();
  tokenParams.append('username', 'admin');
  tokenParams.append('password', 'password');
  
  fetch('/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: tokenParams
  })
    .then(response => response.json())
    .then(data => {
      console.log('Token Response:', data);
      if (data.access_token) {
        // Test protected endpoint
        fetch('/api/v1/system', {
          headers: {
            'Authorization': `Bearer ${data.access_token}`
          }
        })
          .then(response => response.json())
          .then(systemData => {
            console.log('System endpoint response:', systemData);
          })
          .catch(error => {
            console.error('System endpoint error:', error.message);
          });
      }
    })
    .catch(error => {
      console.error('Token endpoint error:', error.message);
    });
};