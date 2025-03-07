# AlphaPulse Dashboard Frontend

This is the frontend application for the AlphaPulse AI Hedge Fund Dashboard. It provides a user interface for monitoring and interacting with the AlphaPulse trading system.

## Features

- Real-time monitoring of portfolio performance
- Alert management and notifications
- Trading activity tracking
- System status monitoring
- Portfolio analysis and visualization
- Responsive design for desktop and mobile

## Technology Stack

- **React**: UI library
- **TypeScript**: Type-safe JavaScript
- **Redux Toolkit**: State management
- **Material UI**: Component library
- **Chart.js**: Data visualization
- **Socket.IO**: Real-time updates
- **Axios**: API communication

## Project Structure

The project follows a feature-based structure:

```
dashboard/
├── src/
│   ├── assets/              # Static assets
│   ├── components/          # Reusable UI components
│   ├── hooks/               # Custom React hooks
│   ├── pages/               # Page components
│   ├── services/            # API and data services
│   ├── store/               # Redux store configuration
│   ├── types/               # TypeScript type definitions
│   └── utils/               # Utility functions
```

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/alpha-pulse.git
cd alpha-pulse/dashboard
```

2. Install dependencies:
```bash
npm install
# or
yarn install
```

3. Set up environment variables:
Create a `.env` file in the dashboard directory with the following variables:
```
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_WS_URL=ws://localhost:8000
```

### Running the Application

```bash
npm start
# or
yarn start
```

The application will be available at `http://localhost:3000`.

### Building for Production

```bash
npm run build
# or
yarn build
```

The build artifacts will be stored in the `build/` directory.

## Authentication

The dashboard uses JWT-based authentication. Users need to log in with their credentials to access the dashboard. The token is stored in localStorage and automatically refreshed when needed.

## API Integration

The dashboard communicates with the AlphaPulse backend API for data retrieval and actions. The API client is configured to handle authentication and token refresh automatically.

## WebSocket Integration

Real-time updates are received through WebSocket connections. The dashboard subscribes to various channels to receive updates for metrics, alerts, portfolio changes, and trading activities.

## Development Guidelines

- Use TypeScript for all new code
- Follow the established component structure
- Use Material UI components for consistency
- Add proper error handling for API calls
- Write unit tests for new components and functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.