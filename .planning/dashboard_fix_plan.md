# Dashboard Fix Plan

## Issues Identified

1. **Missing Dependencies**:
   - `recharts` library is missing (installed)
   - TypeScript errors related to component types

2. **Type Definition Issues**:
   - Mismatched type definitions in Redux slices
   - Missing type definitions for API responses
   - Incorrect imports from Redux slices

3. **Component Errors**:
   - PieChart component has incorrect props
   - MainLayout component has incorrect sidebar props
   - SystemStatusPage has incorrect imports

## Fix Steps

### 1. Immediate Workaround

- Created `.env.mock` to enable mock data mode
- Created `run_dashboard_mock.sh` script to run with mock data

### 2. Redux Store Fixes

The following files need updates:

- `src/store/slices/alertsSlice.ts`:
  - Fix `selectAlerts` export (should be `selectAllAlerts`)
  - Add `selectUnreadAlertCount` (currently using `selectUnreadCount`)

- `src/store/slices/systemSlice.ts`:
  - Fix `fetchSystemStart` export (should use separate functions for logs, metrics, and status)
  - Add proper type definitions for `SystemStatus` and `ComponentStatus`

- `src/store/slices/uiSlice.ts`:
  - Add missing `selectSidebarOpen` and `setSidebarOpen` exports

### 3. Component Fixes

- `src/components/charts/PieChart.tsx`:
  - Update Chart.js options to match current API
  - Fix `cutout` property usage

- `src/components/layout/MainLayout.tsx`:
  - Fix sidebar props and spread types

- `src/pages/system/SystemStatusPage.tsx`:
  - Fix imports from systemSlice
  - Add proper type definitions for system components

### 4. API Client Fixes

- `src/services/api/apiClient.ts`:
  - Update Axios interceptor types to match current Axios version

- `src/services/socket/socketClient.ts`:
  - Add null checks for socket object

### 5. Type Definition Updates

- `src/types/alerts.ts`: 
  - Ensure proper type definitions for alerts
  - Add missing interfaces

- `src/types/system.ts`:
  - Add proper interfaces for system components and metrics

## Implementation Priority

1. Enable mock data mode (completed)
2. Fix Redux slice exports
3. Update component type definitions
4. Fix API client issues
5. Update remaining components

## Testing Plan

1. Test with mock data first
2. Test API integration with fixed components
3. Verify all dashboard pages load correctly
4. Test real-time updates via WebSocket