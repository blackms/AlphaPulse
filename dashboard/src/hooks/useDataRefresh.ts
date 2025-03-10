import { useEffect, useRef, useState } from 'react';

interface UseDataRefreshProps {
  refreshFn: () => void;
  interval?: number;  // in milliseconds
  autoRefresh?: boolean;
  onError?: (error: any) => void;
}

/**
 * A hook that provides auto-refresh functionality for data fetching.
 * It can automatically refresh data at specified intervals and handles
 * error cases appropriately.
 */
const useDataRefresh = ({
  refreshFn,
  interval = 60000, // Default: 1 minute
  autoRefresh = true,
  onError
}: UseDataRefreshProps) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<any>(null);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Cancel any existing timer
  const clearRefreshTimer = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  // Set up a new timer
  const setupRefreshTimer = () => {
    if (autoRefresh) {
      clearRefreshTimer();
      timerRef.current = setTimeout(refresh, interval);
    }
  };

  // Execute the refresh function
  const refresh = async () => {
    if (isRefreshing) return;

    clearRefreshTimer();
    setIsRefreshing(true);
    setError(null);

    try {
      await refreshFn();
      setLastRefreshed(new Date());
    } catch (err) {
      setError(err);
      if (onError) {
        onError(err);
      }
    } finally {
      setIsRefreshing(false);
      setupRefreshTimer();
    }
  };

  // Set up the initial timer when the component mounts
  useEffect(() => {
    // Immediately refresh when mounted
    refresh();

    // Clean up timer on unmount
    return () => {
      clearRefreshTimer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update timer if autoRefresh or interval changes
  useEffect(() => {
    if (autoRefresh) {
      setupRefreshTimer();
    } else {
      clearRefreshTimer();
    }

    return () => {
      clearRefreshTimer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh, interval]);

  return {
    refresh,
    isRefreshing,
    error,
    lastRefreshed,
    toggleAutoRefresh: () => {}, // Could implement this if needed
  };
};

export default useDataRefresh;