import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import {
  Box,
  Button,
  TextField,
  Typography,
  FormControlLabel,
  Checkbox,
  Alert,
} from '@mui/material';
import authService from '../../services/auth/authService';
import { loginSuccess } from '../../store/slices/authSlice';
import { initializeDataFetching } from '../../initApp';

const LoginPage: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validation
    if (!username.trim()) {
      setError('Username is required');
      return;
    }
    
    if (!password) {
      setError('Password is required');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      // Try API login first
      try {
        const result = await authService.login(username, password);
        dispatch(loginSuccess({
          user: result.user,
          tokens: {
            accessToken: result.accessToken,
            refreshToken: result.refreshToken || '',
            expiresAt: result.expiresAt || Date.now() + 3600000
          }
        }));
        localStorage.setItem('isAuthenticated', 'true');
        // Initialize data fetching after successful login
        initializeDataFetching();
        navigate('/dashboard');
        return;
      } catch (apiError) {
        console.error('API login failed, falling back to demo mode', apiError);
      }
      
      // Fallback to demo mode
      if (username === 'demo' && password === 'demo') {
        localStorage.setItem('isAuthenticated', 'true');
        // Initialize data fetching after successful demo login
        initializeDataFetching();
        navigate('/dashboard');
      } else {
        throw new Error('Invalid credentials');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1 }}>
      <Typography variant="h5" component="h1" gutterBottom align="center">
        Sign In
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <TextField
        margin="normal"
        required
        fullWidth
        id="username"
        label="Username"
        name="username"
        autoComplete="username"
        autoFocus
        value={username}
        onChange={(e) => setUsername(e.target.value)}
        disabled={loading}
      />
      
      <TextField
        margin="normal"
        required
        fullWidth
        name="password"
        label="Password"
        type="password"
        id="password"
        autoComplete="current-password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        disabled={loading}
      />
      
      <FormControlLabel
        control={
          <Checkbox
            value="remember"
            color="primary"
            checked={rememberMe}
            onChange={(e) => setRememberMe(e.target.checked)}
            disabled={loading}
          />
        }
        label="Remember me"
      />
      
      <Button
        type="submit"
        fullWidth
        variant="contained"
        sx={{ mt: 3, mb: 2 }}
        disabled={loading}
      >
        {loading ? 'Signing in...' : 'Sign In'}
      </Button>
      
      <Box mt={3} textAlign="center">
        <Typography variant="body2" color="textSecondary">
          Demo credentials: username: "demo", password: "demo"
        </Typography>
      </Box>
    </Box>
  );
};

export default LoginPage;