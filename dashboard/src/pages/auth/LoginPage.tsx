import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  TextField,
  Typography,
  FormControlLabel,
  Checkbox,
  Link,
  Alert,
} from '@mui/material';
import { Login as LoginIcon } from '@mui/icons-material';
import { useDispatch } from 'react-redux';
// In a real implementation, you would import auth actions
// import { login } from '../../store/slices/authSlice';

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
      // For demo purposes, we'll just simulate a login
      // In a real implementation, you would dispatch a login action
      // await dispatch(login({ username, password, rememberMe }));
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // For demo, hardcode successful login with demo/demo
      if (username === 'demo' && password === 'demo') {
        localStorage.setItem('isAuthenticated', 'true');
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
        startIcon={<LoginIcon />}
        sx={{ mt: 3, mb: 2 }}
        disabled={loading}
      >
        {loading ? 'Signing in...' : 'Sign In'}
      </Button>
      
      <Box display="flex" justifyContent="center">
        <Link href="#" variant="body2">
          Forgot password?
        </Link>
      </Box>
      
      <Box mt={3} textAlign="center">
        <Typography variant="body2" color="textSecondary">
          Demo credentials: username: "demo", password: "demo"
        </Typography>
      </Box>
    </Box>
  );
};

export default LoginPage;