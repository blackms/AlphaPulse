import React from 'react';
import { Outlet } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  CssBaseline,
  Typography,
  Link,
  AppBar,
  Toolbar,
} from '@mui/material';

const AuthLayout: React.FC = () => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <CssBaseline />
      
      <AppBar position="static" color="primary" elevation={0}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Alpha Pulse AI Hedge Fund
          </Typography>
        </Toolbar>
      </AppBar>
      
      <Container component="main" maxWidth="sm" sx={{ mt: 8, mb: 4 }}>
        <Paper
          elevation={3}
          sx={{
            my: { xs: 3, md: 6 },
            p: { xs: 2, md: 3 },
          }}
        >
          <Outlet />
        </Paper>
        
        <Box mt={5} textAlign="center">
          <Typography variant="body2" color="textSecondary" align="center">
            &copy; {new Date().getFullYear()} Alpha Pulse AI Hedge Fund
          </Typography>
          <Typography variant="body2" color="textSecondary" align="center">
            <Link color="inherit" href="#">
              Terms of Service
            </Link>{' '}
            |{' '}
            <Link color="inherit" href="#">
              Privacy Policy
            </Link>
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default AuthLayout;