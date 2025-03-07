import React from 'react';
import { Outlet } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import { Box, Container, Typography, Link, Paper } from '@mui/material';

const AuthLayoutRoot = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  minHeight: '100vh',
  backgroundColor: theme.palette.background.default,
}));

const AuthLayoutWrapper = styled(Box)(({ theme }) => ({
  flex: '1 1 auto',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: theme.spacing(3),
}));

const AuthLayoutContainer = styled(Container)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  maxWidth: '450px',
}));

const AuthLayoutPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  width: '100%',
  borderRadius: theme.shape.borderRadius,
  boxShadow: theme.shadows[3],
}));

const Footer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  textAlign: 'center',
  marginTop: theme.spacing(2),
}));

const AuthLayout: React.FC = () => {
  return (
    <AuthLayoutRoot>
      <AuthLayoutWrapper>
        <AuthLayoutContainer>
          <Box sx={{ mb: 3, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h4" component="h1" gutterBottom>
              Alpha Pulse
            </Typography>
            <Typography variant="subtitle1" color="textSecondary">
              AI-Powered Trading Platform
            </Typography>
          </Box>
          <AuthLayoutPaper>
            <Outlet />
          </AuthLayoutPaper>
          <Footer>
            <Typography variant="body2" color="textSecondary">
              © {new Date().getFullYear()} Alpha Pulse. All rights reserved.
            </Typography>
            <Typography variant="body2" color="textSecondary">
              <Link href="#" color="inherit" underline="hover">
                Terms of Service
              </Link>{' '}
              |{' '}
              <Link href="#" color="inherit" underline="hover">
                Privacy Policy
              </Link>
            </Typography>
          </Footer>
        </AuthLayoutContainer>
      </AuthLayoutWrapper>
    </AuthLayoutRoot>
  );
};

export default AuthLayout;