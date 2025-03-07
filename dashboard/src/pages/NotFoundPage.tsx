import React from 'react';
import { Box, Button, Container, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { SentimentVeryDissatisfied as SadFaceIcon, Home as HomeIcon } from '@mui/icons-material';

const NotFoundPage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <Container maxWidth="sm">
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '100vh',
          textAlign: 'center',
          py: 5,
        }}
      >
        <SadFaceIcon sx={{ fontSize: 100, color: 'text.secondary', mb: 4 }} />
        
        <Typography variant="h1" component="h1" gutterBottom>
          404
        </Typography>
        
        <Typography variant="h4" component="h2" gutterBottom>
          Page Not Found
        </Typography>
        
        <Typography variant="body1" color="text.secondary" paragraph sx={{ mb: 4 }}>
          Sorry, we couldn't find the page you're looking for.
          <br />
          The page might have been removed or the URL might be incorrect.
        </Typography>
        
        <Button
          variant="contained"
          color="primary"
          size="large"
          startIcon={<HomeIcon />}
          onClick={() => navigate('/dashboard')}
        >
          Back to Dashboard
        </Button>
      </Box>
    </Container>
  );
};

export default NotFoundPage;