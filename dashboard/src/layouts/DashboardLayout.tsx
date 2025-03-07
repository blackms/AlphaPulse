import React, { useState, useEffect } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { 
  Box, 
  CssBaseline, 
  Toolbar, 
  Container, 
  Grid, 
  AppBar,
  Drawer,
  IconButton,
  Typography,
  Divider,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import MenuIcon from '@mui/icons-material/Menu';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import DashboardIcon from '@mui/icons-material/Dashboard';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import NotificationsIcon from '@mui/icons-material/Notifications';
import SettingsIcon from '@mui/icons-material/Settings';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import LogoutIcon from '@mui/icons-material/Logout';

// Redux
import { selectThemeMode, setSidebarSize, selectSidebarSize } from '../store/slices/uiSlice';
import { selectAllAlerts } from '../store/slices/alertsSlice';
import { selectSystemOverallStatus, ComponentStatus } from '../store/slices/systemSlice';
import { logout } from '../store/slices/authSlice';

// Create temporary themes until we import actual ones
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const lightTheme = createTheme({
  palette: {
    mode: 'light',
  },
});

// Define window size hook temporarily
const useWindowSize = () => {
  const [size, setSize] = useState({ width: window.innerWidth, height: window.innerHeight });
  
  useEffect(() => {
    const handleResize = () => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    };
    
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);
  
  return size;
};

const DashboardLayout: React.FC = () => {
  const dispatch = useDispatch();
  const location = useLocation();
  const navigate = useNavigate();
  const { width } = useWindowSize();
  
  // Redux state
  const themeMode = useSelector(selectThemeMode);
  const sidebarSize = useSelector(selectSidebarSize);
  const alerts = useSelector(selectAllAlerts);
  const systemStatus = useSelector(selectSystemOverallStatus);
  
  // Local state
  const [mobileOpen, setMobileOpen] = useState<boolean>(false);
  const [pageTitle, setPageTitle] = useState<string>('Dashboard');
  
  // Handlers
  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };
  
  const toggleSidebar = () => {
    // Toggle between normal and compact
    dispatch(setSidebarSize(sidebarSize === 'normal' ? 'compact' : 'normal'));
  };

  const handleLogout = () => {
    dispatch(logout());
  };

  const handleNavigation = (path: string) => {
    navigate(path);
    setMobileOpen(false); // Close mobile drawer if open
  };
  
  // Update page title based on route
  useEffect(() => {
    const path = location.pathname;
    const basePath = path.split('/')[1];
    
    switch (basePath) {
      case 'dashboard':
        setPageTitle('Dashboard');
        break;
      case 'portfolio':
        setPageTitle('Portfolio');
        break;
      case 'trading':
        setPageTitle('Trading');
        break;
      case 'alerts':
        setPageTitle('Alerts');
        break;
      case 'system':
        setPageTitle('System');
        break;
      case 'settings':
        setPageTitle('Settings');
        break;
      default:
        setPageTitle('Dashboard');
    }
  }, [location]);
  
  // Auto-collapse sidebar on small screens
  useEffect(() => {
    if (width < 1200 && sidebarSize === 'normal') {
      dispatch(setSidebarSize('compact'));
    }
  }, [width, dispatch, sidebarSize]);
  
  // Determine theme based on user preference
  const theme = themeMode === 'dark' ? darkTheme : lightTheme;

  // Navigation items
  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Portfolio', icon: <AccountBalanceWalletIcon />, path: '/portfolio' },
    { text: 'Trading', icon: <ShowChartIcon />, path: '/trading' },
    { text: 'Alerts', icon: <NotificationsIcon />, path: '/alerts' },
    { text: 'System Status', icon: <MonitorHeartIcon />, path: '/system' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' }
  ];
  
  return (
    <ThemeProvider theme={theme}>
      <Box sx={{ display: 'flex', height: '100vh' }}>
        <CssBaseline />
        
        {/* AppBar */}
        <AppBar
          position="fixed"
          sx={{
            zIndex: (theme) => theme.zIndex.drawer + 1,
            transition: (theme) => theme.transitions.create(['width', 'margin'], {
              easing: theme.transitions.easing.sharp,
              duration: theme.transitions.duration.leavingScreen,
            }),
          }}
        >
          <Toolbar>
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2, display: { sm: 'none' } }}
            >
              <MenuIcon />
            </IconButton>
            <IconButton 
              color="inherit" 
              onClick={toggleSidebar}
              sx={{ mr: 2, display: { xs: 'none', sm: 'block' } }}
            >
              {sidebarSize === 'normal' ? <ChevronLeftIcon /> : <ChevronRightIcon />}
            </IconButton>
            <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
              {pageTitle}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              {/* System status indicator would go here */}
              <Box 
                sx={{ 
                  width: 12, 
                  height: 12, 
                  borderRadius: '50%', 
                  bgcolor: systemStatus === ComponentStatus.HEALTHY ? 'success.main' : 'error.main',
                  mr: 2
                }} 
              />
              
              {/* Alert count would go here */}
              <IconButton color="inherit">
                <NotificationsIcon />
              </IconButton>
            </Box>
          </Toolbar>
        </AppBar>
        
        {/* Sidebar */}
        <Drawer
          variant="permanent"
          sx={{
            width: sidebarSize === 'normal' ? 240 : 72,
            flexShrink: 0,
            [`& .MuiDrawer-paper`]: {
              width: sidebarSize === 'normal' ? 240 : 72,
              boxSizing: 'border-box',
              transition: (theme) => theme.transitions.create('width', {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.enteringScreen,
              }),
            },
            display: { xs: 'none', sm: 'block' }
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton
                    onClick={() => handleNavigation(item.path)}
                    sx={{
                      minHeight: 48,
                      justifyContent: sidebarSize === 'normal' ? 'initial' : 'center',
                      px: 2.5,
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        minWidth: 0,
                        mr: sidebarSize === 'normal' ? 3 : 'auto',
                        justifyContent: 'center',
                      }}
                    >
                      {item.icon}
                    </ListItemIcon>
                    {sidebarSize === 'normal' && <ListItemText primary={item.text} />}
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            <Divider />
            <List>
              <ListItem disablePadding>
                <ListItemButton
                  onClick={handleLogout}
                  sx={{
                    minHeight: 48,
                    justifyContent: sidebarSize === 'normal' ? 'initial' : 'center',
                    px: 2.5,
                  }}
                >
                  <ListItemIcon
                    sx={{
                      minWidth: 0,
                      mr: sidebarSize === 'normal' ? 3 : 'auto',
                      justifyContent: 'center',
                    }}
                  >
                    <LogoutIcon />
                  </ListItemIcon>
                  {sidebarSize === 'normal' && <ListItemText primary="Logout" />}
                </ListItemButton>
              </ListItem>
            </List>
          </Box>
        </Drawer>
        
        {/* Mobile drawer */}
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better mobile performance
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: 240 
            },
          }}
        >
          <Toolbar />
          <Box sx={{ overflow: 'auto' }}>
            <List>
              {menuItems.map((item) => (
                <ListItem key={item.text} disablePadding>
                  <ListItemButton onClick={() => handleNavigation(item.path)}>
                    <ListItemIcon>{item.icon}</ListItemIcon>
                    <ListItemText primary={item.text} />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
            <Divider />
            <List>
              <ListItem disablePadding>
                <ListItemButton onClick={handleLogout}>
                  <ListItemIcon><LogoutIcon /></ListItemIcon>
                  <ListItemText primary="Logout" />
                </ListItemButton>
              </ListItem>
            </List>
          </Box>
        </Drawer>
        
        {/* Main content */}
        <Box
          component="main"
          sx={{
            flexGrow: 1,
            height: '100vh',
            overflow: 'auto',
            backgroundColor: (theme) => theme.palette.background.default,
            pt: { xs: 2, sm: 4 },
            px: { xs: 2, sm: 4 },
            pb: 4,
            position: 'relative',
          }}
        >
          <Toolbar /> {/* Spacer for AppBar */}
          
          <Container maxWidth="xl" sx={{ mb: 4 }}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Outlet />
              </Grid>
            </Grid>
          </Container>
          
          {/* Footer */}
          <Box
            component="footer"
            sx={{
              py: 3,
              px: 2,
              mt: 'auto',
              backgroundColor: (theme) => theme.palette.background.paper,
              borderTop: (theme) => `1px solid ${theme.palette.divider}`,
            }}
          >
            <Container maxWidth="xl">
              <Typography variant="body2" color="text.secondary" align="center">
                AI Hedge Fund Dashboard © {new Date().getFullYear()}
              </Typography>
            </Container>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
};

export default DashboardLayout;