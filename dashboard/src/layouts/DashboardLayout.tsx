import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { styled } from '@mui/material/styles';
import {
  AppBar,
  Box,
  CssBaseline,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Avatar,
  Menu,
  MenuItem,
  Badge,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  AccountBalance as PortfolioIcon,
  SwapVert as TradingIcon,
  Notifications as AlertsIcon,
  Settings as SettingsIcon,
  Speed as SystemIcon,
  Logout as LogoutIcon,
  Person as PersonIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useSelector } from 'react-redux';
import { RootState } from '../store/store';

const drawerWidth = 240;

const Main = styled('main', { shouldForwardProp: (prop) => prop !== 'open' })<{
  open?: boolean;
}>(({ theme, open }) => ({
  flexGrow: 1,
  transition: theme.transitions.create('margin', {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  marginLeft: `-${drawerWidth}px`,
  ...(open && {
    transition: theme.transitions.create('margin', {
      easing: theme.transitions.easing.easeOut,
      duration: theme.transitions.duration.enteringScreen,
    }),
    marginLeft: 0,
  }),
}));

const DrawerHeader = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
  justifyContent: 'center',
}));

const DashboardLayout: React.FC = () => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(true);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const menuOpen = Boolean(anchorEl);
  const alertsCount = useSelector((state: RootState) => state.alerts.alerts.length || 0);

  const handleDrawerToggle = () => {
    setOpen(!open);
  };

  const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleLogout = () => {
    // Handle logout logic here
    handleClose();
  };

  const handleProfile = () => {
    // Navigate to profile page
    handleClose();
  };

  const menuItems = [
    {
      text: 'Dashboard',
      icon: <DashboardIcon />,
      path: '/dashboard',
    },
    {
      text: 'Portfolio',
      icon: <PortfolioIcon />,
      path: '/dashboard/portfolio',
    },
    {
      text: 'Trading',
      icon: <TradingIcon />,
      path: '/dashboard/trading',
    },
    {
      text: 'Alerts',
      icon: 
        <Badge badgeContent={alertsCount} color="error">
          <AlertsIcon />
        </Badge>,
      path: '/dashboard/alerts',
    },
    {
      text: 'System',
      icon: <SystemIcon />,
      path: '/dashboard/system',
    },
    {
      text: 'Settings',
      icon: <SettingsIcon />,
      path: '/dashboard/settings',
    },
  ];

  return (
    <Box sx={{ display: 'flex' }}>
      <CssBaseline />
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${open ? drawerWidth : 0}px)` },
          ml: { sm: `${open ? drawerWidth : 0}px` },
          transition: 'width 225ms cubic-bezier(0.4, 0, 0.6, 1) 0ms',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2 }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            AI Hedge Fund
          </Typography>
          <div>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenu}
              color="inherit"
            >
              <Avatar sx={{ width: 32, height: 32 }}>U</Avatar>
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={menuOpen}
              onClose={handleClose}
            >
              <MenuItem onClick={handleProfile}>
                <ListItemIcon>
                  <PersonIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Profile</ListItemText>
              </MenuItem>
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <LogoutIcon fontSize="small" />
                </ListItemIcon>
                <ListItemText>Logout</ListItemText>
              </MenuItem>
            </Menu>
          </div>
        </Toolbar>
      </AppBar>
      <Drawer
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
          display: { xs: open ? 'block' : 'none', sm: 'block' },
        }}
        variant="persistent"
        anchor="left"
        open={open}
      >
        <DrawerHeader>
          <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
            Alpha Pulse
          </Typography>
        </DrawerHeader>
        <Divider />
        <List>
          {menuItems.map((item, index) => (
            <ListItem key={item.text} disablePadding>
              <ListItemButton
                onClick={() => navigate(item.path)}
                selected={window.location.pathname === item.path}
              >
                <ListItemIcon>{item.icon}</ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Drawer>
      <Main open={open}>
        <DrawerHeader />
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          <Outlet />
        </Box>
      </Main>
    </Box>
  );
};

export default DashboardLayout;