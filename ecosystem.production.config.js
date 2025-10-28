module.exports = {
  apps: [
    {
      name: 'alphapulse-api',
      script: 'poetry',
      args: 'run uvicorn alpha_pulse.api.main:app --host 0.0.0.0 --port 8000 --workers 4',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse',
      interpreter: 'none',
      instances: 4,
      exec_mode: 'cluster',
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env_production: {
        NODE_ENV: 'production',
        DATABASE_URL: 'postgresql://alphapulse:alphapulse@localhost:5432/alphapulse',
        REDIS_URL: 'redis://localhost:6379',
        VAULT_ADDR: 'http://localhost:8200',
        VAULT_TOKEN: 'root',
        RLS_ENABLED: 'true',
        LOG_LEVEL: 'warning'
      },
      error_file: './logs/pm2/api-error.log',
      out_file: './logs/pm2/api-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      min_uptime: '30s',
      max_restarts: 5,
      restart_delay: 5000,
      kill_timeout: 5000
    },
    {
      name: 'alphapulse-dashboard',
      script: 'npm',
      args: 'run serve',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse/dashboard',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      env_production: {
        NODE_ENV: 'production',
        PORT: 3000,
        REACT_APP_API_URL: 'http://localhost:8000'
      },
      error_file: './logs/pm2/dashboard-error.log',
      out_file: './logs/pm2/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    }
  ]
};
