module.exports = {
  apps: [
    {
      name: 'alphapulse-api',
      script: 'poetry',
      args: 'run python src/scripts/test_tenant_api.py',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'development',
        DATABASE_URL: 'postgresql+asyncpg://alphapulse:alphapulse@localhost:5432/alphapulse',
        REDIS_URL: 'redis://localhost:6379',
        VAULT_ADDR: 'http://localhost:8200',
        VAULT_TOKEN: 'root',
        RLS_ENABLED: 'false',
        LOG_LEVEL: 'info',
        JWT_SECRET: 'dev-jwt-secret-change-in-production-min-32-chars',
        ENCRYPTION_KEY: 'dev-encryption-key-change-in-production-32c'
      },
      error_file: './logs/pm2/api-error.log',
      out_file: './logs/pm2/api-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      min_uptime: '10s',
      max_restarts: 10,
      restart_delay: 4000
    },
    {
      name: 'alphapulse-dashboard',
      script: 'npm',
      args: 'start',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse/dashboard',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      env: {
        NODE_ENV: 'development',
        PORT: 3000,
        REACT_APP_API_URL: 'http://localhost:8000'
      },
      error_file: './logs/pm2/dashboard-error.log',
      out_file: './logs/pm2/dashboard-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true
    },
    {
      name: 'postgres',
      script: 'docker',
      args: 'compose -f docker-compose.dev.yml up postgres',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      error_file: './logs/pm2/postgres-error.log',
      out_file: './logs/pm2/postgres-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'redis',
      script: 'docker',
      args: 'compose -f docker-compose.dev.yml up redis',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      error_file: './logs/pm2/redis-error.log',
      out_file: './logs/pm2/redis-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    },
    {
      name: 'vault',
      script: 'docker',
      args: 'compose -f docker-compose.dev.yml up vault',
      cwd: '/Users/a.rocchi/Projects/Personal/AlphaPulse',
      interpreter: 'none',
      instances: 1,
      autorestart: true,
      watch: false,
      error_file: './logs/pm2/vault-error.log',
      out_file: './logs/pm2/vault-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
    }
  ]
};
