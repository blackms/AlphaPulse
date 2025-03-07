#!/bin/bash
# Perform database operations with sudo

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Performing AlphaPulse database operations...${NC}"

# Create a new user
echo -e "${YELLOW}Creating a new user...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.users (username, password_hash, email, role)
VALUES (
    'test_user',
    '\$2a\$12\$1InE4/AkbV4/Ye7nKrLPOOYILsLPjpqiLjHXgH6iBp2o7QEzW.ZpG',
    'test@alphapulse.com',
    'user'
) ON CONFLICT (username) DO UPDATE SET
    email = EXCLUDED.email,
    role = EXCLUDED.role
RETURNING id, username, role;
"

# Create a new portfolio
echo -e "${YELLOW}Creating a new portfolio...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.portfolios (name, description)
VALUES (
    'Test Portfolio',
    'Portfolio for testing purposes'
) ON CONFLICT DO NOTHING
RETURNING id, name;
"

# Get portfolio ID
PORTFOLIO_ID=$(sudo -u postgres psql -d alphapulse -t -c "
SELECT id FROM alphapulse.portfolios WHERE name = 'Test Portfolio' LIMIT 1;
" | tr -d ' ')

echo -e "${YELLOW}Using portfolio ID: ${PORTFOLIO_ID}${NC}"

# Create a new position
echo -e "${YELLOW}Creating a new position...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.positions (portfolio_id, symbol, quantity, entry_price, current_price)
VALUES (
    ${PORTFOLIO_ID},
    'BTC-USD',
    1.5,
    45000.0,
    47000.0
) RETURNING id, symbol, quantity, current_price;
"

# Create a new trade
echo -e "${YELLOW}Creating a new trade...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.trades (portfolio_id, symbol, side, quantity, price, fees, order_type, status)
VALUES (
    ${PORTFOLIO_ID},
    'BTC-USD',
    'buy',
    1.5,
    45000.0,
    22.5,
    'market',
    'filled'
) RETURNING id, symbol, side, quantity, price;
"

# Create a new alert
echo -e "${YELLOW}Creating a new alert...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.alerts (title, message, severity, source, tags)
VALUES (
    'Test Alert',
    'This is a test alert message',
    'info',
    'test_script',
    '[\"test\", \"demo\"]'
) RETURNING id, title, severity;
"

# Insert a metric
echo -e "${YELLOW}Inserting a metric...${NC}"
sudo -u postgres psql -d alphapulse -c "
INSERT INTO alphapulse.metrics (time, metric_name, value, labels)
VALUES (
    now(),
    'test_metric',
    42.0,
    '{\"source\": \"test_script\", \"type\": \"test\"}'
) RETURNING id, metric_name, value;
"

# Query data
echo -e "${YELLOW}Querying users...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, username, role FROM alphapulse.users;"

echo -e "${YELLOW}Querying portfolios...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, name FROM alphapulse.portfolios;"

echo -e "${YELLOW}Querying positions...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, symbol, quantity, current_price FROM alphapulse.positions;"

echo -e "${YELLOW}Querying trades...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, symbol, side, quantity, price FROM alphapulse.trades;"

echo -e "${YELLOW}Querying alerts...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, title, severity FROM alphapulse.alerts;"

echo -e "${YELLOW}Querying metrics...${NC}"
sudo -u postgres psql -d alphapulse -c "SELECT id, metric_name, value FROM alphapulse.metrics;"

echo -e "${GREEN}Database operations completed!${NC}"