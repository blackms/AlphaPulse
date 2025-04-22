#!/usr/bin/env python3
"""
Script to check the database content and verify Bybit data is being saved.
"""
import sqlite3
import sys
import json
from datetime import datetime
from tabulate import tabulate

def connect_to_db(db_path):
    """Connect to SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

def get_tables(conn):
    """Get list of tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table['name'] for table in tables]

def get_table_schema(conn, table_name):
    """Get schema for a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    return cursor.fetchall()

def get_table_data(conn, table_name, limit=10):
    """Get data from a table with limit."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Error querying table {table_name}: {e}")
        return []

def check_bybit_data(conn):
    """Check if Bybit data exists in the database."""
    cursor = conn.cursor()
    
    # Check positions table for Bybit data
    try:
        cursor.execute("""
            SELECT * FROM positions 
            WHERE symbol LIKE 'BTC%' OR symbol LIKE 'ETH%' 
            LIMIT 5;
        """)
        positions = cursor.fetchall()
        
        if positions:
            print("\n=== Bybit Positions Found ===")
            headers = positions[0].keys()
            rows = [dict(row) for row in positions]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            return True
        else:
            print("\nNo Bybit positions found in the database.")
    except sqlite3.Error as e:
        print(f"Error checking Bybit data: {e}")
    
    return False

def check_portfolio_data(conn):
    """Check portfolio data in the database."""
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM portfolios LIMIT 5;")
        portfolios = cursor.fetchall()
        
        if portfolios:
            print("\n=== Portfolios ===")
            headers = portfolios[0].keys()
            rows = [dict(row) for row in portfolios]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            
            # Get positions for each portfolio
            for portfolio in portfolios:
                cursor.execute(f"SELECT * FROM positions WHERE portfolio_id = {portfolio['id']} LIMIT 10;")
                positions = cursor.fetchall()
                
                if positions:
                    print(f"\n=== Positions for Portfolio {portfolio['id']} ===")
                    headers = positions[0].keys()
                    rows = [dict(row) for row in positions]
                    print(tabulate(rows, headers=headers, tablefmt="grid"))
            
            return True
        else:
            print("\nNo portfolio data found in the database.")
    except sqlite3.Error as e:
        print(f"Error checking portfolio data: {e}")
    
    return False

def check_trades(conn):
    """Check trades in the database."""
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM trades ORDER BY executed_at DESC LIMIT 10;")
        trades = cursor.fetchall()
        
        if trades:
            print("\n=== Recent Trades ===")
            headers = trades[0].keys()
            rows = [dict(row) for row in trades]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            return True
        else:
            print("\nNo trades found in the database.")
    except sqlite3.Error as e:
        print(f"Error checking trades: {e}")
    
    return False

def check_metrics(conn):
    """Check metrics in the database."""
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM metrics ORDER BY time DESC LIMIT 10;")
        metrics = cursor.fetchall()
        
        if metrics:
            print("\n=== Recent Metrics ===")
            headers = metrics[0].keys()
            rows = [dict(row) for row in metrics]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            return True
        else:
            print("\nNo metrics found in the database.")
    except sqlite3.Error as e:
        print(f"Error checking metrics: {e}")
    
    return False

def main():
    """Main function."""
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "./data.db"  # Default path
    
    print(f"Checking database: {db_path}")
    conn = connect_to_db(db_path)
    
    print("\n=== Database Tables ===")
    tables = get_tables(conn)
    print("\n".join(tables))
    
    # Check for Bybit data
    bybit_data_exists = check_bybit_data(conn)
    
    # Check portfolio data
    portfolio_data_exists = check_portfolio_data(conn)
    
    # Check trades
    trades_exist = check_trades(conn)
    
    # Check metrics
    metrics_exist = check_metrics(conn)
    
    # Summary
    print("\n=== Database Check Summary ===")
    print(f"Database: {db_path}")
    print(f"Tables found: {len(tables)}")
    print(f"Bybit data found: {'Yes' if bybit_data_exists else 'No'}")
    print(f"Portfolio data found: {'Yes' if portfolio_data_exists else 'No'}")
    print(f"Trades found: {'Yes' if trades_exist else 'No'}")
    print(f"Metrics found: {'Yes' if metrics_exist else 'No'}")
    
    conn.close()

if __name__ == "__main__":
    main()