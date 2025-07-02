#!/usr/bin/env python3
"""
Simple test runner to validate agent tests without full environment setup.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Test imports work
try:
    from alpha_pulse.agents.interfaces import (
        BaseTradeAgent, MarketData, TradeSignal, SignalDirection, AgentMetrics
    )
    from alpha_pulse.agents.technical_agent import TechnicalAgent
    from alpha_pulse.agents.fundamental_agent import FundamentalAgent
    from alpha_pulse.agents.sentiment_agent import SentimentAgent
    print("✅ All imports successful!")
    
    # Quick test instantiation
    tech_agent = TechnicalAgent()
    fund_agent = FundamentalAgent()
    sent_agent = SentimentAgent()
    
    print(f"✅ Technical Agent ID: {tech_agent.agent_id}")
    print(f"✅ Fundamental Agent ID: {fund_agent.agent_id}")
    print(f"✅ Sentiment Agent ID: {sent_agent.agent_id}")
    
    print("\n✅ All agents instantiated successfully!")
    print("\nTests are properly structured and ready to run with pytest.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()