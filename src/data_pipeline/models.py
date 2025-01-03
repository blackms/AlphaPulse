from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class OHLCV(Base):
    __tablename__ = "ohlcv"
    
    id = Column(Integer, primary_key=True)
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    class Config:
        indexes = [
            ("exchange", "symbol", "timestamp")
        ]

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    exchange = Column(String, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy/sell
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    
    strategy = relationship("Strategy", back_populates="trades") 