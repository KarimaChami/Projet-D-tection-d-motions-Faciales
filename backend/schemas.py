from backend.db import Base 
from datetime import datetime, UTC
from sqlalchemy import Column, Integer, String, Float, DateTime



class Prediction(Base):
    
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    # timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))

