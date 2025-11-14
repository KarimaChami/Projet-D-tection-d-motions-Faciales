from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from dotenv import dotenv_values


# Charger les variables du fichier .env
config = dotenv_values(".env")

DB_USER = config["DB_USER"]
DB_PASSWORD = config["DB_PASSWORD"]
DB_NAME = config["DB_NAME"]
DB_HOST = config["DB_HOST"]
DB_PORT = config["DB_PORT"]



DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Prediction(Base):
    
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
