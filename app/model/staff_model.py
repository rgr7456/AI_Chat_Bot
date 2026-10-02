from sqlalchemy import Column, Integer, String
from app.database.base import Base

class Staff(Base):
    __tablename__ = "staff"

    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    mobile_number = Column(String(20), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
