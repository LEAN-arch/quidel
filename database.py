# database.py

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime

# In a real production environment, this would be a PostgreSQL or SQL Server connection string from secrets.
DATABASE_URL = "sqlite:///./assay_vantage.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ORM Models: Defining the Application's Data Structure ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    role = Column(String, default="viewer") # e.g., 'viewer', 'engineer', 'director'

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    status = Column(String)
    start_date = Column(DateTime)
    finish_date = Column(DateTime)
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User")
    protocols = relationship("Protocol", back_populates="project")

class Protocol(Base):
    __tablename__ = "protocols"
    id = Column(Integer, primary_key=True, index=True)
    protocol_id_str = Column(String, unique=True, index=True)
    title = Column(String)
    status = Column(String, default="Draft")
    acceptance_criteria = Column(Text)
    project_id = Column(Integer, ForeignKey("projects.id"))
    project = relationship("Project", back_populates="protocols")
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", foreign_keys=[author_id])
    approver_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    approver = relationship("User", foreign_keys=[approver_id])
    creation_date = Column(DateTime, default=datetime.utcnow)
    approval_date = Column(DateTime, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User")
    action = Column(String)
    details = Column(Text)
    # This is key for 21 CFR Part 11: linking the log to a specific record
    record_type = Column(String, nullable=True)
    record_id = Column(Integer, nullable=True)

# Function to initialize the database with some seed data
def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    # Check if users already exist to prevent re-seeding
    if db.query(User).count() == 0:
        # Create Users
        director = User(username="director", full_name="Assay Director", role="director")
        engineer = User(username="alice", full_name="Alice", role="engineer")
        viewer = User(username="charlie", full_name="Charlie", role="viewer")
        db.add_all([director, engineer, viewer])
        db.commit()

        # Create Projects and Protocols
        proj1 = Project(name="ImmunoPro-A", status="On Track", start_date=datetime.now()-timedelta(days=60), finish_date=datetime.now()+timedelta(days=30), owner_id=engineer.id)
        db.add(proj1)
        db.commit()
        
        proto1 = Protocol(protocol_id_str="IP-PREC-01", title="Precision Study", project_id=proj1.id, author_id=engineer.id, status="Executed - Passed", acceptance_criteria="CV <= 5%")
        db.add(proto1)
        db.commit()
    db.close()
