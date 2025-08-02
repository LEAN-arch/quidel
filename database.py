# database.py (Corrected for IndentationError)

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, ForeignKey, Text, Boolean
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from datetime import datetime, timedelta

DATABASE_URL = "sqlite:///./assay_vantage.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ORM Models ---

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    role = Column(String, default="viewer")

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
    record_type = Column(String, nullable=True)
    record_id = Column(Integer, nullable=True)

# --- Database Initialization Function ---

def init_db():
    """Creates database tables and seeds them with initial data if they don't exist."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Check if users already exist to prevent re-seeding
        if db.query(User).count() == 0:
            # Create Users
            director = User(username="director", full_name="Assay Director", role="director")
            engineer = User(username="alice", full_name="Alice", role="engineer")
            viewer = User(username="charlie", full_name="Charlie", role="viewer")
            db.add_all([director, engineer, viewer])
            db.commit() # Commit users to get their IDs

            # Create Projects
            proj1 = Project(
                name="ImmunoPro-A",
                status="On Track",
                start_date=datetime.now() - timedelta(days=60),
                finish_date=datetime.now() + timedelta(days=30),
                owner_id=engineer.id
            )
            db.add(proj1)
            db.commit() # Commit project to get its ID

            # Create a Protocol associated with the project
            # ** THE FIX IS HERE: The line was previously incorrectly indented **
            proto1 = Protocol(
                protocol_id_str="IP-PREC-01",
                title="Precision Study",
                project_id=proj1.id,
                author_id=engineer.id,
                status="Executed - Passed",
                acceptance_criteria="CV <= 5%"
            )
            db.add(proto1)
            db.commit()
    finally:
        db.close()
