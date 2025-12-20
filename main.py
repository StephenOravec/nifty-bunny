from fastapi import FastAPI, Request, HTTPException
import os
import sqlite3
import secrets
import logging
import hashlib
from google import genai
from google.genai import types

# ----------------------
# Config
# ----------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "bots-476920")
LOCATION = "us-east5"
MODEL_NAME = "gemini-2.5-flash"

# ----------------------
# Logging setup
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ----------------------
# Initialize Gemini client (Vertex AI mode)
# ----------------------
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# ----------------------
# Database setup
# ----------------------
DB_PATH = "/tmp/sessions.db"

def init_db():
    """Initialize SQLite database for session storage."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# ----------------------
# Session management
# ----------------------
class SessionManager:
    """Manages conversation sessions and message history."""
    
    @staticmethod
    def create_session():
        """Create a new session."""
        session_id = secrets.token_urlsafe(16)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO sessions (session_id) VALUES (?)", (session_id,))
        conn.commit()
        conn.close()
        return session_id
    
    @staticmethod
    def session_exists(session_id: str) -> bool:
        """Check if session exists."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sessions WHERE session_id = ?", (session_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists
    
    @staticmethod
    def save_message(session_id: str, role: str, content: str):
        """Save a message to the database."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()
    
    @staticmethod
    def get_messages(session_id: str) -> list:
        """Get all messages for a session."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row[0], "text": row[1]} for row in rows]

session_manager = SessionManager()

# ----------------------
# Privacy helpers
# ----------------------
def hash_session_id(session_id: str) -> str:
    """Hash session ID for privacy-safe logging."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:8]

# ----------------------
# Chat with Gemini (New SDK)
# ----------------------
async def chat_with_gemini(session_id: str, user_message: str) -> str:
    """Chat with Gemini using session memory."""
    session_hash = hash_session_id(session_id)
    
    logger.info(f"Session {session_hash}: Processing request")
    
    memory = session_manager.get_messages(session_id)
    
    system_instruction = (
        "You are Nifty-Bunny, a chatbot inspired by the White Rabbit from Alice in Wonderland. "
        "You adore rabbit-themed NFTs on Ethereum L1 and L2. "
        "Keep responses very brief, conversational, rabbit-themed, and use emoji."
    )
    
    # Build conversation history
    contents = []
    for msg in memory:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["text"])]
            )
        )
    
    # Add current user message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    )
    
    logger.info(f"Session {session_hash}: {len(memory)} messages in context")
    
    try:
        # Generate response
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.9,
            )
        )
        
        reply = response.text
        
        logger.info(f"Session {session_hash}: Generated response ({len(reply)} chars)")
        return reply
        
    except Exception as e:
        logger.exception(f"Session {session_hash}: Gemini error - {type(e).__name__}")
        raise

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI()

# ----------------------
# Startup/Shutdown Events
# ----------------------
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    logger.info("=" * 50)
    logger.info("nifty-bunny starting (privacy-first mode)")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Location: {LOCATION}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"SDK: google-genai (Vertex AI mode)")
    logger.info(f"Database: {DB_PATH} (ephemeral)")
    logger.info("Privacy features:")
    logger.info("  - Hashed session IDs in logs")
    logger.info("  - No message content logged")
    logger.info("  - Ephemeral database (/tmp)")
    logger.info("  - Data not used for training (Vertex AI)")
    logger.info("=" * 50)

@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown information."""
    logger.info("nifty-bunny shutting down")

# ----------------------
# Routes
# ----------------------
@app.post("/chat")
async def chat(request: Request):
    """Handle chat requests."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        message = data.get("message", "").strip()
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Create or validate session
        if not session_id:
            session_id = session_manager.create_session()
            session_hash = hash_session_id(session_id)
            logger.info(f"Session {session_hash}: New session created")
        else:
            if not session_manager.session_exists(session_id):
                session_id = session_manager.create_session()
                session_hash = hash_session_id(session_id)
                logger.info(f"Session {session_hash}: New session created")
            else:
                session_hash = hash_session_id(session_id)
                logger.info(f"Session {session_hash}: Continuing session")
        
        # Get AI response
        reply = await chat_with_gemini(session_id, message)
        
        # Save messages
        session_manager.save_message(session_id, "user", message)
        logger.info(f"Session {session_hash}: Saved user message")
        
        session_manager.save_message(session_id, "assistant", reply)
        logger.info(f"Session {session_hash}: Saved assistant message")
        
        logger.info(f"Session {session_hash}: Request completed")
        
        return {
            "session_id": session_id,
            "response": reply
        }
        
    except Exception as e:
        session_hash = hash_session_id(session_id) if session_id else "unknown"
        logger.exception(f"Session {session_hash}: Chat failed")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}