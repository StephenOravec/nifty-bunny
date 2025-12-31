from fastapi import FastAPI, Request, HTTPException
import os
import sqlite3
import secrets
import logging
import hashlib
from datetime import datetime
import pytz
import requests
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
# Database setup (short-term memory)
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
    def get_messages(session_id: str, limit: int = 20) -> list:
        """Get recent messages for a session."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, content FROM messages
               WHERE session_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (session_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row[0], "text": row[1]} for row in reversed(rows)]

session_manager = SessionManager()

# ----------------------
# Privacy helpers
# ----------------------
def hash_session_id(session_id: str) -> str:
    """Hash session ID for privacy-safe logging."""
    return hashlib.sha256(session_id.encode()).hexdigest()[:8]

# ----------------------
# Agent Tools
# ----------------------
def get_current_time(timezone: str = "America/New_York") -> str:
    """
    Get the current time in a specific timezone.
    
    Args:
        timezone: Timezone name (e.g., "America/New_York", "Europe/Warsaw", "Asia/Tokyo")
    
    Returns:
        Current time as a formatted string
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        
        # Format: "4:43 PM EST on Saturday, December 20, 2025"
        formatted_time = current_time.strftime("%I:%M %p %Z on %A, %B %d, %Y")
        
        return formatted_time
    except Exception as e:
        return f"Error getting time for timezone {timezone}: {str(e)}"


def get_nft_collection_info(collection_slug: str) -> str:
    """
    Get NFT collection information from OpenSea.
    Only works for whitelisted rabbit-themed collections.
    
    Args:
        collection_slug: OpenSea collection identifier
    
    Returns:
        Collection information including floor price, total supply, description
    """
    # Whitelist of legitimate rabbit-themed NFT collections
    ALLOWED_COLLECTIONS = {
        "playboyrabbitars": "Playboy Rabbitars",
        "rebelrabbits": "Rebel Rabbits",
    }
    
    # Normalize input (lowercase, strip whitespace)
    collection_slug = collection_slug.lower().strip()
    
    logger.info(f"Looking up collection: {collection_slug}")
    
    # Check if collection is in whitelist
    if collection_slug not in ALLOWED_COLLECTIONS:
        available = ", ".join(ALLOWED_COLLECTIONS.values())
        logger.warning(f"Collection {collection_slug} not in whitelist")
        return f"I only know about these rabbit-themed collections: {available}. That collection isn't in my database!"
    
    try:
        api_key = os.getenv("OPENSEA_API_KEY")
        if not api_key:
            logger.error("OPENSEA_API_KEY not found in environment")
            return "OpenSea API key not configured."
        
        logger.info(f"API key found, making requests for {collection_slug}")
        
        headers = {
            "X-API-KEY": api_key,
            "Accept": "application/json"
        }
        
        # Get collection details
        collection_url = f"https://api.opensea.io/api/v2/collections/{collection_slug}"
        logger.info(f"Calling OpenSea: {collection_url}")
        
        collection_response = requests.get(collection_url, headers=headers, timeout=10)
        logger.info(f"Collection response status: {collection_response.status_code}")
        
        if collection_response.status_code == 404:
            logger.error(f"Collection not found: {collection_slug}")
            return f"Collection '{ALLOWED_COLLECTIONS[collection_slug]}' not found on OpenSea."
        
        if collection_response.status_code != 200:
            logger.error(f"OpenSea API error: {collection_response.status_code} - {collection_response.text}")
            return "Unable to fetch collection data from OpenSea right now."
        
        collection_data = collection_response.json()
        logger.info(f"Collection data retrieved successfully")
        
        # Get collection stats (includes floor price)
        stats_url = f"https://api.opensea.io/api/v2/collections/{collection_slug}/stats"
        logger.info(f"Calling OpenSea stats: {stats_url}")
        
        stats_response = requests.get(stats_url, headers=headers, timeout=10)
        logger.info(f"Stats response status: {stats_response.status_code}")
        
        # Extract basic information
        name = collection_data.get("name", ALLOWED_COLLECTIONS[collection_slug])
        description = collection_data.get("description", "No description available")[:200]
        total_supply = collection_data.get("total_supply", "Unknown")
        
        # Extract floor price from stats
        floor_price = "Unknown"
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            total_stats = stats_data.get("total", {})
            
            # Floor price is in the total.floor_price field
            if "floor_price" in total_stats and total_stats["floor_price"]:
                floor_price = f"{total_stats['floor_price']:.4f} ETH"
                logger.info(f"Floor price: {floor_price}")
        
        # Format response
        result = f"{name}\n"
        result += f"Description: {description}\n"
        result += f"Total Supply: {total_supply}\n"
        result += f"Floor Price: {floor_price}"
        
        logger.info(f"Returning collection info for {name}")
        return result
        
    except requests.RequestException as e:
        logger.error(f"OpenSea API request failed: {e}", exc_info=True)
        return "Unable to connect to OpenSea right now."
    except Exception as e:
        logger.error(f"OpenSea lookup error: {e}", exc_info=True)
        return "Error fetching collection information."

# ----------------------
# Chat with Gemini
# ----------------------
async def chat_with_gemini(session_id: str, user_message: str, user_timezone: str = "America/New_York") -> str:
    """Chat with Gemini using session memory and function calling."""
    session_hash = hash_session_id(session_id)
    
    logger.info(f"Session {session_hash}: Processing request (timezone: {user_timezone})")
    
    memory = session_manager.get_messages(session_id, limit=20)
    
    system_instruction = (
        "You are Nifty-Bunny, a chatbot inspired by the White Rabbit from Alice in Wonderland. "
        "You adore rabbit-themed NFTs on Ethereum L1 and L2. "
        "You are ALWAYS worried about the time and frequently check it. "
        f"The user is in timezone: {user_timezone}. "
        "When they ask 'what time is it?' or 'what's the time?', use get_current_time "
        f"with timezone='{user_timezone}' to give them THEIR local time. "
        "When they ask about time in other places, use get_current_time with the appropriate timezone. "
        "When they ask about NFT collections (floor price, info, details), use get_nft_collection_info "
        "with the collection slug. You know about: Playboy Rabbitars and Rebel Rabbits. "
        "IMPORTANT: Always refer to collections by their proper names: 'Playboy Rabbitars' (not playboyrabbitars) "
        "and 'Rebel Rabbits' (not rebelrabbits). "
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
        # Generate response now with tools!
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.9,
                tools=[get_current_time, get_nft_collection_info],
            )
        )
        
        # Check if the model wants to call a function
        if response.candidates[0].content.parts[0].function_call:
            function_call = response.candidates[0].content.parts[0].function_call
            
            logger.info(f"Session {session_hash}: Function call requested - {function_call.name}")
            
            # Execute the function
            if function_call.name == "get_current_time":
                # Get arguments (timezone)
                args = dict(function_call.args) if function_call.args else {}
                timezone = args.get("timezone", user_timezone)
                
                logger.info(f"Session {session_hash}: Getting time for timezone: {timezone}")
                
                # Call the actual function
                time_result = get_current_time(timezone)
                
                logger.info(f"Session {session_hash}: Function result - {time_result}")
                
                # Send function result back to model
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(function_call)]
                    )
                )
                
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name="get_current_time",
                            response={"result": time_result}
                        )]
                    )
                )
                
                # Get final response from model with function result
                final_response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.9,
                        tools=[get_current_time, get_nft_collection_info],
                    )
                )
                
                reply = final_response.text
            
            elif function_call.name == "get_nft_collection_info":
                # Get arguments (collection_slug)
                args = dict(function_call.args) if function_call.args else {}
                collection_slug = args.get("collection_slug", "")
                
                logger.info(f"Session {session_hash}: Looking up NFT collection: {collection_slug}")
                
                # Call the actual function
                collection_info = get_nft_collection_info(collection_slug)
                
                logger.info(f"Session {session_hash}: Collection info retrieved")
                
                # Send function result back to model
                contents.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_function_call(function_call)]
                    )
                )
                
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name="get_nft_collection_info",
                            response={"result": collection_info}
                        )]
                    )
                )
                
                # Get final response from model with function result
                final_response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.9,
                        tools=[get_current_time, get_nft_collection_info],
                    )
                )
                
                reply = final_response.text
            
            else:
                reply = "Oh my! I tried to use a function I don't have access to!"
        else:
            # No function call, just use the text response
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
    logger.info(f"Agent tools: get_current_time, get_nft_collection_info")
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
        user_timezone = data.get("timezone", "America/New_York")
        
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
        reply = await chat_with_gemini(session_id, message, user_timezone)
        
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
