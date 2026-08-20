from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

# Setup Logger (Critical for Cloud Debugging)
logger = logging.getLogger("uvicorn")

class Database:
    client: AsyncIOMotorClient = None

db_instance = Database()

def get_database_client():
    return db_instance.client


def get_db():
    db_name = getattr(settings, "DB_NAME", None) or "rag_db"
    return db_instance.client[db_name]

def get_user_collection():
    """Authorization state: which provider account, and whether they may act."""
    return get_db()["users"]

def get_profile_collection():
    """User-editable display data, 1:1 with users."""
    return get_db()["profiles"]


def get_refresh_token_collection():
    """One document per issued refresh token, hashed. See app/repositories/session_repo.py."""
    return get_db()["refresh_tokens"]


def get_connector_collection():
    return get_db()["connector"]


def get_vector_collection():
    return get_db()["vector_documents"]


def get_conversations_collection():
    """One document per chat thread: title, owner, counters, timestamps."""
    return get_db()["conversations"]


def get_messages_collection():
    """One document per message, ordered within a thread by `seq`."""
    return get_db()["messages"]


async def ensure_chat_indexes():
    """
    Creates the indexes the chat history depends on. Idempotent — MongoDB
    ignores a create_index call for an index that already exists.

    The unique (conversation_id, seq) index is doing real work, not just
    speeding up reads: it is the last line of defence against two concurrent
    requests on the same conversation being handed the same sequence number.
    """
    try:
        logger.info("⏳ Ensuring chat history indexes...")

        await get_messages_collection().create_index(
            [("conversation_id", 1), ("seq", 1)],
            unique=True,
            name="conversation_seq",
        )
        await get_conversations_collection().create_index(
            [("user_id", 1), ("updated_at", -1)],
            name="user_recent",
        )

        logger.info("✅ Chat history indexes ready.")

    except Exception as e:
        logger.error(f"❌ Failed to create chat history indexes: {e}")
        raise e


async def ensure_auth_indexes():
    """
    Creates the indexes the auth system depends on. Idempotent.

    The unique indexes are load-bearing, not just fast lookups:

      * (clerk_user_id) and (email) are what make first-sign-in an upsert
        rather than a read-then-write race — two concurrent first requests
        from the same person cannot create two accounts.
      * (token_hash) is what makes refresh-token lookup a single indexed hit
        instead of a collection scan, on an endpoint hit by every client.

    The TTL index on expires_at lets MongoDB reap dead tokens on its own, so
    the collection does not grow without bound. Note it only removes expired
    rows — revoked-but-unexpired tokens stay, which is what reuse detection
    needs in order to recognise a stolen token when it comes back.
    """
    try:
        logger.info("⏳ Ensuring auth indexes...")

        await get_user_collection().create_index(
            [("clerk_user_id", 1)], unique=True, name="clerk_user_id_unique"
        )
        await get_user_collection().create_index(
            [("email", 1)], unique=True, name="email_unique"
        )
        await get_profile_collection().create_index(
            [("user_id", 1)], unique=True, name="profile_user_unique"
        )
        await get_refresh_token_collection().create_index(
            [("token_hash", 1)], unique=True, name="token_hash_unique"
        )
        await get_refresh_token_collection().create_index(
            [("user_id", 1), ("family_id", 1)], name="user_family"
        )
        await get_refresh_token_collection().create_index(
            [("expires_at", 1)], expireAfterSeconds=0, name="refresh_ttl"
        )

        logger.info("✅ Auth indexes ready.")

    except Exception as e:
        logger.error(f"❌ Failed to create auth indexes: {e}")
        raise e


async def connect_to_mongo():
    try:
        logger.info("⏳ Connecting to MongoDB...")
        db_instance.client = AsyncIOMotorClient(settings.MONGO_URL)
        
        # THE PING TEST (Crucial for Cloud)
        await db_instance.client.admin.command('ping')
        logger.info("✅ MongoDB Connected Successfully!")
        
    except Exception as e:
        logger.error(f"❌ MongoDB Connection Failed: {e}")
        raise e


async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()
        logger.info("🔒 MongoDB connection closed.")


