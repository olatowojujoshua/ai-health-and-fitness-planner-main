# config/appconfig_local.py
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(project_root / "config" / "logs" / "config.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

REQUIRED_VARS = ["FIRECRAWL_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY", "EXAAI_API_KEY"]

# Load .env if present (do NOT hard fail)
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    logger.info(f"Loaded .env from {env_path}")
else:
    logger.warning(f"No .env found at {env_path}. Using environment variables only.")

# Read keys (missing -> empty string)
config = {var: (os.getenv(var) or "") for var in REQUIRED_VARS}

# Log which are missing (but don't exit)
missing = [k for k, v in config.items() if not v]
if missing:
    logger.warning(f"Missing env vars: {', '.join(missing)}. Some features will be disabled until set.")

# Export
FIRECRAWL_API_KEY = config["FIRECRAWL_API_KEY"]
GROQ_API_KEY = config["GROQ_API_KEY"]
GOOGLE_API_KEY = config["GOOGLE_API_KEY"]
EXAAI_API_KEY = config["EXAAI_API_KEY"]

__all__ = REQUIRED_VARS
import os
import re
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent

from qdrant_client import QdrantClient


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).resolve().parent.parent / 'config/logs/config.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Path configuration for .env located at the project root
    env_path = project_root / '.env'
    
    if not env_path.exists():
        logger.critical(f"Missing .env file at {env_path}")
        sys.exit(1)

    load_dotenv(env_path)

    # Required environment variables
    REQUIRED_VARS = [
        'FIRECRAWL_API_KEY',
        'GROQ_API_KEY',
        'GOOGLE_API_KEY',
        'EXAAI_API_KEY'
    ]

    # Load environment variables
    config = {var: os.getenv(var) for var in REQUIRED_VARS}

    missing_vars = [var for var in REQUIRED_VARS if config[var] is None]
    if missing_vars:
        logger.critical(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    # Export variables
    FIRECRAWL_API_KEY = config['FIRECRAWL_API_KEY']
    GROQ_API_KEY = config['GROQ_API_KEY']
    GOOGLE_API_KEY = config['GOOGLE_API_KEY']
    EXAAI_API_KEY = config['EXAAI_API_KEY']

    # Secure logging for sensitive variables
    sensitive_vars = ['FIRECRAWL_API_KEY', 'GROQ_API_KEY', 'GOOGLE_API_KEY', 'EXAAI_API_KEY']
    for var in REQUIRED_VARS:
        value = locals().get(var, '')
        logged_value = f"{value[:2]}****{value[-2:]}" if var in sensitive_vars and len(value) > 4 else str(value)
        logger.info(f"{var}: {logged_value}")

    logger.info("Configuration Keys loaded successfully")

except Exception as e:
    logger.critical(f"Configuration Keys initialization failed: {str(e)}")
    sys.exit(1)

__all__ = REQUIRED_VARS
