"""
Configuration module for the Multi-Agent Battery Market Analysis System.
Loads environment variables and defines system-wide constants.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Validate required keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

# LLM Configuration
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0

# Embedding Model Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"

# Storage Paths
VECTOR_DB_PATH = "./vectordb"
DOCS_PATH = "./docs"

# Retry Configuration
MAX_RETRIES = 2

# Web Search Configuration
WEB_SEARCH_MAX_RESULTS = 5

# RAG Configuration
RAG_TOP_K = 5
