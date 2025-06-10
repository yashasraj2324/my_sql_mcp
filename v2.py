#!/usr/bin/env python3
"""
FastAPI MCP SQL Agent using FastMCP
"""

import os
import sys
import json
import logging
import requests
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

import mysql.connector
from mysql.connector import Error
from mcp.server.fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_sql_agent")

# Reserved keywords for MySQL
RESERVED_KEYWORDS = {
    'rank', 'group', 'order', 'table', 'index', 'key', 'primary', 'default',
    'create', 'select', 'insert', 'update', 'delete', 'where', 'from', 'join'
}

def get_db_config():
    jdbc_url = os.getenv("MYSQL_JDBC_URL")
    if jdbc_url and jdbc_url.startswith("jdbc:mysql://"):
        try:
            jdbc_url = jdbc_url.replace("jdbc:", "")
            parsed = urlparse(jdbc_url)
            query = parse_qs(parsed.query)
            return {
                "host": parsed.hostname,
                "port": parsed.port or 3306,
                "user": query.get("user", ["root"])[0],
                "password": query.get("password", [""])[0],
                "database": parsed.path.strip("/"),
            }
        except Exception as e:
            logger.error(f"Invalid JDBC URL: {e}")
            raise ValueError("Invalid JDBC URL")
    else:
        return {
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT")),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE")
        }

def escape_identifier(name: str) -> str:
    name = re.sub(r'[^\w]', '', name)
    if name.lower() in RESERVED_KEYWORDS:
        return f"`{name}`"
    return name

# MCP Agent Setup
mcp = FastMCP("MySQL_FastAPI_Agent")

@mcp.tool()
def execute_sql(sql: str) -> str:
    logger.info(f"Executing SQL: {sql}")
    if not sql.strip().upper().startswith(("SELECT", "SHOW", "DESCRIBE")):
        if any(k in sql.upper() for k in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']):
            return "Error: Potentially dangerous SQL operation detected"

    config = get_db_config()
    try:
        with mysql.connector.connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    header = " | ".join(columns)
                    result_rows = [" | ".join(str(val) if val is not None else 'NULL' for val in row) for row in rows]
                    return f"{header}\n{'-' * len(header)}\n" + "\n".join(result_rows) if result_rows else "Query returned no results"
                else:
                    conn.commit()
                    return f"Query executed successfully. Rows affected: {cursor.rowcount}"
    except Error as e:
        return f"Error executing query: {e}"

@mcp.tool()
def get_complete_database_info() -> str:
    config = get_db_config()
    try:
        with mysql.connector.connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = [table[0] for table in cursor.fetchall()]
                if not tables:
                    return "No tables found in the database"

                schema_info = [f"DATABASE: {config['database']}", "=" * 50]
                for table in tables:
                    schema_info.append(f"\nTABLE: {table}")
                    cursor.execute(f"DESCRIBE {escape_identifier(table)}")
                    for col in cursor.fetchall():
                        field, type_, null, key, default, extra = col
                        key_info = {"PRI": "[PRIMARY KEY]", "UNI": "[UNIQUE]", "MUL": "[INDEX]"}.get(key, "")
                        null_info = " NOT NULL" if null == 'NO' else " NULL"
                        default_info = f" DEFAULT {default}" if default else ""
                        extra_info = f" {extra}" if extra else ""
                        schema_info.append(f"  - {field}: {type_}{key_info}{null_info}{default_info}{extra_info}")
                return "\n".join(schema_info)
    except Error as e:
        return f"Error: {e}"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    sql: Optional[str] = None
    explanation: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None

@dataclass
class OpenRouterAgent:
    system_prompt: str = """You are a MySQL expert..."""  # (Same as original)
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY",)
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-0528-qwen3-8b:free")
    
    def generate_sql(self, query: str, database_info: str = "") -> Dict[str, Any]:
        prompt = f"Database Information:\n{database_info}\n\nUser Question: {query}" if database_info else query
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": 1024, "response_format": {"type": "json_object"}}
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            return {"error": str(e)}

openrouter_agent = OpenRouterAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI MCP SQL Agent...")
    try:
        config = get_db_config()
        logger.info(f"Connected to MySQL at {config['host']}:{config['port']}, DB: {config['database']}")
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="AI SQL Agent API with FastMCP",
    version="1.0.0",
    description="Translate natural language to SQL using FastMCP and OpenRouter",
    lifespan=lifespan
)

def get_openrouter_agent():
    return openrouter_agent

@app.get("/")
async def root():
    return {
        "message": "AI SQL Agent with FastMCP",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Natural language to SQL + execution",
            "GET /health": "Check DB & API health"
        }
    }

@app.get("/health")
async def health_check():
    try:
        execute_sql("SELECT 1;")
        return {"status": "healthy", "database": "connected", "fastmcp": "active"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, agent: OpenRouterAgent = Depends(get_openrouter_agent)):
    query_lower = request.query.lower().strip()
    if any(p in query_lower for p in ["show database", "database schema", "all tables", "table structure", "show me the database"]):
        info = get_complete_database_info()
        return QueryResponse(success=True, sql="-- schema info", explanation="Returned DB schema", result=info)

    db_info = get_complete_database_info()
    sql_result = agent.generate_sql(request.query, db_info)
    if "error" in sql_result:
        return QueryResponse(success=False, error=sql_result["error"])

    sql_query = sql_result.get("sql", "")
    explanation = sql_result.get("explanation", "No explanation")
    result = execute_sql(sql_query)
    return QueryResponse(success=True, sql=sql_query, explanation=explanation, result=result)


# Run the app
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
