"""
tests/conftest.py
Shared pytest fixtures.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import get_settings


@pytest.fixture(scope="session", autouse=True)
def settings():
    s = get_settings()
    s.environment = "testing"
    return s


@pytest.fixture
async def client():
    """Async HTTP client pointed at the FastAPI test app."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Valid API key headers for authenticated requests."""
    return {"X-Api-Key": get_settings().api_key}


@pytest.fixture
def mock_redis():
    """Mock Redis client — prevents real Redis calls in unit tests."""
    mock = AsyncMock()
    mock.rpush = AsyncMock(return_value=1)
    mock.expire = AsyncMock(return_value=True)
    mock.lrange = AsyncMock(return_value=[])
    mock.delete = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def sample_agent_state():
    """A minimal valid AgentState for testing individual agent nodes."""
    return {
        "query": "What is Flash Attention?",
        "session_id": "test-session-123",
        "messages": [],
        "sub_tasks": [],
        "routing": {},
        "web_results": [],
        "rag_results": [],
        "code_output": None,
        "final_answer": None,
        "citations": [],
        "agents_used": [],
        "error": None,
        "iteration_count": 0,
    }
