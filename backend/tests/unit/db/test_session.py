import pytest
from unittest.mock import AsyncMock, patch

from app.db.session import get_database_session


@pytest.mark.asyncio
@patch("app.db.session.async_session_factory")
async def test_get_database_session_yields_and_closes(mock_factory) -> None:
    mock_session = AsyncMock()
    mock_factory.return_value.__aenter__.return_value = mock_session
    
    iterator = get_database_session()
    session = await iterator.__anext__()
    
    assert session is mock_session
    
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()
    
    mock_factory.return_value.__aexit__.assert_called_once()
    mock_session.rollback.assert_not_called()


@pytest.mark.asyncio
@patch("app.db.session.async_session_factory")
async def test_get_database_session_rolls_back_on_exception(mock_factory) -> None:
    mock_session = AsyncMock()
    mock_factory.return_value.__aenter__.return_value = mock_session
    
    iterator = get_database_session()
    session = await iterator.__anext__()
    
    assert session is mock_session
    
    with pytest.raises(ValueError):
        await iterator.athrow(ValueError("test error"))
    
    mock_session.rollback.assert_awaited_once()
