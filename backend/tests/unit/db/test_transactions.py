import pytest
from unittest.mock import AsyncMock

from app.core.exceptions import AppError
from app.db.transactions import transactional


@pytest.mark.asyncio
async def test_transactional_success_commits() -> None:
    session = AsyncMock()
    session.in_transaction.return_value = False
    
    async with transactional(session) as tx_session:
        assert tx_session is session
        
    session.begin.assert_called_once()


@pytest.mark.asyncio
async def test_transactional_failure_rolls_back() -> None:
    session = AsyncMock()
    session.in_transaction.return_value = False
    
    with pytest.raises(ValueError):
        async with transactional(session):
            raise ValueError("Test error")
            
    session.begin.assert_called_once()


@pytest.mark.asyncio
async def test_transactional_prevents_nested() -> None:
    session = AsyncMock()
    session.in_transaction.return_value = True
    
    with pytest.raises(AppError) as exc_info:
        async with transactional(session):
            pass
            
    assert exc_info.value.code == "TRANSACTION_ERROR"
    session.begin.assert_not_called()
