import pytest
from unittest.mock import AsyncMock

from app.repositories.base import BaseRepository


class DummyModel:
    pass


@pytest.mark.asyncio
async def test_repository_get_by_id() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    obj = DummyModel()
    session.get.return_value = obj
    
    result = await repo.get_by_id(123)
    assert result is obj
    session.get.assert_awaited_once_with(DummyModel, 123)


@pytest.mark.asyncio
async def test_repository_get_by_id_missing() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    session.get.return_value = None
    
    result = await repo.get_by_id(123)
    assert result is None


def test_repository_add() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    obj = DummyModel()
    repo.add(obj)
    
    session.add.assert_called_once_with(obj)


@pytest.mark.asyncio
async def test_repository_delete() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    obj = DummyModel()
    await repo.delete(obj)
    
    session.delete.assert_awaited_once_with(obj)


@pytest.mark.asyncio
async def test_repository_flush() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    await repo.flush()
    
    session.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_repository_refresh() -> None:
    session = AsyncMock()
    repo = BaseRepository[DummyModel, int](session, DummyModel)
    
    obj = DummyModel()
    await repo.refresh(obj)
    
    session.refresh.assert_awaited_once_with(obj)
