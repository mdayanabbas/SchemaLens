import uuid
from dataclasses import dataclass
from enum import StrEnum


class ConnectorMode(StrEnum):
    METADATA = "metadata"
    TEST = "test"
    EXECUTION = "execution"


@dataclass(frozen=True)
class ConnectorPoolKey:
    organization_id: uuid.UUID
    connection_id: uuid.UUID
    credential_fingerprint: str
    connector_mode: ConnectorMode

    def __str__(self) -> str:
        return f"{self.organization_id}:{self.connection_id}:{self.connector_mode}:{self.credential_fingerprint}"
