from abc import ABC, abstractmethod

from harness.tracing.models import Trace


class TraceStorageBase(ABC):
    """Abstract persistence layer for traces.

    Implement this to swap backends (SQLite, Postgres, cloud, mock).
    Pass your implementation to configure_storage() at startup.
    The default provided implementation is SQLiteTraceStorage.
    """

    @abstractmethod
    def save_trace(self, trace: Trace) -> None:
        """Persist a completed trace and all its child events."""
        ...

    @abstractmethod
    def get_traces_by_session(self, session_id: str) -> list[dict]:
        """Return all traces for a session, newest first."""
        ...