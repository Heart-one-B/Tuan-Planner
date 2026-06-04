from __future__ import annotations

from pathlib import Path

from src.utils.path_tool import get_abs_path


_PREFERENCE_DOC_PATH = get_abs_path("data/user_preference_profile.md")


def load_user_preference_profile(max_chars: int = 3000) -> str:
    """Load the confirmed user preference profile as soft planning context."""
    path = Path(_PREFERENCE_DOC_PATH)
    if not path.exists():
        return ""

    try:
        content = path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return content[-max_chars:].strip()
