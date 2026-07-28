"""Fail-closed API-key authentication bound to workspaces."""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkspacePrincipal:
    workspace_id: str


class ApiKeyAuthenticator:
    def __init__(self, key_to_workspace: dict[str, str]) -> None:
        if not key_to_workspace:
            raise ValueError("at least one API key must be configured")
        self._digest_to_workspace: dict[str, str] = {}
        for api_key, workspace_id in key_to_workspace.items():
            if len(api_key) < 20:
                raise ValueError("API keys must contain at least 20 characters")
            uuid.UUID(workspace_id)
            self._digest_to_workspace[_digest(api_key)] = workspace_id

    @classmethod
    def from_json(cls, value: str | None) -> ApiKeyAuthenticator:
        if not value:
            raise ValueError("SOURCELENS_API_KEYS_JSON is required")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("API-key configuration must be valid JSON") from error
        if not isinstance(parsed, dict) or not all(
            isinstance(key, str) and isinstance(workspace, str)
            for key, workspace in parsed.items()
        ):
            raise ValueError("API-key configuration must map strings to workspace UUIDs")
        return cls(parsed)

    def authenticate(self, api_key: str | None) -> WorkspacePrincipal | None:
        if not api_key:
            return None
        candidate = _digest(api_key)
        for digest, workspace_id in self._digest_to_workspace.items():
            if hmac.compare_digest(candidate, digest):
                return WorkspacePrincipal(workspace_id)
        return None


def _digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()
