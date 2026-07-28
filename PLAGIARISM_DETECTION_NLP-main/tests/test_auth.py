import json
import uuid

import pytest

from plagiarism_detection.auth import ApiKeyAuthenticator


def test_api_key_is_bound_to_workspace() -> None:
    workspace_id = str(uuid.uuid4())
    authenticator = ApiKeyAuthenticator(
        {"a-secure-api-key-with-entropy": workspace_id}
    )

    assert (
        authenticator.authenticate("a-secure-api-key-with-entropy").workspace_id
        == workspace_id
    )
    assert authenticator.authenticate("wrong-key-with-enough-characters") is None
    assert authenticator.authenticate(None) is None


def test_configuration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="required"):
        ApiKeyAuthenticator.from_json(None)
    with pytest.raises(ValueError):
        ApiKeyAuthenticator.from_json("not-json")
    with pytest.raises(ValueError):
        ApiKeyAuthenticator({"short": str(uuid.uuid4())})


def test_json_configuration() -> None:
    workspace_id = str(uuid.uuid4())
    authenticator = ApiKeyAuthenticator.from_json(
        json.dumps({"another-secure-api-key": workspace_id})
    )

    assert authenticator.authenticate("another-secure-api-key") is not None
