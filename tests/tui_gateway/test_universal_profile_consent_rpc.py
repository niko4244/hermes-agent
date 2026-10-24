"""Potentially billable profile mutations must request explicit consent."""

from unittest.mock import Mock, patch

import pytest

from tui_gateway import server


@pytest.mark.parametrize("action", ["activate", "save", "import"])
def test_remote_profile_mutation_does_not_write_before_confirmation(action):
    from hermes_cli.universal_models.service import CloudConsentRequired

    service = Mock()
    service.profile.side_effect = CloudConsentRequired("private-url-and-token")
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        response = server._methods["orchestration.profile"](
            "request", {"action": action, "name": "remote-team", "data": None},
        )
    assert response["result"]["confirm_required"] is True
    assert "provider charges" in response["result"]["warning"]
    assert "private-url-and-token" not in str(response)
    service.profile.assert_called_once_with(action, "remote-team", None, confirm_cloud=False)


def test_explicit_profile_confirmation_is_passed_to_registry_exactly_once():
    service = Mock()
    service.profile.return_value = {"name": "remote-team", "roles": []}
    with patch("hermes_cli.universal_models.service.ModelService", return_value=service):
        response = server._methods["orchestration.profile"](
            "request", {"action": "activate", "name": "remote-team", "confirm_cloud": True},
        )
    assert response["result"]["name"] == "remote-team"
    service.profile.assert_called_once_with("activate", "remote-team", None, confirm_cloud=True)


@pytest.mark.parametrize("invalid", [1, "true", None, [], {}])
def test_profile_consent_rejects_non_boolean_values(invalid):
    with patch("hermes_cli.universal_models.service.ModelService") as service:
        response = server._methods["orchestration.profile"](
            "request", {"action": "activate", "name": "remote-team", "confirm_cloud": invalid},
        )
    assert response["error"]["code"] == 4002
    service.assert_not_called()
