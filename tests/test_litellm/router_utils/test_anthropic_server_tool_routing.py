from typing import Any

import pytest

from litellm import Router
from litellm.router_utils.common_utils import filter_web_search_deployments


@pytest.fixture
def mixed_capability_deployments() -> list[dict[str, Any]]:
    return [
        {
            "model_name": "mixed-claude",
            "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake-key"},
            "model_info": {
                "id": "supports-both",
                "supports_web_search": True,
                "supports_web_fetch": True,
            },
        },
        {
            "model_name": "mixed-claude",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "fake-key",
            },
            "model_info": {
                "id": "search-only",
                "supports_web_search": True,
                "supports_web_fetch": False,
            },
        },
        {
            "model_name": "mixed-claude",
            "litellm_params": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "fake-key",
            },
            "model_info": {
                "id": "supports-neither",
                "supports_web_search": False,
                "supports_web_fetch": False,
            },
        },
    ]


def _deployment_ids(deployments: list[dict] | dict) -> list[str]:
    assert isinstance(deployments, list)
    return [deployment["model_info"]["id"] for deployment in deployments]


@pytest.mark.parametrize(
    "tool_type",
    [
        "web_search",
        "web_search_preview",
        "web_search_20250305",
        "web_search_20990101",
    ],
)
def test_web_search_tool_filters_unsupported_deployments(
    mixed_capability_deployments: list[dict[str, Any]], tool_type: str
) -> None:
    result = filter_web_search_deployments(
        mixed_capability_deployments,
        {"tools": [{"type": tool_type, "name": "web_search"}]},
    )

    assert _deployment_ids(result) == ["supports-both", "search-only"]


@pytest.mark.parametrize("tool_type", ["web_fetch_20250910", "web_fetch_20990101"])
def test_anthropic_web_fetch_tool_filters_unsupported_deployments(
    mixed_capability_deployments: list[dict[str, Any]], tool_type: str
) -> None:
    result = filter_web_search_deployments(
        mixed_capability_deployments,
        {"tools": [{"type": tool_type, "name": "web_fetch"}]},
    )

    assert _deployment_ids(result) == ["supports-both"]


def test_search_and_fetch_require_both_capabilities(
    mixed_capability_deployments: list[dict[str, Any]],
) -> None:
    result = filter_web_search_deployments(
        mixed_capability_deployments,
        {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
            ]
        },
    )

    assert _deployment_ids(result) == ["supports-both"]


def test_missing_capability_metadata_remains_eligible() -> None:
    deployments = [
        {
            "model_name": "mixed-claude",
            "model_info": {"id": "metadata-unknown"},
        }
    ]

    result = filter_web_search_deployments(
        deployments,
        {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
            ]
        },
    )

    assert result == deployments


def test_specific_deployment_dict_is_not_filtered() -> None:
    deployment = {
        "model_name": "mixed-claude",
        "model_info": {
            "id": "explicit-deployment",
            "supports_web_search": False,
            "supports_web_fetch": False,
        },
    }

    result = filter_web_search_deployments(
        deployment,
        {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
            ]
        },
    )

    assert result == deployment


def test_non_server_tool_keeps_all_deployments(
    mixed_capability_deployments: list[dict[str, Any]],
) -> None:
    result = filter_web_search_deployments(
        mixed_capability_deployments,
        {"tools": [{"type": "function", "function": {"name": "web_search_custom"}}]},
    )

    assert result == mixed_capability_deployments


async def test_router_filters_anthropic_server_tool_before_selection(
    mixed_capability_deployments: list[dict[str, Any]],
) -> None:
    router = Router(model_list=mixed_capability_deployments)

    healthy_deployments = await router.async_get_healthy_deployments(
        model="mixed-claude",
        request_kwargs={
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "web_fetch_20250910", "name": "web_fetch"},
            ]
        },
    )

    assert _deployment_ids(healthy_deployments) == ["supports-both"]
