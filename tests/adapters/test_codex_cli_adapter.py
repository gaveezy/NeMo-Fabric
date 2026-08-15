# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.codex_cli import adapter


def runtime_input(payload):
    return (
        AgentConfig.from_mapping(payload["config"]),
        RuntimeContext.from_mapping(payload["runtime_context"]),
        payload["base_dir"],
    )


def lifecycle_start_payload(payload):
    config, context, _ = runtime_input(payload)
    return {
        **payload,
        "config": config,
        "runtime_context": context,
        "request": None,
    }


def lifecycle_invocation(payload):
    return {
        "runtime_context": runtime_input(payload)[1],
        "request": payload["request"],
    }


async def invoke_once_async(payload):
    runtime = adapter.CodexCliRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        return await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()


def invoke_once(payload):
    return asyncio.run(invoke_once_async(payload))


def runtime_start_error(payload):
    async def scenario() -> adapter.lifecycle.LifecycleError:
        runtime = adapter.CodexCliRuntime()
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(lifecycle_start_payload(payload))
        return caught.value

    return asyncio.run(scenario())


@pytest.fixture(name="codex_payload")
def codex_payload_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "agent_name": "codex-cli-test",
        "base_dir": str(tmp_path),
        "config": {
            "harness": {
                "settings": {
                    "sandbox": "workspace-write",
                    "skip_git_repo_check": True,
                    "config_overrides": {
                        "features.web_search": False,
                        "model_reasoning_effort": "high",
                    },
                },
            },
            "models": {
                "default": {
                    "provider": "openai",
                    "model": "openai/gpt-5.4",
                }
            },
        },
        "runtime_context": {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-codex-cli-1",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(workspace),
            },
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        "request": {"input": "Inspect the change."},
    }


def write_mock_codex(path: Path, *, log_path: Path, echo_resumed_thread=True):
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
with open({str(log_path)!r}, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + chr(10))
if {echo_resumed_thread!r} and "resume" in args:
    thread_id = args[args.index("resume") + 1]
else:
    thread_id = "thread-fake"
prompt = sys.stdin.read().strip()
events = [
    {{"type": "thread.started", "thread_id": thread_id}},
    {{"type": "turn.started"}},
    {{
        "type": "item.completed",
        "item": {{
            "id": "item-1",
            "type": "agent_message",
            "text": thread_id + ":" + prompt,
        }},
    }},
    {{
        "type": "turn.completed",
        "usage": {{"input_tokens": 1, "cached_input_tokens": 0, "output_tokens": 1}},
    }},
]
for event in events:
    print(json.dumps(event))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def argv_log(log_path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relay_settings(tmp_path: Path, plugin_config=None):
    return adapter.CodexCliRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay" / "gateway.log",
        ),
        plugin_config=plugin_config or {"version": 1, "components": []},
    )


def test_build_command_maps_normalized_config(codex_payload):
    config, _, base_dir = runtime_input(codex_payload)

    command = adapter.build_command(
        config,
        base_dir,
        profile_name=None,
        relay_enabled=False,
    )

    assert command[0] == "codex"
    assert command[1:3] == ["exec", "--json"]
    assert command[command.index("--sandbox") + 1] == "workspace-write"
    assert command[command.index("--model") + 1] == "gpt-5.4"
    assert "--skip-git-repo-check" in command
    assert "--profile" not in command
    assert "--dangerously-bypass-hook-trust" not in command
    assert command[-1] == "-"


def test_build_command_resumes_thread_with_profile_and_relay(codex_payload):
    config, _, base_dir = runtime_input(codex_payload)

    command = adapter.build_command(
        config,
        base_dir,
        profile_name="fabric-runtime-1",
        relay_enabled=True,
        thread_id="thread-123",
    )

    assert command[command.index("--profile") + 1] == "fabric-runtime-1"
    assert "--dangerously-bypass-hook-trust" in command
    assert command[-3:] == ["resume", "thread-123", "-"]


def test_write_profile_config_merges_overrides(codex_payload, tmp_path):
    config, context, _ = runtime_input(codex_payload)

    generated = adapter.write_profile_config(config, context, None)

    assert generated is not None
    name, path = generated
    assert name == "fabric-runtime-1"
    assert path == tmp_path / "codex-home" / "fabric-runtime-1.config.toml"
    assert tomllib.loads(path.read_text(encoding="utf-8")) == {
        "features": {"web_search": False},
        "model_reasoning_effort": "high",
    }


def test_profile_config_routes_codex_through_relay_gateway(codex_payload, tmp_path):
    config, context, _ = runtime_input(codex_payload)
    relay = relay_settings(tmp_path)

    document = adapter.profile_config(config, context, relay)

    assert document["model_provider"] == "nemo-relay-openai"
    assert document["model_providers"]["nemo-relay-openai"] == {
        "name": "NeMo Relay OpenAI",
        "base_url": "http://127.0.0.1:43210",
        "wire_api": "responses",
        "requires_openai_auth": True,
        "supports_websockets": False,
    }
    assert document["features"] == {
        "hooks": True,
        "multi_agent_v2": {"enabled": False},
        "web_search": False,
    }
    hook = document["hooks"]["SessionStart"][0]["hooks"][0]
    assert hook["command"].endswith("hook-forward codex")
    assert document["model_reasoning_effort"] == "high"


def test_profile_config_routes_custom_provider_through_relay(
    codex_payload, monkeypatch, tmp_path
):
    monkeypatch.setenv("CUSTOM_KEY", "credential")
    codex_payload["config"]["models"]["default"] = {
        "provider": "custom",
        "model": "custom-model",
        "api_key_env": "CUSTOM_KEY",
        "base_url": "https://models.example.test/v1",
    }
    config, context, _ = runtime_input(codex_payload)
    relay = relay_settings(tmp_path)

    document = adapter.profile_config(config, context, relay)

    assert document["model_provider"] == "custom"
    provider = document["model_providers"]["custom"]
    assert provider["base_url"] == "http://127.0.0.1:43210"
    assert provider["env_key"] == "CUSTOM_KEY"
    assert provider["wire_api"] == "responses"
    assert provider["supports_websockets"] is False


def test_prepare_codex_relay_configures_gateway(codex_payload, monkeypatch, tmp_path):
    codex_payload["config"]["models"]["default"]["base_url"] = (
        "https://api.openai.example.test/v1/"
    )
    codex_payload["runtime_context"]["telemetry"] = {"relay_enabled": True}
    relay_wrapper = tmp_path / "relay-config.json"
    relay_wrapper.write_text(json.dumps({"relay": {"config": {}}}), encoding="utf-8")
    monkeypatch.setenv("FABRIC_RELAY_CONFIG_PATH", str(relay_wrapper))
    relay_executable = tmp_path / "bin" / "nemo-relay"
    relay_executable.parent.mkdir()
    relay_executable.touch()
    monkeypatch.setattr(
        adapter.relay_gateway,
        "resolve_relay_command",
        MagicMock(return_value=relay_executable),
    )
    monkeypatch.setattr(
        adapter.relay_gateway,
        "relay_cli_contract",
        MagicMock(
            return_value=adapter.relay_gateway.RelayCliContract(
                version=(0, 6, 0), observability_version=2
            )
        ),
    )
    monkeypatch.setattr(
        adapter.relay_gateway, "find_available_tcp_port", MagicMock(return_value=43210)
    )
    config, context, base_dir = runtime_input(codex_payload)

    relay = adapter.prepare_codex_relay("codex-cli-test", config, context, base_dir)

    assert relay is not None
    assert relay.gateway.url == "http://127.0.0.1:43210"
    assert relay.gateway.openai_base_url == "https://api.openai.example.test/v1"
    relay_config = relay.gateway.config_path.read_text(encoding="utf-8")
    assert 'command = "codex"' in relay_config


async def test_runtime_reuses_codex_thread_across_invocations(
    codex_payload, monkeypatch, tmp_path
):
    log_path = tmp_path / "codex-argv.jsonl"
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))

    runtime = adapter.CodexCliRuntime()
    await runtime.start(lifecycle_start_payload(codex_payload))
    try:
        first = await runtime.invoke(lifecycle_invocation(codex_payload))
        codex_payload["request"] = {"input": "Continue."}
        second = await runtime.invoke(lifecycle_invocation(codex_payload))
    finally:
        await runtime.stop()

    assert first["completed"] is True
    assert first["response"] == "thread-fake:Inspect the change."
    assert first["adapter"] == "cli"
    assert second["response"] == "thread-fake:Continue."
    assert first["thread_id"] == second["thread_id"] == "thread-fake"
    commands = argv_log(log_path)
    assert "resume" not in commands[0]
    assert commands[1][-3:] == ["resume", "thread-fake", "-"]
    profile_path = tmp_path / "codex-home" / "fabric-runtime-1.config.toml"
    assert not profile_path.exists()


def test_thread_mismatch_is_reported(codex_payload, monkeypatch, tmp_path):
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(
        mock_codex,
        log_path=tmp_path / "codex-argv.jsonl",
        echo_resumed_thread=False,
    )
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))

    async def scenario():
        runtime = adapter.CodexCliRuntime()
        runtime._thread_id = "thread-persisted"
        await runtime.start(lifecycle_start_payload(codex_payload))
        try:
            return await runtime.invoke(lifecycle_invocation(codex_payload))
        finally:
            await runtime.stop()

    output = asyncio.run(scenario())

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_cli_thread_mismatch"


def test_child_environment_maps_openai_credentials(codex_payload, monkeypatch):
    monkeypatch.setenv("FABRIC_UNRELATED_SECRET", "do-not-forward")
    monkeypatch.setenv("MY_OPENAI_KEY", "credential")
    codex_payload["config"]["models"]["default"]["api_key_env"] = "MY_OPENAI_KEY"
    config, context, _ = runtime_input(codex_payload)

    environment = adapter.child_environment(
        config, context, relay_gateway_url="http://127.0.0.1:43210"
    )

    assert environment["FABRIC_UNRELATED_SECRET"] == ""
    assert environment["OPENAI_API_KEY"] == "credential"
    assert environment["NEMO_RELAY_GATEWAY_URL"] == "http://127.0.0.1:43210"


def test_invalid_sandbox_fails_start(codex_payload):
    codex_payload["config"]["harness"]["settings"]["sandbox"] = "invalid"

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"


def test_custom_provider_requires_credentials(codex_payload):
    codex_payload["config"]["models"]["default"] = {
        "provider": "custom",
        "model": "custom-model",
    }

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"


def test_missing_thread_identity_is_reported(codex_payload, monkeypatch, tmp_path):
    mock_codex = tmp_path / "mock-codex"
    mock_codex.write_text(
        """#!/usr/bin/env python3
import json
import sys

sys.stdin.read()
print(json.dumps({
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "untracked"},
}))
""",
        encoding="utf-8",
    )
    mock_codex.chmod(0o755)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_cli_missing_thread"


def test_missing_response_is_reported(codex_payload, monkeypatch, tmp_path):
    mock_codex = tmp_path / "mock-codex"
    mock_codex.write_text(
        """#!/usr/bin/env python3
import json
import sys

sys.stdin.read()
print(json.dumps({"type": "thread.started", "thread_id": "thread-123"}))
print(json.dumps({"type": "turn.completed", "usage": {}}))
""",
        encoding="utf-8",
    )
    mock_codex.chmod(0o755)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_cli_missing_response"


def test_missing_cli_is_reported(codex_payload, monkeypatch, tmp_path):
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(tmp_path / "missing-codex"))

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_cli_not_found"


def test_structured_input_is_rejected(codex_payload, monkeypatch, tmp_path):
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=tmp_path / "codex-argv.jsonl")
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))
    codex_payload["request"] = {"input": {"messages": []}}

    output = invoke_once(codex_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "codex_cli_invalid_request"


def test_lifecycle_entrypoint_serves_ordered_operations(
    codex_payload, monkeypatch, tmp_path
):
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=tmp_path / "codex-argv.jsonl")
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))
    requests = [
        {
            "operation": "start",
            "payload": {
                "agent_name": codex_payload["agent_name"],
                "base_dir": codex_payload["base_dir"],
                "config": codex_payload["config"],
                "runtime_context": codex_payload["runtime_context"],
                "capability_plan": {},
            },
        },
        {
            "operation": "invoke",
            "payload": {
                "runtime_context": codex_payload["runtime_context"],
                "request": codex_payload["request"],
            },
        },
        {"operation": "stop", "payload": {"runtime_id": "runtime-1"}},
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.codex_cli.adapter"],
        input="".join(json.dumps(request) + "\n" for request in requests),
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )

    responses = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [response["operation"] for response in responses] == [
        "start",
        "invoke",
        "stop",
    ]
    assert all(
        response["outcome"]["status"] == "succeeded" for response in responses
    )
    output = responses[1]["outcome"]["output"]
    assert output["response"] == "thread-fake:Inspect the change."


async def test_fabric_runtime_invokes_codex_cli_then_resumes(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    log_path = tmp_path / "codex-argv.jsonl"
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))
    config = {
        "metadata": {"name": "codex-cli-runtime-test"},
        "harness": {
            "adapter_id": "nvidia.fabric.codex.cli",
            "resolution": "preinstalled",
            "settings": {"skip_git_repo_check": True},
        },
        "models": {"default": {"provider": "openai", "model": "openai/gpt-5.4"}},
        "runtime": {"artifacts": str(tmp_path / "artifacts")},
        "environment": {
            "provider": "local",
            "workspace": str(tmp_path),
            "artifacts": str(tmp_path / "artifacts"),
        },
    }

    from nemo_fabric import FabricConfig

    async with await Fabric().start_runtime(
        FabricConfig.from_mapping(config),
        base_dir=tmp_path,
    ) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.output["response"] == "thread-fake:first"
    assert second.output["response"] == "thread-fake:second"
    assert first.output["thread_id"] == second.output["thread_id"] == "thread-fake"
    commands = argv_log(log_path)
    assert "resume" not in commands[0]
    assert commands[1][-3:] == ["resume", "thread-fake", "-"]


def test_plan_resolves_codex_cli_descriptor(tmp_path):
    from nemo_fabric import FabricConfig

    plan = Fabric().plan(
        FabricConfig.from_mapping(
            {
                "metadata": {"name": "codex-cli-plan-test"},
                "harness": {
                    "adapter_id": "nvidia.fabric.codex.cli",
                    "resolution": "preinstalled",
                },
            }
        ),
        base_dir=tmp_path,
    )

    assert plan.adapter.adapter_id == "nvidia.fabric.codex.cli"
    assert plan.adapter.harness == "codex"


def make_skill(tmp_path: Path, name: str) -> Path:
    skill = tmp_path / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    return skill


def test_profile_config_maps_instructions_and_mcp(codex_payload):
    codex_payload["config"]["instructions"] = {
        "system": {"content": "Review carefully.", "mode": "replace"}
    }
    codex_payload["config"]["mcp"] = {
        "servers": {
            "files": {
                "transport": "stdio",
                "url": "mcp-files",
                "args": ["--root", "."],
                "env": {"FILES_TOKEN": "token"},
            },
            "search": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/search",
            },
        }
    }
    config, context, _ = runtime_input(codex_payload)

    document = adapter.profile_config(config, context, None)

    assert document["instructions"] == "Review carefully."
    assert document["mcp_servers"]["files"] == {
        "command": "mcp-files",
        "args": ["--root", "."],
        "env": {"FILES_TOKEN": "token"},
    }
    assert document["mcp_servers"]["search"] == {
        "url": "https://mcp.example.test/search"
    }


def test_mcp_authentication_is_rejected(codex_payload):
    codex_payload["config"]["mcp"] = {
        "servers": {
            "secure": {
                "transport": "http",
                "url": "https://mcp.example.test",
                "authentication": {"type": "oauth2"},
            }
        }
    }

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"


def test_invalid_skill_path_is_rejected(codex_payload):
    codex_payload["config"]["skills"] = {"paths": ["missing-skill"]}

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"


async def test_skills_are_linked_into_workspace_and_cleaned_up(
    codex_payload, monkeypatch, tmp_path
):
    skill = make_skill(tmp_path, "review")
    codex_payload["config"]["skills"] = {"paths": ["skills/review"]}
    log_path = tmp_path / "codex-argv.jsonl"
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))
    workspace = Path(codex_payload["runtime_context"]["environment"]["workspace"])

    runtime = adapter.CodexCliRuntime()
    await runtime.start(lifecycle_start_payload(codex_payload))
    link = workspace / ".agents" / "skills" / "review"
    assert link.is_symlink()
    assert link.resolve() == skill.resolve()
    try:
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
    finally:
        await runtime.stop()

    assert output["completed"] is True
    assert not link.exists()
    assert not (workspace / ".agents").exists()


def test_skill_link_collision_is_rejected(codex_payload, tmp_path):
    make_skill(tmp_path, "review")
    codex_payload["config"]["skills"] = {"paths": ["skills/review"]}
    workspace = Path(codex_payload["runtime_context"]["environment"]["workspace"])
    existing = workspace / ".agents" / "skills" / "review"
    existing.mkdir(parents=True)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"
    assert existing.is_dir()


def test_profile_config_maps_extra_settings(codex_payload):
    codex_payload["config"]["harness"]["settings"].update(
        {
            "approval_mode": "deny_all",
            "personality": "pragmatic",
            "reasoning_effort": "xhigh",
            "service_tier": "priority",
            "config_overrides": {},
        }
    )
    config, context, _ = runtime_input(codex_payload)

    document = adapter.profile_config(config, context, None)

    assert document["approval_policy"] == "never"
    assert document["personality"] == "pragmatic"
    assert document["model_reasoning_effort"] == "xhigh"
    assert document["service_tier"] == "priority"


def test_config_overrides_win_over_named_settings(codex_payload):
    # Dotted config_overrides are the raw escape hatch and take precedence
    # over the normalized named settings.
    codex_payload["config"]["harness"]["settings"]["reasoning_effort"] = "xhigh"
    config, context, _ = runtime_input(codex_payload)

    document = adapter.profile_config(config, context, None)

    assert document["model_reasoning_effort"] == "high"


def test_auto_review_keeps_native_approval_default(codex_payload):
    codex_payload["config"]["harness"]["settings"]["approval_mode"] = "auto_review"
    config, context, _ = runtime_input(codex_payload)

    document = adapter.profile_config(config, context, None)

    assert "approval_policy" not in document


@pytest.mark.parametrize(
    "settings",
    [
        {"approval_mode": "ask"},
        {"personality": "terse"},
        {"reasoning_effort": "extreme"},
        {"service_tier": ""},
        {"output_schema": []},
    ],
)
def test_invalid_extra_settings_fail_start(codex_payload, settings):
    codex_payload["config"]["harness"]["settings"].update(settings)

    error = runtime_start_error(codex_payload)

    assert error.code == "codex_cli_invalid_configuration"


async def test_output_schema_is_staged_and_cleaned_up(
    codex_payload, monkeypatch, tmp_path
):
    schema = {"type": "object", "properties": {"summary": {"type": "string"}}}
    codex_payload["config"]["harness"]["settings"]["output_schema"] = schema
    log_path = tmp_path / "codex-argv.jsonl"
    mock_codex = tmp_path / "mock-codex"
    write_mock_codex(mock_codex, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CODEX_BIN", str(mock_codex))

    runtime = adapter.CodexCliRuntime()
    await runtime.start(lifecycle_start_payload(codex_payload))
    schema_path = runtime._output_schema_path
    assert json.loads(schema_path.read_text(encoding="utf-8")) == schema
    try:
        output = await runtime.invoke(lifecycle_invocation(codex_payload))
    finally:
        await runtime.stop()

    assert output["completed"] is True
    command = argv_log(log_path)[0]
    assert command[command.index("--output-schema") + 1] == str(schema_path)
    assert not schema_path.parent.exists()
