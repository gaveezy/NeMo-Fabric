# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nemo_fabric import Fabric
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.claude_code_cli import adapter


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
    runtime = adapter.ClaudeCodeCliRuntime()
    await runtime.start(lifecycle_start_payload(payload))
    try:
        return await runtime.invoke(lifecycle_invocation(payload))
    finally:
        await runtime.stop()


def invoke_once(payload):
    return asyncio.run(invoke_once_async(payload))


def runtime_start_error(payload):
    async def scenario() -> adapter.lifecycle.LifecycleError:
        runtime = adapter.ClaudeCodeCliRuntime()
        with pytest.raises(adapter.lifecycle.LifecycleError) as caught:
            await runtime.start(lifecycle_start_payload(payload))
        return caught.value

    return asyncio.run(scenario())


@pytest.fixture(name="claude_payload")
def claude_payload_fixture(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return {
        "agent_name": "claude-code-cli-test",
        "base_dir": str(tmp_path),
        "config": {
            "harness": {
                "settings": {"permission_mode": "dontAsk"},
            },
            "models": {
                "default": {
                    "provider": "anthropic",
                    "model": "anthropic/claude-fable-5",
                }
            },
            "instructions": {
                "system": {"content": "Review carefully.", "mode": "replace"}
            },
            "runtime": {"max_turns": 4},
            "tools": {"blocked": ["Bash", "WebSearch"]},
        },
        "runtime_context": {
            "runtime_id": "runtime-1",
            "invocation_id": "invocation-1",
            "request_id": "request-1",
            "environment": {
                "environment_id": "environment-claude-code-cli-1",
                "provider": "local",
                "control_location": "in_env_control",
                "ownership": "caller_owned",
                "workspace": str(workspace),
            },
            "artifacts": {"root": str(tmp_path / "artifacts")},
        },
        "request": {"input": "Inspect the change."},
    }


def write_mock_claude(path: Path, *, log_path: Path, returncode=0, emit_result=True):
    path.write_text(
        f"""#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
with open({str(log_path)!r}, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + chr(10))
session = args[args.index("--resume") + 1] if "--resume" in args else "session-fake"
prompt = sys.stdin.read().strip()
print(json.dumps({{"type": "system", "subtype": "init", "session_id": session}}))
if {emit_result!r}:
    print(json.dumps({{
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "session_id": session,
        "result": session + ":" + prompt,
        "usage": {{"input_tokens": 1, "output_tokens": 2}},
        "total_cost_usd": 0.001,
        "duration_ms": 10,
        "duration_api_ms": 8,
    }}))
sys.exit({returncode})
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


def test_build_command_maps_normalized_config(claude_payload):
    config, _, base_dir = runtime_input(claude_payload)

    command = adapter.build_command(
        config,
        base_dir,
        session_id=None,
        settings_path=None,
    )

    assert command[0] == "claude"
    assert command[1:5] == ["--print", "--output-format", "stream-json", "--verbose"]
    assert command[command.index("--model") + 1] == "claude-fable-5"
    assert command[command.index("--system-prompt") + 1] == "Review carefully."
    assert command[command.index("--max-turns") + 1] == "4"
    assert command[command.index("--disallowed-tools") + 1] == "Bash,WebSearch"
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert "--resume" not in command
    assert "--settings" not in command


def test_build_command_appends_session_and_settings(claude_payload, tmp_path):
    config, _, base_dir = runtime_input(claude_payload)

    command = adapter.build_command(
        config,
        base_dir,
        session_id="session-1",
        settings_path=tmp_path / "settings.json",
    )

    assert command[command.index("--resume") + 1] == "session-1"
    assert command[command.index("--settings") + 1] == str(tmp_path / "settings.json")


async def test_runtime_invokes_and_resumes_session(
    claude_payload, monkeypatch, tmp_path
):
    log_path = tmp_path / "claude-argv.jsonl"
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))

    runtime = adapter.ClaudeCodeCliRuntime()
    await runtime.start(lifecycle_start_payload(claude_payload))
    try:
        first = await runtime.invoke(lifecycle_invocation(claude_payload))
        claude_payload["request"] = {"input": "Continue."}
        second = await runtime.invoke(lifecycle_invocation(claude_payload))
    finally:
        await runtime.stop()

    assert first["completed"] is True
    assert first["response"] == "session-fake:Inspect the change."
    assert first["session_id"] == "session-fake"
    assert first["adapter"] == "cli"
    assert first["usage"] == {"input_tokens": 1, "output_tokens": 2}
    assert second["response"] == "session-fake:Continue."
    commands = argv_log(log_path)
    assert "--resume" not in commands[0]
    assert commands[1][commands[1].index("--resume") + 1] == "session-fake"


def test_child_environment_is_deny_by_default(claude_payload, monkeypatch):
    monkeypatch.setenv("FABRIC_UNRELATED_SECRET", "do-not-forward")
    monkeypatch.setenv("MY_ANTHROPIC_KEY", "credential")
    claude_payload["config"]["models"]["default"]["api_key_env"] = "MY_ANTHROPIC_KEY"
    config, context, _ = runtime_input(claude_payload)

    environment = adapter.child_environment(
        config, context, relay_gateway_url="http://127.0.0.1:43210"
    )

    assert environment["FABRIC_UNRELATED_SECRET"] == ""
    assert environment["ANTHROPIC_API_KEY"] == "credential"
    assert environment["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:43210"
    assert environment["NEMO_RELAY_GATEWAY_URL"] == "http://127.0.0.1:43210"


def test_custom_provider_requires_credentials_and_endpoint(claude_payload):
    claude_payload["config"]["models"]["default"] = {
        "provider": "custom",
        "model": "custom-model",
    }

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


def test_invalid_permission_mode_fails_start(claude_payload):
    claude_payload["config"]["harness"]["settings"]["permission_mode"] = "invalid"

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


def test_prepare_claude_relay_stages_hook_settings(
    claude_payload, monkeypatch, tmp_path
):
    claude_payload["config"]["models"]["default"] = {
        "provider": "custom",
        "model": "custom-model",
        "api_key_env": "CUSTOM_KEY",
        "base_url": "https://models.example.test/v1",
    }
    claude_payload["runtime_context"]["telemetry"] = {"relay_enabled": True}
    monkeypatch.setenv("CUSTOM_KEY", "credential")
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
    config, context, base_dir = runtime_input(claude_payload)

    relay = adapter.prepare_claude_relay("claude-code-cli-test", config, context, base_dir)

    assert relay is not None
    assert relay.gateway.url == "http://127.0.0.1:43210"
    assert relay.gateway.anthropic_base_url == "https://models.example.test"
    settings = json.loads(relay.settings_path.read_text(encoding="utf-8"))
    hook = settings["hooks"]["SessionStart"][0]["hooks"][0]
    assert hook["command"].endswith("hook-forward claude")
    assert settings["hooks"]["PreToolUse"][0]["matcher"] == "*"
    relay_config = (relay.gateway.config_path).read_text(encoding="utf-8")
    assert 'command = "claude"' in relay_config


async def test_relay_invocation_reports_runtime_and_artifacts(
    claude_payload, monkeypatch, tmp_path
):
    log_path = tmp_path / "claude-argv.jsonl"
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))
    relay = adapter.ClaudeCodeCliRelaySettings(
        gateway=adapter.relay_gateway.RelayGatewayLaunch(
            executable=tmp_path / "nemo-relay",
            config_path=tmp_path / "relay" / "config.toml",
            bind="127.0.0.1:43210",
            url="http://127.0.0.1:43210",
            log_path=tmp_path / "relay" / "gateway.log",
        ),
        plugin_config={"version": 1, "components": []},
        settings_path=tmp_path / "relay" / "claude-settings.json",
    )
    relay.settings_path.parent.mkdir(parents=True)
    relay.settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        adapter, "prepare_claude_relay", MagicMock(return_value=relay)
    )
    gateway_process = MagicMock()
    monkeypatch.setattr(
        adapter.relay_gateway,
        "start_relay_gateway",
        MagicMock(return_value=gateway_process),
    )
    stop_gateway = MagicMock()
    monkeypatch.setattr(adapter.relay_gateway, "stop_relay_gateway", stop_gateway)

    output = await invoke_once_async(claude_payload)

    assert output["completed"] is True
    assert output["relay_runtime"]["enabled"] is True
    assert output["relay_runtime"]["gateway_url"] == "http://127.0.0.1:43210"
    assert output["relay_artifacts"] == []
    commands = argv_log(log_path)
    assert commands[0][commands[0].index("--settings") + 1] == str(relay.settings_path)
    stop_gateway.assert_called_once_with(gateway_process)
    assert not relay.settings_path.exists()


def test_process_failure_returns_structured_error(
    claude_payload, monkeypatch, tmp_path
):
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(
        mock_claude,
        log_path=tmp_path / "claude-argv.jsonl",
        returncode=2,
        emit_result=False,
    )
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))

    output = invoke_once(claude_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "claude_code_cli_process_failed"
    assert output["error"]["metadata"]["exit_code"] == 2


def test_missing_result_is_reported(claude_payload, monkeypatch, tmp_path):
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(
        mock_claude, log_path=tmp_path / "claude-argv.jsonl", emit_result=False
    )
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))

    output = invoke_once(claude_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "claude_code_cli_missing_result"


def test_missing_cli_is_reported(claude_payload, monkeypatch, tmp_path):
    monkeypatch.setenv(
        "FABRIC_TEST_CLAUDE_CLI_PATH", str(tmp_path / "missing-claude")
    )

    output = invoke_once(claude_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "claude_code_cli_not_found"


def test_structured_input_is_rejected(claude_payload, monkeypatch, tmp_path):
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=tmp_path / "claude-argv.jsonl")
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))
    claude_payload["request"] = {"input": {"messages": []}}

    output = invoke_once(claude_payload)

    assert output["failed"] is True
    assert output["error"]["code"] == "claude_code_cli_invalid_request"


def test_lifecycle_entrypoint_serves_ordered_operations(
    claude_payload, monkeypatch, tmp_path
):
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=tmp_path / "claude-argv.jsonl")
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))
    requests = [
        {
            "operation": "start",
            "payload": {
                "agent_name": claude_payload["agent_name"],
                "base_dir": claude_payload["base_dir"],
                "config": claude_payload["config"],
                "runtime_context": claude_payload["runtime_context"],
                "capability_plan": {},
            },
        },
        {
            "operation": "invoke",
            "payload": {
                "runtime_context": claude_payload["runtime_context"],
                "request": claude_payload["request"],
            },
        },
        {"operation": "stop", "payload": {"runtime_id": "runtime-1"}},
    ]

    completed = subprocess.run(
        [sys.executable, "-m", "nemo_fabric_adapters.claude_code_cli.adapter"],
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
    assert output["response"] == "session-fake:Inspect the change."


async def test_fabric_runtime_invokes_claude_code_cli_then_resumes(
    monkeypatch, tmp_path
):
    log_path = tmp_path / "claude-argv.jsonl"
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))
    config = {
        "metadata": {"name": "claude-code-cli-runtime-test"},
        "harness": {
            "adapter_id": "nvidia.fabric.claude.code.cli",
            "resolution": "preinstalled",
        },
        "models": {
            "default": {"provider": "anthropic", "model": "claude-fable-5"}
        },
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

    assert first.output["response"] == "session-fake:first"
    assert second.output["response"] == "session-fake:second"
    assert first.output["session_id"] == second.output["session_id"]
    commands = argv_log(log_path)
    assert "--resume" not in commands[0]
    assert commands[1][commands[1].index("--resume") + 1] == "session-fake"


def test_plan_resolves_claude_code_cli_descriptor(tmp_path):
    from nemo_fabric import FabricConfig

    plan = Fabric().plan(
        FabricConfig.from_mapping(
            {
                "metadata": {"name": "claude-code-cli-plan-test"},
                "harness": {
                    "adapter_id": "nvidia.fabric.claude.code.cli",
                    "resolution": "preinstalled",
                },
            }
        ),
        base_dir=tmp_path,
    )

    assert plan.adapter.adapter_id == "nvidia.fabric.claude.code.cli"
    assert plan.adapter.harness == "claude"


def make_skill(tmp_path: Path, name: str) -> Path:
    skill = tmp_path / "skills" / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# skill\n", encoding="utf-8")
    return skill


def test_mcp_config_is_staged_with_projected_credentials(claude_payload):
    claude_payload["config"]["mcp"] = {
        "servers": {
            "files": {
                "transport": "stdio",
                "url": "mcp-files",
                "args": ["--root", "."],
                "env": {"FILES_TOKEN": "secret-value"},
            },
            "search": {
                "transport": "streamable-http",
                "url": "https://mcp.example.test/search",
                "custom_headers": {"X-Api-Key": "${SEARCH_KEY}"},
            },
        }
    }
    config, context, base_dir = runtime_input(claude_payload)

    staged = adapter._stage_mcp_config(config, context, base_dir)

    assert staged is not None
    document = json.loads(staged.config_path.read_text(encoding="utf-8"))
    files = document["mcpServers"]["files"]
    assert files["type"] == "stdio"
    assert files["command"] == "mcp-files"
    assert "secret-value" not in staged.config_path.read_text(encoding="utf-8")
    projected = files["env"]["FILES_TOKEN"]
    assert projected.startswith("${NEMO_FABRIC_CLAUDE_CODE_CLI_MCP_")
    assert staged.environment[projected[2:-1]] == "secret-value"
    assert document["mcpServers"]["search"] == {
        "type": "http",
        "url": "https://mcp.example.test/search",
        "headers": {"X-Api-Key": "${SEARCH_KEY}"},
    }
    adapter._cleanup_mcp_config(staged.config_path)
    assert not staged.config_path.parent.exists()


def test_mcp_authentication_is_rejected(claude_payload):
    claude_payload["config"]["mcp"] = {
        "servers": {
            "secure": {
                "transport": "http",
                "url": "https://mcp.example.test",
                "authentication": {"type": "oauth2"},
            }
        }
    }

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


def test_invalid_skill_path_is_rejected(claude_payload):
    claude_payload["config"]["skills"] = {"paths": ["missing-skill"]}

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


async def test_skills_and_mcp_reach_the_cli_and_are_cleaned_up(
    claude_payload, monkeypatch, tmp_path
):
    make_skill(tmp_path, "review")
    claude_payload["config"]["skills"] = {"paths": ["skills/review"]}
    claude_payload["config"]["mcp"] = {
        "servers": {"search": {"transport": "http", "url": "https://mcp.example.test"}}
    }
    log_path = tmp_path / "claude-argv.jsonl"
    mock_claude = tmp_path / "mock-claude"
    write_mock_claude(mock_claude, log_path=log_path)
    monkeypatch.setenv("FABRIC_TEST_CLAUDE_CLI_PATH", str(mock_claude))

    runtime = adapter.ClaudeCodeCliRuntime()
    await runtime.start(lifecycle_start_payload(claude_payload))
    mcp_config_path = runtime._mcp.config_path
    plugin_root = runtime._skill_plugin_root
    assert (plugin_root / "skills" / "review" / "SKILL.md").is_file()
    try:
        output = await runtime.invoke(lifecycle_invocation(claude_payload))
    finally:
        await runtime.stop()

    assert output["completed"] is True
    command = argv_log(log_path)[0]
    assert command[command.index("--mcp-config") + 1] == str(mcp_config_path)
    assert "--strict-mcp-config" in command
    assert command[command.index("--plugin-dir") + 1] == str(plugin_root)
    assert command[command.index("--allowedTools") + 1] == "Skill"
    assert not mcp_config_path.parent.exists()
    assert not plugin_root.exists()


def test_build_command_maps_budget_and_setting_sources(claude_payload):
    claude_payload["config"]["harness"]["settings"].update(
        {"max_budget_usd": 2.5, "setting_sources": ["user", "project"]}
    )
    config, _, base_dir = runtime_input(claude_payload)

    command = adapter.build_command(config, base_dir)

    assert command[command.index("--max-budget-usd") + 1] == "2.5"
    assert "--setting-sources=user,project" in command


@pytest.mark.parametrize(
    "settings",
    [
        {"max_budget_usd": 0},
        {"max_budget_usd": True},
        {"setting_sources": "project"},
        {"setting_sources": ["invalid"]},
    ],
)
def test_invalid_budget_and_sources_fail_start(claude_payload, settings):
    claude_payload["config"]["harness"]["settings"].update(settings)

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


def test_tool_definitions_are_projected_onto_mcp(claude_payload):
    claude_payload["config"]["tools"] = {
        "definitions": {
            "lint": {
                "kind": "mcp_stdio",
                "ref": "acme-lint-mcp",
                "settings": {"args": ["--serve"], "env": {"LINT_TOKEN": "token"}},
            }
        },
        "blocked": ["Bash"],
    }
    config, context, base_dir = runtime_input(claude_payload)

    staged = adapter._stage_mcp_config(config, context, base_dir)

    assert staged is not None
    document = json.loads(staged.config_path.read_text(encoding="utf-8"))
    lint = document["mcpServers"]["lint"]
    assert lint["type"] == "stdio"
    assert lint["command"] == "acme-lint-mcp"
    assert lint["args"] == ["--serve"]
    projected = lint["env"]["LINT_TOKEN"]
    assert projected.startswith("${NEMO_FABRIC_CLAUDE_CODE_CLI_MCP_")
    assert staged.environment[projected[2:-1]] == "token"
    adapter._cleanup_mcp_config(staged.config_path)


def test_tool_definition_kind_is_bounded(claude_payload):
    claude_payload["config"]["tools"] = {
        "definitions": {
            "lint": {"kind": "python_entrypoint", "ref": "acme.tools:lint"}
        }
    }

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"


def test_tool_definition_name_collision_is_rejected(claude_payload):
    claude_payload["config"]["mcp"] = {
        "servers": {"lint": {"transport": "http", "url": "https://mcp.example.test"}}
    }
    claude_payload["config"]["tools"] = {
        "definitions": {"lint": {"kind": "mcp_stdio", "ref": "acme-lint-mcp"}}
    }

    error = runtime_start_error(claude_payload)

    assert error.code == "claude_code_cli_invalid_configuration"
