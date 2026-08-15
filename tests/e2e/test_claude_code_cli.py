# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Claude Code CLI adapter boundary and opt-in real-binary integration tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from _utils.utils import assert_atof_model
from nemo_fabric import (
    EnvironmentConfig,
    Fabric,
    FabricConfig,
    HarnessConfig,
    MetadataConfig,
    ModelConfig,
    RelayAtifConfig,
    RelayAtofConfig,
    RelayAtofFileSinkConfig,
    RelayObservabilityConfig,
    RuntimeConfig,
    ToolsConfig,
)

ADAPTER_ID = "nvidia.fabric.claude.code.cli"


@pytest.fixture(name="use_current_python_for_adapter_discovery", autouse=True)
def use_current_python_for_adapter_discovery_fixture(restore_environ) -> None:
    restore_environ["ADAPTER_PYTHON"] = sys.executable


def write_mock_claude(path: Path, *, log_path: Path, env_log_path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import os
import sys

args = sys.argv[1:]
with open({str(log_path)!r}, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(args) + chr(10))
with open({str(env_log_path)!r}, "a", encoding="utf-8") as stream:
    stream.write(json.dumps({{
        "ANTHROPIC_BASE_URL": os.environ.get("ANTHROPIC_BASE_URL"),
        "NEMO_RELAY_GATEWAY_URL": os.environ.get("NEMO_RELAY_GATEWAY_URL"),
    }}) + chr(10))
session = args[args.index("--resume") + 1] if "--resume" in args else "session-e2e"
prompt = sys.stdin.read().strip()
print(json.dumps({{"type": "system", "subtype": "init", "session_id": session}}))
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
}}))
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def write_mock_relay_gateway(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("nemo-relay 0.6.0")
    raise SystemExit(0)
Path({str(log_path)!r}).write_text(json.dumps(args), encoding="utf-8")
bind = args[args.index("--bind") + 1]
host, port = bind.rsplit(":", 1)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/healthz" else 404)
        self.end_headers()

    def log_message(self, format, *args):
        pass

HTTPServer((host, int(port)), Handler).serve_forever()
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def fabric_config(
    tmp_path: Path,
    *,
    mock_cli: bool = True,
    relay: bool = False,
    atif: bool = True,
    nemo_relay_command: str | Path | None = None,
) -> FabricConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment_env: dict[str, str] = {}
    if mock_cli:
        cli_path = tmp_path / "mock-claude"
        write_mock_claude(
            cli_path,
            log_path=tmp_path / "claude-args.jsonl",
            env_log_path=tmp_path / "claude-env.jsonl",
        )
        environment_env["FABRIC_TEST_CLAUDE_CLI_PATH"] = str(cli_path)
    if nemo_relay_command is not None:
        environment_env["FABRIC_TEST_NEMO_RELAY_COMMAND"] = str(nemo_relay_command)
    config = FabricConfig(
        metadata=MetadataConfig(name="claude-code-cli-e2e"),
        harness=HarnessConfig(
            adapter_id=ADAPTER_ID,
            resolution="preinstalled",
            settings={"permission_mode": "dontAsk", "setting_sources": []},
        ),
        models={
            "default": ModelConfig(provider="anthropic", model="claude-fable-5")
        },
        runtime=RuntimeConfig(artifacts=tmp_path / "artifacts"),
        environment=EnvironmentConfig(
            provider="local",
            workspace=tmp_path / "workspace",
            artifacts=tmp_path / "artifacts",
            env=environment_env,
        ),
    )
    (tmp_path / "workspace").mkdir(exist_ok=True)
    if relay:
        config.enable_relay(
            observability=RelayObservabilityConfig(
                atof=RelayAtofConfig(enabled=True, sinks=[RelayAtofFileSinkConfig()]),
                atif=RelayAtifConfig(enabled=atif),
            )
        )
    return config


def add_capabilities(config: FabricConfig, tmp_path: Path) -> None:
    skill_path = tmp_path / "skills" / "review"
    skill_path.mkdir(parents=True, exist_ok=True)
    (skill_path / "SKILL.md").write_text("# Review\n", encoding="utf-8")
    config.add_skill_path(skill_path)
    config.add_mcp_server(
        "docs", transport="streamable-http", url="https://mcp.example.test"
    )
    config.tools = ToolsConfig(
        blocked=["WebSearch"],
        definitions={
            "lint": {"kind": "mcp_stdio", "ref": "acme-lint-mcp"},
        },
    )


def cli_arguments(tmp_path: Path) -> list[list[str]]:
    return [
        json.loads(line)
        for line in (tmp_path / "claude-args.jsonl").read_text().splitlines()
    ]


async def test_fabric_session_resumes_and_maps_capabilities(tmp_path):
    config = fabric_config(tmp_path)
    add_capabilities(config, tmp_path)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.status == second.status == "succeeded"
    assert first.output["adapter"] == "cli"
    assert first.output["session_id"] == second.output["session_id"] == "session-e2e"
    assert first.output["response"] == "session-e2e:first"
    assert second.output["response"] == "session-e2e:second"
    assert first.metadata["adapter_runner"] == "persistent_local_host"
    arguments = cli_arguments(tmp_path)
    assert "--resume" not in arguments[0]
    assert arguments[1][arguments[1].index("--resume") + 1] == "session-e2e"
    for args in arguments:
        assert "--mcp-config" in args
        assert "--strict-mcp-config" in args
        assert "--plugin-dir" in args
        assert args[args.index("--allowedTools") + 1] == "Skill"
        assert args[args.index("--disallowed-tools") + 1] == "WebSearch"
    mcp_config_path = Path(arguments[0][arguments[0].index("--mcp-config") + 1])
    plugin_root = Path(arguments[0][arguments[0].index("--plugin-dir") + 1])
    assert not mcp_config_path.exists()
    assert not plugin_root.exists()


async def test_relay_routes_mock_claude_cli_through_gateway(tmp_path):
    gateway_log = tmp_path / "gateway-args.json"
    mock_relay = tmp_path / "mock-nemo-relay"
    write_mock_relay_gateway(mock_relay, gateway_log)
    # The mock gateway records its launch arguments but writes no ATIF, so
    # disable ATIF to avoid the adapter's finalization wait.
    config = fabric_config(
        tmp_path, relay=True, atif=False, nemo_relay_command=mock_relay
    )

    result = await Fabric().run(config, base_dir=tmp_path, input="inspect")

    assert result.status == "succeeded"
    assert result.output["relay_runtime"]["enabled"] is True
    gateway_args = json.loads(gateway_log.read_text())
    assert "--bind" in gateway_args
    arguments = cli_arguments(tmp_path)
    settings_path = Path(arguments[0][arguments[0].index("--settings") + 1])
    assert settings_path.name == "claude-settings.json"
    assert not settings_path.exists()
    claude_env = json.loads((tmp_path / "claude-env.jsonl").read_text())
    assert claude_env["ANTHROPIC_BASE_URL"] == result.output["relay_runtime"][
        "gateway_url"
    ]
    assert claude_env["NEMO_RELAY_GATEWAY_URL"] == claude_env["ANTHROPIC_BASE_URL"]


async def test_doctor_probes_claude_binary(tmp_path, restore_environ):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    write_mock_claude(
        binary_dir / "claude",
        log_path=tmp_path / "claude-args.jsonl",
        env_log_path=tmp_path / "claude-env.jsonl",
    )
    restore_environ["PATH"] = os.pathsep.join(
        [str(binary_dir), os.environ.get("PATH", "")]
    )
    config = fabric_config(tmp_path, mock_cli=False)

    report = await Fabric().doctor(config, base_dir=tmp_path)

    assert report.status == "pass"
    assert any(
        check.name == "requirement.binary" and "claude" in check.message
        for check in report.checks
    )


def _mock_backend_config(tmp_path: Path, api_server: str) -> FabricConfig:
    config = fabric_config(tmp_path, mock_cli=False)
    config.models["default"].provider = "fabric-test"
    config.models["default"].model = "fabric-echo"
    config.models["default"].api_key_env = "FABRIC_TEST_API_KEY"
    config.models["default"].base_url = f"{api_server}/v1"
    config.environment.env["FABRIC_TEST_API_KEY"] = "test"
    return config


requires_claude_binary = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="the claude executable is required on PATH",
)


@requires_claude_binary
async def test_real_claude_cli_against_mock_backend(api_server, tmp_path):
    config = _mock_backend_config(tmp_path, api_server)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="first")
        second = await runtime.invoke(input="second")

    assert first.status == second.status == "succeeded"
    assert first.output["adapter"] == "cli"
    assert first.output["session_id"]
    assert first.output["response"]


@requires_claude_binary
@pytest.mark.skipif(
    shutil.which("nemo-relay") is None,
    reason="the nemo-relay executable is required on PATH",
)
@pytest.mark.usefixtures("nemo_relay")
async def test_real_claude_cli_relay_semantics(api_server, tmp_path):
    config = _mock_backend_config(tmp_path, api_server)
    config.enable_relay(
        observability=RelayObservabilityConfig(
            atof=RelayAtofConfig(enabled=True, sinks=[RelayAtofFileSinkConfig()]),
            atif=RelayAtifConfig(enabled=True),
        )
    )

    result = await Fabric().run(config, base_dir=tmp_path, input="inspect")

    assert result.status == "succeeded", result.to_mapping()
    assert {artifact["kind"] for artifact in result.output["relay_artifacts"]} == {
        "atof",
        "atif",
    }
    assert_atof_model(result.output, "fabric-echo")


@pytest.mark.skipif(
    os.environ.get("RUN_FABRIC_CLAUDE_CODE_CLI_INTEGRATION") != "1",
    reason="set RUN_FABRIC_CLAUDE_CODE_CLI_INTEGRATION=1 to run against the live API",
)
async def test_live_claude_cli_session_continuity(tmp_path):
    config = fabric_config(tmp_path, mock_cli=False)

    async with await Fabric().start_runtime(config, base_dir=tmp_path) as runtime:
        first = await runtime.invoke(input="Remember token FABRIC-CLI-CONTINUITY-7")
        second = await runtime.invoke(
            input="Reply only with the token I asked you to remember"
        )

    assert first.status == second.status == "succeeded"
    assert "FABRIC-CLI-CONTINUITY-7" in second.output["response"]
