# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Claude Code CLI through the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from nemo_fabric_adapter_contract.codec import ContractValidationError
from nemo_fabric_adapter_contract.models import AgentConfig
from nemo_fabric_adapter_contract.models import AgentModelConfig
from nemo_fabric_adapter_contract.models import RuntimeContext
from nemo_fabric_adapters.common import lifecycle
from nemo_fabric_adapters.common import relay_artifacts
from nemo_fabric_adapters.common import relay_gateway
from nemo_fabric_adapters.common import relay_hooks
from nemo_fabric_adapters.common import utils as common_utils

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 1800.0
TIMEOUT_RETURNCODE = 124
LAUNCH_FAILURE_RETURNCODE = 127
PERMISSION_MODES = {
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
    "auto",
}
SETTING_SOURCES = {"user", "project", "local"}
# The only caller-defined tools Claude Code can execute natively are MCP
# servers, so named tool definitions are bounded to stdio MCP commands.
TOOL_DEFINITION_KIND = "mcp_stdio"
TOOL_DEFINITION_SETTING_NAMES = {"args", "env"}
INHERITED_ENV_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID",
    "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
    "ANTHROPIC_ORGANIZATION_ID",
    "ANTHROPIC_PROFILE",
    "ANTHROPIC_SERVICE_ACCOUNT_ID",
    "ANTHROPIC_WORKSPACE_ID",
    "APPDATA",
    "CLAUDE_CONFIG_DIR",
    "COMSPEC",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


@dataclass(frozen=True)
class ClaudeCodeCliMcpSettings:
    """Staged MCP configuration and its process-only environment values."""

    config_path: Path
    environment: dict[str, str]


@dataclass(frozen=True)
class ClaudeCodeCliRelaySettings:
    """Relay gateway and staged Claude settings owned by one adapter run."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]
    settings_path: Path


class ClaudeCodeCliAdapterError(Exception):
    """Expected adapter error with a stable public code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.metadata = metadata or {}


class AdapterInputError(ClaudeCodeCliAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(ClaudeCodeCliAdapterError):
    """Invalid Claude Code CLI adapter configuration."""


class AdapterRelayError(ClaudeCodeCliAdapterError):
    """NeMo Relay setup or lifecycle failure."""


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness else {}


def request_prompt(payload: dict[str, Any]) -> str:
    value = (payload.get("request") or {}).get("input")
    if not isinstance(value, str):
        raise AdapterInputError(
            "claude_code_cli_invalid_request", "Claude Code input must be text"
        )
    return value


def resolve_cwd(context: RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _selected_model_config(config: AgentConfig) -> AgentModelConfig | None:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    return model


def selected_model(config: AgentConfig) -> str | None:
    model = _selected_model_config(config)
    if model is None:
        return None
    return (
        model.model.removeprefix("anthropic/")
        if model.provider == "anthropic"
        else model.model
    )


def _anthropic_base_url(model: AgentModelConfig | None) -> str | None:
    if model is None or not model.base_url:
        return None
    base_url = model.base_url.rstrip("/")
    return (
        base_url.removesuffix("/v1") if model.provider != "anthropic" else base_url
    )


def _model_environment(
    config: AgentConfig, environment: dict[str, str]
) -> dict[str, str]:
    model = _selected_model_config(config)
    if model is None:
        return {}
    api_key_env = model.api_key_env
    api_key = (
        environment.get(api_key_env) or os.environ.get(api_key_env)
        if api_key_env
        else None
    )
    if model.provider != "anthropic" and api_key_env is None:
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration",
            "selected model api_key_env is required for a custom "
            "Anthropic Messages-compatible provider",
        )
    if api_key_env is not None and not api_key:
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration",
            f"{api_key_env} is required for the selected model provider",
        )
    base_url = _anthropic_base_url(model)
    if model.provider != "anthropic" and not base_url:
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration",
            "selected model base_url is required for a custom "
            "Anthropic Messages-compatible provider",
        )
    values: dict[str, str] = {}
    if api_key:
        values["ANTHROPIC_API_KEY"] = api_key
    if base_url:
        values["ANTHROPIC_BASE_URL"] = base_url
    if model.provider != "anthropic":
        values["ANTHROPIC_AUTH_TOKEN"] = ""
    return values


def child_environment(
    config: AgentConfig,
    context: RuntimeContext,
    *,
    relay_gateway_url: str | None = None,
) -> dict[str, str]:
    values = dict.fromkeys(os.environ, "")
    values.update(
        {name: os.environ[name] for name in INHERITED_ENV_NAMES if name in os.environ}
    )
    telemetry_env = context.telemetry.env if context.telemetry else {}
    values.update(telemetry_env)
    model = _selected_model_config(config)
    api_key_env = model.api_key_env if model is not None else None
    if api_key_env is not None and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    configured = context.environment.env
    values.update(configured)
    model_environment = _model_environment(config, values)
    conflicts = sorted(
        name
        for name, value in model_environment.items()
        if name in configured and configured[name] != value
    )
    if conflicts:
        fields = ", ".join(f"environment.env.{name}" for name in conflicts)
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration",
            f"{fields} conflicts with the selected model configuration; "
            "configure model credentials and endpoints through models.<role>, "
            "or remove the duplicate environment.env values",
        )
    values.update(model_environment)
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
        values["ANTHROPIC_BASE_URL"] = relay_gateway_url
    return values


def _artifact_root(context: RuntimeContext, base_dir: str) -> Path:
    root = context.artifacts.root
    if root:
        return Path(str(root))
    return Path(base_dir) / "artifacts" / "claude-code-cli"


def _native_mcp_servers(config: AgentConfig) -> dict[str, dict[str, Any]]:
    servers = config.mcp.servers if config.mcp else {}
    result: dict[str, dict[str, Any]] = {}
    for name, server in sorted(servers.items()):
        url = os.path.expandvars(server.url).strip()
        if not url:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"MCP server {name} URL is required",
            )
        transport = server.transport.strip().lower().replace("_", "-")
        if transport == "stdio":
            result[name] = {"type": "stdio", "command": url, "args": server.args}
            if env := server.env:
                result[name]["env"] = dict(env)
        elif transport in {"http", "streamable-http"}:
            result[name] = {"type": "http", "url": url}
        elif transport == "sse":
            result[name] = {"type": "sse", "url": url}
        else:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"unsupported MCP transport: {server.transport}",
            )
        if headers := server.custom_headers:
            try:
                common_utils.validate_http_headers(name, headers)
            except ValueError as error:
                raise AdapterConfigError(
                    "claude_code_cli_invalid_configuration", str(error)
                ) from error
            result[name]["headers"] = dict(headers)
        if server.authentication is not None:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"MCP server {name!r} authentication is not supported by the "
                "Claude Code CLI adapter",
            )
    return result


def _tool_definition_servers(config: AgentConfig) -> dict[str, dict[str, Any]]:
    definitions = config.tools.definitions if config.tools else {}
    result: dict[str, dict[str, Any]] = {}
    for name, definition in sorted(definitions.items()):
        if definition.kind != TOOL_DEFINITION_KIND:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"tool definition {name!r} kind must be {TOOL_DEFINITION_KIND!r}",
            )
        unsupported = sorted(set(definition.settings) - TOOL_DEFINITION_SETTING_NAMES)
        if unsupported:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"tool definition {name!r} settings {unsupported} are not supported",
            )
        args = definition.settings.get("args") or []
        if not isinstance(args, list) or any(
            not isinstance(item, str) for item in args
        ):
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"tool definition {name!r} args must be a list of strings",
            )
        env = definition.settings.get("env")
        if env is not None and (
            not isinstance(env, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in env.items()
            )
        ):
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"tool definition {name!r} env must map strings to strings",
            )
        server: dict[str, Any] = {
            "type": "stdio",
            "command": definition.ref,
            "args": list(args),
        }
        if env:
            server["env"] = dict(env)
        result[name] = server
    return result


def _combined_mcp_servers(config: AgentConfig) -> dict[str, dict[str, Any]]:
    """Merge configured MCP servers with tool definitions projected onto MCP."""

    servers = _native_mcp_servers(config)
    for name, server in _tool_definition_servers(config).items():
        if name in servers:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"tool definition {name!r} collides with an MCP server name",
            )
        servers[name] = server
    return servers


def _stage_mcp_config(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> ClaudeCodeCliMcpSettings | None:
    # A dictionary passed on the command line would expose MCP credentials to
    # process inspection, so stage a file and pass only its path in argv. Each
    # configured env credential is replaced by a generated ${VAR} reference and
    # provided through the deliberately scoped child environment, keeping the
    # staged file free of secret values. Claude Code expands the references.
    servers = _combined_mcp_servers(config)
    if not servers:
        return None
    fabric_runtime_id = context.runtime_id
    environment: dict[str, str] = {}

    def project_environment_value(server_name: str, value_name: str, value: str) -> str:
        projection_key = (
            sha256(f"{fabric_runtime_id}\0{server_name}\0{value_name}".encode())
            .hexdigest()
            .upper()
        )
        projected_name = f"NEMO_FABRIC_CLAUDE_CODE_CLI_MCP_{projection_key}"
        environment[projected_name] = value
        return f"${{{projected_name}}}"

    for server_name, server in servers.items():
        raw_environment = server.get("env")
        if raw_environment is None:
            continue
        projected_environment: dict[str, str] = {}
        for variable_name, value in sorted(raw_environment.items()):
            if not isinstance(variable_name, str) or not variable_name:
                raise AdapterConfigError(
                    "claude_code_cli_invalid_configuration",
                    f"MCP server {server_name} env names must be non-empty strings",
                )
            if not isinstance(value, str):
                raise AdapterConfigError(
                    "claude_code_cli_invalid_configuration",
                    f"MCP server {server_name} env values must be strings",
                )
            projected_environment[variable_name] = project_environment_value(
                server_name, variable_name, value
            )
        server["env"] = projected_environment

    config_root = (
        _artifact_root(context, base_dir)
        / ".fabric"
        / "claude-code-cli"
        / "mcp"
        / sha256(fabric_runtime_id.encode()).hexdigest()
    )
    if config_root.exists():
        shutil.rmtree(config_root)
    config_root.mkdir(parents=True, mode=0o700)
    config_root.chmod(0o700)
    config_path = config_root / "mcp.json"
    try:
        descriptor = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"mcpServers": servers}, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except BaseException:
        shutil.rmtree(config_root, ignore_errors=True)
        raise
    return ClaudeCodeCliMcpSettings(config_path=config_path, environment=environment)


def _cleanup_mcp_config(config_path: Path | None) -> None:
    if config_path is None:
        return
    try:
        shutil.rmtree(config_path.parent)
    except OSError:
        LOGGER.exception("Claude Code CLI MCP runtime configuration could not be removed")


def _native_skill_paths(config: AgentConfig, base_dir: str) -> list[Path]:
    values = config.skills.paths if config.skills else []

    paths: list[Path] = []
    names: set[str] = set()
    config_root = Path(base_dir)
    for value in values:
        skill_path = Path(value)
        if not skill_path.is_absolute():
            skill_path = config_root / skill_path
        skill_path = skill_path.resolve()
        if not skill_path.is_dir() or not (skill_path / "SKILL.md").is_file():
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                "NeMo Fabric skill path must be a directory containing SKILL.md: "
                f"{skill_path}",
            )
        name = skill_path.name
        if not name or name in names:
            raise AdapterConfigError(
                "claude_code_cli_invalid_configuration",
                f"NeMo Fabric skill names must be unique: {name}",
            )
        names.add(name)
        paths.append(skill_path)
    return paths


def _stage_skill_plugin(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> Path | None:
    skill_paths = _native_skill_paths(config, base_dir)
    if not skill_paths:
        return None
    plugin_key = sha256(context.runtime_id.encode()).hexdigest()
    plugin_root = (
        _artifact_root(context, base_dir)
        / ".fabric"
        / "claude-code-cli"
        / "plugins"
        / plugin_key
    )
    if plugin_root.exists():
        shutil.rmtree(plugin_root)
    (plugin_root / ".claude-plugin").mkdir(parents=True)
    (plugin_root / "skills").mkdir()
    (plugin_root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps(
            {
                "name": "nemo-fabric-skills",
                "description": "Skills provided by NeMo Fabric",
                "version": "1.0.0",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for skill_path in skill_paths:
        shutil.copytree(skill_path, plugin_root / "skills" / skill_path.name)
    return plugin_root


def _cleanup_skill_plugin(plugin_root: Path | None) -> None:
    if plugin_root is None:
        return
    try:
        shutil.rmtree(plugin_root)
    except OSError:
        LOGGER.exception("Claude Code CLI skill plugin could not be removed")


def permission_mode(config: AgentConfig) -> str | None:
    value = _settings(config).get("permission_mode")
    if value is None:
        return None
    if value not in PERMISSION_MODES:
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration", "permission_mode is invalid"
        )
    return value


def max_budget_usd(config: AgentConfig) -> float | None:
    value = _settings(config).get("max_budget_usd")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration", "max_budget_usd must be positive"
        )
    return float(value)


def setting_sources(config: AgentConfig) -> list[str] | None:
    value = _settings(config).get("setting_sources")
    if value is None:
        return None
    if not isinstance(value, list) or any(
        source not in SETTING_SOURCES for source in value
    ):
        raise AdapterConfigError(
            "claude_code_cli_invalid_configuration", "setting_sources is invalid"
        )
    return list(value)


def timeout_seconds() -> float:
    return DEFAULT_TIMEOUT_SECONDS


def resolve_claude_command(base_dir: str) -> str:
    override = os.environ.get("FABRIC_TEST_CLAUDE_CLI_PATH")
    if not override:
        return "claude"
    path = Path(override)
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return str(path)


def build_command(
    config: AgentConfig,
    base_dir: str,
    *,
    session_id: str | None = None,
    settings_path: Path | None = None,
    mcp_config_path: Path | None = None,
    skill_plugin_root: Path | None = None,
) -> list[str]:
    command = [
        resolve_claude_command(base_dir),
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    model = selected_model(config)
    if model:
        command.extend(["--model", model])
    if config.instructions and config.instructions.system:
        command.extend(["--system-prompt", config.instructions.system.content])
    if config.runtime and config.runtime.max_turns is not None:
        command.extend(["--max-turns", str(config.runtime.max_turns)])
    blocked = config.tools.blocked if config.tools else []
    if blocked:
        command.extend(["--disallowed-tools", ",".join(blocked)])
    mode = permission_mode(config)
    if mode is not None:
        command.extend(["--permission-mode", mode])
    budget = max_budget_usd(config)
    if budget is not None:
        command.extend(["--max-budget-usd", str(budget)])
    sources = setting_sources(config)
    if sources is not None:
        command.append(f"--setting-sources={','.join(sources)}")
    if settings_path is not None:
        command.extend(["--settings", str(settings_path)])
    if mcp_config_path is not None:
        command.extend(["--mcp-config", str(mcp_config_path)])
        command.append("--strict-mcp-config")
    if skill_plugin_root is not None:
        command.extend(["--plugin-dir", str(skill_plugin_root)])
        # Headless runs cannot approve tools interactively; pre-approve the
        # Skill tool so staged skills stay invocable.
        command.extend(["--allowedTools", "Skill"])
    if session_id is not None:
        command.extend(["--resume", session_id])
    return command


def _stage_relay_settings(settings_path: Path, executable: Path) -> None:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            relay_hooks.render_relay_hooks("claude", executable),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare_claude_relay(
    agent_name: str,
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
) -> ClaudeCodeCliRelaySettings | None:
    """Generate Relay gateway configuration and the staged Claude hook settings."""

    if context.telemetry is None or not context.telemetry.relay_enabled:
        return None
    command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(base_dir).resolve(), command
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "claude_code_cli_relay_unavailable",
            "NeMo Relay CLI executable was not found",
        ) from error

    try:
        relay_contract = relay_gateway.relay_cli_contract(executable)
        plugin_config = common_utils.load_relay_plugin_config(
            {
                "agent_name": agent_name,
                "base_dir": base_dir,
                "config": config.to_mapping(),
                "runtime_context": context.to_mapping(),
            }
        )
        config_path, plugin_config_path = common_utils.write_relay_configs(
            relay_config={"agents": {"claude": {"command": "claude"}}},
            plugin_config=plugin_config,
            observability_version=relay_contract.observability_version,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise AdapterRelayError(
            "claude_code_cli_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        ) from error
    if config_path is None or plugin_config_path is None:
        raise AdapterRelayError(
            "claude_code_cli_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        )

    port = relay_gateway.find_available_tcp_port()
    bind = f"127.0.0.1:{port}"
    settings_path = config_path.parent / "claude-settings.json"
    try:
        _stage_relay_settings(settings_path, executable)
    except OSError as error:
        raise AdapterRelayError(
            "claude_code_cli_relay_configuration_failed",
            "Claude Relay hook configuration could not be generated",
        ) from error
    return ClaudeCodeCliRelaySettings(
        gateway=relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=config_path,
            bind=bind,
            url=f"http://{bind}",
            log_path=config_path.parent / "gateway.log",
            anthropic_base_url=_anthropic_base_url(_selected_model_config(config)),
        ),
        plugin_config=plugin_config,
        settings_path=settings_path,
    )


def validate_runtime_payload(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> str:
    """Validate runtime-owned configuration before starting CLI or Relay processes."""

    fabric_runtime_id = context.runtime_id
    resolve_cwd(context, base_dir)
    selected_model(config)
    permission_mode(config)
    max_budget_usd(config)
    setting_sources(config)
    _combined_mcp_servers(config)
    _native_skill_paths(config, base_dir)
    child_environment(config, context)
    build_command(config, base_dir)
    return fabric_runtime_id


def parse_stream_events(stdout: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Parse Claude Code stream-json output into events and the terminal result."""

    events: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "result":
            result = event
    return events, result


def _result_failed(result: dict[str, Any]) -> bool:
    subtype = result.get("subtype")
    return bool(result.get("is_error")) or (
        isinstance(subtype, str) and subtype.startswith("error_")
    )


def _failure(code: str, message: str, **metadata: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "retryable": False}
    if metadata:
        error["metadata"] = metadata
    return {
        "harness": "claude",
        "adapter": "cli",
        "mode": "claude_code_cli_runtime",
        "response": None,
        "completed": False,
        "failed": True,
        "error": error,
        "events": [],
    }


def adapter_failure(error: ClaudeCodeCliAdapterError) -> dict[str, Any]:
    return _failure(error.code, error.message, **error.metadata)


def normalize_result(
    events: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    failed = _result_failed(result)
    error = None
    if failed:
        error = {
            "code": "claude_code_cli_result_failed",
            "message": "Claude Code returned an error result",
            "retryable": False,
            "metadata": {"subtype": result.get("subtype")},
        }
    return {
        "harness": "claude",
        "adapter": "cli",
        "mode": "claude_code_cli_runtime",
        "response": result.get("result"),
        "session_id": result.get("session_id"),
        "usage": result.get("usage") or {},
        "cost_usd": result.get("total_cost_usd"),
        "duration_ms": result.get("duration_ms"),
        "duration_api_ms": result.get("duration_api_ms"),
        "num_turns": result.get("num_turns"),
        "subtype": result.get("subtype"),
        "completed": not failed,
        "failed": failed,
        "error": error,
        "events": events,
    }


def _relay_output(
    output: dict[str, Any],
    relay: ClaudeCodeCliRelaySettings,
    *,
    artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "claude-code-cli/nemo-relay",
        "config_path": os.environ.get("FABRIC_RELAY_CONFIG_PATH"),
        "gateway_config_path": str(relay.gateway.config_path),
        "gateway_url": relay.gateway.url,
        "gateway_log_path": str(relay.gateway.log_path),
    }
    output["relay_artifacts"] = (
        common_utils.collect_relay_artifacts(relay.plugin_config)
        if artifacts is None
        else artifacts
    )
    return output


def _start_relay_gateway(
    context: RuntimeContext,
    base_dir: str,
    relay: ClaudeCodeCliRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway, cwd=resolve_cwd(context, base_dir)
        )
    except relay_gateway.RelayGatewayError as error:
        raise AdapterRelayError(
            "claude_code_cli_relay_start_failed",
            "NeMo Relay gateway failed to start",
            metadata={"gateway_log_path": str(relay.gateway.log_path)},
        ) from error


def _cleanup_relay(
    relay: ClaudeCodeCliRelaySettings | None,
    process: subprocess.Popen[Any] | None,
) -> AdapterRelayError | None:
    cleanup_error: AdapterRelayError | None = None
    if process is not None:
        try:
            relay_gateway.stop_relay_gateway(process)
        except relay_gateway.RelayGatewayError:
            cleanup_error = AdapterRelayError(
                "claude_code_cli_relay_stop_failed",
                "NeMo Relay gateway failed to stop",
                metadata={
                    "gateway_log_path": str(relay.gateway.log_path)
                    if relay is not None
                    else ""
                },
            )
    if relay is not None:
        try:
            relay.settings_path.unlink(missing_ok=True)
        except OSError:
            if cleanup_error is None:
                cleanup_error = AdapterRelayError(
                    "claude_code_cli_relay_cleanup_failed",
                    "Claude Relay hook configuration could not be removed",
                )
    return cleanup_error


def _as_lifecycle_error(error: ClaudeCodeCliAdapterError) -> lifecycle.LifecycleError:
    return lifecycle.LifecycleError(
        error.code,
        error.message,
        metadata=error.metadata,
    )


def _runtime_context(payload: dict[str, Any]) -> RuntimeContext:
    try:
        return RuntimeContext.from_mapping(payload.get("runtime_context"))
    except ContractValidationError as error:
        raise lifecycle.LifecycleError(
            "claude_code_cli_invalid_runtime_context",
            "Claude Code CLI runtime context is invalid",
        ) from error


class ClaudeCodeCliRuntime:
    """One Claude Code CLI conversation owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._config: AgentConfig | None = None
        self._context: RuntimeContext | None = None
        self._base_dir: str | None = None
        self._fabric_runtime_id: str | None = None
        self._claude_session_id: str | None = None
        self._relay: ClaudeCodeCliRelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None
        self._mcp: ClaudeCodeCliMcpSettings | None = None
        self._skill_plugin_root: Path | None = None
        self._started = False
        self._unusable = False

    async def start(self, payload: dict[str, Any]) -> None:
        if self._started:
            raise lifecycle.LifecycleError(
                "claude_code_cli_runtime_already_started",
                "Claude Code CLI runtime is already started",
            )
        try:
            agent_config = payload["config"]
            context = _runtime_context(payload)
            base_dir = common_utils.base_dir(payload)
            fabric_runtime_id = validate_runtime_payload(
                agent_config, context, base_dir
            )
            relay = prepare_claude_relay(
                common_utils.agent_name(payload), agent_config, context, base_dir
            )
            self._relay = relay
            self._gateway_process = _start_relay_gateway(context, base_dir, relay)
            self._mcp = await asyncio.to_thread(
                _stage_mcp_config, agent_config, context, base_dir
            )
            self._skill_plugin_root = await asyncio.to_thread(
                _stage_skill_plugin, agent_config, context, base_dir
            )
        except ClaudeCodeCliAdapterError as error:
            self._cleanup_failed_start()
            raise _as_lifecycle_error(error) from error
        except OSError as error:
            self._cleanup_failed_start()
            raise lifecycle.LifecycleError(
                "claude_code_cli_configuration_failed",
                "Claude Code CLI runtime configuration could not be staged",
            ) from error
        except BaseException:
            self._cleanup_failed_start()
            raise

        self._config = agent_config
        self._context = context
        self._base_dir = base_dir
        self._fabric_runtime_id = fabric_runtime_id
        self._started = True

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        config = self._config
        context = self._context
        base_dir = self._base_dir
        if (
            not self._started
            or config is None
            or context is None
            or base_dir is None
            or self._fabric_runtime_id is None
        ):
            raise lifecycle.LifecycleError(
                "claude_code_cli_runtime_not_started",
                "Claude Code CLI runtime is not started",
            )
        runtime_context = _runtime_context(invocation)
        if runtime_context.runtime_id != self._fabric_runtime_id:
            raise lifecycle.LifecycleError(
                "claude_code_cli_runtime_mismatch",
                "Claude Code CLI invocation does not match the connected runtime",
            )
        if self._unusable:
            return _failure(
                "claude_code_cli_runtime_unavailable",
                "Claude Code CLI runtime cannot accept another invocation after a runtime failure",
            )

        try:
            prompt = request_prompt(invocation)
            relay = self._relay
            atif_before = (
                relay_artifacts.snapshot_atif_files(relay.plugin_config)
                if relay is not None
                and relay_artifacts.expects_local_atif(relay.plugin_config)
                else None
            )
            output = await self._run_turn(config, runtime_context, base_dir, prompt)
            if (
                output.get("completed")
                and relay is not None
                and atif_before is not None
            ):
                finalized = await relay_artifacts.wait_for_finalized_atif(
                    relay.plugin_config, atif_before
                )
                if finalized is None:
                    self._unusable = True
                    return _relay_output(
                        adapter_failure(
                            AdapterRelayError(
                                "claude_code_cli_relay_atif_timeout",
                                "NeMo Relay did not finalize an ATIF artifact before the deadline",
                                metadata={
                                    "timeout_seconds": relay_artifacts.ATIF_FINALIZATION_TIMEOUT_SECONDS,
                                },
                            )
                        ),
                        relay,
                        artifacts=[],
                    )
        except AdapterRelayError as error:
            self._unusable = True
            output = adapter_failure(error)
        except ClaudeCodeCliAdapterError as error:
            output = adapter_failure(error)

        if self._relay is not None:
            output = _relay_output(output, self._relay)
        return output

    async def _run_turn(
        self,
        config: AgentConfig,
        context: RuntimeContext,
        base_dir: str,
        prompt: str,
    ) -> dict[str, Any]:
        command = build_command(
            config,
            base_dir,
            session_id=self._claude_session_id,
            settings_path=self._relay.settings_path if self._relay else None,
            mcp_config_path=self._mcp.config_path if self._mcp else None,
            skill_plugin_root=self._skill_plugin_root,
        )
        environment = child_environment(
            config,
            context,
            relay_gateway_url=self._relay.gateway.url if self._relay else None,
        )
        if self._mcp is not None:
            environment.update(self._mcp.environment)
        cwd = resolve_cwd(context, base_dir)
        timeout = timeout_seconds()
        try:
            completed = await asyncio.to_thread(
                subprocess.run,
                command,
                cwd=cwd,
                env=environment,
                input=prompt,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _failure(
                "claude_code_cli_timed_out",
                f"Claude Code CLI timed out after {timeout:g} seconds",
                returncode=TIMEOUT_RETURNCODE,
            )
        except OSError:
            return _failure(
                "claude_code_cli_not_found",
                "Claude Code executable could not start",
                returncode=LAUNCH_FAILURE_RETURNCODE,
            )

        events, result = parse_stream_events(completed.stdout)
        if result is None:
            if completed.returncode != 0:
                return _failure(
                    "claude_code_cli_process_failed",
                    "Claude Code process failed",
                    exit_code=completed.returncode,
                )
            return _failure(
                "claude_code_cli_missing_result",
                "Claude Code returned no terminal result",
            )
        output = normalize_result(events, result)
        if not output["failed"]:
            session_id = output.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                return _failure(
                    "claude_code_cli_missing_session",
                    "Claude Code did not return a session identity",
                )
            # Each headless run reports the session that must be resumed for the
            # next turn, so track the latest identity instead of pinning one.
            self._claude_session_id = session_id
        return output

    async def stop(self) -> None:
        self._config = None
        self._context = None
        self._base_dir = None
        self._fabric_runtime_id = None
        self._claude_session_id = None
        self._started = False
        self._unusable = True
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        _cleanup_mcp_config(self._mcp.config_path if self._mcp else None)
        self._mcp = None
        _cleanup_skill_plugin(self._skill_plugin_root)
        self._skill_plugin_root = None
        if cleanup_error is not None:
            raise _as_lifecycle_error(cleanup_error)

    def _cleanup_failed_start(self) -> None:
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        _cleanup_mcp_config(self._mcp.config_path if self._mcp else None)
        self._mcp = None
        _cleanup_skill_plugin(self._skill_plugin_root)
        self._skill_plugin_root = None
        if cleanup_error is not None:
            LOGGER.error(
                "Claude Code CLI runtime cleanup after start failure also failed: %s",
                cleanup_error.code,
            )


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(ClaudeCodeCliRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
