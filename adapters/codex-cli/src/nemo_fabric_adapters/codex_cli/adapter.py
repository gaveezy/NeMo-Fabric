# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the Codex CLI (``codex exec``) through the Fabric adapter contract."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

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
SANDBOXES = ("read-only", "workspace-write", "danger-full-access")
DEFAULT_SANDBOX = "read-only"
INHERITED_ENV_NAMES = {
    "APPDATA",
    "CODEX_HOME",
    "CODEX_SQLITE_HOME",
    "COMSPEC",
    "DBUS_SESSION_BUS_ADDRESS",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOCALAPPDATA",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "PATH",
    "PATHEXT",
    "SHELL",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "XDG_RUNTIME_DIR",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}


@dataclass(frozen=True)
class CodexCliRelaySettings:
    """Runtime-scoped Relay state consumed by the Codex CLI adapter."""

    gateway: relay_gateway.RelayGatewayLaunch
    plugin_config: dict[str, Any]


class CodexCliAdapterError(Exception):
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


class AdapterInputError(CodexCliAdapterError):
    """Invalid Fabric invocation input."""


class AdapterConfigError(CodexCliAdapterError):
    """Invalid Codex CLI adapter configuration."""


class AdapterRelayError(CodexCliAdapterError):
    """NeMo Relay setup or lifecycle failure."""


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AdapterConfigError(
            "codex_cli_invalid_configuration", f"{name} must be a mapping"
        )
    return value


def _settings(config: AgentConfig) -> dict[str, Any]:
    return config.harness.settings if config.harness else {}


def request_prompt(payload: dict[str, Any]) -> str:
    value = (payload.get("request") or {}).get("input")
    if not isinstance(value, str):
        raise AdapterInputError(
            "codex_cli_invalid_request", "Codex input must be text"
        )
    return value


def resolve_cwd(context: RuntimeContext, base_dir: str) -> Path:
    path = Path(context.environment.workspace or base_dir)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path.resolve()


def _selected_model_config(config: AgentConfig) -> AgentModelConfig:
    model = config.models.get("default")
    if model is None and len(config.models) == 1:
        model = next(iter(config.models.values()))
    if model is None:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "Codex requires a default model or exactly one model",
        )
    return model


def selected_model(config: AgentConfig) -> str:
    model = _selected_model_config(config)
    return (
        model.model.removeprefix("openai/")
        if model.provider == "openai"
        else model.model
    )


def sandbox(config: AgentConfig) -> str:
    value = _settings(config).get("sandbox", DEFAULT_SANDBOX)
    if value not in SANDBOXES:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            f"sandbox must be one of: {', '.join(SANDBOXES)}",
        )
    return value


def skip_git_repo_check(config: AgentConfig) -> bool:
    value = _settings(config).get("skip_git_repo_check", False)
    if not isinstance(value, bool):
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "skip_git_repo_check must be a boolean",
        )
    return value


def timeout_seconds() -> float:
    return DEFAULT_TIMEOUT_SECONDS


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".codex"


def generated_profile(context: RuntimeContext) -> tuple[str, Path]:
    name = f"fabric-{context.runtime_id}"
    return name, codex_home() / f"{name}.config.toml"


def resolve_codex_command(base_dir: str) -> str:
    override = os.environ.get("FABRIC_TEST_CODEX_BIN")
    if not override:
        return "codex"
    path = Path(override)
    if not path.is_absolute():
        path = (Path(base_dir) / path).resolve()
    return str(path)


def custom_model_provider_config(
    config: AgentConfig, context: RuntimeContext
) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    provider = model_config.provider
    if provider == "openai":
        return {}
    api_key_env = model_config.api_key_env
    if api_key_env is None:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "selected model api_key_env is required for a custom "
            "Responses-compatible provider",
        )
    if not (context.environment.env.get(api_key_env) or os.environ.get(api_key_env)):
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            f"{api_key_env} is required for the selected model provider",
        )
    base_url = model_config.base_url
    if not base_url:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "selected model base_url is required for a custom "
            "Responses-compatible provider",
        )
    return {
        "model_provider": provider,
        "model_providers": {
            provider: {
                "name": provider,
                "base_url": base_url.rstrip("/"),
                "env_key": api_key_env,
                "wire_api": "responses",
            }
        },
    }


def openai_model_provider_config(config: AgentConfig) -> dict[str, Any]:
    model_config = _selected_model_config(config)
    if model_config.provider != "openai":
        return {}
    base_url = model_config.base_url
    return {"openai_base_url": base_url.rstrip("/")} if base_url else {}


def _merge_config(target: dict[str, Any], layer: dict[str, Any]) -> None:
    for key, value in layer.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_config(existing, value)
        else:
            target[key] = value


def _json_value(value: Any, *, name: str) -> Any:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration", f"{name} must be JSON-compatible"
        ) from error
    return value


def _apply_config_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> None:
    for dotted_key, value in sorted(overrides.items()):
        if not isinstance(dotted_key, str):
            raise AdapterConfigError(
                "codex_cli_invalid_configuration",
                "config_overrides keys must be strings",
            )
        parts = dotted_key.split(".")
        if any(not part for part in parts):
            raise AdapterConfigError(
                "codex_cli_invalid_configuration",
                f"invalid Codex config override key {dotted_key!r}",
            )
        target = config
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise AdapterConfigError(
                    "codex_cli_invalid_configuration",
                    f"Codex config override {dotted_key!r} conflicts with {part!r}",
                )
            target = existing
        target[parts[-1]] = _json_value(value, name=f"config_overrides.{dotted_key}")


def native_codex_telemetry_config(context: RuntimeContext) -> dict[str, Any]:
    telemetry = context.telemetry
    if telemetry is None or "native" not in telemetry.metadata.get(
        "telemetry_providers", []
    ):
        return {}

    telemetry_config = telemetry.metadata.get("native_config", {})
    if not isinstance(telemetry_config, dict):
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "runtime_context.telemetry.metadata.native_config must be a mapping",
        )
    for component in telemetry_config.get("components") or []:
        if (
            not isinstance(component, dict)
            or component.get("kind") != "observability"
            or not component.get("enabled", True)
        ):
            continue
        component_config = component.get("config") or {}
        opentelemetry = component_config.get("opentelemetry") or {}
        if not isinstance(opentelemetry, dict) or not opentelemetry.get("enabled"):
            continue

        otel: dict[str, Any] = {}
        resource_attributes = opentelemetry.get("resource_attributes") or {}
        environment = resource_attributes.get("deployment.environment")
        if environment is not None:
            otel["environment"] = environment

        endpoint = opentelemetry.get("endpoint")
        if endpoint:
            transport = opentelemetry.get("transport", "http_binary")
            exporters = {
                "http_binary": ("otlp-http", "binary"),
                "grpc": ("otlp-grpc", "grpc"),
                "http_json": ("otlp-http", "json"),
            }
            try:
                exporter, protocol = exporters[transport]
            except (KeyError, TypeError) as error:
                raise AdapterConfigError(
                    "codex_cli_invalid_configuration",
                    f"unsupported Codex native OpenTelemetry transport {transport!r}",
                ) from error
            otel["trace_exporter"] = {
                exporter: {"endpoint": endpoint, "protocol": protocol}
            }
        return {"otel": otel}
    return {}


def prepare_codex_relay(
    agent_name: str,
    config: AgentConfig,
    context: RuntimeContext,
    base_dir: str,
) -> CodexCliRelaySettings | None:
    """Generate runtime-scoped Relay gateway configuration."""

    if context.telemetry is None or not context.telemetry.relay_enabled:
        return None
    command = os.environ.get("FABRIC_TEST_NEMO_RELAY_COMMAND", "nemo-relay")
    try:
        executable = relay_gateway.resolve_relay_command(
            Path(base_dir).resolve(), command
        )
    except FileNotFoundError as error:
        raise AdapterRelayError(
            "codex_cli_relay_unavailable", "NeMo Relay CLI executable was not found"
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
            relay_config={"agents": {"codex": {"command": "codex"}}},
            plugin_config=plugin_config,
            observability_version=relay_contract.observability_version,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        raise AdapterRelayError(
            "codex_cli_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        ) from error
    if config_path is None or plugin_config_path is None:
        raise AdapterRelayError(
            "codex_cli_relay_configuration_failed",
            "NeMo Relay runtime configuration is unavailable",
        )

    base_url = _selected_model_config(config).base_url
    port = relay_gateway.find_available_tcp_port()
    bind = f"127.0.0.1:{port}"
    return CodexCliRelaySettings(
        gateway=relay_gateway.RelayGatewayLaunch(
            executable=executable,
            config_path=config_path,
            bind=bind,
            url=f"http://{bind}",
            log_path=config_path.parent / "gateway.log",
            openai_base_url=base_url.rstrip("/") if base_url else None,
        ),
        plugin_config=plugin_config,
    )


def profile_config(
    config: AgentConfig,
    context: RuntimeContext,
    relay: CodexCliRelaySettings | None,
) -> dict[str, Any]:
    """Build the generated Codex profile configuration for one runtime."""

    result = native_codex_telemetry_config(context)
    _merge_config(result, custom_model_provider_config(config, context))
    _merge_config(result, openai_model_provider_config(config))
    overrides = _mapping(
        _settings(config).get("config_overrides"),
        name="harness.settings.config_overrides",
    )
    _apply_config_overrides(result, overrides)
    if relay is not None:
        provider = _selected_model_config(config).provider
        transport_config = (
            {"openai_base_url": relay.gateway.url}
            if provider == "openai"
            else {"model_providers": {provider: {"base_url": relay.gateway.url}}}
        )
        _merge_config(
            result,
            {
                **transport_config,
                "features": {
                    "hooks": True,
                    # Relay disables delegated multi-agent execution because
                    # Codex encrypts delegated task content before it reaches
                    # the gateway, making those spans opaque.
                    "multi_agent_v2": {"enabled": False},
                },
                "hooks": relay_hooks.render_relay_hooks(
                    "codex", relay.gateway.executable
                )["hooks"],
            },
        )
    return result


def write_profile_config(
    config: AgentConfig,
    context: RuntimeContext,
    relay: CodexCliRelaySettings | None,
) -> tuple[str, Path] | None:
    """Write the generated profile file; return its name and path when needed."""

    document = profile_config(config, context, relay)
    if not document:
        return None
    name, path = generated_profile(context)
    try:
        rendered = tomli_w.dumps(document)
    except (TypeError, ValueError) as error:
        raise AdapterConfigError(
            "codex_cli_invalid_configuration",
            "Codex configuration values must be TOML-compatible",
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return name, path


def build_command(
    config: AgentConfig,
    base_dir: str,
    *,
    profile_name: str | None,
    relay_enabled: bool,
    thread_id: str | None = None,
) -> list[str]:
    """Build one ``codex exec`` invocation reading the prompt from stdin."""

    command = [resolve_codex_command(base_dir), "exec", "--json"]
    command.extend(["--sandbox", sandbox(config)])
    if profile_name is not None:
        command.extend(["--profile", profile_name])
        if relay_enabled:
            # Codex only trusts hooks that the user enabled interactively.
            # Fabric generated and vetted every hook command in the profile, so
            # bypass that interactive trust prompt for this non-interactive run.
            command.append("--dangerously-bypass-hook-trust")
    command.extend(["--model", selected_model(config)])
    if skip_git_repo_check(config):
        command.append("--skip-git-repo-check")
    if thread_id:
        command.extend(["resume", thread_id, "-"])
    else:
        command.append("-")
    return command


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
    model_config = _selected_model_config(config)
    api_key_env = model_config.api_key_env
    if api_key_env is not None and api_key_env in os.environ:
        values[api_key_env] = os.environ[api_key_env]
    values.update(context.environment.env)
    if (
        model_config.provider == "openai"
        and api_key_env is not None
        and api_key_env in values
    ):
        values["OPENAI_API_KEY"] = values[api_key_env]
    if relay_gateway_url is not None:
        values["NEMO_RELAY_GATEWAY_URL"] = relay_gateway_url
    return values


def validate_runtime_payload(
    config: AgentConfig, context: RuntimeContext, base_dir: str
) -> str:
    """Validate runtime-owned configuration before starting CLI or Relay processes."""

    fabric_runtime_id = context.runtime_id
    resolve_cwd(context, base_dir)
    selected_model(config)
    sandbox(config)
    skip_git_repo_check(config)
    child_environment(config, context)
    profile_config(config, context, None)
    return fabric_runtime_id


def parse_events(stdout: str) -> dict[str, Any]:
    """Parse ``codex exec --json`` JSONL output into one turn summary."""

    events: list[dict[str, Any]] = []
    thread_id = None
    response = None
    usage = None
    error = None
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
        event_type = event.get("type")
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
        elif event_type == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                response = item.get("text")
        elif event_type == "turn.completed":
            usage = event.get("usage")
        elif event_type in {"turn.failed", "error"}:
            failure = event.get("error") or event.get("message") or event
            error = (
                failure.get("message") if isinstance(failure, dict) else str(failure)
            )
    return {
        "events": events,
        "thread_id": str(thread_id) if thread_id else None,
        "response": response,
        "usage": usage,
        "error": error,
    }


def _failure(code: str, message: str, **metadata: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "retryable": False}
    if metadata:
        error["metadata"] = metadata
    return {
        "harness": "codex",
        "adapter": "cli",
        "mode": "codex_cli_runtime",
        "response": None,
        "completed": False,
        "failed": True,
        "error": error,
        "events": [],
    }


def adapter_failure(error: CodexCliAdapterError) -> dict[str, Any]:
    return _failure(error.code, error.message, **error.metadata)


def _relay_output(
    output: dict[str, Any],
    relay: CodexCliRelaySettings,
    *,
    artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    output["relay_runtime"] = {
        "enabled": True,
        "emitter": "codex-cli/nemo-relay",
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
    relay: CodexCliRelaySettings | None,
) -> subprocess.Popen[Any] | None:
    if relay is None:
        return None
    try:
        return relay_gateway.start_relay_gateway(
            launch=relay.gateway, cwd=resolve_cwd(context, base_dir)
        )
    except relay_gateway.RelayGatewayError as error:
        raise AdapterRelayError(
            "codex_cli_relay_start_failed",
            "NeMo Relay gateway failed to start",
            metadata={"gateway_log_path": str(relay.gateway.log_path)},
        ) from error


def _cleanup_relay(
    relay: CodexCliRelaySettings | None,
    process: subprocess.Popen[Any] | None,
) -> AdapterRelayError | None:
    if process is None:
        return None
    try:
        relay_gateway.stop_relay_gateway(process)
    except relay_gateway.RelayGatewayError:
        return AdapterRelayError(
            "codex_cli_relay_stop_failed",
            "NeMo Relay gateway failed to stop",
            metadata={
                "gateway_log_path": str(relay.gateway.log_path)
                if relay is not None
                else ""
            },
        )
    return None


def _as_lifecycle_error(error: CodexCliAdapterError) -> lifecycle.LifecycleError:
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
            "codex_cli_invalid_runtime_context",
            "Codex CLI runtime context is invalid",
        ) from error


class CodexCliRuntime:
    """One Codex CLI thread owned by a Fabric runtime."""

    def __init__(self) -> None:
        self._config: AgentConfig | None = None
        self._context: RuntimeContext | None = None
        self._base_dir: str | None = None
        self._fabric_runtime_id: str | None = None
        self._thread_id: str | None = None
        self._profile_name: str | None = None
        self._profile_path: Path | None = None
        self._relay: CodexCliRelaySettings | None = None
        self._gateway_process: subprocess.Popen[Any] | None = None
        self._started = False
        self._unusable = False

    async def start(self, payload: dict[str, Any]) -> None:
        if self._started:
            raise lifecycle.LifecycleError(
                "codex_cli_runtime_already_started",
                "Codex CLI runtime is already started",
            )
        try:
            agent_config = payload["config"]
            context = _runtime_context(payload)
            base_dir = common_utils.base_dir(payload)
            fabric_runtime_id = validate_runtime_payload(
                agent_config, context, base_dir
            )
            relay = prepare_codex_relay(
                common_utils.agent_name(payload), agent_config, context, base_dir
            )
            self._relay = relay
            self._gateway_process = _start_relay_gateway(context, base_dir, relay)
            generated = await asyncio.to_thread(
                write_profile_config, agent_config, context, relay
            )
            if generated is not None:
                self._profile_name, self._profile_path = generated
        except CodexCliAdapterError as error:
            await self._cleanup_failed_start()
            raise _as_lifecycle_error(error) from error
        except OSError as error:
            await self._cleanup_failed_start()
            raise lifecycle.LifecycleError(
                "codex_cli_configuration_failed",
                "Codex CLI runtime configuration could not be written",
            ) from error
        except BaseException:
            await self._cleanup_failed_start()
            raise

        self._config = agent_config
        self._context = context
        self._base_dir = base_dir
        self._fabric_runtime_id = fabric_runtime_id
        self._started = True

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        config = self._config
        base_dir = self._base_dir
        if (
            not self._started
            or config is None
            or base_dir is None
            or self._fabric_runtime_id is None
        ):
            raise lifecycle.LifecycleError(
                "codex_cli_runtime_not_started",
                "Codex CLI runtime is not started",
            )
        runtime_context = _runtime_context(invocation)
        if runtime_context.runtime_id != self._fabric_runtime_id:
            raise lifecycle.LifecycleError(
                "codex_cli_runtime_mismatch",
                "Codex CLI invocation does not match the connected runtime",
            )
        if self._unusable:
            return _failure(
                "codex_cli_runtime_unavailable",
                "Codex CLI runtime cannot accept another invocation after a runtime failure",
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
                                "codex_cli_relay_atif_timeout",
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
        except CodexCliAdapterError as error:
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
            profile_name=self._profile_name,
            relay_enabled=self._relay is not None,
            thread_id=self._thread_id,
        )
        environment = child_environment(
            config,
            context,
            relay_gateway_url=self._relay.gateway.url if self._relay else None,
        )
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
                "codex_cli_timed_out",
                f"Codex CLI timed out after {timeout:g} seconds",
                returncode=TIMEOUT_RETURNCODE,
            )
        except OSError:
            return _failure(
                "codex_cli_not_found",
                "Codex CLI executable could not start",
                returncode=LAUNCH_FAILURE_RETURNCODE,
            )

        parsed = parse_events(completed.stdout)
        thread_id = parsed["thread_id"]
        if completed.returncode != 0:
            return _failure(
                "codex_cli_process_failed",
                parsed["error"] or "Codex CLI exited with a non-zero status",
                exit_code=completed.returncode,
            )
        if parsed["error"] is not None:
            return _failure("codex_cli_turn_failed", parsed["error"])
        if parsed["response"] is None:
            return _failure(
                "codex_cli_missing_response",
                "Codex invocation did not return a final agent message",
            )
        if not thread_id:
            return _failure(
                "codex_cli_missing_thread",
                "Codex invocation did not return a thread identity",
            )
        if self._thread_id is not None and thread_id != self._thread_id:
            return _failure(
                "codex_cli_thread_mismatch",
                "Codex resumed a different thread than the connected runtime",
                expected=self._thread_id,
                received=thread_id,
            )
        self._thread_id = thread_id
        return {
            "harness": "codex",
            "adapter": "cli",
            "mode": "codex_cli_runtime",
            "cwd": str(cwd),
            "model": selected_model(config),
            "thread_id": thread_id,
            "response": parsed["response"],
            "usage": parsed["usage"],
            "returncode": completed.returncode,
            "completed": True,
            "failed": False,
            "error": None,
            "events": parsed["events"],
        }

    async def stop(self) -> None:
        self._config = None
        self._context = None
        self._base_dir = None
        self._fabric_runtime_id = None
        self._thread_id = None
        self._started = False
        self._unusable = True

        profile_path = self._profile_path
        self._profile_name = None
        self._profile_path = None
        profile_error: OSError | None = None
        if profile_path is not None:
            try:
                profile_path.unlink(missing_ok=True)
            except OSError as error:
                profile_error = error
                LOGGER.exception("Codex CLI generated profile could not be removed")

        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        if cleanup_error is not None:
            raise _as_lifecycle_error(cleanup_error)
        if profile_error is not None:
            raise lifecycle.LifecycleError(
                "codex_cli_cleanup_failed",
                "Codex CLI generated profile could not be removed",
            ) from profile_error

    async def _cleanup_failed_start(self) -> None:
        profile_path = self._profile_path
        self._profile_name = None
        self._profile_path = None
        if profile_path is not None:
            try:
                profile_path.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception("Codex CLI generated profile could not be removed")
        cleanup_error = _cleanup_relay(self._relay, self._gateway_process)
        self._relay = None
        self._gateway_process = None
        if cleanup_error is not None:
            LOGGER.error(
                "Codex CLI Relay cleanup after start failure also failed: %s",
                cleanup_error.code,
            )


def main() -> None:
    """Serve the persistent local-host lifecycle protocol."""

    lifecycle.serve(CodexCliRuntime, config_loader=AgentConfig.from_mapping)


if __name__ == "__main__":
    main()
