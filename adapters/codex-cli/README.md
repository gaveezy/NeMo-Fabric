<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Codex CLI Adapter

The `nvidia.fabric.codex.cli` adapter runs the Codex command-line interface
(`codex exec --json`) directly behind NeMo Fabric's normalized invocation
contract, without the Codex Python SDK. Each invocation executes one
non-interactive Codex turn; later turns of the same runtime resume the same
Codex thread with `codex exec resume`.

Use this adapter when the execution environment provides a managed `codex`
executable and installing the Python SDK is undesirable. For SDK-managed
execution, use the `nvidia.fabric.codex` adapter instead.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[codex-cli]"` | Yes | Yes | No | No |
| `pip install nemo-fabric-adapters-codex-cli` | No | Yes | No | No |

The harness is the external `codex` executable; install the Codex CLI
separately and keep it on `PATH`. The `harness` and `full` extras are published
for uniformity and install no additional packages.

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

## Authentication

NeMo Fabric preserves Codex's native credential resolution, including a cached
`codex login`. For the `openai` provider, `models.<role>.api_key_env` maps the
named credential onto `OPENAI_API_KEY` for the runtime-scoped child
environment. For another provider name, configure both
`models.<role>.api_key_env` and `models.<role>.base_url`; the endpoint must
implement the OpenAI Responses protocol. The runtime-scoped mapping does not
change the parent environment.

## Supported Configuration

The adapter accepts `models`, `models.base_url`, `instructions.system`, `mcp`,
and `skills`. Harness settings support `sandbox`, `skip_git_repo_check`,
`approval_mode` (`deny_all` pins the native approval policy to `never`;
`auto_review` keeps the Codex default), `personality`, `reasoning_effort`
(mapped to `model_reasoning_effort`), `service_tier`, `output_schema` (staged
to a file passed with `--output-schema`), and dotted `config_overrides`, which
take precedence over the named settings. Runtime-scoped configuration is
written to a generated Codex profile
(`$CODEX_HOME/fabric-<runtime>.config.toml`) that is removed when the runtime
stops.

`instructions.system` maps to the Codex `instructions` configuration key,
which replaces the request-level system instructions. MCP servers (stdio,
HTTP, and streamable HTTP) map to the profile's `mcp_servers` table; MCP
`authentication` is not supported because `codex exec` has no non-interactive
login path — use the SDK-based `nvidia.fabric.codex` adapter for OAuth-backed
servers.

Skills are surfaced through Codex's documented workspace discovery root: the
adapter symlinks each configured skill into `<workspace>/.agents/skills/<name>`
and removes the links (and any directories it created) when the runtime stops.
A pre-existing workspace entry with the same name fails startup instead of
being replaced.

## Relay Observability

With NeMo Relay enabled, the adapter launches the Relay gateway, routes model
traffic through it via the generated profile, enables Codex hooks that call
`nemo-relay hook-forward codex`, and passes `--dangerously-bypass-hook-trust`
because NeMo Fabric generated and vetted every hook command. Completed turns
wait for a finalized ATIF artifact and report `relay_runtime` and
`relay_artifacts` in the output.
