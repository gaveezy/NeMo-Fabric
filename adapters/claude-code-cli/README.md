<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Fabric Claude Code CLI Adapter

The `nvidia.fabric.claude.code.cli` adapter runs the Claude Code command-line
interface (`claude`) directly behind NeMo Fabric's normalized invocation
contract, without the Claude Agent SDK. Each invocation executes one headless
`claude --print --output-format stream-json` turn and resumes the same
conversation across turns of one runtime.

Use this adapter when the execution environment provides a managed `claude`
executable and installing the Python SDK is undesirable. For SDK-managed
execution, use the `nvidia.fabric.claude` adapter instead.

## Install

The following table shows which components each installation provides:

| Installation | Runtime | Adapter | Harness | NeMo Relay CLI |
| --- | --- | --- | --- | --- |
| `pip install "nemo-fabric[claude-code-cli]"` | Yes | Yes | No | No |
| `pip install nemo-fabric-adapters-claude-code-cli` | No | Yes | No | No |

The harness is the external `claude` executable; install
[Claude Code](https://claude.com/product/claude-code) separately and keep it on
`PATH`. The `harness` and `full` extras are published for uniformity and
install no additional packages.

For split runtime and adapter environments, configure `ADAPTER_PYTHON` and use
matching NeMo Fabric release versions. Refer to the
[installation guide](https://docs.nvidia.com/nemo/fabric/getting-started/install#install-an-adapter-and-harness-without-the-runtime).

## Authentication

NeMo Fabric preserves Claude's native credential resolution. Use an existing
Claude Code login for local development or `ANTHROPIC_API_KEY` for a static API
credential. The native `anthropic` provider needs no explicit endpoint. For
another provider name, configure both `models.<role>.api_key_env` and
`models.<role>.base_url`; the endpoint must implement the Anthropic Messages
protocol. The runtime-scoped environment mapping does not change the parent
environment.

## Supported Configuration

The adapter accepts `models`, `models.base_url`, `instructions.system`,
`runtime.max_turns`, `tools.blocked`, `tools.definitions`, `mcp`, and
`skills`. Harness settings support `permission_mode`, `max_budget_usd`
(`--max-budget-usd`), and `setting_sources` (`--setting-sources`).

Named tool definitions use kind `mcp_stdio`: each definition's `ref` is an
executable serving the tool over stdio MCP (optional `settings.args` and
`settings.env`), staged into the same runtime-scoped MCP configuration as
`mcp.servers`. Definition names must not collide with configured MCP server
names.

MCP servers (stdio, HTTP, streamable HTTP, and SSE) are staged into a
runtime-scoped `--mcp-config` document used with `--strict-mcp-config`;
configured credential values are replaced by generated environment references
so secrets stay out of the staged file. MCP `authentication` is not supported;
use the SDK-based `nvidia.fabric.claude` adapter for OAuth-backed servers.

Skills are staged as a generated local plugin passed with `--plugin-dir`, and
the `Skill` tool is pre-approved for the headless run. `tools.enabled`
remains unsupported.

## Relay Observability

With NeMo Relay enabled, the adapter launches the Relay gateway, routes model
traffic through it (`ANTHROPIC_BASE_URL`), and forwards Claude hook events
through a generated `--settings` document whose hooks call
`nemo-relay hook-forward claude`. Completed turns wait for a finalized ATIF
artifact and report `relay_runtime` and `relay_artifacts` in the output.
