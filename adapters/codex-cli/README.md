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

The adapter accepts `models` and `models.base_url`. Harness settings support
`sandbox`, `skip_git_repo_check`, and dotted `config_overrides` merged into a
generated, runtime-scoped Codex profile (`$CODEX_HOME/fabric-<runtime>.config.toml`)
that is removed when the runtime stops. Skills, MCP servers, and structured
instructions are not supported by this adapter; use the SDK-based
`nvidia.fabric.codex` adapter for those capabilities.

## Relay Observability

With NeMo Relay enabled, the adapter launches the Relay gateway, routes model
traffic through it via the generated profile, enables Codex hooks that call
`nemo-relay hook-forward codex`, and passes `--dangerously-bypass-hook-trust`
because NeMo Fabric generated and vetted every hook command. Completed turns
wait for a finalized ATIF artifact and report `relay_runtime` and
`relay_artifacts` in the output.
