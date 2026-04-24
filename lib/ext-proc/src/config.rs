// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use clap::Parser;

/// Command-line / environment-variable configuration for the ext_proc binary.
#[derive(Debug, Clone, Parser)]
#[command(name = "dynamo-ext-proc", about = "Envoy ext_proc KV-cache routing service")]
pub struct Config {
    /// Address to listen on for the ext_proc gRPC service.
    #[arg(long, env = "EXT_PROC_LISTEN_ADDR", default_value = "[::]:9003")]
    pub listen_addr: String,

    /// KV cache block size (number of tokens per block).
    /// Must match the block size used by the workers.
    #[arg(long, env = "EXT_PROC_BLOCK_SIZE", default_value = "16")]
    pub block_size: u32,

    /// Number of event-worker threads for the `ThreadPoolIndexer`.
    #[arg(long, env = "EXT_PROC_EVENT_WORKERS", default_value = "4")]
    pub event_workers: usize,

    /// Header name that ext_proc sets to communicate the chosen backend endpoint.
    /// Envoy AI Gateway reads `x-gateway-destination-endpoint` for direct routing.
    #[arg(
        long,
        env = "EXT_PROC_DESTINATION_HEADER",
        default_value = "x-gateway-destination-endpoint"
    )]
    pub destination_header: String,

    /// Header name that carries the worker instance ID (for compatibility with
    /// the existing Go EPP / Dynamo frontend).
    #[arg(
        long,
        env = "EXT_PROC_WORKER_ID_HEADER",
        default_value = "x-worker-instance-id"
    )]
    pub worker_id_header: String,

    /// Header name that carries the DP rank.
    #[arg(
        long,
        env = "EXT_PROC_DP_RANK_HEADER",
        default_value = "x-dp-rank"
    )]
    pub dp_rank_header: String,
}
