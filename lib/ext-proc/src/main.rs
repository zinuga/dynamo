// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary entry point for the Dynamo ext_proc service.
//!
//! # Usage
//!
//! ```bash
//! dynamo-ext-proc \
//!   --listen-addr "[::]:9003" \
//!   --block-size 16 \
//!   --event-workers 4
//! ```
//!
//! Workers register themselves by calling the HTTP registration endpoint exposed
//! by the standalone KV-router indexer (or by Kubernetes discovery).  For
//! Envoy AI Gateway deployments the registration is handled automatically by the
//! Dynamo operator.

use std::{net::SocketAddr, sync::Arc};

use clap::Parser;
use tonic::transport::Server;

use dynamo_ext_proc::{
    config::Config,
    proto::external_processor_server::ExternalProcessorServer,
    service::KvRoutingExtProc,
    worker_map::WorkerMap,
};
use dynamo_kv_router::{ConcurrentRadixTreeCompressed, ThreadPoolIndexer};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::parse();
    tracing::info!(?config, "starting dynamo-ext-proc");

    let addr: SocketAddr = config.listen_addr.parse()?;

    let indexer = Arc::new(ThreadPoolIndexer::new(
        ConcurrentRadixTreeCompressed::new(),
        config.event_workers,
        config.block_size,
    ));

    let worker_map = Arc::new(WorkerMap::new());

    let svc = KvRoutingExtProc::new(Arc::new(config), indexer, worker_map);

    tracing::info!(%addr, "ext_proc gRPC server listening");

    Server::builder()
        .add_service(ExternalProcessorServer::new(svc))
        .serve(addr)
        .await?;

    Ok(())
}
