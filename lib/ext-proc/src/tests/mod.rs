// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests: spin up a real gRPC server and drive it with a test client.

use std::sync::Arc;

use tokio_stream::iter as stream_iter;
use tonic::transport::Server;
use tonic::Request;

use dynamo_kv_router::{
    ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
    indexer::KvIndexerInterface,
    protocols::{
        BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
        KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
        compute_block_hash_for_seq,
    },
};

use crate::{
    config::Config,
    proto::{
        external_processor_client::ExternalProcessorClient,
        external_processor_server::ExternalProcessorServer,
        processing_request::Request as PhaseReq,
        processing_response::Response as PhaseResp,
        HeaderValue, HttpBody, HttpHeaders, ProcessingRequest,
    },
    service::KvRoutingExtProc,
    worker_map::WorkerMap,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_config(block_size: u32) -> Arc<Config> {
    Arc::new(Config {
        listen_addr: "[::]:0".into(),
        block_size,
        event_workers: 1,
        destination_header: "x-gateway-destination-endpoint".into(),
        worker_id_header: "x-worker-instance-id".into(),
        dp_rank_header: "x-dp-rank".into(),
    })
}

/// Spawn a test gRPC server on an ephemeral port.
/// Returns a connected client and the shared WorkerMap.
async fn start_server(
    block_size: u32,
) -> (ExternalProcessorClient<tonic::transport::Channel>, Arc<WorkerMap>) {
    let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = Arc::new(
        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, block_size),
    );
    let worker_map = Arc::new(WorkerMap::new());
    let config = test_config(block_size);
    let svc = KvRoutingExtProc::new(config, Arc::clone(&indexer), Arc::clone(&worker_map));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        Server::builder()
            .add_service(ExternalProcessorServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Brief pause for the server to start accepting.
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();

    (ExternalProcessorClient::new(channel), worker_map)
}

fn header_req(key: &str, value: &str) -> ProcessingRequest {
    ProcessingRequest {
        request: Some(PhaseReq::RequestHeaders(HttpHeaders {
            headers: vec![HeaderValue {
                key: key.to_string(),
                value: value.to_string(),
                raw_value: vec![],
            }],
            end_of_stream: false,
        })),
    }
}

fn body_req(body: impl Into<Vec<u8>>) -> ProcessingRequest {
    ProcessingRequest {
        request: Some(PhaseReq::RequestBody(HttpBody {
            body: body.into(),
            end_of_stream: true,
        })),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn integration_request_headers_pass_through() {
    let (mut client, _) = start_server(4).await;

    let requests = vec![header_req("content-type", "application/json")];
    let mut resp_stream = client
        .process(Request::new(stream_iter(requests)))
        .await
        .unwrap()
        .into_inner();

    let resp = resp_stream.message().await.unwrap().unwrap();
    assert!(
        matches!(resp.response, Some(PhaseResp::RequestHeaders(_))),
        "expected RequestHeaders response"
    );
}

#[tokio::test]
async fn integration_body_injects_routing_headers() {
    let (mut client, worker_map) = start_server(4).await;
    worker_map.register(42, 0, "pod-42:8000".into());

    let body_json = serde_json::json!({
        "messages": [{"role": "user", "content": "hello world"}]
    });
    let body_bytes = serde_json::to_vec(&body_json).unwrap();

    let requests = vec![body_req(body_bytes)];
    let mut resp_stream = client
        .process(Request::new(stream_iter(requests)))
        .await
        .unwrap()
        .into_inner();

    let resp = resp_stream.message().await.unwrap().unwrap();
    let mutation = if let Some(PhaseResp::RequestBody(b)) = resp.response {
        b.response.unwrap().header_mutation.unwrap()
    } else {
        panic!("expected RequestBody with header mutation");
    };

    let keys: Vec<&str> = mutation
        .set_headers
        .iter()
        .map(|h| h.header.as_ref().unwrap().key.as_str())
        .collect();
    assert!(keys.contains(&"x-gateway-destination-endpoint"));
    assert!(keys.contains(&"x-worker-instance-id"));
    assert!(keys.contains(&"x-dp-rank"));

    let dest_val = mutation
        .set_headers
        .iter()
        .find(|h| h.header.as_ref().unwrap().key == "x-gateway-destination-endpoint")
        .unwrap()
        .header
        .as_ref()
        .unwrap()
        .value
        .as_str();
    assert_eq!(dest_val, "pod-42:8000");
}

#[tokio::test]
async fn integration_precomputed_tokens_via_header() {
    let (mut client, worker_map) = start_server(2).await;
    worker_map.register(55, 0, "pod-55:8000".into());

    let requests = vec![
        header_req("x-dynamo-token-ids", "10,20,30,40"),
        body_req(b"{}".as_ref()),
    ];
    let mut resp_stream = client
        .process(Request::new(stream_iter(requests)))
        .await
        .unwrap()
        .into_inner();

    // headers phase
    let _headers_resp = resp_stream.message().await.unwrap().unwrap();

    // body phase should have routing headers (token IDs extracted from header)
    let body_resp = resp_stream.message().await.unwrap().unwrap();
    assert!(
        matches!(body_resp.response, Some(PhaseResp::RequestBody(_))),
        "expected RequestBody response"
    );
    if let Some(PhaseResp::RequestBody(b)) = body_resp.response {
        assert!(b.response.unwrap().header_mutation.is_some());
    }
}

#[tokio::test]
async fn integration_no_workers_body_passes_through() {
    let (mut client, _worker_map) = start_server(4).await;
    // No workers registered → routing falls through.

    let body_json = serde_json::json!({
        "messages": [{"role": "user", "content": "hello"}]
    });
    let requests = vec![body_req(serde_json::to_vec(&body_json).unwrap())];
    let mut resp_stream = client
        .process(Request::new(stream_iter(requests)))
        .await
        .unwrap()
        .into_inner();

    let resp = resp_stream.message().await.unwrap().unwrap();
    if let Some(PhaseResp::RequestBody(b)) = resp.response {
        assert!(
            b.response.unwrap().header_mutation.is_none(),
            "no header mutation expected when no workers registered"
        );
    } else {
        panic!("expected RequestBody response");
    }
}

#[tokio::test]
async fn integration_kv_overlap_routes_to_matching_worker() {
    const BLOCK_SIZE: u32 = 2;

    let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = Arc::new(
        ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, BLOCK_SIZE),
    );
    let worker_map = Arc::new(WorkerMap::new());
    worker_map.register(1, 0, "worker-1:8000".into());
    worker_map.register(2, 0, "worker-2:8000".into());

    // Tokens [10, 20, 30, 40] → 2 blocks at BLOCK_SIZE=2.
    let tokens: Vec<u32> = vec![10, 20, 30, 40];
    let hashes = compute_block_hash_for_seq(&tokens, BLOCK_SIZE, BlockHashOptions::default());

    // Register block hashes with worker 1.
    indexer
        .apply_event(RouterEvent {
            worker_id: 1,
            storage_tier: StorageTier::Device,
            event: KvCacheEvent {
                event_id: 0,
                dp_rank: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: hashes
                        .iter()
                        .map(|h| KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(h.0),
                            tokens_hash: *h,
                            mm_extra_info: None,
                        })
                        .collect(),
                }),
            },
        })
        .await;
    indexer.flush().await;

    let config = test_config(BLOCK_SIZE);
    let svc = KvRoutingExtProc::new(config, Arc::clone(&indexer), Arc::clone(&worker_map));

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        Server::builder()
            .add_service(ExternalProcessorServer::new(svc))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    let channel = tonic::transport::Channel::from_shared(format!("http://{addr}"))
        .unwrap()
        .connect_lazy();
    let mut client = ExternalProcessorClient::new(channel);

    // Send token IDs via header so we bypass text tokenisation.
    let token_ids_str: String = tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",");

    let requests = vec![
        header_req("x-dynamo-token-ids", &token_ids_str),
        body_req(b"{}".as_ref()),
    ];
    let mut resp_stream = client
        .process(Request::new(stream_iter(requests)))
        .await
        .unwrap()
        .into_inner();

    let _hdr = resp_stream.message().await.unwrap().unwrap();
    let body_resp = resp_stream.message().await.unwrap().unwrap();

    let mutation = if let Some(PhaseResp::RequestBody(b)) = body_resp.response {
        b.response.unwrap().header_mutation.unwrap()
    } else {
        panic!("expected RequestBody response");
    };

    let dest = mutation
        .set_headers
        .iter()
        .find(|h| h.header.as_ref().unwrap().key == "x-gateway-destination-endpoint")
        .and_then(|h| h.header.as_ref())
        .map(|h| h.value.as_str())
        .expect("destination header present");

    // Worker 1 has all the KV blocks; it must win.
    assert_eq!(dest, "worker-1:8000");
}
