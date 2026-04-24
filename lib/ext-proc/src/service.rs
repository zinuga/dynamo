// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! gRPC ExternalProcessor service implementation.
//!
//! One `Process` stream is opened per HTTP request.  The handler receives
//! messages in phase order:
//!
//! 1. `request_headers` — inspect headers; look for `x-dynamo-token-ids`.
//! 2. `request_body`    — tokenise prompt, query KV indexer, set routing headers.
//! 3. `response_headers` / `response_body` — passed through unchanged.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use dynamo_kv_router::indexer::KvIndexerInterface;

use crate::{
    config::Config,
    proto::{
        external_processor_server::ExternalProcessor,
        processing_request::Request as PhaseRequest,
        processing_response::Response as PhaseResponse,
        BodyResponse, CommonResponse, HeaderMutation, HeaderValue, HeaderValueOption,
        HeadersResponse, ProcessingRequest, ProcessingResponse, TrailersResponse,
    },
    routing::{route_request, tokens_from_body, tokens_from_header},
    worker_map::WorkerMap,
};

/// Header sent by the client that carries pre-computed token IDs (comma-separated u32).
/// When present it bypasses text-based tokenisation in the body.
const TOKEN_IDS_HEADER: &str = "x-dynamo-token-ids";

// ---------------------------------------------------------------------------
// Service struct
// ---------------------------------------------------------------------------

/// The ext_proc gRPC service.  One instance is shared across all concurrent
/// request streams.
pub struct KvRoutingExtProc {
    pub config: Arc<Config>,
    pub indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
    pub worker_map: Arc<WorkerMap>,
}

impl KvRoutingExtProc {
    pub fn new(
        config: Arc<Config>,
        indexer: Arc<dyn KvIndexerInterface + Send + Sync>,
        worker_map: Arc<WorkerMap>,
    ) -> Self {
        Self { config, indexer, worker_map }
    }
}

// ---------------------------------------------------------------------------
// ExternalProcessor implementation
// ---------------------------------------------------------------------------

#[tonic::async_trait]
impl ExternalProcessor for KvRoutingExtProc {
    type ProcessStream = ReceiverStream<Result<ProcessingResponse, Status>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let config = Arc::clone(&self.config);
        let indexer = Arc::clone(&self.indexer);
        let worker_map = Arc::clone(&self.worker_map);

        let (tx, rx) = mpsc::channel::<Result<ProcessingResponse, Status>>(16);
        let mut stream = request.into_inner();

        tokio::spawn(async move {
            let mut precomputed_tokens: Option<Vec<u32>> = None;

            while let Ok(Some(phase_req)) = stream.message().await {
                let resp = handle_phase(
                    phase_req,
                    &config,
                    &*indexer,
                    &worker_map,
                    &mut precomputed_tokens,
                )
                .await;

                if tx.send(Ok(resp)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// ---------------------------------------------------------------------------
// Per-phase handler
// ---------------------------------------------------------------------------

pub(crate) async fn handle_phase(
    req: ProcessingRequest,
    config: &Config,
    indexer: &dyn KvIndexerInterface,
    worker_map: &WorkerMap,
    precomputed_tokens: &mut Option<Vec<u32>>,
) -> ProcessingResponse {
    match req.request {
        // ----- Phase 1: Request headers -----
        Some(PhaseRequest::RequestHeaders(ref hdr_msg)) => {
            // Cache pre-computed token IDs from a well-known header.
            for hdr in &hdr_msg.headers {
                if hdr.key.eq_ignore_ascii_case(TOKEN_IDS_HEADER) {
                    if let Some(tokens) = tokens_from_header(&hdr.value) {
                        *precomputed_tokens = Some(tokens);
                    }
                }
            }
            pass_through_request_headers()
        }

        // ----- Phase 2: Request body — routing decision made here -----
        Some(PhaseRequest::RequestBody(ref body_msg)) => {
            let tokens = precomputed_tokens.take().or_else(|| tokens_from_body(&body_msg.body));

            let Some(tokens) = tokens else {
                tracing::warn!("ext_proc: no tokens extracted, passing through");
                return pass_through_request_body();
            };

            match route_request(&tokens, config.block_size, indexer, worker_map).await {
                Some(decision) => {
                    tracing::info!(
                        worker_id = decision.worker.worker_id,
                        dp_rank   = decision.worker.dp_rank,
                        overlap   = decision.overlap_blocks,
                        endpoint  = %decision.endpoint,
                        "ext_proc: routing decision"
                    );
                    request_body_with_routing_headers(
                        &config.destination_header,
                        &config.worker_id_header,
                        &config.dp_rank_header,
                        &decision.endpoint,
                        decision.worker.worker_id,
                        decision.worker.dp_rank,
                    )
                }
                None => {
                    tracing::warn!("ext_proc: no routing decision, passing through");
                    pass_through_request_body()
                }
            }
        }

        // ----- Response phases: pass through -----
        Some(PhaseRequest::ResponseHeaders(_)) => pass_through_response_headers(),
        Some(PhaseRequest::ResponseBody(_)) => pass_through_response_body(),
        Some(PhaseRequest::RequestTrailers(_)) => pass_through_request_trailers(),
        Some(PhaseRequest::ResponseTrailers(_)) => pass_through_response_trailers(),

        None => ProcessingResponse { response: None },
    }
}

// ---------------------------------------------------------------------------
// Response builders
// ---------------------------------------------------------------------------

fn pass_through_request_headers() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::RequestHeaders(HeadersResponse {
            response: Some(empty_common()),
        })),
    }
}

fn pass_through_response_headers() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::ResponseHeaders(HeadersResponse {
            response: Some(empty_common()),
        })),
    }
}

fn pass_through_request_body() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::RequestBody(BodyResponse {
            response: Some(empty_common()),
        })),
    }
}

fn pass_through_response_body() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::ResponseBody(BodyResponse {
            response: Some(empty_common()),
        })),
    }
}

fn pass_through_request_trailers() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::RequestTrailers(TrailersResponse {
            header_mutation: None,
        })),
    }
}

fn pass_through_response_trailers() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(PhaseResponse::ResponseTrailers(TrailersResponse {
            header_mutation: None,
        })),
    }
}

fn empty_common() -> CommonResponse {
    CommonResponse {
        header_mutation: None,
        body_mutation: None,
        clear_route_cache: false,
    }
}

/// Build a `request_body` response that injects KV routing headers.
fn request_body_with_routing_headers(
    destination_header: &str,
    worker_id_header: &str,
    dp_rank_header: &str,
    endpoint: &str,
    worker_id: u64,
    dp_rank: u32,
) -> ProcessingResponse {
    let set_headers = vec![
        make_header(destination_header, endpoint),
        make_header(worker_id_header, &worker_id.to_string()),
        make_header(dp_rank_header, &dp_rank.to_string()),
    ];

    ProcessingResponse {
        response: Some(PhaseResponse::RequestBody(BodyResponse {
            response: Some(CommonResponse {
                header_mutation: Some(HeaderMutation {
                    set_headers,
                    remove_headers: vec![],
                }),
                body_mutation: None,
                clear_route_cache: true,
            }),
        })),
    }
}

fn make_header(key: &str, value: &str) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.to_string(),
            value: value.to_string(),
            raw_value: vec![],
        }),
        keep_empty_value: false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::Config,
        proto::{
            processing_request::Request as PhaseReq,
            processing_response::Response as PhaseResp,
            HttpBody, HttpHeaders, HeaderValue,
        },
        worker_map::WorkerMap,
    };
    use dynamo_kv_router::{
        ConcurrentRadixTreeCompressed, ThreadPoolIndexer,
        indexer::KvIndexerInterface,
        protocols::{
            BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
            KvCacheStoreData, KvCacheStoredBlockData, RouterEvent, StorageTier,
            compute_block_hash_for_seq,
        },
    };
    use std::sync::Arc;

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

    fn make_indexer(block_size: u32) -> Arc<dyn KvIndexerInterface + Send + Sync> {
        Arc::new(ThreadPoolIndexer::new(
            ConcurrentRadixTreeCompressed::new(),
            1,
            block_size,
        ))
    }

    fn make_header_req(key: &str, value: &str) -> ProcessingRequest {
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

    fn make_body_req(body: &[u8]) -> ProcessingRequest {
        ProcessingRequest {
            request: Some(PhaseReq::RequestBody(HttpBody {
                body: body.to_vec(),
                end_of_stream: true,
            })),
        }
    }

    // --- request_headers pass-through ---

    #[tokio::test]
    async fn headers_phase_returns_request_headers_response() {
        let config = test_config(4);
        let indexer = make_indexer(4);
        let worker_map = Arc::new(WorkerMap::new());
        let mut tokens = None;

        let resp = handle_phase(make_header_req("content-type", "application/json"),
            &config, &*indexer, &worker_map, &mut tokens).await;

        assert!(matches!(resp.response, Some(PhaseResp::RequestHeaders(_))));
        if let Some(PhaseResp::RequestHeaders(h)) = resp.response {
            assert!(h.response.unwrap().header_mutation.is_none());
        }
    }

    // --- token-ids header extraction ---

    #[tokio::test]
    async fn token_id_header_extracted_in_headers_phase() {
        let config = test_config(4);
        let indexer = make_indexer(4);
        let worker_map = Arc::new(WorkerMap::new());
        let mut tokens: Option<Vec<u32>> = None;

        handle_phase(
            make_header_req(TOKEN_IDS_HEADER, "1,2,3,4"),
            &config, &*indexer, &worker_map, &mut tokens,
        ).await;

        assert_eq!(tokens, Some(vec![1, 2, 3, 4]));
    }

    // --- no workers: body passes through ---

    #[tokio::test]
    async fn no_workers_body_passes_through() {
        let config = test_config(4);
        let indexer = make_indexer(4);
        let worker_map = Arc::new(WorkerMap::new());
        let mut tokens = None;

        let body = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
        let resp = handle_phase(make_body_req(body), &config, &*indexer, &worker_map, &mut tokens).await;

        if let Some(PhaseResp::RequestBody(b)) = resp.response {
            assert!(b.response.unwrap().header_mutation.is_none());
        } else {
            panic!("expected RequestBody response");
        }
    }

    // --- with worker: routing headers injected ---

    #[tokio::test]
    async fn routing_headers_injected_when_worker_present() {
        let config = test_config(4);
        let indexer = make_indexer(4);
        let worker_map = Arc::new(WorkerMap::new());
        worker_map.register(1, 0, "pod-1:8000".into());
        let mut tokens = None;

        let body = br#"{"messages":[{"role":"user","content":"hello"}]}"#;
        let resp = handle_phase(make_body_req(body), &config, &*indexer, &worker_map, &mut tokens).await;

        let mutation = if let Some(PhaseResp::RequestBody(b)) = resp.response {
            b.response.unwrap().header_mutation.unwrap()
        } else {
            panic!("expected RequestBody with mutation");
        };

        let keys: Vec<&str> = mutation.set_headers.iter()
            .map(|h| h.header.as_ref().unwrap().key.as_str())
            .collect();
        assert!(keys.contains(&"x-gateway-destination-endpoint"));
        assert!(keys.contains(&"x-worker-instance-id"));
        assert!(keys.contains(&"x-dp-rank"));
        assert!(!mutation.set_headers.is_empty());
    }

    // --- pre-computed tokens take precedence over body ---

    #[tokio::test]
    async fn precomputed_tokens_consumed_over_body_parsing() {
        let config = test_config(2);
        let indexer = make_indexer(2);
        let worker_map = Arc::new(WorkerMap::new());
        worker_map.register(5, 0, "pod-5:8000".into());
        let mut tokens: Option<Vec<u32>> = Some(vec![10, 20, 30, 40]);

        let body = br#"{"messages":[{"role":"user","content":"UNUSED"}]}"#;
        let resp = handle_phase(make_body_req(body), &config, &*indexer, &worker_map, &mut tokens).await;

        // Pre-computed tokens should have been consumed.
        assert!(tokens.is_none());
        // Routing headers must be present.
        assert!(matches!(resp.response, Some(PhaseResp::RequestBody(_))));
        if let Some(PhaseResp::RequestBody(b)) = resp.response {
            assert!(b.response.unwrap().header_mutation.is_some());
        }
    }

    // --- response_headers phase returns ResponseHeaders response ---

    #[tokio::test]
    async fn response_headers_phase_returns_response_headers_variant() {
        let config = test_config(4);
        let indexer = make_indexer(4);
        let worker_map = Arc::new(WorkerMap::new());
        let mut tokens = None;

        let req = ProcessingRequest {
            request: Some(PhaseReq::ResponseHeaders(HttpHeaders {
                headers: vec![],
                end_of_stream: false,
            })),
        };

        let resp = handle_phase(req, &config, &*indexer, &worker_map, &mut tokens).await;
        assert!(matches!(resp.response, Some(PhaseResp::ResponseHeaders(_))));
    }

    // --- KV overlap selects correct worker ---

    #[tokio::test]
    async fn kv_overlap_routes_to_matching_worker() {
        const BLOCK_SIZE: u32 = 2;
        let indexer: Arc<dyn KvIndexerInterface + Send + Sync> = Arc::new(
            ThreadPoolIndexer::new(ConcurrentRadixTreeCompressed::new(), 1, BLOCK_SIZE),
        );
        let worker_map = Arc::new(WorkerMap::new());
        worker_map.register(10, 0, "worker-10:8000".into());
        worker_map.register(20, 0, "worker-20:8000".into());

        // Tokens [200, 201, 202, 203] → 2 blocks of size 2.
        let tokens: Vec<u32> = vec![200, 201, 202, 203];
        let hashes = compute_block_hash_for_seq(&tokens, BLOCK_SIZE, BlockHashOptions::default());

        // Teach worker 10 about these blocks.
        indexer.apply_event(RouterEvent {
            worker_id: 10,
            storage_tier: StorageTier::Device,
            event: KvCacheEvent {
                event_id: 0,
                dp_rank: 0,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    blocks: hashes.iter().map(|h| KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(h.0),
                        tokens_hash: *h,
                        mm_extra_info: None,
                    }).collect(),
                }),
            },
        }).await;
        indexer.flush().await;

        let config = test_config(BLOCK_SIZE);
        let mut precomputed: Option<Vec<u32>> = Some(tokens);

        let resp = handle_phase(
            make_body_req(b"{}"),
            &config, &*indexer, &worker_map, &mut precomputed,
        ).await;

        let mutation = if let Some(PhaseResp::RequestBody(b)) = resp.response {
            b.response.unwrap().header_mutation.unwrap()
        } else {
            panic!("expected RequestBody with mutation");
        };

        let dest = mutation.set_headers.iter()
            .find(|h| h.header.as_ref().unwrap().key == "x-gateway-destination-endpoint")
            .and_then(|h| h.header.as_ref())
            .map(|h| h.value.as_str())
            .expect("destination header present");

        assert_eq!(dest, "worker-10:8000",
            "worker 10 has KV overlap and should be selected");
    }
}
