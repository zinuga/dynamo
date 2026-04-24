// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV-cache-aware routing logic.
//!
//! This module is intentionally decoupled from gRPC so that the core routing
//! algorithm can be tested independently of the service layer.

use std::sync::Arc;

use dynamo_kv_router::{
    indexer::KvIndexerInterface,
    protocols::{BlockHashOptions, OverlapScores, WorkerWithDpRank, compute_block_hash_for_seq},
};
use serde::Deserialize;

use crate::worker_map::WorkerMap;

// ---------------------------------------------------------------------------
// Public routing API
// ---------------------------------------------------------------------------

/// Outcome of a routing decision.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected worker.
    pub worker: WorkerWithDpRank,
    /// The pod endpoint (host:port) to route this request to.
    pub endpoint: String,
    /// Number of cached blocks that overlap with this request's prompt.
    pub overlap_blocks: u32,
}

/// Select the best worker for a request given its token sequence.
///
/// Returns `None` when no workers are registered.  Falls back to the least-
/// loaded worker when no overlap is found (all overlap scores are zero).
pub async fn route_request(
    tokens: &[u32],
    block_size: u32,
    indexer: &dyn KvIndexerInterface,
    worker_map: &WorkerMap,
) -> Option<RoutingDecision> {
    if worker_map.is_empty() {
        tracing::warn!("route_request: no workers registered, cannot route");
        return None;
    }

    // Compute per-block hashes for the token sequence.
    let block_hashes = compute_block_hash_for_seq(tokens, block_size, BlockHashOptions::default());

    // Query the indexer for overlap scores across all workers.
    let scores = match indexer.find_matches(block_hashes).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("route_request: find_matches failed: {e}");
            return None;
        }
    };

    select_from_scores(&scores, worker_map)
}

/// Select the best worker from pre-computed overlap scores.
///
/// Strategy:
/// 1. Workers with the highest overlap score win.
/// 2. On ties (or when all scores are zero), pick the worker with the smallest
///    tree size (least loaded heuristic).
/// 3. Only workers present in `worker_map` are considered.
pub fn select_from_scores(
    scores: &OverlapScores,
    worker_map: &WorkerMap,
) -> Option<RoutingDecision> {
    // Collect (worker, overlap, tree_size) for workers that have a known endpoint.
    let candidates: Vec<(WorkerWithDpRank, u32, usize)> = worker_map
        .all_workers()
        .into_iter()
        .map(|w| {
            let overlap = scores.scores.get(&w).copied().unwrap_or(0);
            let tree_sz = scores.tree_sizes.get(&w).copied().unwrap_or(usize::MAX);
            (w, overlap, tree_sz)
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Pick the worker with max overlap; break ties by min tree size.
    let best = candidates
        .iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.2.cmp(&a.2)))?;

    let endpoint = worker_map.endpoint_for(best.0)?;
    Some(RoutingDecision {
        worker: best.0,
        endpoint,
        overlap_blocks: best.1,
    })
}

// ---------------------------------------------------------------------------
// Request body parsing
// ---------------------------------------------------------------------------

/// OpenAI-compatible chat-completions request (only the fields we care about).
#[derive(Deserialize)]
struct ChatCompletionsRequest {
    #[serde(default)]
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    content: MessageContent,
}

/// `content` can be a plain string or a structured array.
#[derive(Deserialize, Default)]
#[serde(untagged)]
enum MessageContent {
    String(String),
    Parts(Vec<ContentPart>),
    #[default]
    Null,
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    part_type: String,
    text: Option<String>,
}

/// Extract a flat byte sequence from an OpenAI chat-completions body.
///
/// Each character of each message's text content is mapped to a `u32` token.
/// This is a best-effort approximation: accurate KV cache matching requires the
/// same tokenizer as the inference worker.  Pass actual token IDs via the
/// `x-dynamo-token-ids` header (comma-separated) to get exact matching.
pub fn tokens_from_body(body: &[u8]) -> Option<Vec<u32>> {
    let req: ChatCompletionsRequest = serde_json::from_slice(body).ok()?;
    let mut tokens: Vec<u32> = Vec::new();

    for msg in &req.messages {
        match &msg.content {
            MessageContent::String(s) => {
                tokens.extend(s.bytes().map(|b| b as u32));
            }
            MessageContent::Parts(parts) => {
                for part in parts {
                    if part.part_type == "text" {
                        if let Some(text) = &part.text {
                            tokens.extend(text.bytes().map(|b| b as u32));
                        }
                    }
                }
            }
            MessageContent::Null => {}
        }
    }

    if tokens.is_empty() { None } else { Some(tokens) }
}

/// Parse pre-computed token IDs from a comma-separated header value.
///
/// Workers / clients that already have token IDs available can pass them via
/// the `x-dynamo-token-ids` header to bypass text-based tokenisation.
pub fn tokens_from_header(header_value: &str) -> Option<Vec<u32>> {
    let tokens: Vec<u32> = header_value
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if tokens.is_empty() { None } else { Some(tokens) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_kv_router::protocols::{OverlapScores, WorkerWithDpRank};

    // Helper: build an OverlapScores with given per-worker overlaps.
    fn make_scores(entries: &[(WorkerWithDpRank, u32, usize)]) -> OverlapScores {
        let mut s = OverlapScores::new();
        for (w, overlap, sz) in entries {
            s.scores.insert(*w, *overlap);
            s.tree_sizes.insert(*w, *sz);
        }
        s
    }

    fn ww(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::new(id, 0)
    }

    fn map_with(workers: &[(u64, &str)]) -> WorkerMap {
        let m = WorkerMap::new();
        for (id, ep) in workers {
            m.register(*id, 0, ep.to_string());
        }
        m
    }

    // ------------------------------------------------------------------

    #[test]
    fn no_workers_returns_none() {
        let map = WorkerMap::new();
        let scores = OverlapScores::new();
        assert!(select_from_scores(&scores, &map).is_none());
    }

    #[test]
    fn single_worker_always_selected() {
        let map = map_with(&[(1, "10.0.0.1:8000")]);
        let scores = make_scores(&[(ww(1), 5, 100)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.worker, ww(1));
        assert_eq!(d.endpoint, "10.0.0.1:8000");
        assert_eq!(d.overlap_blocks, 5);
    }

    #[test]
    fn highest_overlap_wins() {
        let map = map_with(&[(1, "ep1"), (2, "ep2"), (3, "ep3")]);
        let scores = make_scores(&[(ww(1), 3, 100), (ww(2), 10, 100), (ww(3), 7, 100)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.worker, ww(2));
        assert_eq!(d.overlap_blocks, 10);
    }

    #[test]
    fn tie_broken_by_smallest_tree_size() {
        let map = map_with(&[(1, "ep1"), (2, "ep2")]);
        // Same overlap (5), but worker 2 has a smaller tree → less loaded.
        let scores = make_scores(&[(ww(1), 5, 200), (ww(2), 5, 50)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.worker, ww(2));
    }

    #[test]
    fn zero_overlap_falls_back_to_smallest_tree() {
        let map = map_with(&[(1, "ep1"), (2, "ep2")]);
        let scores = make_scores(&[(ww(1), 0, 500), (ww(2), 0, 100)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.worker, ww(2));
    }

    #[test]
    fn unregistered_worker_ignored_in_selection() {
        // Worker 99 has a great score but is not in the map.
        let map = map_with(&[(1, "ep1")]);
        let scores = make_scores(&[(ww(99), 100, 1), (ww(1), 2, 200)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.worker, ww(1));
    }

    // ------------------------------------------------------------------
    // Token extraction

    #[test]
    fn tokens_from_simple_body() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}]}"#;
        let tokens = tokens_from_body(body).unwrap();
        // "hi" → [104, 105]
        assert_eq!(tokens, vec![104u32, 105]);
    }

    #[test]
    fn tokens_from_multi_message_body() {
        let body = br#"{"messages":[
            {"role":"system","content":"sys"},
            {"role":"user","content":"hi"}
        ]}"#;
        let tokens = tokens_from_body(body).unwrap();
        // "sys" = [115, 121, 115], "hi" = [104, 105]
        assert_eq!(tokens, vec![115, 121, 115, 104, 105]);
    }

    #[test]
    fn tokens_from_structured_content() {
        let body = br#"{"messages":[{"role":"user","content":[
            {"type":"text","text":"hi"},
            {"type":"image_url","url":"..."}
        ]}]}"#;
        let tokens = tokens_from_body(body).unwrap();
        assert_eq!(tokens, vec![104u32, 105]); // only "hi" → text parts only
    }

    #[test]
    fn tokens_from_empty_body_returns_none() {
        let body = br#"{"messages":[]}"#;
        assert!(tokens_from_body(body).is_none());
    }

    #[test]
    fn tokens_from_header_parses_csv() {
        let hdr = "1,2,3,4,5";
        assert_eq!(tokens_from_header(hdr), Some(vec![1u32, 2, 3, 4, 5]));
    }

    #[test]
    fn tokens_from_header_empty_returns_none() {
        assert!(tokens_from_header("").is_none());
    }

    #[test]
    fn tokens_from_header_ignores_invalid_entries() {
        let hdr = "10, , 20, abc, 30";
        assert_eq!(tokens_from_header(hdr), Some(vec![10u32, 20, 30]));
    }

    // ------------------------------------------------------------------
    // Block-hash routing end-to-end (no real indexer — uses manual overlap)

    #[test]
    fn routing_decision_contains_correct_endpoint() {
        let map = map_with(&[(7, "pod-7:8000"), (8, "pod-8:8000")]);
        let scores = make_scores(&[(ww(7), 0, 100), (ww(8), 15, 50)]);
        let d = select_from_scores(&scores, &map).unwrap();
        assert_eq!(d.endpoint, "pod-8:8000");
        assert_eq!(d.overlap_blocks, 15);
    }
}
