// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry mapping worker IDs to their pod endpoints.
//!
//! Workers call [`WorkerMap::register`] on startup (via an HTTP sidecar or
//! discovery hook) and [`WorkerMap::deregister`] on shutdown.  The ext_proc
//! service uses [`WorkerMap::endpoint_for`] to translate a routing decision
//! (WorkerWithDpRank) into a concrete `host:port` string that Envoy can route
//! to directly.

use std::sync::Arc;

use dashmap::DashMap;
use dynamo_kv_router::protocols::{DpRank, WorkerId, WorkerWithDpRank};

/// Concurrent map from `(worker_id, dp_rank)` → `host:port` string.
#[derive(Clone, Default)]
pub struct WorkerMap {
    inner: Arc<DashMap<WorkerWithDpRank, String>>,
}

impl WorkerMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a worker endpoint.
    ///
    /// If the worker is already registered the endpoint is updated.
    pub fn register(&self, worker_id: WorkerId, dp_rank: DpRank, endpoint: String) {
        let key = WorkerWithDpRank::new(worker_id, dp_rank);
        self.inner.insert(key, endpoint);
        tracing::info!(worker_id, dp_rank, "worker registered");
    }

    /// Remove a worker from the map.
    pub fn deregister(&self, worker_id: WorkerId, dp_rank: DpRank) {
        let key = WorkerWithDpRank::new(worker_id, dp_rank);
        self.inner.remove(&key);
        tracing::info!(worker_id, dp_rank, "worker deregistered");
    }

    /// Look up the endpoint for a scheduling result.
    pub fn endpoint_for(&self, worker: WorkerWithDpRank) -> Option<String> {
        self.inner.get(&worker).map(|v| v.clone())
    }

    /// All registered workers.
    pub fn all_workers(&self) -> Vec<WorkerWithDpRank> {
        self.inner.iter().map(|e| *e.key()).collect()
    }

    /// Number of registered workers.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_lookup() {
        let map = WorkerMap::new();
        map.register(1, 0, "10.0.0.1:8000".into());
        let w = WorkerWithDpRank::new(1, 0);
        assert_eq!(map.endpoint_for(w).as_deref(), Some("10.0.0.1:8000"));
    }

    #[test]
    fn deregister_removes_entry() {
        let map = WorkerMap::new();
        map.register(2, 0, "10.0.0.2:8000".into());
        map.deregister(2, 0);
        let w = WorkerWithDpRank::new(2, 0);
        assert!(map.endpoint_for(w).is_none());
    }

    #[test]
    fn overwrite_on_re_register() {
        let map = WorkerMap::new();
        map.register(3, 0, "10.0.0.3:8000".into());
        map.register(3, 0, "10.0.0.3:9000".into());
        let w = WorkerWithDpRank::new(3, 0);
        assert_eq!(map.endpoint_for(w).as_deref(), Some("10.0.0.3:9000"));
    }

    #[test]
    fn multiple_workers() {
        let map = WorkerMap::new();
        map.register(10, 0, "10.0.0.10:8000".into());
        map.register(11, 0, "10.0.0.11:8000".into());
        map.register(12, 1, "10.0.0.12:8000".into());
        assert_eq!(map.len(), 3);
        assert!(!map.is_empty());
    }

    #[test]
    fn dp_rank_is_part_of_key() {
        let map = WorkerMap::new();
        map.register(5, 0, "host-a:8000".into());
        map.register(5, 1, "host-b:8000".into());
        let w0 = WorkerWithDpRank::new(5, 0);
        let w1 = WorkerWithDpRank::new(5, 1);
        assert_eq!(map.endpoint_for(w0).as_deref(), Some("host-a:8000"));
        assert_eq!(map.endpoint_for(w1).as_deref(), Some("host-b:8000"));
    }
}
