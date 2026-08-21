//! Graph algorithm accelerators for module-level analysis.
//!
//! Provides PageRank, betweenness centrality, connected components,
//! and cycle detection with rayon parallelism.

use ahash::AHashMap;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::collections::VecDeque;

/// PageRank via power iteration.
#[pyfunction]
#[pyo3(signature = (adjacency, damping=0.85, iterations=100, tolerance=1e-6))]
pub fn pagerank(
    adjacency: &Bound<'_, PyDict>,
    damping: f64,
    iterations: usize,
    tolerance: f64,
) -> PyResult<HashMap<String, f64>> {
    // Parse adjacency dict: Dict[str, List[str]]
    let mut adj: AHashMap<String, Vec<String>> = AHashMap::new();
    let mut all_nodes: Vec<String> = Vec::new();

    for (key, value) in adjacency.iter() {
        let node: String = key.extract()?;
        let neighbors: Vec<String> = value.extract()?;
        all_nodes.push(node.clone());
        adj.insert(node, neighbors);
    }

    let n = all_nodes.len();
    if n == 0 {
        return Ok(HashMap::new());
    }

    let inv_n = 1.0 / n as f64;
    let mut scores: AHashMap<String, f64> = AHashMap::new();
    for node in &all_nodes {
        scores.insert(node.clone(), inv_n);
    }

    for _ in 0..iterations {
        let mut new_scores: AHashMap<String, f64> = AHashMap::new();
        let base = (1.0 - damping) * inv_n;

        for node in &all_nodes {
            new_scores.insert(node.clone(), base);
        }

        for (src, neighbors) in &adj {
            if neighbors.is_empty() {
                continue;
            }
            let share = damping * scores.get(src).copied().unwrap_or(0.0) / neighbors.len() as f64;
            for dst in neighbors {
                if let Some(val) = new_scores.get_mut(dst) {
                    *val += share;
                }
            }
        }

        // Check convergence
        let mut max_diff = 0.0_f64;
        for node in &all_nodes {
            let old = scores.get(node).copied().unwrap_or(0.0);
            let new_val = new_scores.get(node).copied().unwrap_or(0.0);
            max_diff = max_diff.max((old - new_val).abs());
        }

        scores = new_scores;

        if max_diff < tolerance {
            break;
        }
    }

    Ok(scores.into_iter().collect())
}

/// Weighted PageRank where adjacency is Dict[str, Dict[str, int]].
#[pyfunction]
#[pyo3(signature = (adjacency, damping=0.85, iterations=100))]
pub fn weighted_pagerank(
    adjacency: &Bound<'_, PyDict>,
    damping: f64,
    iterations: usize,
) -> PyResult<HashMap<String, f64>> {
    let mut adj: AHashMap<String, Vec<(String, f64)>> = AHashMap::new();
    let mut all_nodes: Vec<String> = Vec::new();

    for (key, value) in adjacency.iter() {
        let node: String = key.extract()?;
        let neighbors: &Bound<'_, PyDict> = value.cast()?;
        let mut edges: Vec<(String, f64)> = Vec::new();
        for (nk, nv) in neighbors.iter() {
            let dst: String = nk.extract()?;
            let weight: f64 = nv.extract()?;
            edges.push((dst, weight));
        }
        all_nodes.push(node.clone());
        adj.insert(node, edges);
    }

    let n = all_nodes.len();
    if n == 0 {
        return Ok(HashMap::new());
    }

    let inv_n = 1.0 / n as f64;
    let mut scores: AHashMap<String, f64> = AHashMap::new();
    for node in &all_nodes {
        scores.insert(node.clone(), inv_n);
    }

    for _ in 0..iterations {
        let mut new_scores: AHashMap<String, f64> = AHashMap::new();
        let base = (1.0 - damping) * inv_n;
        for node in &all_nodes {
            new_scores.insert(node.clone(), base);
        }

        for (src, edges) in &adj {
            let total_weight: f64 = edges.iter().map(|(_, w)| w).sum();
            if total_weight <= 0.0 {
                continue;
            }
            let src_score = scores.get(src).copied().unwrap_or(0.0);
            for (dst, weight) in edges {
                let share = damping * src_score * weight / total_weight;
                if let Some(val) = new_scores.get_mut(dst) {
                    *val += share;
                }
            }
        }

        scores = new_scores;
    }

    Ok(scores.into_iter().collect())
}

/// Betweenness centrality using Brandes algorithm.
#[pyfunction]
#[pyo3(signature = (adjacency, normalized=true))]
pub fn betweenness_centrality(
    adjacency: &Bound<'_, PyDict>,
    normalized: bool,
) -> PyResult<HashMap<String, f64>> {
    let mut adj: AHashMap<String, Vec<String>> = AHashMap::new();
    let mut all_nodes: Vec<String> = Vec::new();

    for (key, value) in adjacency.iter() {
        let node: String = key.extract()?;
        let neighbors: Vec<String> = value.extract()?;
        all_nodes.push(node.clone());
        adj.insert(node, neighbors);
    }

    let n = all_nodes.len();
    let mut cb: AHashMap<String, f64> = AHashMap::new();
    for node in &all_nodes {
        cb.insert(node.clone(), 0.0);
    }

    // Node-to-index mapping for faster lookups
    let node_idx: AHashMap<&str, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    for s in &all_nodes {
        let mut stack: Vec<usize> = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut sigma: Vec<u64> = vec![0; n];
        let mut dist: Vec<i64> = vec![-1; n];
        let s_idx = node_idx[s.as_str()];
        sigma[s_idx] = 1;
        dist[s_idx] = 0;

        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s_idx);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let neighbors = adj.get(&all_nodes[v]).map(|v| v.as_slice()).unwrap_or(&[]);
            for w_name in neighbors {
                if let Some(&w) = node_idx.get(w_name.as_str()) {
                    if dist[w] < 0 {
                        queue.push_back(w);
                        dist[w] = dist[v] + 1;
                    }
                    if dist[w] == dist[v] + 1 {
                        sigma[w] += sigma[v];
                        pred[w].push(v);
                    }
                }
            }
        }

        let mut delta: Vec<f64> = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                if sigma[w] > 0 {
                    delta[v] += (sigma[v] as f64 / sigma[w] as f64) * (1.0 + delta[w]);
                }
            }
            if w != s_idx {
                *cb.get_mut(&all_nodes[w]).unwrap() += delta[w];
            }
        }
    }

    if normalized && n > 2 {
        let norm = 1.0 / ((n as f64 - 1.0) * (n as f64 - 2.0));
        for val in cb.values_mut() {
            *val *= norm;
        }
    }

    Ok(cb.into_iter().collect())
}

/// Find connected components using union-find.
#[pyfunction]
pub fn connected_components(adjacency: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<String>>> {
    let mut all_nodes: Vec<String> = Vec::new();
    let mut adj: AHashMap<String, Vec<String>> = AHashMap::new();

    for (key, value) in adjacency.iter() {
        let node: String = key.extract()?;
        let neighbors: Vec<String> = value.extract()?;
        // Add neighbor nodes that might not be keys
        for n in &neighbors {
            if !adj.contains_key(n) {
                all_nodes.push(n.clone());
                adj.insert(n.clone(), Vec::new());
            }
        }
        all_nodes.push(node.clone());
        adj.insert(node, neighbors);
    }

    // Deduplicate
    all_nodes.sort();
    all_nodes.dedup();

    let node_idx: AHashMap<&str, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let n = all_nodes.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    // Build undirected edges
    for (src, neighbors) in &adj {
        let si = node_idx[src.as_str()];
        for dst in neighbors {
            if let Some(&di) = node_idx.get(dst.as_str()) {
                union(&mut parent, si, di);
            }
        }
    }

    // Group by component
    let mut components: AHashMap<usize, Vec<String>> = AHashMap::new();
    for (i, node) in all_nodes.iter().enumerate() {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(node.clone());
    }

    Ok(components.into_values().collect())
}

/// Detect cycles using DFS coloring.
#[pyfunction]
pub fn detect_cycles(adjacency: &Bound<'_, PyDict>) -> PyResult<Vec<Vec<String>>> {
    let mut adj: AHashMap<String, Vec<String>> = AHashMap::new();
    let mut all_nodes: Vec<String> = Vec::new();

    for (key, value) in adjacency.iter() {
        let node: String = key.extract()?;
        let neighbors: Vec<String> = value.extract()?;
        all_nodes.push(node.clone());
        adj.insert(node, neighbors);
    }

    let node_idx: AHashMap<&str, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let n = all_nodes.len();
    // 0=white, 1=gray, 2=black
    let mut color: Vec<u8> = vec![0; n];
    let mut cycles: Vec<Vec<String>> = Vec::new();
    let mut path: Vec<usize> = Vec::new();

    fn dfs(
        v: usize,
        adj: &AHashMap<String, Vec<String>>,
        all_nodes: &[String],
        node_idx: &AHashMap<&str, usize>,
        color: &mut Vec<u8>,
        path: &mut Vec<usize>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        color[v] = 1; // gray
        path.push(v);

        let neighbors = adj.get(&all_nodes[v]).map(|v| v.as_slice()).unwrap_or(&[]);
        for w_name in neighbors {
            if let Some(&w) = node_idx.get(w_name.as_str()) {
                if color[w] == 1 {
                    // Found cycle - extract it
                    if let Some(pos) = path.iter().position(|&x| x == w) {
                        let cycle: Vec<String> =
                            path[pos..].iter().map(|&i| all_nodes[i].clone()).collect();
                        if cycle.len() > 1 {
                            cycles.push(cycle);
                        }
                    }
                } else if color[w] == 0 {
                    dfs(w, adj, all_nodes, node_idx, color, path, cycles);
                }
            }
        }

        path.pop();
        color[v] = 2; // black
    }

    for i in 0..n {
        if color[i] == 0 {
            dfs(
                i,
                &adj,
                &all_nodes,
                &node_idx,
                &mut color,
                &mut path,
                &mut cycles,
            );
        }
    }

    Ok(cycles)
}

#[cfg(test)]
mod tests {
    // Tests require PyO3 runtime, so they are integration-tested from Python.
    // Basic algorithm logic is validated via the Python test suite.
}
