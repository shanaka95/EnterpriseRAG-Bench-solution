import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  Position,
  MarkerType,
  NodeProps,
  Handle,
} from 'reactflow';
import 'reactflow/dist/style.css';
import dagre from 'dagre';
import axios from 'axios';
import './styles/App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

// Max nodes to render in the graph at once (for performance)
const MAX_RENDER_NODES = 200;
const MAX_RENDER_DEPTH = 5;

interface ClusterNodeData {
  id: string;
  parent_id: string | null;
  medoid_doc_id: string | null;
  doc_count: number;
  keywords: string[];
  is_leaf: boolean;
  depth: number;
}

interface TreeData {
  nodes: ClusterNodeData[];
  edges: { source: string; target: string }[];
}

interface QueryResult {
  query: string;
  doc_ids: string[];
  paths_traversed: string[][];
  scores: Record<string, number>;
}

// Custom node component - lightweight for large graphs
const ClusterNodeComponent: React.FC<NodeProps> = ({ data, selected }) => {
  const isHighlighted = data.highlighted as boolean;
  const score = data.score as number | undefined;
  const isLeaf = data.is_leaf as boolean;

  return (
    <div className={`cluster-node ${isLeaf ? 'leaf-node' : 'internal-node'} ${isHighlighted ? 'highlighted' : ''} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Top} />
      <div className="node-header">
        <span className="node-depth">D{data.depth}</span>
        <span className="node-count">{data.doc_count} docs</span>
        {score !== undefined && (
          <span className="node-score">{score.toFixed(3)}</span>
        )}
      </div>
      <div className="node-keywords">
        {(data.keywords || []).slice(0, 3).map((kw: string, i: number) => (
          <span key={i} className="keyword-tag">{kw}</span>
        ))}
        {(data.keywords || []).length > 3 && (
          <span className="keyword-more">+{data.keywords.length - 3}</span>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
};

const nodeTypes = { clusterNode: ClusterNodeComponent };

function layoutNodes(nodes: Node[], edges: Edge[]): Node[] {
  if (nodes.length === 0) return nodes;

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 80 });

  nodes.forEach((node) => g.setNode(node.id, { width: 160, height: 60 }));
  edges.forEach((edge) => g.setEdge(edge.source, edge.target));
  dagre.layout(g);

  return nodes.map((node) => {
    const pos = g.node(node.id);
    return { ...node, position: { x: pos.x - 80, y: pos.y - 30 } };
  });
}

function App() {
  const [allNodes, setAllNodes] = useState<ClusterNodeData[]>([]);
  const [allEdges, setAllEdges] = useState<{ source: string; target: string }[]>([]);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [query, setQuery] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [queryLoading, setQueryLoading] = useState(false);
  const [highlightedNodes, setHighlightedNodes] = useState<Set<string>>(new Set());
  const [stats, setStats] = useState<any>(null);
  const [sseStatus, setSseStatus] = useState('disconnected');
  const [ingestStatus, setIngestStatus] = useState('');
  const [depthFilter, setDepthFilter] = useState(MAX_RENDER_DEPTH);
  const [showLeafOnly, setShowLeafOnly] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Build renderable nodes/edges from data, with performance filtering
  const buildFlowData = useCallback((treeNodes: ClusterNodeData[], treeEdges: { source: string; target: string }[], maxDepth: number) => {
    const nodeMap = new Map(treeNodes.map(n => [n.id, n]));
    const filteredNodes = treeNodes.filter(n => {
      if (n.depth > maxDepth) return false;
      if (showLeafOnly && !n.is_leaf) return false;
      return true;
    });

    // If still too many, take top N by doc_count
    let displayNodes = filteredNodes;
    if (displayNodes.length > MAX_RENDER_NODES) {
      displayNodes = displayNodes
        .sort((a, b) => b.doc_count - a.doc_count)
        .slice(0, MAX_RENDER_NODES);
    }

    const displayIds = new Set(displayNodes.map(n => n.id));
    const displayEdges = treeEdges.filter(e => displayIds.has(e.source) && displayIds.has(e.target));

    const flowNodes: Node[] = displayNodes.map(n => ({
      id: n.id,
      type: 'clusterNode',
      data: {
        ...n,
        highlighted: false,
        score: undefined,
      },
      position: { x: 0, y: 0 },
    }));

    const flowEdges: Edge[] = displayEdges.map(e => ({
      id: `${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      animated: false,
      style: { stroke: '#475569', strokeWidth: 1.5 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#475569' },
    }));

    const layouted = layoutNodes(flowNodes, flowEdges);
    setNodes(layouted);
    setEdges(flowEdges);
  }, [setNodes, setEdges, showLeafOnly]);

  // Fetch tree data
  const fetchTree = useCallback(async () => {
    try {
      const res = await axios.get<TreeData>(`${API_BASE}/api/v1/tree`);
      const treeData = res.data;
      if (treeData.nodes.length === 0) return;

      setAllNodes(treeData.nodes);
      setAllEdges(treeData.edges);
      buildFlowData(treeData.nodes, treeData.edges, depthFilter);
    } catch (err) {
      console.error('Failed to fetch tree:', err);
    }
  }, [buildFlowData, depthFilter]);

  // Fetch stats
  const fetchStats = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/v1/stats`);
      setStats(res.data);
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  }, []);

  // Rebuild flow when filter changes
  useEffect(() => {
    if (allNodes.length > 0) {
      buildFlowData(allNodes, allEdges, depthFilter);
    }
  }, [depthFilter, showLeafOnly, allNodes, allEdges, buildFlowData]);

  // SSE connection for real-time updates
  useEffect(() => {
    const connectSSE = () => {
      if (eventSourceRef.current) eventSourceRef.current.close();
      const es = new EventSource(`${API_BASE}/api/v1/stream`);
      es.onopen = () => setSseStatus('connected');
      es.onerror = () => {
        setSseStatus('disconnected');
        es.close();
        setTimeout(connectSSE, 5000);
      };
      es.addEventListener('tree_update', () => {
        fetchTree();
        fetchStats();
      });
      eventSourceRef.current = es;
    };

    connectSSE();
    fetchTree();
    fetchStats();

    const interval = setInterval(() => {
      fetchTree();
      fetchStats();
    }, 15000); // Refresh every 15s

    return () => {
      clearInterval(interval);
      if (eventSourceRef.current) eventSourceRef.current.close();
    };
  }, [fetchTree, fetchStats]);

  // Query the tree
  const handleQuery = async () => {
    if (!query.trim()) return;
    setQueryLoading(true);
    setHighlightedNodes(new Set());

    try {
      const res = await axios.post<QueryResult>(`${API_BASE}/api/v1/query`, {
        query: query.trim(),
      });
      const result = res.data;
      setQueryResult(result);

      // Highlight traversed paths
      const nodeSet = new Set<string>();
      const edgeSet = new Set<string>();

      result.paths_traversed.forEach((path) => {
        path.forEach((nodeId) => nodeSet.add(nodeId));
        for (let i = 0; i < path.length - 1; i++) {
          edgeSet.add(`${path[i]}-${path[i + 1]}`);
        }
      });

      setHighlightedNodes(nodeSet);

      setNodes((nds) =>
        nds.map((n) => ({
          ...n,
          data: {
            ...n.data,
            highlighted: nodeSet.has(n.id),
            score: result.scores[n.id],
          },
        }))
      );

      setEdges((eds) =>
        eds.map((e) => ({
          ...e,
          animated: edgeSet.has(e.id),
          style: edgeSet.has(e.id)
            ? { stroke: '#22d3ee', strokeWidth: 3 }
            : { stroke: '#475569', strokeWidth: 1.5 },
          markerEnd: edgeSet.has(e.id)
            ? { type: MarkerType.ArrowClosed, color: '#22d3ee' }
            : { type: MarkerType.ArrowClosed, color: '#475569' },
        }))
      );
    } catch (err) {
      console.error('Query failed:', err);
    } finally {
      setQueryLoading(false);
    }
  };

  const displayedNodeCount = allNodes.filter(n => n.depth <= depthFilter && (!showLeafOnly || n.is_leaf)).length;

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🌳 Hierarchical Soft-Clustering RAG</h1>
        <div className="header-stats">
          {stats && (
            <>
              <span className="stat">{stats.total_nodes} nodes</span>
              <span className="stat">{stats.leaf_nodes} leaves</span>
              <span className="stat">{stats.total_documents} docs</span>
              <span className="stat">depth {stats.max_depth}</span>
            </>
          )}
          <span className={`sse-status ${sseStatus}`}>SSE: {sseStatus}</span>
        </div>
      </header>

      <div className="main-layout">
        <div className="side-panel">
          <div className="panel-section">
            <h2>🔍 Query</h2>
            <div className="query-input-group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleQuery()}
                placeholder="Enter query..."
                className="query-input"
              />
              <button onClick={handleQuery} disabled={queryLoading} className="query-btn">
                {queryLoading ? '⏳' : '🔍'}
              </button>
            </div>
            {queryResult && (
              <div className="query-results">
                <div className="result-header">
                  Found <strong>{queryResult.doc_ids.length}</strong> documents
                </div>
                <div className="result-paths">
                  <strong>Paths:</strong> {queryResult.paths_traversed.length} paths traversed
                </div>
                <div className="result-docs">
                  {queryResult.doc_ids.slice(0, 15).map((id) => (
                    <span key={id} className="doc-id-chip">{id.substring(0, 20)}...</span>
                  ))}
                  {queryResult.doc_ids.length > 15 && (
                    <span className="doc-id-more">+{queryResult.doc_ids.length - 15} more</span>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="panel-section">
            <h2>⚙️ Graph Filters</h2>
            <div className="filter-group">
              <label>Max Depth: {depthFilter}</label>
              <input
                type="range"
                min="1"
                max={stats?.max_depth || 10}
                value={depthFilter}
                onChange={(e) => setDepthFilter(parseInt(e.target.value))}
                className="depth-slider"
              />
            </div>
            <div className="filter-info">
              Showing {displayedNodeCount} of {allNodes.length} nodes
            </div>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={showLeafOnly}
                onChange={(e) => setShowLeafOnly(e.target.checked)}
              />
              Show leaves only
            </label>
          </div>

          <div className="panel-section">
            <h2>📥 Ingest</h2>
            <button onClick={async () => {
              setIngestStatus('Ingesting test docs...');
              try {
                await axios.post(`${API_BASE}/api/v1/ingest`, {
                  documents: [
                    { id: 'test-' + Date.now(), text: 'Test document about GPU pricing and capacity.', source: 'test' },
                  ],
                });
                setIngestStatus('Ingested! Tree rebuilding...');
              } catch (err) { setIngestStatus('Failed'); }
            }} className="ingest-btn">Ingest Test Doc</button>
            {ingestStatus && <div className="ingest-status">{ingestStatus}</div>}
          </div>

          <div className="panel-section">
            <h2>📊 Legend</h2>
            <div className="legend">
              <div className="legend-item"><span className="legend-color internal"></span> Internal Node</div>
              <div className="legend-item"><span className="legend-color leaf"></span> Leaf Node</div>
              <div className="legend-item"><span className="legend-color highlighted"></span> Query Path</div>
            </div>
          </div>
        </div>

        <div className="graph-area">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            fitView
            minZoom={0.01}
            maxZoom={2}
            proOptions={{ hideAttribution: true }}
          >
            <Background color="#1e293b" gap={20} />
            <Controls />
            <MiniMap
              nodeColor={(n) => {
                if (n.data?.highlighted) return '#22d3ee';
                return n.data?.is_leaf ? '#6366f1' : '#475569';
              }}
              maskColor="rgba(15, 23, 42, 0.8)"
              pannable
              zoomable
            />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

export default App;
