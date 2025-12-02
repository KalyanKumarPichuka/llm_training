# full_hierarchical_workflow_gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData, Data
from torch_geometric.nn import HGTConv, Linear, DenseGraphConv, DMoNPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. Improved HGT Encoder: 768 → 512 → 256 → 128
# ============================================================

class ImprovedHGTEncoder(nn.Module):
    """
    HGT encoder with:
      - Per-node-type input projection: in_dim (e.g. 768) → 512
      - HGTConv stack at 512
      - Final compression: 512 → 256 → 128
      - Residual connections + LayerNorm + dropout
    """
    def __init__(self, metadata, hidden_1=512, hidden_2=256, out_dim=128,
                 num_layers=3, heads=4, dropout=0.2):
        super().__init__()
        node_types, edge_types = metadata

        self.node_types = node_types
        self.edge_types = edge_types
        self.h1 = hidden_1
        self.h2 = hidden_2
        self.out_dim = out_dim
        self.dropout = dropout

        # Lazy per-type projections: input_dim → 512
        self.input_proj = nn.ModuleDict()
        self._proj_initialized = False

        # HGTConv stack at 512-d
        self.hgt_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_1,
                    out_channels=hidden_1,
                    metadata=metadata,
                    heads=heads,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_1))

        # Final compression to 256 → 128
        self.to_256 = nn.Linear(hidden_1, hidden_2)
        self.to_128 = nn.Linear(hidden_2, out_dim)

    def _lazy_init(self, x_dict):
        if self._proj_initialized:
            return
        for ntype, x in x_dict.items():
            in_dim = x.size(-1)          # e.g. 768
            self.input_proj[ntype] = nn.Linear(in_dim, self.h1)
        self._proj_initialized = True

    def forward(self, x_dict, edge_index_dict):
        # 1) project to 512
        self._lazy_init(x_dict)
        h_dict = {nt: self.input_proj[nt](x) for nt, x in x_dict.items()}

        # 2) HGTConv layers at 512
        for conv, norm in zip(self.hgt_layers, self.norms):
            h_prev = {k: v for k, v in h_dict.items()}
            h_dict = conv(h_dict, edge_index_dict)
            for k in h_dict.keys():
                h_dict[k] = norm(h_dict[k] + h_prev[k])  # residual
                h_dict[k] = F.relu(h_dict[k])
                h_dict[k] = F.dropout(h_dict[k], p=self.dropout, training=self.training)

        # 3) compress 512 → 256 → 128
        h_dict = {k: F.relu(self.to_256(v)) for k, v in h_dict.items()}
        h_dict = {k: self.to_128(v) for k, v in h_dict.items()}

        return h_dict  # {node_type: [num_nodes, 128]}


# ============================================================
# 2. DMoN clustering helper (for any node type)
# ============================================================

class DMoNClusterNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_clusters=8):
        super().__init__()
        self.conv1 = DenseGraphConv(in_dim, hidden_dim)
        self.conv2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.pool = DMoNPooling(
            in_channels=hidden_dim,
            out_channels=num_clusters,
        )

    def forward(self, x, edge_index):
        # x: [N, d], edge_index: [2, E]
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)  # single graph

        x_dense, _ = to_dense_batch(x, batch)         # [1, N, d]
        adj_dense = to_dense_adj(edge_index, batch)   # [1, N, N]

        h = self.conv1(x_dense, adj_dense).relu()
        h = self.conv2(h, adj_dense).relu()

        loss, x_pooled, adj_pooled, S, O, C = self.pool(h, adj_dense, mask=None)
        return loss, S[0]      # S[0]: [N, K], soft assignments


def run_dmon_clustering(x, edge_index, num_clusters=8, num_epochs=100, lr=1e-3):
    """
    x: [N, d] embeddings
    edge_index: [2, E]
    Returns: hard cluster ids [N], soft assignments [N, K]
    """
    model = DMoNClusterNet(in_dim=x.size(-1), hidden_dim=128, num_clusters=num_clusters).to(x.device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        opt.zero_grad()
        loss, S = model(x, edge_index)
        loss.backward()
        opt.step()

        if epoch % 20 == 0:
            print(f"[DMoN] Epoch {epoch:03d} | Loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        loss, S = model(x, edge_index)

    cluster_ids = S.argmax(dim=1).cpu()
    return cluster_ids, S.cpu()


# ============================================================
# 3. Utility: build kNN graph if you don't have type-type edges
# ============================================================

def build_knn_graph(x, k=5):
    """
    Build a simple undirected k-NN graph (cosine similarity) for embeddings x.
    x: [N, d]
    Returns: edge_index [2, E]
    """
    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.t()                 # [N, N]
    N = x.size(0)

    # mask self-similarity
    sim.fill_diagonal_(-1.0)

    # pick top-k neighbors for each node
    knn_vals, knn_idx = sim.topk(k, dim=-1)   # [N, k]

    edges = []
    for i in range(N):
        for j in knn_idx[i]:
            j = j.item()
            edges.append((i, j))
            edges.append((j, i))  # undirected

    edge_index = torch.tensor(edges, dtype=torch.long, device=x.device).t().contiguous()
    return edge_index


# ============================================================
# 4. Parent mappings: section→intent, step→section
# ============================================================

def build_section_to_intent_mapping(data: HeteroData):
    """
    Uses ('intent','contains','section') edges to build:
    section_to_intent: [num_sections] with parent intent index
    Assumes each section belongs to one main intent (choose first if multiple).
    """
    num_sections = data["section"].num_nodes
    section_to_intent = torch.full((num_sections,), -1, dtype=torch.long, device=device)

    if ("intent", "contains", "section") not in data.edge_index_dict:
        raise RuntimeError("Missing ('intent','contains','section') edge type in HeteroData.")

    edge_index = data["intent", "contains", "section"].edge_index.to(device)
    src, dst = edge_index

    for intent_id, section_id in zip(src, dst):
        if section_to_intent[section_id] == -1:
            section_to_intent[section_id] = intent_id

    return section_to_intent


def build_step_to_section_mapping(data: HeteroData):
    """
    Uses ('section','has_step','step') edges to build:
    step_to_section: [num_steps] with parent section index
    """
    num_steps = data["step"].num_nodes
    step_to_section = torch.full((num_steps,), -1, dtype=torch.long, device=device)

    if ("section", "has_step", "step") not in data.edge_index_dict:
        raise RuntimeError("Missing ('section','has_step','step') edge type in HeteroData.")

    edge_index = data["section", "has_step", "step"].edge_index.to(device)
    src, dst = edge_index

    for section_id, step_id in zip(src, dst):
        if step_to_section[step_id] == -1:
            step_to_section[step_id] = section_id

    return step_to_section


# ============================================================
# 5. Main pipeline
# ============================================================

def main():
    # --------------------------------------------------------
    # (A) Build your HeteroData here
    # --------------------------------------------------------
    #
    # You must construct `data` with:
    #   - node types: 'intent', 'section', 'step', ...
    #   - data['intent'].x : [num_intents, 768]
    #   - data['section'].x: [num_sections, 768]
    #   - data['step'].x   : [num_steps, 768]
    #   - edges:
    #       ('intent','contains','section')
    #       ('section','has_step','step')
    #   - optionally other relations (persona, application, etc.)
    #
    # Example (placeholder):
    #
    # data = HeteroData()
    # data['intent'].x = torch.randn(num_intents, 768)
    # data['section'].x = torch.randn(num_sections, 768)
    # data['step'].x = torch.randn(num_steps, 768)
    # data[('intent','contains','section')].edge_index = ...
    # data[('section','has_step','step')].edge_index = ...
    #
    # --------------------------------------------------------

    data = HeteroData()
    # TODO: replace these with real tensors
    num_intents = 100
    num_sections = 300
    num_steps = 1000

    data["intent"].x = torch.randn(num_intents, 768)
    data["section"].x = torch.randn(num_sections, 768)
    data["step"].x = torch.randn(num_steps, 768)

    # Dummy connectivity examples (replace with your real edges):
    # Intent -> Section
    data["intent", "contains", "section"].edge_index = torch.randint(
        0, num_intents, (2, num_sections), dtype=torch.long
    )
    # Section -> Step
    data["section", "has_step", "step"].edge_index = torch.randint(
        0, num_sections, (2, num_steps), dtype=torch.long
    )

    data = data.to(device)

    # --------------------------------------------------------
    # (B) Train HGT encoder (simple self-supervised regularization)
    # --------------------------------------------------------
    encoder = ImprovedHGTEncoder(
        metadata=data.metadata(),
        hidden_1=512,
        hidden_2=256,
        out_dim=128,
        num_layers=3,
        heads=4,
        dropout=0.2,
    ).to(device)

    opt_enc = torch.optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-5)

    def train_hgt(num_epochs=20):
        for epoch in range(1, num_epochs + 1):
            encoder.train()
            opt_enc.zero_grad()

            x_dict = {nt: data[nt].x for nt in data.node_types}
            edge_index_dict = data.edge_index_dict

            h_dict = encoder(x_dict, edge_index_dict)

            # Simple embedding regularization (you can plug link prediction here)
            loss = 0.0
            for h in h_dict.values():
                loss = loss + h.pow(2).mean()

            loss.backward()
            opt_enc.step()
            if epoch % 5 == 0:
                print(f"[HGT] Epoch {epoch:03d} | Loss {loss.item():.4f}")

    train_hgt(num_epochs=20)

    encoder.eval()
    with torch.no_grad():
        x_dict = {nt: data[nt].x for nt in data.node_types}
        edge_index_dict = data.edge_index_dict
        h_dict = encoder(x_dict, edge_index_dict)

    intent_z = h_dict["intent"]   # [num_intents, 128]
    section_z = h_dict["section"] # [num_sections, 128]
    step_z = h_dict["step"]       # [num_steps, 128]

    # --------------------------------------------------------
    # (C) Build type-level graphs (intent, section, step)
    # --------------------------------------------------------

    # 1) Intent-intent graph
    # If you already have intent-intent edges in data, use those instead.
    # Here: build kNN on intent_z as a fallback.
    edge_index_intent = build_knn_graph(intent_z, k=5)

    # 2) Section-section graph
    # Could be based on sections belonging to same intent & order, or kNN.
    edge_index_section = build_knn_graph(section_z, k=5)

    # 3) Step-step graph
    edge_index_step = build_knn_graph(step_z, k=5)

    # --------------------------------------------------------
    # (D) Build parent mappings
    # --------------------------------------------------------
    section_to_intent = build_section_to_intent_mapping(data)  # [num_sections]
    step_to_section = build_step_to_section_mapping(data)      # [num_steps]

    # --------------------------------------------------------
    # (E) Level 1: DMoN on intents → top-level workflow families
    # --------------------------------------------------------

    num_intent_clusters = 8  # hyperparameter
    intent_cluster_ids, intent_S = run_dmon_clustering(
        intent_z, edge_index_intent, num_clusters=num_intent_clusters, num_epochs=80
    )

    print("Intent cluster IDs:", intent_cluster_ids.shape)

    # --------------------------------------------------------
    # (F) Level 2: DMoN on sections WITHIN each intent family
    # --------------------------------------------------------

    num_section_clusters_per_intent_cluster = 6  # hyperparameter
    num_intent_clusters = int(intent_cluster_ids.max().item()) + 1

    section_clusters_by_intent_cluster = {}

    for ic in range(num_intent_clusters):
        intents_in_ic = torch.where(intent_cluster_ids == ic)[0].to(device)
        if intents_in_ic.numel() == 0:
            continue

        print(f"\n=== DMoN on sections for Intent Cluster {ic} ===")

        # sections whose parent intent is in this cluster
        section_mask = torch.isin(section_to_intent, intents_in_ic)
        section_indices = torch.where(section_mask)[0]

        if section_indices.numel() < num_section_clusters_per_intent_cluster:
            print(f"  Skipping: only {section_indices.numel()} sections in this intent cluster.")
            continue

        # Subgraph of sections
        edge_index_sec_sub, _ = subgraph(
            subset=section_indices,
            edge_index=edge_index_section,
            relabel_nodes=True,
        )
        section_z_sub = section_z[section_indices]

        sec_cluster_ids_sub, sec_S_sub = run_dmon_clustering(
            section_z_sub,
            edge_index_sec_sub,
            num_clusters=num_section_clusters_per_intent_cluster,
            num_epochs=80,
        )

        section_clusters_by_intent_cluster[ic] = {
            "section_indices": section_indices.cpu(),          # global ids
            "section_cluster_ids": sec_cluster_ids_sub,        # local cluster ids
        }

    # --------------------------------------------------------
    # (G) Level 3: DMoN on steps WITHIN each (intent_cluster, section_cluster)
    # --------------------------------------------------------

    num_step_clusters_per_section_cluster = 5   # hyperparameter

    step_clusters_by_intent_and_section = {}  # (intent_cluster, local_section_cluster) → info

    for ic, sec_info in section_clusters_by_intent_cluster.items():
        section_indices_global = sec_info["section_indices"]        # [num_secs_in_ic]
        section_cluster_ids = sec_info["section_cluster_ids"]       # [num_secs_in_ic]

        # group sections by their local section cluster ID
        unique_sec_clusters = section_cluster_ids.unique().tolist()

        for sec_c in unique_sec_clusters:
            sec_mask = (section_cluster_ids == sec_c)
            sections_in_sec_cluster = section_indices_global[sec_mask]

            print(f"\n=== DMoN on steps for Intent Cluster {ic}, Section Cluster {sec_c} ===")

            # steps whose parent section is in this section cluster
            step_mask = torch.isin(step_to_section, sections_in_sec_cluster.to(device))
            step_indices = torch.where(step_mask)[0]

            if step_indices.numel() < num_step_clusters_per_section_cluster:
                print(f"  Skipping: only {step_indices.numel()} steps in this (ic={ic}, sec_c={sec_c}).")
                continue

            # Build step subgraph
            edge_index_step_sub, _ = subgraph(
                subset=step_indices,
                edge_index=edge_index_step,
                relabel_nodes=True,
            )
            step_z_sub = step_z[step_indices]

            step_cluster_ids_sub, step_S_sub = run_dmon_clustering(
                step_z_sub,
                edge_index_step_sub,
                num_clusters=num_step_clusters_per_section_cluster,
                num_epochs=80,
            )

            step_clusters_by_intent_and_section[(ic, int(sec_c))] = {
                "step_indices": step_indices.cpu(),
                "step_cluster_ids": step_cluster_ids_sub,
            }

    # --------------------------------------------------------
    # (H) You now have 3-level hierarchy:
    # --------------------------------------------------------
    #
    # 1) Intent clusters: intent_cluster_ids[i] → which top-level family intent i belongs to
    #
    # 2) For each intent cluster ic:
    #       section_clusters_by_intent_cluster[ic] = {
    #           "section_indices": global_section_ids,
    #           "section_cluster_ids": local_section_cluster_for_each
    #       }
    #
    # 3) For each (ic, sec_cluster):
    #       step_clusters_by_intent_and_section[(ic, sec_cluster)] = {
    #           "step_indices": global_step_ids,
    #           "step_cluster_ids": local_step_cluster_for_each
    #       }
    #
    # You can now:
    #   - auto-label each cluster with GPT
    #   - build a JSON/graph ontology
    #   - visualize with Dash / Neo4j / Cytoscape
    #
    # --------------------------------------------------------

    print("\n=== Pipeline finished ===")
    print(f"Num intent clusters: {num_intent_clusters}")
    print(f"Num intent clusters that have section clustering: {len(section_clusters_by_intent_cluster)}")
    print(f"Num (intent, section_cluster) groups that have step clustering: {len(step_clusters_by_intent_and_section)}")


if __name__ == "__main__":
    main()
