import torch.nn as nn
from torch_geometric.nn import HGTConv, Linear

class HGTEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=256, out_dim=128, num_layers=2, heads=4):
        """
        metadata: data.metadata() -> (node_types, edge_types)
        """
        super().__init__()
        node_types, edge_types = metadata

        # 1) Per-node-type linear projections to common hidden_dim
        self.input_proj = nn.ModuleDict()
        for ntype in node_types:
            in_dim = None  # Will infer from data in forward if you prefer
            # We'll set lazily → see forward; or you can pass dims in ctor.

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.heads = heads

        # HGTConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads
                )
            )

        # Final projection (optional)
        self.out_proj = nn.ModuleDict({
            ntype: Linear(hidden_dim, out_dim)
            for ntype in node_types
        })

        # flag for lazy init
        self._proj_initialized = False

    def _lazy_init_projs(self, x_dict):
        if self._proj_initialized:
            return
        self.input_proj = nn.ModuleDict()
        for ntype, x in x_dict.items():
            self.input_proj[ntype] = Linear(x.size(-1), self.hidden_dim)
        self._proj_initialized = True

    def forward(self, x_dict, edge_index_dict):
        # 0) lazy init projections if needed
        self._lazy_init_projs(x_dict)

        # 1) project features
        x_dict = {
            ntype: self.input_proj[ntype](x)
            for ntype, x in x_dict.items()
        }

        # 2) HGTConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: v.relu() for k, v in x_dict.items()}

        # 3) output projection
        out_dict = {
            ntype: self.out_proj[ntype](x)
            for ntype, x in x_dict.items()
        }

        return out_dict  # hetero embeddings per node type

device = "cuda" if torch.cuda.is_available() else "cpu"
data = data.to(device)

encoder = HGTEncoder(metadata=data.metadata(), hidden_dim=256, out_dim=128).to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

def train_hgt(num_epochs=50):
    encoder.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        x_dict = {ntype: data[ntype].x for ntype in data.node_types}
        edge_index_dict = data.edge_index_dict

        out_dict = encoder(x_dict, edge_index_dict)

        # Dummy loss: just keep embeddings small and non-degenerate
        loss = sum(v.pow(2).mean() for v in out_dict.values())

        loss.backward()
        optimizer.step()
        print(f"[HGT] Epoch {epoch:03d}, Loss {loss.item():.4f}")

train_hgt(num_epochs=20)

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import DMoNPooling, DenseGraphConv

# 1) get final intent embeddings from trained encoder
encoder.eval()
with torch.no_grad():
    x_dict = {ntype: data[ntype].x for ntype in data.node_types}
    edge_index_dict = data.edge_index_dict
    out_dict = encoder(x_dict, edge_index_dict)
    x_intent = out_dict['intent']  # shape: [num_intents, out_dim]

num_intents = x_intent.size(0)

# 2) Build an intent-intent graph (this is where you plug your logic)
# Here: dummy undirected graph connecting random pairs
edge_index_intent = torch.randint(0, num_intents, (2, num_intents * 3), device=device)

import torch.nn.functional as F

class IntentDMoN(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_clusters=20):
        super().__init__()

        self.conv1 = DenseGraphConv(in_channels, hidden_channels)
        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)

        self.pool = DMoNPooling(
            in_channels=[hidden_channels, hidden_channels],
            out_channels=num_clusters
        )

    def forward(self, x, edge_index):
        # Single big graph → batch all zeros
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Create dense adjacency & dense node feature batch
        x_dense, mask = to_dense_batch(x, batch)          # [B, N, F]
        adj_dense = to_dense_adj(edge_index, batch)       # [B, N, N]

        # Dense GNN
        h = self.conv1(x_dense, adj_dense).relu()
        h = self.conv2(h, adj_dense).relu()

        # DMoN pooling
        cluster_loss, x_pooled, adj_pooled, S, O, C = self.pool(h, adj_dense, mask)

        # x_pooled: [B, num_clusters, hidden_channels]
        # S: soft assignment matrix [B, N, num_clusters]
        return cluster_loss, S, x_pooled
dmon = IntentDMoN(in_channels=x_intent.size(-1), hidden_channels=128, num_clusters=20).to(device)
opt_dmon = torch.optim.Adam(dmon.parameters(), lr=0.001)

def train_dmon(num_epochs=50):
    for epoch in range(num_epochs):
        dmon.train()
        opt_dmon.zero_grad()

        cluster_loss, S, x_pooled = dmon(x_intent, edge_index_intent)
        loss = cluster_loss  # optionally add other regularizers

        loss.backward()
        opt_dmon.step()
        print(f"[DMoN] Epoch {epoch:03d}, Loss {loss.item():.4f}")

train_dmon(30)

dmon.eval()
with torch.no_grad():
    cluster_loss, S, x_pooled = dmon(x_intent, edge_index_intent)

# S: [B, N, K] → here B=1
S = S[0]  # [N, K]
intent_cluster_ids = S.argmax(dim=1).cpu()  # [N]

print("Intent cluster assignments:", intent_cluster_ids)
