import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import train_test_split_edges, negative_sampling, to_undirected
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedHGTEncoder(nn.Module):
    def __init__(self, metadata, hidden_dim=256, num_layers=3, heads=4, dropout=0.2):
        super().__init__()
        node_types, edge_types = metadata

        self.hidden_dim = hidden_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.dropout = dropout

        # Lazy input projection: one Linear per node type
        self.input_proj = nn.ModuleDict()
        self._proj_initialized = False

        # HGT layers + LayerNorm + residuals
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=heads
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def _lazy_init_projs(self, x_dict):
        if self._proj_initialized:
            return
        for ntype, x in x_dict.items():
            in_dim = x.size(-1)
            self.input_proj[ntype] = Linear(in_dim, self.hidden_dim)
        self._proj_initialized = True

    def forward(self, x_dict, edge_index_dict):
        # project inputs
        self._lazy_init_projs(x_dict)
        h_dict = {
            ntype: self.input_proj[ntype](x)
            for ntype, x in x_dict.items()
        }

        # multi-layer HGT with residual + norm + dropout
        for conv, norm in zip(self.convs, self.norms):
            h_prev = {k: v for k, v in h_dict.items()}
            h_dict = conv(h_dict, edge_index_dict)
            for k in h_dict.keys():
                h_dict[k] = norm(h_dict[k] + h_prev[k])   # residual
                h_dict[k] = F.relu(h_dict[k])
                h_dict[k] = F.dropout(h_dict[k], p=self.dropout, training=self.training)

        return h_dict  # dict: {node_type: [num_nodes_t, hidden_dim]}
# === USER CONFIG ===
TARGET_NTYPE = "intent"           # or "section" / "step"
TARGET_EDGE_INDEX = None          # fill with your tensor [2, E] for that node type

# If you have an edge_index for the target node type in hetero:
# e.g. data[("intent", "related_to", "intent")].edge_index
# then:
# TARGET_EDGE_INDEX = data["intent", "related_to", "intent"].edge_index

assert TARGET_EDGE_INDEX is not None, "You must provide TARGET_EDGE_INDEX for link prediction."

# Build homogeneous Data object for link prediction / metrics
def make_target_data(x_target, edge_index_target):
    # ensure undirected
    edge_index_target = to_undirected(edge_index_target)
    data_hom = Data(
        x=x_target,
        edge_index=edge_index_target
    )
    return data_hom


def link_pred_loss(z, pos_edge_index):
    # Positive scores
    src, dst = pos_edge_index
    pos_scores = (z[src] * z[dst]).sum(dim=-1)  # dot product
    pos_labels = z.new_ones(pos_scores.size(0))

    # Negative samples
    neg_edge_index = negative_sampling(
        pos_edge_index,
        num_nodes=z.size(0),
        num_neg_samples=pos_edge_index.size(1)
    )
    src_n, dst_n = neg_edge_index
    neg_scores = (z[src_n] * z[dst_n]).sum(dim=-1)
    neg_labels = z.new_zeros(neg_scores.size(0))

    scores = torch.cat([pos_scores, neg_scores], dim=0)
    labels = torch.cat([pos_labels, neg_labels], dim=0)

    return F.binary_cross_entropy_with_logits(scores, labels)
def evaluate_linkpred(z, pos_edge_index, neg_edge_index):
    # Calculate probabilities
    def edge_scores(edge_index):
        src, dst = edge_index
        with torch.no_grad():
            s = torch.sigmoid((z[src] * z[dst]).sum(dim=-1)).cpu().numpy()
        return s

    pos_scores = edge_scores(pos_edge_index)
    neg_scores = edge_scores(neg_edge_index)

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_pred = np.concatenate([pos_scores, neg_scores])

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap


def cluster_and_scores(z, edge_index, num_clusters=10):
    """
    Computes:
    - k-means clusters
    - silhouette score on embeddings
    - modularity over graph communities
    """
    z_np = z.detach().cpu().numpy()

    # KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    cluster_ids = kmeans.fit_predict(z_np)

    # Silhouette
    try:
        sil = silhouette_score(z_np, cluster_ids)
    except Exception:
        sil = float("nan")

    # Build networkx graph for modularity
    hom_data = Data(x=z, edge_index=edge_index)
    G = to_networkx(hom_data, to_undirected=True)
    # Build community list: list of sets
    communities = []
    for c in range(num_clusters):
        nodes_in_c = [int(i) for i, cid in enumerate(cluster_ids) if cid == c]
        if nodes_in_c:
            communities.append(set(nodes_in_c))

    try:
        mod = nx.algorithms.community.modularity(G, communities)
    except Exception:
        mod = float("nan")

    return cluster_ids, sil, mod
# === Instantiate encoder ===
encoder = ImprovedHGTEncoder(
    metadata=data.metadata(),
    hidden_dim=256,
    num_layers=3,
    heads=4,
    dropout=0.2
).to(device)

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-5)

# === Prepare splits for link prediction on target graph ===
# We'll create a homogeneous Data just for splitting edges
x_dummy = torch.zeros(data[TARGET_NTYPE].num_nodes, 1)  # embeddings don't matter for split
data_target_for_split = Data(x=x_dummy, edge_index=TARGET_EDGE_INDEX).to(device)
data_split = train_test_split_edges(data_target_for_split)

train_pos_edge_index = data_split.train_pos_edge_index
val_pos_edge_index   = data_split.val_pos_edge_index
val_neg_edge_index   = data_split.val_neg_edge_index
test_pos_edge_index  = data_split.test_pos_edge_index
test_neg_edge_index  = data_split.test_neg_edge_index

print("Train edges:", train_pos_edge_index.size(1))
print("Val edges:",   val_pos_edge_index.size(1))
print("Test edges:",  test_pos_edge_index.size(1))


def train_epoch():
    encoder.train()
    optimizer.zero_grad()

    # HGT forward on full hetero graph
    x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    h_dict = encoder(x_dict, edge_index_dict)

    # Take embeddings of target node type
    z_target = h_dict[TARGET_NTYPE]

    # Link prediction loss on training edges
    lp_loss = link_pred_loss(z_target, train_pos_edge_index)

    # L2 reg on embeddings
    reg_loss = z_target.pow(2).mean()

    loss = lp_loss + 1e-3 * reg_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_epoch(split="val", num_clusters=10):
    encoder.eval()
    x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}
    h_dict = encoder(x_dict, edge_index_dict)
    z_target = h_dict[TARGET_NTYPE]

    # Choose which edges to evaluate
    if split == "val":
        pos_edge, neg_edge = val_pos_edge_index, val_neg_edge_index
    else:
        pos_edge, neg_edge = test_pos_edge_index, test_neg_edge_index

    auc, ap = evaluate_linkpred(z_target, pos_edge, neg_edge)
    _, sil, mod = cluster_and_scores(z_target, TARGET_EDGE_INDEX, num_clusters=num_clusters)

    return {
        "auc": auc,
        "ap": ap,
        "silhouette": sil,
        "modularity": mod
    }


# === Main training loop ===
best_val_auc = 0.0
best_state = None

for epoch in range(1, 101):
    loss = train_epoch()
    if epoch % 5 == 0:
        metrics_val = eval_epoch(split="val", num_clusters=10)
        metrics_test = eval_epoch(split="test", num_clusters=10)

        print(
            f"Epoch {epoch:03d} | Loss {loss:.4f} | "
            f"Val AUC {metrics_val['auc']:.4f} AP {metrics_val['ap']:.4f} "
            f"Sil {metrics_val['silhouette']:.4f} Mod {metrics_val['modularity']:.4f} | "
            f"Test AUC {metrics_test['auc']:.4f} AP {metrics_test['ap']:.4f}"
        )

        # Early model selection based on validation AUC + modularity
        score = metrics_val['auc'] + 0.1 * metrics_val['modularity']
        if score > best_val_auc:
            best_val_auc = score
            best_state = {k: v.cpu() for k, v in encoder.state_dict().items()}

# Load best encoder if desired
if best_state is not None:
    encoder.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print("Loaded best encoder state based on validation metrics.")
