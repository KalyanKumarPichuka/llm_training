import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling


class LinkPredictor(nn.Module):
    """
    A cleaner, modular link prediction model:
    - GCN encoder learns meaningful node representations
    - Edge scorer predicts probability for an edge existing
    """

    def __init__(self, in_channels, hidden_dim=128, latent_dim=64, dropout=0.2):
        super().__init__()

        # Encoder network (GCN-based)
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

        # Scoring MLP for final edge probability
        self.fc = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 1)
        )

        self.dropout = dropout

    def encode(self, x, edge_index):
        """Runs GNN encoder and returns node embeddings."""
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def score_edges(self, z, edge_index):
        """
        Computes edge scores:
        z: learned embeddings
        edge_index: edges to evaluate (shape: [2, E])
        """
        src, dst = edge_index[0], edge_index[1]
        pair_features = torch.cat([z[src], z[dst]], dim=-1)
        return torch.sigmoid(self.fc(pair_features)).view(-1)

    def forward(self, x, edge_index, eval_edge_index):
        """
        Full prediction forward pass.
        """
        z = self.encode(x, edge_index)
        return self.score_edges(z, eval_edge_index)

    def compute_loss(self, z, edge_index_pos):
        """Computes binary classification loss with negative sampling."""
        # Positive edges
        pos_scores = self.score_edges(z, edge_index_pos)
        pos_labels = torch.ones(pos_scores.size(0), device=pos_scores.device)

        # Negative sampled edges
        edge_index_neg = negative_sampling(
            edge_index_pos,
            num_nodes=z.size(0),
            num_neg_samples=edge_index_pos.size(1),
        )
        neg_scores = self.score_edges(z, edge_index_neg)
        neg_labels = torch.zeros(neg_scores.size(0), device=neg_scores.device)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0)

        return F.binary_cross_entropy(scores, labels)


device = "cuda" if torch.cuda.is_available() else "cpu"

predictor = LinkPredictor(in_channels=intent_embeddings.size(-1)).to(device)
optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)

x = intent_embeddings.to(device)
pos_edge_index = edge_index_intent.to(device)   # real known edges

for epoch in range(50):
    predictor.train()
    optimizer.zero_grad()

    z = predictor.encode(x, pos_edge_index)
    loss = predictor.compute_loss(z, pos_edge_index)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
predictor.eval()
with torch.no_grad():
    z = predictor.encode(x, pos_edge_index)

# Score all possible node pairs (careful: O(N^2), best for smaller graphs)
all_pairs = torch.combinations(torch.arange(x.size(0), device=device), r=2).T
scores = predictor.score_edges(z, all_pairs)

edge_index_intent = torch.cat([pos_edge_index, predicted_edges], dim=1)
