import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:, :4]
print(X.shape)
plt.scatter(X[:, 0], X[:, 1], c="red", marker="o", label="see")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)
plt.show()


class FeatureNet(nn.Module):
    def __init__(self, in_features: int, latent_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DifferentiableKMeans(nn.Module):
    def __init__(self, n_clusters: int, in_features: int, latent_dim: int = 4):
        super().__init__()
        self.feature_extractor = FeatureNet(in_features, latent_dim)
        self.register_buffer("centroids", torch.zeros(n_clusters, latent_dim))

    @torch.no_grad()
    def initialize_centroids(self, data: torch.Tensor) -> None:
        embeddings = self.feature_extractor(data)
        perm = torch.randperm(embeddings.size(0), device=embeddings.device)
        self.centroids.copy_(embeddings[perm[: self.centroids.size(0)]])

    def forward(self, x: torch.Tensor):
        embeddings = self.feature_extractor(x)
        distances = torch.cdist(embeddings, self.centroids)
        return embeddings, distances


def centroid_separation_loss(
    centroids: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    if centroids.size(0) < 2:
        return torch.tensor(0.0, device=centroids.device)

    pairwise = torch.cdist(centroids, centroids)
    idx = torch.triu_indices(
        pairwise.size(0), pairwise.size(1), offset=1, device=centroids.device
    )
    pairwise = pairwise[idx[0], idx[1]]
    if pairwise.numel() == 0:
        return torch.tensor(0.0, device=centroids.device)

    penalty = torch.relu(margin - pairwise)
    return (penalty**2).mean()


def train(
    model: DifferentiableKMeans,
    data: torch.Tensor,
    epochs: int = 600,
    lr: float = 1e-2,
    separation_weight: float = 5e-2,
    update_interval: int = 10,
):
    optimizer = torch.optim.AdamW(
        model.feature_extractor.parameters(), lr=lr, weight_decay=1e-4
    )
    model.train()
    for step in range(epochs):
        optimizer.zero_grad()
        embeddings, distances = model(data)
        assignments = distances.argmin(dim=1)
        target = model.centroids[assignments]
        recon_loss = (embeddings - target).pow(2).mean()
        sep_loss = centroid_separation_loss(model.centroids)
        loss = recon_loss + separation_weight * sep_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % update_interval == 0:
            with torch.no_grad():
                refreshed_embeddings = model.feature_extractor(data)
                refreshed_distances = torch.cdist(refreshed_embeddings, model.centroids)
                refreshed_assignments = refreshed_distances.argmin(dim=1)
                for k in range(model.centroids.size(0)):
                    mask = refreshed_assignments == k
                    indices = torch.nonzero(mask, as_tuple=False).flatten()
                    if indices.numel() > 0:
                        points = refreshed_embeddings.index_select(0, indices)
                        model.centroids[k].copy_(points.mean(dim=0))
                    else:
                        rand_idx = torch.randint(
                            0, refreshed_embeddings.size(0), (1,)
                        ).item()
                        model.centroids[k].copy_(refreshed_embeddings[rand_idx])
    return model


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(24)

data = torch.from_numpy(X).float().to(device)
model = DifferentiableKMeans(n_clusters=3, in_features=X.shape[1], latent_dim=4).to(
    device
)
model.initialize_centroids(data)
train(model, data)

with torch.no_grad():
    _, final_distances = model(data)
label_pred = final_distances.argmin(dim=1).cpu().numpy()
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="label0")
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker="*", label="label1")
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker="+", label="label2")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)
plt.show()
