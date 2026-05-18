"""
Delta Filing — Custom Financial Embedding Model
==================================================
Fine-tunes a sentence embedding model for SEC filing semantic search.
Hand-written InfoNCE contrastive loss — no external training framework.

Purpose: Embed filing sections for RAG retrieval. Similar filing sections
(same topic, same risk type) should be close in embedding space.

Architecture:
  Base model: sentence-transformers/all-MiniLM-L6-v2 (22M params)
  Loss: InfoNCE (contrastive) — hand-written
  Training: Positive pairs from same-topic filing sections,
            negatives from in-batch sampling

Usage:
    # Generate training pairs from cached filings
    python train_embedding.py --generate-data

    # Train the embedding model
    python train_embedding.py --train

    # Test embedding quality
    python train_embedding.py --test
"""

import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


# ============================================================
#  Config
# ============================================================

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, runs on CPU
OUTPUT_DIR = "./models/financial_embeddings"
PAIR_DATA = "./training_data/embedding_pairs.jsonl"

EMBED_DIM = 384  # MiniLM output dimension
MAX_LENGTH = 256  # Tokens per passage
BATCH_SIZE = 32
EPOCHS = 15
LR = 2e-5
TEMPERATURE = 0.07  # InfoNCE temperature


# ============================================================
#  Training Data Generation
# ============================================================

# SEC filing section categories for generating pairs
SECTION_TOPICS = {
    "supply_chain": [
        "The Company relies on a complex global supply chain for components and manufacturing.",
        "Supply chain disruptions could materially affect our ability to deliver products on time.",
        "Our dependence on third-party suppliers for key components creates concentration risk.",
        "Manufacturing operations in China represent approximately 85% of our production capacity.",
        "We maintain relationships with multiple suppliers to mitigate single-source dependency.",
    ],
    "regulatory": [
        "Regulatory changes in the European Union may impact our business model significantly.",
        "The Digital Markets Act imposes new obligations on platform operators.",
        "SEC compliance requirements continue to evolve, increasing our compliance costs.",
        "Antitrust litigation in multiple jurisdictions poses financial and operational risks.",
        "Data privacy regulations including GDPR and CCPA affect how we process user information.",
    ],
    "currency": [
        "International sales account for approximately 60% of net revenue, exposing us to currency risk.",
        "Fluctuations in foreign exchange rates can materially impact reported revenues and expenses.",
        "We use derivative instruments to hedge a portion of our foreign currency exposure.",
        "The strengthening of the US dollar adversely affected international revenue by $2.1 billion.",
        "Currency translation effects reduced operating income by approximately 3% year-over-year.",
    ],
    "competition": [
        "The technology industry is intensely competitive and subject to rapid technological change.",
        "We compete with companies that have significantly greater financial and technical resources.",
        "Our competitive position depends on our ability to attract and retain talented employees.",
        "New market entrants with innovative AI capabilities threaten our established market share.",
        "Price competition in our core markets has intensified, putting pressure on margins.",
    ],
    "cybersecurity": [
        "Cybersecurity threats continue to evolve in sophistication and frequency.",
        "A security breach could result in significant financial losses and reputational damage.",
        "We invest substantially in information security infrastructure and incident response.",
        "Third-party service providers may introduce additional cybersecurity vulnerabilities.",
        "Regulatory requirements for cybersecurity disclosure have increased our reporting obligations.",
    ],
    "financial_performance": [
        "Revenue increased 15% to $245.1 billion driven by cloud services growth of 23%.",
        "Operating income grew 24% to $109.4 billion, reflecting improved operational efficiency.",
        "Gross margin expanded 200 basis points to 45.96% due to favorable product mix.",
        "Free cash flow generation of $84.2 billion enabled accelerated share repurchases.",
        "Research and development expenses of $29.9 billion represented 8% of total revenue.",
    ],
    "macro_risk": [
        "Global macroeconomic conditions including inflation and recession risk may reduce demand.",
        "Rising interest rates could increase our borrowing costs and reduce capital investment.",
        "Geopolitical tensions and trade restrictions create uncertainty in our operations.",
        "Consumer confidence and spending patterns directly impact our product sales.",
        "Supply chain inflation has increased our cost of goods sold by approximately 4%.",
    ],
    "ai_risk": [
        "Artificial intelligence regulation could limit our ability to deploy AI-powered features.",
        "AI hallucination and bias risks require significant investment in safety measures.",
        "The rapid evolution of AI capabilities creates both opportunities and competitive threats.",
        "Our AI investments of $12.5 billion may not generate expected returns.",
        "Ethical concerns about AI use could lead to reputational damage and regulatory action.",
    ],
}


def generate_training_pairs(output_path: str, pairs_per_topic: int = 50):
    """Generate contrastive learning pairs from filing section templates.

    Positive pairs: Two passages from the same topic category.
    Negative pairs: Handled by in-batch negatives during training
                   (no need to explicitly generate).
    """
    pairs = []

    topics = list(SECTION_TOPICS.keys())
    for topic in topics:
        passages = SECTION_TOPICS[topic]
        # Generate positive pairs: same topic
        for _ in range(pairs_per_topic):
            if len(passages) >= 2:
                a, b = random.sample(passages, 2)
                pairs.append({
                    "anchor": a,
                    "positive": b,
                    "topic": topic,
                })

    random.shuffle(pairs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(pairs)} pairs across {len(topics)} topics")
    print(f"Saved to {output_path}")
    return pairs


# ============================================================
#  Dataset
# ============================================================

class ContrastiveDataset(Dataset):
    """Dataset of (anchor, positive) text pairs for contrastive learning."""

    def __init__(self, path: str, tokenizer, max_length: int = 256):
        with open(path) as f:
            self.pairs = [json.loads(line) for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        anchor = self.tokenizer(
            pair["anchor"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        positive = self.tokenizer(
            pair["positive"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "anchor_ids": anchor["input_ids"].squeeze(0),
            "anchor_mask": anchor["attention_mask"].squeeze(0),
            "positive_ids": positive["input_ids"].squeeze(0),
            "positive_mask": positive["attention_mask"].squeeze(0),
        }


# ============================================================
#  Embedding Model
# ============================================================

class FinancialEmbeddingModel(nn.Module):
    """Sentence embedding model for financial text.

    Uses mean pooling over transformer outputs (masked by attention).
    Optional projection head for contrastive training.
    """

    def __init__(self, model_name: str, projection_dim: int = 128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 384 for MiniLM

        # Projection head: maps embeddings to a space optimized for contrastive learning
        # This is discarded after training — we use the encoder output directly for retrieval
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

    def mean_pool(self, token_embeddings, attention_mask):
        """Mean pooling: average token embeddings, ignoring padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Get sentence embedding (without projection — for inference)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.mean_pool(outputs.last_hidden_state, attention_mask)

    def forward(self, input_ids, attention_mask):
        """Get projected embedding (for training with contrastive loss)."""
        embedding = self.encode(input_ids, attention_mask)
        projected = self.projection(embedding)
        return F.normalize(projected, p=2, dim=1)  # L2 normalize for cosine similarity


# ============================================================
#  Contrastive Loss — Hand-written InfoNCE
# ============================================================

def info_nce_loss(anchor_embeddings, positive_embeddings, temperature=0.07):
    """InfoNCE (Noise Contrastive Estimation) loss.

    This is the core contrastive learning loss function, hand-written.
    Used by SimCLR, CLIP, and most modern embedding models.

    For each anchor, the positive is its matching pair.
    All other items in the batch serve as negatives (in-batch negatives).

    Math:
        similarity_matrix[i][j] = cosine_sim(anchor_i, positive_j) / temperature
        loss = -log(exp(sim(anchor_i, positive_i)) / sum_j(exp(sim(anchor_i, positive_j))))

    This is equivalent to cross-entropy where:
        - The "correct class" for anchor_i is positive_i
        - All other positives_j (j ≠ i) are "wrong classes"

    Args:
        anchor_embeddings: (batch_size, embed_dim) — L2 normalized
        positive_embeddings: (batch_size, embed_dim) — L2 normalized
        temperature: scalar — controls sharpness of the distribution.
                     Lower = sharper = harder negatives matter more.

    Returns:
        loss: scalar
        accuracy: what fraction of anchors correctly identify their positive
    """
    batch_size = anchor_embeddings.shape[0]

    # Cosine similarity matrix: (batch_size, batch_size)
    # similarity[i][j] = how similar anchor_i is to positive_j
    # Since embeddings are L2 normalized, dot product = cosine similarity
    similarity = torch.mm(anchor_embeddings, positive_embeddings.t()) / temperature

    # Labels: anchor_i should match positive_i (diagonal)
    labels = torch.arange(batch_size, device=similarity.device)

    # Cross-entropy loss: correct answer is the diagonal
    # This pushes anchor_i closer to positive_i and away from all positive_j (j ≠ i)
    loss = F.cross_entropy(similarity, labels)

    # Accuracy: how often does the model correctly identify the matching positive?
    predictions = similarity.argmax(dim=1)
    accuracy = (predictions == labels).float().mean()

    return loss, accuracy


# ============================================================
#  Training
# ============================================================

def train(args):
    print("=" * 60)
    print("  Delta Filing — Financial Embedding Training")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Base model: {BASE_MODEL}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Dataset
    print("\n  Loading data...")
    dataset = ContrastiveDataset(PAIR_DATA, tokenizer, MAX_LENGTH)
    # Split 90/10
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    print(f"  Train: {train_size}, Val: {val_size}")

    # Model
    print("\n  Loading model...")
    model = FinancialEmbeddingModel(BASE_MODEL).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Training loop
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")

    print(f"\n  Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_losses, train_accs = [], []

        for batch in train_loader:
            anchor_emb = model(
                batch["anchor_ids"].to(device),
                batch["anchor_mask"].to(device),
            )
            positive_emb = model(
                batch["positive_ids"].to(device),
                batch["positive_mask"].to(device),
            )

            loss, acc = info_nce_loss(anchor_emb, positive_emb, TEMPERATURE)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(acc.item())

        # Validate
        model.eval()
        val_losses, val_accs = [], []
        with torch.no_grad():
            for batch in val_loader:
                anchor_emb = model(
                    batch["anchor_ids"].to(device),
                    batch["anchor_mask"].to(device),
                )
                positive_emb = model(
                    batch["positive_ids"].to(device),
                    batch["positive_mask"].to(device),
                )
                loss, acc = info_nce_loss(anchor_emb, positive_emb, TEMPERATURE)
                val_losses.append(loss.item())
                val_accs.append(acc.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_val_acc = sum(val_accs) / len(val_accs)

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            # Save best model
            torch.save(model.encoder.state_dict(), os.path.join(OUTPUT_DIR, "encoder_best.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)

        print(f"  Epoch {epoch}/{EPOCHS} | "
              f"train loss={avg_train_loss:.4f} acc={avg_train_acc:.1%} | "
              f"val loss={avg_val_loss:.4f} acc={avg_val_acc:.1%}"
              f"{'  ★' if is_best else ''}")

    # Save final
    torch.save(model.encoder.state_dict(), os.path.join(OUTPUT_DIR, "encoder_final.pt"))
    print(f"\n  Training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Saved to: {OUTPUT_DIR}")


# ============================================================
#  Testing — Embedding Quality
# ============================================================

def test_embeddings():
    """Test that the trained embeddings cluster by topic."""
    print("=" * 60)
    print("  Embedding Quality Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    encoder = AutoModel.from_pretrained(BASE_MODEL).to(device)
    weights_path = os.path.join(OUTPUT_DIR, "encoder_best.pt")
    if os.path.exists(weights_path):
        encoder.load_state_dict(torch.load(weights_path, map_location=device))
    encoder.eval()

    def embed(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=MAX_LENGTH, padding=True).to(device)
        with torch.no_grad():
            outputs = encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            embedding = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        return F.normalize(embedding, p=2, dim=1)

    # Test: same-topic passages should be more similar than cross-topic
    test_cases = [
        ("supply_chain",
         "Supply chain disruptions have affected our manufacturing timeline.",
         "We rely on suppliers in Asia for critical components."),
        ("cybersecurity",
         "A data breach could expose sensitive customer information.",
         "We invest heavily in cybersecurity infrastructure."),
        ("currency",
         "Foreign exchange fluctuations reduced our international revenue.",
         "We hedge currency exposure using derivative instruments."),
    ]

    print("\n  Same-topic similarity (should be high):")
    same_topic_sims = []
    for topic, a, b in test_cases:
        sim = F.cosine_similarity(embed(a), embed(b)).item()
        same_topic_sims.append(sim)
        print(f"    {topic}: {sim:.3f}")

    print("\n  Cross-topic similarity (should be lower):")
    cross_topic_sims = []
    for i in range(len(test_cases)):
        for j in range(i + 1, len(test_cases)):
            sim = F.cosine_similarity(
                embed(test_cases[i][1]), embed(test_cases[j][1])
            ).item()
            cross_topic_sims.append(sim)
            print(f"    {test_cases[i][0]} vs {test_cases[j][0]}: {sim:.3f}")

    avg_same = sum(same_topic_sims) / len(same_topic_sims)
    avg_cross = sum(cross_topic_sims) / len(cross_topic_sims)
    margin = avg_same - avg_cross

    print(f"\n  Avg same-topic:  {avg_same:.3f}")
    print(f"  Avg cross-topic: {avg_cross:.3f}")
    print(f"  Margin:          {margin:.3f}")
    print(f"  {'PASS' if margin > 0.05 else 'FAIL'}: Same-topic should be notably higher")


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate contrastive training pairs")
    parser.add_argument("--train", action="store_true",
                        help="Train the embedding model")
    parser.add_argument("--test", action="store_true",
                        help="Test embedding quality")
    parser.add_argument("--all", action="store_true",
                        help="Generate data + train + test")
    args = parser.parse_args()

    if args.all or args.generate_data:
        generate_training_pairs(PAIR_DATA)
    if args.all or args.train:
        train(args)
    if args.all or args.test:
        test_embeddings()
    if not any([args.generate_data, args.train, args.test, args.all]):
        parser.print_help()
