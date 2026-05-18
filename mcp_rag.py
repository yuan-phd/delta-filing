"""
Delta Filing — RAG MCP Server
================================
MCP server that provides semantic search over SEC filing sections
using FAISS vector store and the trained financial embedding model.

Tools:
  1. search_similar_sections(query, top_k) — Find filing sections similar to a query
  2. search_by_topic(topic, top_k) — Find sections by topic category
  3. add_section(ticker, section, text, date) — Index a new filing section

Architecture:
  User query → Embedding model → FAISS index → Top-k similar sections

Usage:
    # Build index from cached filings
    python mcp_rag.py --build-index

    # Run as MCP server
    python mcp_rag.py
"""

import json
import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

try:
    import faiss
except ImportError:
    print("FAISS not installed. Run: pip install faiss-cpu")
    faiss = None

try:
    from mcp.server.fastmcp import FastMCP
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


# ============================================================
#  Config
# ============================================================

EMBED_MODEL_DIR = "./models/financial_embeddings"
EMBED_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "./faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "filings.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
EMBED_DIM = 384
MAX_LENGTH = 256


# ============================================================
#  Embedding Engine
# ============================================================

class EmbeddingEngine:
    """Handles text → embedding conversion using the trained model."""

    def __init__(self, model_dir: str = EMBED_MODEL_DIR,
                 base_model: str = EMBED_BASE_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load tokenizer
        tokenizer_path = model_dir if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")) else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load encoder
        self.encoder = AutoModel.from_pretrained(base_model).to(self.device)
        weights_path = os.path.join(model_dir, "encoder_best.pt")
        if os.path.exists(weights_path):
            self.encoder.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            print(f"  Loaded fine-tuned weights from {weights_path}")
        else:
            print(f"  Using base model weights (no fine-tuned weights found)")
        self.encoder.eval()

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text into a vector."""
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            embedding = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy().flatten()

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts."""
        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH, padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).expand(
                outputs.last_hidden_state.size()
            ).float()
            embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


# ============================================================
#  FAISS Vector Store
# ============================================================

class FilingVectorStore:
    """FAISS-based vector store for SEC filing sections."""

    def __init__(self, embed_engine: EmbeddingEngine):
        self.engine = embed_engine
        self.index = None
        self.metadata = []  # List of dicts with section info

        # Load existing index if available
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            self.load()

    def create_index(self):
        """Create a new empty FAISS index."""
        # Using IndexFlatIP (inner product = cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.metadata = []

    def add(self, ticker: str, section: str, text: str,
            filing_date: str, filing_type: str = "10-K"):
        """Add a filing section to the index."""
        if self.index is None:
            self.create_index()

        # Embed the text
        embedding = self.engine.embed(text)

        # Add to FAISS
        self.index.add(embedding.reshape(1, -1).astype(np.float32))

        # Store metadata
        self.metadata.append({
            "ticker": ticker,
            "section": section,
            "filing_date": filing_date,
            "filing_type": filing_type,
            "text_preview": text[:500],
            "text_length": len(text),
        })

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for sections similar to a query."""
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed query
        query_vec = self.engine.embed(query).reshape(1, -1).astype(np.float32)

        # Search
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            result = self.metadata[idx].copy()
            result["similarity"] = float(score)
            results.append(result)

        return results

    def search_by_ticker(self, ticker: str, query: str,
                         top_k: int = 5) -> list[dict]:
        """Search within a specific company's filings."""
        # Search broadly, then filter
        all_results = self.search(query, top_k=top_k * 3)
        filtered = [r for r in all_results if r["ticker"] == ticker.upper()]
        return filtered[:top_k]

    def save(self):
        """Save index and metadata to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  Saved index ({self.index.ntotal} vectors) to {INDEX_DIR}")

    def load(self):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)
        print(f"  Loaded index ({self.index.ntotal} vectors) from {INDEX_DIR}")

    def stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"total_vectors": 0}

        tickers = set(m["ticker"] for m in self.metadata)
        sections = set(m["section"] for m in self.metadata)
        return {
            "total_vectors": self.index.ntotal,
            "unique_companies": len(tickers),
            "unique_sections": len(sections),
            "companies": sorted(tickers),
        }


# ============================================================
#  Build Index from Cached Filings
# ============================================================

def build_index_from_cache():
    """Build FAISS index from locally cached SEC filings."""
    from edgar import get_filings, get_section

    print("=" * 60)
    print("  Building FAISS Index from Cached Filings")
    print("=" * 60)

    engine = EmbeddingEngine()
    store = FilingVectorStore(engine)
    store.create_index()

    # Companies to index
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA",
               "JPM", "V", "JNJ", "WMT", "PFE", "DIS", "NFLX", "CRM"]

    # Sections to index
    sections = ["1A", "7"]  # Risk Factors and MD&A

    indexed = 0
    for ticker in tickers:
        for section_id in sections:
            try:
                section = get_section(ticker, "10-K", section_id, filing_index=0)
                if section and section.get("text"):
                    # Split into chunks for better retrieval
                    text = section["text"]
                    # Index the full section
                    store.add(
                        ticker=ticker,
                        section=section["section_name"],
                        text=text[:2000],  # First 2000 chars
                        filing_date=section["filing_date"],
                    )
                    indexed += 1
                    print(f"  Indexed {ticker} {section['section_name']} "
                          f"({section['filing_date']})")
            except Exception as e:
                print(f"  Skipped {ticker} section {section_id}: {e}")

    store.save()
    print(f"\n  Total indexed: {indexed} sections")
    print(f"  Stats: {store.stats()}")


# ============================================================
#  MCP Server
# ============================================================

def create_mcp_server():
    """Create MCP server with RAG tools."""
    if not HAS_MCP:
        print("MCP not installed. Run: pip install mcp")
        return None

    mcp = FastMCP("delta-filing-rag")

    # Initialize on first use
    engine = None
    store = None

    def get_store():
        nonlocal engine, store
        if engine is None:
            engine = EmbeddingEngine()
            store = FilingVectorStore(engine)
        return store

    @mcp.tool()
    def search_similar_sections(query: str, top_k: int = 5) -> str:
        """Search for SEC filing sections semantically similar to the query.

        Args:
            query: Natural language description of what to find
            top_k: Number of results to return (default 5)

        Returns:
            JSON with matching sections and similarity scores
        """
        s = get_store()
        results = s.search(query, top_k)
        return json.dumps({
            "query": query,
            "results": results,
            "total_indexed": s.index.ntotal if s.index else 0,
        }, indent=2)

    @mcp.tool()
    def search_company_sections(ticker: str, query: str,
                                top_k: int = 3) -> str:
        """Search within a specific company's indexed filing sections.

        Args:
            ticker: Stock ticker (e.g., "AAPL")
            query: What to search for within that company's filings
            top_k: Number of results (default 3)

        Returns:
            JSON with matching sections from that company
        """
        s = get_store()
        results = s.search_by_ticker(ticker, query, top_k)
        return json.dumps({
            "ticker": ticker,
            "query": query,
            "results": results,
        }, indent=2)

    @mcp.tool()
    def index_stats() -> str:
        """Get statistics about the filing vector index.

        Returns:
            JSON with total vectors, companies, sections indexed
        """
        s = get_store()
        return json.dumps(s.stats(), indent=2)

    return mcp


# ============================================================
#  Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true",
                        help="Build FAISS index from cached filings")
    parser.add_argument("--test", action="store_true",
                        help="Test search functionality")
    parser.add_argument("--serve", action="store_true",
                        help="Run as MCP server")
    args = parser.parse_args()

    if args.build_index:
        build_index_from_cache()

    elif args.test:
        print("Testing RAG search...")
        engine = EmbeddingEngine()
        store = FilingVectorStore(engine)
        print(f"  Index stats: {store.stats()}")

        queries = [
            "supply chain risks and manufacturing",
            "cybersecurity threats and data breaches",
            "revenue growth and cloud services",
            "foreign currency exchange rate impact",
        ]
        for q in queries:
            results = store.search(q, top_k=3)
            print(f"\n  Query: {q}")
            for r in results:
                print(f"    {r['ticker']} {r['section']} ({r['filing_date']}) "
                      f"sim={r['similarity']:.3f}")

    elif args.serve:
        mcp = create_mcp_server()
        if mcp:
            mcp.run()

    else:
        parser.print_help()
