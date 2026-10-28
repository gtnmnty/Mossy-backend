# Feature 6: Ingestion script - builds the document index that rag.py searches.
# Run this once whenever your documents change: python -m app.scripts.ingest
# Flow: read every .md/.txt file in backend/docs/ -> split each into overlapping chunks
# -> turn each chunk into an embedding (Voyage) -> save the chunks and embeddings to
# backend/data/, where rag.py loads them.

import json
import re
import sys

import numpy as np
import voyageai

from app.core.config import settings
from app.services.rag import DATA_DIR, DOCS_DIR

# Splits text into chunks of about `size` characters, breaking on blank lines so an
# idea isn't cut mid-paragraph. Each chunk keeps the last `overlap` characters of the one
# before it, so an idea that falls near a chunk boundary still appears whole somewhere.
def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        while len(para) > size:                    # one giant paragraph: cut it up
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(para[:size])
            para = para[size - overlap:]
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur.strip())
            cur = cur[-overlap:]                   # carry a little context into the next chunk
        cur += para + "\n\n"
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def main():
    files = sorted(p for p in DOCS_DIR.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    if not files:
        sys.exit(f"Put .md or .txt files in {DOCS_DIR} first.")

    chunks = []
    for f in files:
        for i, c in enumerate(chunk(f.read_text(encoding="utf-8"))):
            chunks.append({"id": f"{f.name}:{i}", "source": f.name, "text": c})

    vo = voyageai.Client(api_key=settings.voyage_api_key)
    vecs = []
    for i in range(0, len(chunks), 64):            # batches keep each call under the token limit
        batch = [c["text"] for c in chunks[i:i + 64]]
        vecs += vo.embed(batch, model=settings.embed_model, input_type="document").embeddings
        print(f"embedded {min(i + 64, len(chunks))}/{len(chunks)}")

    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(DATA_DIR / "embeddings.npy", np.array(vecs, dtype=np.float32))
    print(f"done: {len(files)} files, {len(chunks)} chunks")


if __name__ == "__main__":
    # Without this guard, main() ran the moment this file was imported, so anything
    # that just imports the module (a test, an IDE, another script) would trigger a
    # real, billed embedding run. Run it with: python -m app.scripts.ingest
    main()