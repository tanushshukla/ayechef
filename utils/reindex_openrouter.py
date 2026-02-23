#!/usr/bin/env python3
"""
Re-index all recipe embeddings via OpenRouter API.

Uses batched API calls (multiple texts per request) for efficiency.
Safe to interrupt — progress is saved after each batch.

Usage:
    python -u utils/reindex_openrouter.py [--batch-size 500] [--api-batch 20]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import json
import sqlite3
import time
import numpy as np
import aiohttp
from datetime import datetime

from config import (
    DATA_DIR, CHAT_API_URL, CHAT_API_KEY, EMBEDDING_PROVIDER, EMBEDDING_MODEL,
    OPENROUTER_EMBEDDING_MODEL, EMBEDDING_DIMENSION,
)
from recipe_ann_index import RecipeANNIndex

DB_PATH = str(DATA_DIR / "recipe_index.db")
EMBED_URL = f"{CHAT_API_URL}/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CHAT_API_KEY}",
    "HTTP-Referer": "https://github.com/deepseekcoder2/ayechef",
    "X-Title": "Aye Chef Meal Planner",
}


def log(msg):
    print(msg, flush=True)


async def embed_api_batch(session, texts, max_retries=3):
    """Send a batch of texts to OpenRouter embeddings API in a single call."""
    payload = {
        "model": OPENROUTER_EMBEDDING_MODEL,
        "input": texts,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(
                EMBED_URL, json=payload, headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    return [
                        np.array(item["embedding"], dtype=np.float32)
                        for item in sorted_data
                    ]
                if resp.status in (429, 500, 502, 503, 504):
                    wait = 2 ** (attempt + 1)
                    log(f"  ⚠️  API {resp.status}, retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue
                body = await resp.text()
                log(f"  ❌ API error {resp.status}: {body[:200]}")
                return None
        except asyncio.TimeoutError:
            log(f"  ⚠️  Timeout (attempt {attempt+1}/{max_retries})")
            continue
        except Exception as e:
            log(f"  ❌ Request failed: {e}")
            return None

    log(f"  ❌ Failed after {max_retries} attempts")
    return None


def get_all_recipes(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, searchable_text FROM recipes WHERE searchable_text IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    return rows


def update_embeddings(db_path, updates):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for recipe_id, embedding in updates:
        cursor.execute("UPDATE recipes SET embedding = ? WHERE id = ?", (embedding.tobytes(), recipe_id))
    conn.commit()
    conn.close()


def rebuild_ann_index(db_path):
    log("\n🏗️  Rebuilding ANN index...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM recipes WHERE embedding IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()

    recipe_ids, embeddings = [], []
    for rid, emb_bytes in rows:
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        if len(emb) == EMBEDDING_DIMENSION:
            recipe_ids.append(rid)
            embeddings.append(emb)

    ann = RecipeANNIndex(dimension=EMBEDDING_DIMENSION)
    ann.build_index(embeddings, recipe_ids)
    ann.save()
    log(f"✅ ANN index rebuilt with {len(recipe_ids)} vectors")


def write_metadata():
    meta = {
        "provider": EMBEDDING_PROVIDER,
        "model": OPENROUTER_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openrouter" else EMBEDDING_MODEL,
        "dimension": EMBEDDING_DIMENSION,
        "last_indexed": datetime.now().isoformat(),
    }
    with open(DATA_DIR / "embedding_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


async def reindex(batch_size=500, api_batch=20, concurrency=3):
    log("=" * 60)
    log("OpenRouter Embedding Re-Index")
    log("=" * 60)
    log(f"Model:        {OPENROUTER_EMBEDDING_MODEL}")
    log(f"Batch size:   {batch_size} recipes per save")
    log(f"API batch:    {api_batch} texts per API call")
    log(f"Concurrency:  {concurrency} parallel API calls")
    log("")

    recipes = get_all_recipes(DB_PATH)
    total = len(recipes)
    log(f"📊 Total recipes: {total}")
    log("")

    done = 0
    failed = 0
    batch_num = 0
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        while done < total:
            batch_num += 1
            batch = recipes[done:done + batch_size]
            batch_ids = [r[0] for r in batch]
            batch_texts = [r[1] for r in batch]

            log(f"--- Batch {batch_num}: recipes {done+1}–{done+len(batch)} of {total} ---")
            batch_start = time.time()

            # Split into API-sized sub-batches and process concurrently
            sub_batches = [
                batch_texts[i:i + api_batch]
                for i in range(0, len(batch_texts), api_batch)
            ]

            sem = asyncio.Semaphore(concurrency)

            async def process_sub(sub):
                async with sem:
                    return await embed_api_batch(session, sub)

            sub_results = await asyncio.gather(*[process_sub(sb) for sb in sub_batches])

            # Flatten results
            all_embeddings = []
            batch_failed = 0
            for sub_result in sub_results:
                if sub_result is None:
                    batch_failed += api_batch
                    all_embeddings.extend([None] * api_batch)
                else:
                    all_embeddings.extend(sub_result)

            # Trim to actual batch size (last sub-batch may be smaller)
            all_embeddings = all_embeddings[:len(batch)]

            updates = []
            for rid, emb in zip(batch_ids, all_embeddings):
                if emb is not None:
                    updates.append((rid, emb))
                else:
                    batch_failed += 1

            if updates:
                update_embeddings(DB_PATH, updates)

            batch_elapsed = time.time() - batch_start
            done += len(batch)
            failed += batch_failed
            total_elapsed = time.time() - start_time
            rate = done / total_elapsed if total_elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            log(f"  ✅ {len(updates)} embedded, {batch_failed} failed")
            log(f"  ⏱️  {batch_elapsed:.1f}s this batch ({len(updates)/batch_elapsed:.0f} recipes/sec)")
            log(f"  📈 {done}/{total} ({done/total*100:.1f}%) — ETA {eta/60:.1f} min")
            log("")

    total_elapsed = time.time() - start_time
    log("=" * 60)
    log("EMBEDDING COMPLETE")
    log("=" * 60)
    log(f"  Recipes:  {done}")
    log(f"  Failed:   {failed}")
    log(f"  Time:     {total_elapsed/60:.1f} min")
    log(f"  Rate:     {done/total_elapsed:.0f} recipes/sec")
    log("")

    rebuild_ann_index(DB_PATH)
    write_metadata()
    log("\n✅ Done. Check your OpenRouter balance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--api-batch", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(reindex(args.batch_size, args.api_batch, args.concurrency))


if __name__ == "__main__":
    main()
