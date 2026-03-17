"""Prepare the Dolci-Think-DPO-7B dataset for propella-annotations.

Loads the DPO dataset, keeps only the chosen (preferred) responses,
flattens the chat messages into plain text (User: ... / Assistant: ...),
and writes sharded parquet files with 'id' and 'text' columns ready for
inference-hive + the format_propella_prompt UDF.

Usage:
    python prepare_datasets/prepare_dolci_dpo.py \
        --output /leonardo_work/AIFAC_L01_028/datasets/dolci-think-dpo-prepared \
        --num-shards 8
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


CHUNK_SIZE = 50_000


def flatten_chosen(row: dict) -> str:
    """Flatten the chosen (preferred) messages from Dolci-Think-DPO into plain text."""
    parts = []
    for msg in row["chosen"]:
        role = msg["role"].capitalize()
        parts.append(f"{role}: {msg['content']}")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Dolci-Think-DPO-7B for propella-annotations"
    )
    parser.add_argument(
        "--dataset-path",
        default="allenai/Dolci-Think-DPO-7B",
        help="HF dataset identifier or local path to downloaded dataset",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for sharded parquet files",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of output shards",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process the first N rows (useful for benchmarking)",
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset(args.dataset_path, split=args.dataset_split)

    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))
        print(f"Limited to first {len(ds):,} rows")

    n_rows = len(ds)
    print(f"Processing {n_rows:,} rows")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([("id", pa.string()), ("text", pa.string())])
    rows_per_shard = n_rows // args.num_shards

    shard_idx = 0
    rows_in_shard = 0
    writer = None
    total = 0

    def open_writer(idx):
        path = output_dir / f"shard_{idx:06d}.parquet"
        return pq.ParquetWriter(str(path), schema, compression="zstd")

    chunk = []

    for i, example in enumerate(ds):
        text = flatten_chosen(example)
        doc_id = example.get("id", str(i))
        chunk.append({"id": str(doc_id), "text": text})
        rows_in_shard += 1

        # Roll to next shard if current one is full (except for last shard)
        if shard_idx < args.num_shards - 1 and rows_in_shard >= rows_per_shard:
            # Flush remaining chunk to current shard before rolling
            if chunk:
                if writer is None:
                    writer = open_writer(shard_idx)
                table = pa.Table.from_pylist(chunk, schema=schema)
                writer.write_table(table)
                total += len(chunk)
                chunk = []
            if writer is not None:
                writer.close()
            shard_idx += 1
            writer = None
            rows_in_shard = 0

        # Flush chunk to current shard
        if len(chunk) >= CHUNK_SIZE or i == n_rows - 1:
            if writer is None:
                writer = open_writer(shard_idx)

            table = pa.Table.from_pylist(chunk, schema=schema)
            writer.write_table(table)
            total += len(chunk)
            chunk = []

        if (i + 1) % CHUNK_SIZE == 0 or i == n_rows - 1:
            pct = (i + 1) / n_rows * 100
            print(f"\r  [{pct:5.1f}%] {total:,} / {n_rows:,} rows written", end="", flush=True)

    # Close final shard
    if writer is not None:
        writer.close()

    print(f"\nSaved {total:,} rows in {shard_idx + 1} shards to {args.output}")


if __name__ == "__main__":
    main()
