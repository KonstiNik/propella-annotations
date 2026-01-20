from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.dataset as pads

from tqdm.auto import tqdm


# d_in = Path("/leonardo_work/AIFAC_L01_028/datasets/HuggingFaceFW/fineweb-2/data/pol_Latn")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/fineweb-2-984-shards/pol_Latn")
# d_in = Path("/leonardo_work/AIFAC_L01_028/datasets/HuggingFaceFW/finepdfs/data/nob_Latn/train")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/finepdfs-synt/data/nob_Latn/train-8-shards")
# d_in = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-10k-sample")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-10k-sample-8-shards")
# d_in = Path("/leonardo_work/AIFAC_L01_028/datasets/german-commons")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/german-commons-492-shards")
d_in = Path("/leonardo_work/AIFAC_L01_028/datasets/HPLT3-parquet/fin_Latn")
d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/HPLT3-parquet-984-shards/fin_Latn")
num_shards = 246
# num_shards = 492
# num_shards = 246
# num_shards = 123
# num_shards = 8

print(f"d_in: {d_in}")
print(f"d_out: {d_out}")
print(f"num_shards: {num_shards}")

d_out.mkdir(exist_ok=False, parents=True)

data_files = sorted(list(d_in.glob("**/*.parquet")))
print(f"Found {len(data_files)} files in {d_in}")
ds = pads.dataset(data_files)

num_rows = ds.count_rows()
print(f"Number of rows in dataset: {num_rows:_}")

num_rows_per_shard = num_rows // num_shards
print(f"Number of rows per output shard: {num_rows_per_shard:_}")

schema = ds.schema

batches_iter = ds.to_batches(batch_size=10_000)

def create_writer():
    writer = pq.ParquetWriter(temp_output_file, schema, compression="zstd")
    return writer

writer = None
shard=0
temp_output_file = d_out / "shard.incomplete"
rows_written = 0
with tqdm(total=num_rows, desc="Sharding dataset") as pbar:
    for batch in batches_iter:
        if writer is None:
            writer = create_writer()

        offset_in_batch = 0
        remaining_in_batch = batch.num_rows

        # Write the batch in slices so we never exceed the shard size
        while remaining_in_batch > 0:
            remaining_in_shard = num_rows_per_shard - rows_written if shard < num_shards - 1 else remaining_in_batch
            rows_to_write = min(remaining_in_batch, remaining_in_shard)

            if rows_to_write > 0:
                slice_to_write = batch.slice(offset_in_batch, rows_to_write)
                writer.write_batch(slice_to_write)
                rows_written += rows_to_write
                offset_in_batch += rows_to_write
                remaining_in_batch -= rows_to_write
                pbar.update(rows_to_write)

            # If current shard is filled (for all but the last shard), roll to next shard
            if (shard < num_shards - 1) and (rows_written == num_rows_per_shard):
                writer.close()
                temp_output_file.rename(d_out / f"shard_{shard:06d}.parquet")
                shard += 1
                writer = create_writer()
                rows_written = 0
    if writer is not None:
        writer.close()
        temp_output_file.rename(d_out / f"shard_{shard:06d}.parquet")
ds = pads.dataset(d_out)
print(f"Number of rows in output dataset: {ds.count_rows():_}")
print(f"Number of output files: {len(list(ds.get_fragments()))}")
