from pathlib import Path

import polars as pl

# d = Path("/leonardo_work/AIFAC_L01_028/datasets/fineweb-2-propella-4b-responses/ita_Latn/*.parquet")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/fineweb-2-propella-4b-annotations/ita_Latn")
# d = Path("/leonardo_work/AIFAC_L01_028/datasets/finepdfs-propella-4b-responses/ces_Latn/*.parquet")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/finepdfs-propella-4b-annotations/ces_Latn")
# d = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-10k-propella-4b-responses/*.parquet")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-10k-propella-4b-annotations")
# d = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-high-actual-propella-4b-responses/high-actual/*.parquet")
# d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/nemotron-cc-high-actual-propella-4b-annotations/high-actual")
d = Path("/leonardo_work/AIFAC_L01_028/datasets/german-commons-propella-4b-responses/*.parquet")
d_out = Path("/leonardo_work/AIFAC_L01_028/datasets/german-commons-propella-4b-annotations")
d_out.mkdir(exist_ok=True, parents=True)

# Polars schema matching AnnotationResponse from propella.py
annotation_schema = pl.Struct({
    "content_integrity": pl.Utf8,
    "content_ratio": pl.Utf8,
    "content_length": pl.Utf8,
    "one_sentence_description": pl.Utf8,
    "content_type": pl.List(pl.Utf8),
    "business_sector": pl.List(pl.Utf8),
    "technical_content": pl.List(pl.Utf8),
    "information_density": pl.Utf8,
    "content_quality": pl.Utf8,
    "audience_level": pl.Utf8,
    "commercial_bias": pl.Utf8,
    "time_sensitivity": pl.Utf8,
    "content_safety": pl.Utf8,
    "educational_value": pl.Utf8,
    "reasoning_indicators": pl.Utf8,
    "pii_presence": pl.Utf8,
    "regional_relevance": pl.List(pl.Utf8),
    "country_relevance": pl.List(pl.Utf8),
})

df = pl.scan_parquet(d) #.head(50_000)
df = df.select(
    pl.col('id'),
    pl.col("response")
    .struct.field('choices')
    .list.get(0)
    .struct.field('message')
    .struct.field('content')
    .str.json_decode(dtype=annotation_schema)
    .alias('annotations'),
)
df = df.unnest("annotations")


def filename(ctx):
    return f"shard{ctx.index_in_partition:06d}.parquet"
df.sink_parquet(pl.PartitionBy(d_out, file_path_provider=filename, max_rows_per_file=50_000_000))

print("Counting number of rows in output dataset...")
df = df = pl.scan_parquet(d_out)
rows = df.select(pl.len()).collect().item()
print(f"Rows: {rows:_}")