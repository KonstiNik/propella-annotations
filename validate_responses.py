import polars as pl
from tqdm.auto import tqdm
from propella import AnnotationResponse


d = "/leonardo_work/AIFAC_L01_028/datasets/fineweb-2-propella-4b-responses/deu_Latn/shard000982_part000000.parquet"

df = pl.read_parquet(d)

for row in tqdm(df.rows(named=True)):
    response = row["response"]
    try:
        response_content = response["choices"][0]["message"]["content"]
        result = AnnotationResponse.model_validate_json(response_content)
    except Exception as e:
        print(f"Validation error: {e}")
        print(f"Response: {response}")
        print(f"Row: {row}")
        break