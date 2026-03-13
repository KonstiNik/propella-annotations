# Propella Annotations

Annotate datasets with [Propella](https://huggingface.co/ellamind/propella-1-4b) quality properties using [inference-hive](https://github.com/ellamind/inference-hive) for scalable batch inference on SLURM clusters.

## Prerequisites

- Access to a SLURM cluster with GPUs (tested on Leonardo, 4x A100 per node)
- [pixi](https://pixi.sh) installed

## Setup

Clone this repository and then clone `inference-hive` inside it:

```bash
git clone git@github.com:OpenEuroLLM/propella-annotations.git
cd propella-annotations
git clone https://github.com/ellamind/inference-hive.git inference-hive
```

Copy the required files into the inference-hive working directory and install dependencies:

```bash
cp pixi.toml inference-hive/
cd inference-hive
cp ../udf.py inference_hive/udf.py
cp ../propella.py inference_hive/propella.py
cp ../property_descriptions.md inference_hive/property_descriptions.md
pixi install --all
```

## Download model and data

After setup, download the Propella model and any datasets you need. Run these from the `inference-hive/` directory so `pixi` is available:

```bash
# Download the Propella model
pixi run -e cuda-sglang hf download ellamind/propella-1-4b \
    --local-dir /path/to/models/propella-1-4b

# Download a dataset (example: Dolci-Instruct-SFT)
pixi run -e cuda-sglang hf download allenai/Dolci-Instruct-SFT \
    --repo-type dataset \
    --local-dir /path/to/data/dolci-instruct-sft
```

## Running an existing dataset

This works, if the dataset is already prepared and has a corresponding config in `ih_configs/`. For example, the `propella-4b-dolci-instruct.yaml` config points to the Dolci-Instruct-SFT dataset.
Pick a config from `ih_configs/` and run:

```bash
pixi shell -e cuda-sglang

python validate_config.py --config ../ih_configs/<config>.yaml
python validate_data.py --config ../ih_configs/<config>.yaml
python create_run.py --config ../ih_configs/<config>.yaml --output <run-name>
python submit.py --run-dir <run-name> --limit 1  # test with 1 job first
```

Check the logs. If everything looks good, submit without `--limit`:

```bash
python submit.py --run-dir <run-name>
```

## Adding a new dataset

To annotate a new dataset, you need two things:

1. **Prepared data** — sharded parquet files with `id` (string) and `text` (string) columns
2. **An inference-hive config** — a YAML file in `ih_configs/` pointing to the prepared data

### Step 1: Prepare the data

Your dataset must be converted to parquet files containing:
- `id`: a unique string identifier per row
- `text`: the document text to annotate

The `format_propella_prompt` UDF (defined in `udf.py`) handles formatting the text into the Propella chat prompt at inference time, so your preparation script only needs to output plain text.

If your dataset needs custom preprocessing (e.g. flattening chat messages), write a preparation script in `prepare_datasets/`. See `prepare_datasets/prepare_dolci_instruct.py` for an example.

For large datasets, submit the preparation as a SLURM job rather than running on the login node.

### Step 2: Create a config

Copy an existing config from `ih_configs/` and update:
- `job_name`: a descriptive name for the SLURM job
- `dataset_path`: path to your prepared parquet directory
- `id_column_name`: the ID column name (typically `"id"`)
- `input_column_name`: set to `"messages"` (the UDF creates this from `text`)
- `output_path`: where to write the annotation results

Everything else (model, server settings, schema, UDF) can stay the same.

### Step 3: Validate and run

Follow the steps in [Running an existing dataset](#running-an-existing-dataset) with your new config.

## Example: Dolci-Instruct-SFT

The [Dolci-Instruct-SFT](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) dataset contains chat conversations that need to be flattened into plain text before annotation.

```bash
# 1. Prepare the full dataset (submits a SLURM job)
bash prepare_datasets/submit_prepare_dolci_instruct.sh

# 2. Once the job completes, validate and run
cd inference-hive
pixi shell -e cuda-sglang
python validate_config.py --config ../ih_configs/propella-4b-dolci-instruct.yaml
python validate_data.py --config ../ih_configs/propella-4b-dolci-instruct.yaml
python create_run.py --config ../ih_configs/propella-4b-dolci-instruct.yaml --output dolci-instruct-run1
python submit.py --run-dir dolci-instruct-run1 --limit 1
```

### Benchmarking with a small sample

To test performance before running the full dataset, prepare a small subset:

```bash
# Prepare only 5k rows in a single shard
bash prepare_datasets/submit_prepare_dolci_instruct.sh \
    --output /path/to/dolci-instruct-sft-prepared-5k \
    --num-shards 1 \
    --max-rows 5000
```

Then create a config pointing to that output (or copy `propella-4b-dolci-instruct.yaml` and update `dataset_path`) and run as usual.

## Benchmarks

All benchmarks use `OpenEuroLLM/propella-1-4b` with Dolci-Instruct-SFT data, SGLang with `llguidance` grammar backend and JSON schema constrained decoding.

### Leonardo Booster (CINECA) — 4x A100 64GB

**DP=4 (data parallel)**

| Documents | mem-fraction-static | Inference time | Throughput (docs/s) | Throughput (docs/h) |
|----------:|--------------------:|---------------:|--------------------:|--------------------:|
|      5000 |                0.65 |          1.3m  |                62.0 |             223,193 |

**TP=4 (tensor parallel)**

| Documents | mem-fraction-static | Inference time | Throughput (docs/s) | Throughput (docs/h) |
|----------:|--------------------:|---------------:|--------------------:|--------------------:|
|      5000 |                0.65 |          2.8m  |                29.6 |             106,469 |

**1x GPU (no parallelism)**

| Documents | mem-fraction-static | Inference time | Throughput (docs/s) | Throughput (docs/h) |
|----------:|--------------------:|---------------:|--------------------:|--------------------:|
|      5000 |                0.65 |          3.9m  |                21.1 |              76,081 |

### HoreKa (KIT) — 4x A100 40GB

**DP=4 (data parallel)**

| Documents | mem-fraction-static | Inference time | Throughput (docs/s) | Throughput (docs/h) |
|----------:|--------------------:|---------------:|--------------------:|--------------------:|
|      5000 |                0.65 |          1.8m  |                41.7 |             150,166 |

## Troubleshooting

### Pixi environment breaks system `curl` (HoreKa)

**Symptom**: Health checks silently fail — the sglang server starts ("fired up and ready to roll") but is never detected as healthy. The job loops until the SLURM time limit kills it.

**Cause**: The pixi environment sets `LD_LIBRARY_PATH` to include its own OpenSSL. System `curl` picks up pixi's `libssl`/`libcrypto` instead of the system ones, but system `/lib64/libldap.so.2` still expects the system OpenSSL's `EVP_md2` symbol, which pixi's OpenSSL doesn't provide:

```
curl: symbol lookup error: /lib64/libldap.so.2: undefined symbol: EVP_md2, version OPENSSL_3.0.0
```

**Fix**: The health check in `create_run.py` uses `python -c "import urllib.request; ..."` instead of `curl`. Python lives entirely within the pixi environment, avoiding the library conflict.

### Triton JIT linker error (HoreKa)

**Symptom**: sglang crashes during startup with a Triton compilation error mixing i386 and x86-64 architectures.

**Cause**: System modules (e.g. `compiler/intel`) pollute `LIBRARY_PATH`, causing Triton's JIT linker to pick up wrong 32-bit object files.

**Fix**: Add `LIBRARY_PATH: ""` to `env_vars` in the inference-hive config YAML to clear the polluted path before the server starts.

