# Propella Annotations

## Setup

Clone this repository and then clone `inference-hive` separately:

```bash
git clone git@github.com:OpenEuroLLM/propella-annotations.git
cd propella-annotations
git clone <inference-hive-repo-url> inference-hive

cp pixi.toml inference_hive/

cd inference-hive
cp ../udf.py inference_hive/udf.py
cp ../propella.py inference_hive/propella.py
cp ../property_descriptions.md inference_hive/property_descriptions.md
pixi install --all

pixi shell -e cuda-sglang

python validate_config.py --config ../ih_configs/propella-4b-hplt3-fin.yaml
python validate_data.py --config ../ih_configs/propella-4b-hplt3-fin.yaml
python create_run.py --config ../ih_configs/propella-4b-hplt3-fin.yaml  --output hplt3-fin-run1
python submit.py --run-dir hplt3-fin-run1 --limit 1 # test one first, check logs, submit without limit if it works:
# 2026-01-22 17:56:19.365 | INFO     | __main__:run_inference:476 - Starting inference
# 2026-01-22 17:57:21.377 | INFO     | __main__:report_progress:417 - Progress: 186/201_455 completed (186 new, 0 existing), Overall: 4.8 reqs/s (17_267 reqs/h), Last 60s: 4.8 reqs/s (17_267 reqs/h), ETA: 11.7h
```
