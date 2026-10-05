# Analysis I/O Smoke Test Report

Status: PASS

Formal entry points exercised:

- `01_collect_run_outputs.py`
- `02_compute_ranking_metrics.py`
- `05_prepare_mixed_model_data.py`

Summary:

```json
{
  "commands": [
    {
      "command": "python3 research-lab/experiments/registered-workflow/stats/scripts/01_collect_run_outputs.py --manifest research-lab/experiments/registered-workflow/run-manifests/smoke_test_manifest.csv --output-dir research-lab/experiments/scripts/outputs/smoke/analysis_io",
      "stdout": "{\n  \"manifest\": \"research-lab/experiments/registered-workflow/run-manifests/smoke_test_manifest.csv\",\n  \"metrics_table\": \"research-lab/experiments/scripts/outputs/smoke/analysis_io/run_metrics_table.csv\",\n  \"output_dir\": \"research-lab/experiments/scripts/outputs/smoke/analysis_io\",\n  \"prediction_rows\": 192,\n  \"predictions_table\": \"research-lab/experiments/scripts/outputs/smoke/analysis_io/predictions_table.csv\",\n  \"runs_collected\": 8\n}\nCOLLECT_RUN_OUTPUTS_OK"
    },
    {
      "command": "python3 research-lab/experiments/registered-workflow/stats/scripts/02_compute_ranking_metrics.py --metrics-table research-lab/experiments/scripts/outputs/smoke/analysis_io/run_metrics_table.csv --output-dir research-lab/experiments/scripts/outputs/smoke/analysis_io",
      "stdout": "{\n  \"ranking_rows\": 8,\n  \"selection_rows\": 2\n}\nCOMPUTE_RANKING_METRICS_OK"
    },
    {
      "command": "python3 research-lab/experiments/registered-workflow/stats/scripts/05_prepare_mixed_model_data.py --metrics-table research-lab/experiments/scripts/outputs/smoke/analysis_io/run_metrics_table.csv --output-dir research-lab/experiments/scripts/outputs/smoke/analysis_io",
      "stdout": "{\n  \"rows\": 8,\n  \"output\": \"research-lab/experiments/scripts/outputs/smoke/analysis_io/mixed_model_input.csv\"\n}\nPREPARE_MIXED_MODEL_DATA_OK"
    }
  ],
  "manifest": "research-lab/experiments/registered-workflow/run-manifests/smoke_test_manifest.csv",
  "output_dir": "research-lab/experiments/scripts/outputs/smoke/analysis_io",
  "required_outputs": [
    "research-lab/experiments/scripts/outputs/smoke/analysis_io/run_metrics_table.csv",
    "research-lab/experiments/scripts/outputs/smoke/analysis_io/predictions_table.csv",
    "research-lab/experiments/scripts/outputs/smoke/analysis_io/model_ranking_table.csv",
    "research-lab/experiments/scripts/outputs/smoke/analysis_io/selection_frequency_table.csv",
    "research-lab/experiments/scripts/outputs/smoke/analysis_io/mixed_model_input.csv"
  ],
  "status": "PASS"
}
```
