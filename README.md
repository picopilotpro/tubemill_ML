# tubemill_ML
This repo is used to perform EDA, analytics, ML engineering for tubemill machine sensor data

## Dataset linkage
Simple version:

Production data is the main table (base).
For each production run, we add info from other files like this:

Downtime
Match by same run ID (product_ref_id).
If missing, match by machine.
EMS readings for each downtime event are linked separately by time, using
`Begin_time` through `End_time`.

Process parameters
Match by run ID (transaction_id = product_ref_id).

Energy (EMS)
Take EMS rows whose timestamp is between the production run's start and end time.
Match by timestamp only; EMS meter IDs are not assumed to be production machine IDs.

PPC/Scrap
Take PPC rows whose timestamp is between production start and end time.
Also match machine if available.

So: Production row = center, other datasets are attached either by run ID or by time range.

## EDA script for machine efficiency

Use [tube_mill_eda.py](C:/pi_systems/tubemill_ML/tube_mill_eda.py) to:
- preprocess input data from TSV / SQL / Excel into pandas DataFrames
- relate Production, Downtime, Process, EMS, and PPC datasets
- compare machine efficiency only when enough complete, comparable machine metrics are available
- generate EDA plots and summary CSV outputs
- generate a self-contained HTML report with charts and descriptive metrics

EMS run energy uses `Machine KWH - Master KWH` as the cumulative consumed-energy reading,
then calculates the change between readings at the start and end of each production run.
The same interval calculation is applied to downtime events. The HTML report shows
meter-level counter/power details, production-run coverage, and downtime-event coverage.
`downtime_events_enriched.csv` contains the EMS rows and calculated energy/power for each event.
Since EMS meter IDs are not linked to machine IDs, downtime energy is associated by time window
and should not be interpreted as machine-specific.

Numeric duration fields are not assumed to be minutes or nanoseconds. If production
`Run_Time` / `Net_Run_Time` or downtime `Duration` values are numeric, specify each
source's actual unit independently:

```powershell
--production-duration-unit minutes --downtime-duration-unit seconds
```

Supported units are `seconds`, `minutes`, `hours`, and `days`. Without the applicable
option, numeric duration-based metrics are reported as unavailable instead of being
assigned an arbitrary unit. Use the actual units in your files, not necessarily the example above.

### Example run

```bash
python tube_mill_eda.py ^
  --production-path "data/production.tsv" ^
  --downtime-path "data/downtime.xlsx" ^
  --process-path "data/process.sqlite" --process-table process_params ^
  --ems-path "data/ems.tsv" ^
  --output-dir "eda_outputs"
```

### Key outputs

- `machine_efficiency_summary.csv`
- `production_runs_enriched.csv`
- `downtime_events_enriched.csv` (when downtime data is supplied)
- `eda_report.html` (self-contained report; machine rankings/bands are withheld when data is insufficient)
- `run_efficiency_band_mapping.csv` (if linkable run IDs exist)
- `process_parameter_summary.csv` (per-tag profile from production-linked process readings)
- `process_factor_gap_by_tag.csv` (if process parameter mapping exists)
- plots:
  - `efficiency_score_by_machine.png`
  - `output_vs_energy.png`
  - `machine_metric_correlation_heatmap.png`
  - `top_downtime_reasons.png` (downtime hours and matched EMS energy by loss reason; energy uses a secondary axis)
  - `process_parameter_gap.png`

### local run

c:\pi_systems\tubemill_ML\.venv\Scripts\python.exe C:\pi_systems\tubemill_ML\tube_mill_eda.py --production-path "C:\Users\rgvai\Downloads\drive-download-20261118T093400Z-1-001\Production Downtime Test Data (102026).xlsx" --production-sheet "Production Data" --downtime-path "C:\Users\rgvai\Downloads\drive-download-20261118T093400Z-1-001\Production Downtime Test Data (102026).xlsx" --downtime-sheet "Downtime Data" --process-path "C:\Users\rgvai\Downloads\drive-download-20261118T093400Z-1-001\Process Parameter Data\ptl_pp_comp_112026_backup.sql" --ems-path "C:\Users\rgvai\Downloads\drive-download-20261118T093400Z-1-001\Energy Data\power_analysis_102026.tsv" --ppc-path "C:\Users\rgvai\Downloads\drive-download-20261118T093400Z-1-001\Prod Scrap Data\tubemill_ppc_data.tsv" --output-dir "C:\pi_systems\tubemill_ML\eda_outputs"

### shape

[production] shape after preprocessing: (2378, 26)
[downtime] shape after preprocessing: (1082, 24)
[process] shape after preprocessing: (3818251, 12)
[ems] shape after preprocessing: (338344, 107)
[ppc] shape after preprocessing: (581736, 4)