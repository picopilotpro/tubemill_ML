from __future__ import annotations

import argparse
import base64
import csv
import html
import re
import sqlite3
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import time as datetime_time
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        norm = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip()).strip("_").lower()
        renamed[col] = norm
    return df.rename(columns=renamed)


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def machine_id_match(values: pd.Series, target: object) -> pd.Series:
    target_number = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]
    values_number = pd.to_numeric(values, errors="coerce")
    if pd.notna(target_number):
        return values_number == target_number
    return values.astype("string").str.strip() == str(target).strip()


def read_input_to_dataframe(
    input_path: str,
    table_name: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    path = Path(input_path)
    suffix = path.suffix.lower()

    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".csv"}:
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(path) as conn:
            selected_table = table_name
            if not selected_table:
                query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1"
                row = pd.read_sql_query(query, conn).iloc[0]
                selected_table = row["name"]
            return pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
    if suffix == ".sql":
        sql_text = path.read_text(encoding="utf-8")
        conn = sqlite3.connect(":memory:")
        try:
            try:
                conn.executescript(sql_text)
                selected_table = table_name
                if not selected_table:
                    query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1"
                    row = pd.read_sql_query(query, conn).iloc[0]
                    selected_table = row["name"]
                return pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
            except sqlite3.OperationalError:
                # Fallback for MySQL-style dumps containing DATABASE/ENGINE/LOCK statements.
                return parse_sql_dump_to_dataframe(sql_text=sql_text, table_name=table_name)
        finally:
            conn.close()

    raise ValueError(
        f"Unsupported file format '{suffix}'. Use TSV, CSV, Excel, SQL dump, or SQLite DB files."
    )


def coerce_datetime_columns(df: pd.DataFrame, candidates: list[str]) -> None:
    day_first_columns = {
        "date",
        "prod_start_time",
        "prod_end_time",
        "begin_time",
        "end_time",
    }
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
    ]
    for col in candidates:
        if col in df.columns:
            values = df[col]
            if pd.api.types.is_datetime64_any_dtype(values):
                df[col] = pd.to_datetime(values, errors="coerce")
                continue

            text = values.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
            for date_format in formats:
                remaining = parsed.isna() & text.notna()
                if not remaining.any():
                    break
                parsed.loc[remaining] = pd.to_datetime(
                    text.loc[remaining], format=date_format, errors="coerce"
                )
            remaining = parsed.isna() & text.notna()
            if remaining.any():
                parsed.loc[remaining] = pd.to_datetime(
                    text.loc[remaining],
                    errors="coerce",
                    dayfirst=col in day_first_columns,
                )
            df[col] = parsed


def coerce_numeric_columns(df: pd.DataFrame, candidates: list[str]) -> None:
    for col in candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def preprocess_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = normalize_columns(df).copy()

    object_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})

    datetime_candidates = [
        "date",
        "prod_start_time",
        "prod_end_time",
        "begin_time",
        "end_time",
        "timestamp",
        "date_time",
        "tm1_timestamp",
        "_timestamp",
    ]
    numeric_candidates = [
        "pq_qty",
        "scrap_qty",
        "tm1_total_qty",
        "tm1_scrap_qty",
        "target_speed",
        "actual_speed",
        "od",
        "thickness",
        "length1",
        "machine_kwh",
        "master_kwh",
        "_value",
        "kw",
        "kva",
        "kvar",
        "watts",
    ]

    coerce_datetime_columns(df, datetime_candidates)
    coerce_numeric_columns(df, numeric_candidates)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[{dataset_name}] shape after preprocessing: {df.shape}")
    return df


def split_sql_tuples(values_block: str) -> list[str]:
    tuples: list[str] = []
    depth = 0
    in_string = False
    start = -1
    i = 0
    while i < len(values_block):
        ch = values_block[i]
        if ch == "'":
            if in_string and i + 1 < len(values_block) and values_block[i + 1] == "'":
                i += 2
                continue
            in_string = not in_string
        elif not in_string:
            if ch == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and start >= 0:
                    tuples.append(values_block[start : i + 1])
                    start = -1
        i += 1
    return tuples


def parse_sql_tuple_values(tuple_text: str) -> list[object]:
    inner = tuple_text.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    # MySQL escapes quotes as doubled single-quotes; convert for CSV parser.
    inner = inner.replace("''", "\\'")
    reader = csv.reader([inner], delimiter=",", quotechar="'", escapechar="\\")
    values = next(reader)
    parsed: list[object] = []
    for val in values:
        v = val.strip()
        if v.upper() == "NULL":
            parsed.append(None)
            continue
        if re.fullmatch(r"-?\d+", v):
            parsed.append(int(v))
            continue
        if re.fullmatch(r"-?\d+\.\d+", v):
            parsed.append(float(v))
            continue
        parsed.append(v)
    return parsed


def parse_sql_dump_to_dataframe(sql_text: str, table_name: Optional[str] = None) -> pd.DataFrame:
    insert_pattern = re.compile(
        r"INSERT\s+INTO\s+`?([a-zA-Z0-9_]+)`?\s*(\((.*?)\))?\s*VALUES\s*(.*?);",
        flags=re.IGNORECASE | re.DOTALL,
    )
    table_rows: dict[str, list[list[object]]] = defaultdict(list)
    table_cols: dict[str, Optional[list[str]]] = {}

    for match in insert_pattern.finditer(sql_text):
        tbl = match.group(1)
        if table_name and tbl.lower() != table_name.lower():
            continue
        raw_cols = match.group(3)
        cols = None
        if raw_cols:
            cols = [c.strip().strip("`").strip('"') for c in raw_cols.split(",")]
        tuples = split_sql_tuples(match.group(4))
        for t in tuples:
            table_rows[tbl].append(parse_sql_tuple_values(t))
        if tbl not in table_cols:
            table_cols[tbl] = cols

    if not table_rows:
        target = table_name or "any table"
        raise ValueError(
            f"Could not parse INSERT rows from SQL dump for {target}. "
            "Please pass a TSV/CSV/Excel/SQLite file or set --process-table correctly."
        )

    if table_name:
        selected = None
        for tbl in table_rows:
            if tbl.lower() == table_name.lower():
                selected = tbl
                break
        if not selected:
            raise ValueError(f"Table '{table_name}' not found in SQL dump INSERT statements.")
    else:
        selected = max(table_rows.keys(), key=lambda t: len(table_rows[t]))

    rows = table_rows[selected]
    cols = table_cols.get(selected)
    if cols and rows and len(cols) == len(rows[0]):
        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(rows)


def duration_to_hours(
    series: pd.Series,
    numeric_unit: Optional[str] = None,
) -> pd.Series:
    if series.empty:
        return pd.Series(index=series.index, dtype=float)

    if pd.api.types.is_timedelta64_dtype(series.dtype):
        return series.dt.total_seconds() / 3600.0

    numeric_values = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric_values.notna()
    result = pd.Series(np.nan, index=series.index, dtype=float)

    if numeric_mask.any():
        if numeric_unit is None:
            warnings.warn(
                "Numeric duration values were found but the applicable duration-unit option was not supplied; "
                "duration-based metrics will be unavailable.",
                UserWarning,
                stacklevel=2,
            )
        else:
            unit_to_hours = {
                "seconds": 1 / 3600,
                "minutes": 1 / 60,
                "hours": 1,
                "days": 24,
            }
            result.loc[numeric_mask] = numeric_values.loc[numeric_mask] * unit_to_hours[numeric_unit]

    non_numeric_mask = series.notna() & ~numeric_mask
    if non_numeric_mask.any():
        def parse_explicit_duration(value: object) -> object:
            if isinstance(value, datetime_time):
                return pd.Timedelta(
                    hours=value.hour,
                    minutes=value.minute,
                    seconds=value.second,
                    microseconds=value.microsecond,
                )
            return pd.to_timedelta(value, errors="coerce")

        timedelta_values = pd.to_timedelta(
            series.loc[non_numeric_mask].map(parse_explicit_duration),
            errors="coerce",
        )
        result.loc[non_numeric_mask] = timedelta_values.dt.total_seconds() / 3600.0
    return result


def add_ems_energy_columns(ems_df: pd.DataFrame) -> pd.DataFrame:
    ems = ems_df.copy()
    time_col = first_existing_column(ems, ["date_time", "timestamp", "_timestamp", "tm1_timestamp"])
    if not time_col:
        return ems

    machine_col = first_existing_column(ems, ["equipment_id", "eq_id", "meter_id", "eq_code"])
    machine_kwh_col = first_existing_column(ems, ["machine_kwh", "kwh"])
    master_kwh_col = first_existing_column(ems, ["master_kwh"])

    if machine_kwh_col:
        sort_cols = [machine_col, time_col] if machine_col else [time_col]
        ems = ems.sort_values(sort_cols).copy()
        machine_kwh = pd.to_numeric(ems[machine_kwh_col], errors="coerce")

        if master_kwh_col:
            master_kwh = pd.to_numeric(ems[master_kwh_col], errors="coerce")
            ems["shift_energy_kwh"] = (machine_kwh - master_kwh).where(
                machine_kwh >= master_kwh
            )
        elif machine_col:
            ems["energy_kwh"] = machine_kwh.groupby(ems[machine_col]).diff()
        else:
            ems["energy_kwh"] = machine_kwh.diff()
        if "energy_kwh" in ems:
            ems["energy_kwh"] = ems["energy_kwh"].where(ems["energy_kwh"] >= 0)

    return ems


class EMSIntervalIndex:
    def __init__(self, ems_df: pd.DataFrame) -> None:
        time_col = first_existing_column(
            ems_df, ["date_time", "timestamp", "_timestamp", "tm1_timestamp"]
        )
        self.rows = (
            ems_df.loc[ems_df[time_col].notna()]
            .sort_values(time_col, kind="mergesort")
            if time_col
            else ems_df.iloc[0:0]
        )
        self.times = pd.DatetimeIndex(self.rows[time_col]) if time_col else pd.DatetimeIndex([])
        self.power_prefix = np.zeros(len(self.rows) + 1, dtype=float)
        self.power_count_prefix = np.zeros(len(self.rows) + 1, dtype=np.int64)
        self.energy_prefix = np.zeros(len(self.rows) + 1, dtype=float)
        self.energy_count_prefix = np.zeros(len(self.rows) + 1, dtype=np.int64)
        power_col = first_existing_column(ems_df, ["kw", "watts"])
        if power_col:
            power = pd.to_numeric(self.rows[power_col], errors="coerce")
            if power_col == "watts":
                power = power / 1000
            self.power_prefix, self.power_count_prefix = self._prefix_sums(power)
        if "shift_energy_kwh" not in ems_df.columns and "energy_kwh" in ems_df.columns:
            energy = pd.to_numeric(self.rows["energy_kwh"], errors="coerce")
            self.energy_prefix, self.energy_count_prefix = self._prefix_sums(energy)

        self.counter_series: list[tuple[pd.DatetimeIndex, np.ndarray]] = []
        if time_col and "shift_energy_kwh" in ems_df.columns:
            meter_col = first_existing_column(
                ems_df, ["equipment_id", "eq_id", "meter_id", "eq_code"]
            )
            if meter_col:
                groups = self.rows.groupby(meter_col, dropna=False, sort=False)
            else:
                groups = [(None, self.rows)]
            for _, meter_rows in groups:
                counters = meter_rows.loc[meter_rows["shift_energy_kwh"].notna()]
                if not counters.empty:
                    self.counter_series.append(
                        (
                            pd.DatetimeIndex(counters[time_col]),
                            pd.to_numeric(
                                counters["shift_energy_kwh"], errors="coerce"
                            ).to_numpy(dtype=float),
                        )
                    )

    @staticmethod
    def _prefix_sums(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(numeric)
        sums = np.concatenate(([0.0], np.cumsum(np.where(valid, numeric, 0.0))))
        counts = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
        return sums, counts

    def measure(self, start_time: object, end_time: object) -> dict[str, object]:
        metrics: dict[str, object] = {
            "match_rows": 0,
            "energy_kwh": np.nan,
            "avg_kw": np.nan,
        }
        if pd.isna(start_time) or pd.isna(end_time) or end_time < start_time:
            return metrics

        left = int(self.times.searchsorted(start_time, side="left"))
        right = int(self.times.searchsorted(end_time, side="right"))
        metrics["match_rows"] = right - left
        if right <= left:
            return metrics

        if self.counter_series:
            deltas = []
            for counter_times, counter_values in self.counter_series:
                start_index = int(counter_times.searchsorted(start_time, side="right")) - 1
                end_index = int(counter_times.searchsorted(end_time, side="right")) - 1
                if start_index < 0 or end_index < 0:
                    continue
                start_value = counter_values[start_index]
                end_value = counter_values[end_index]
                if np.isfinite(start_value) and np.isfinite(end_value) and end_value >= start_value:
                    deltas.append(end_value - start_value)
            if deltas:
                metrics["energy_kwh"] = float(sum(deltas))
        else:
            energy_count = self.energy_count_prefix[right] - self.energy_count_prefix[left]
            if energy_count:
                metrics["energy_kwh"] = (
                    self.energy_prefix[right] - self.energy_prefix[left]
                )

        power_count = self.power_count_prefix[right] - self.power_count_prefix[left]
        if power_count:
            metrics["avg_kw"] = (
                self.power_prefix[right] - self.power_prefix[left]
            ) / power_count
        return metrics


def aggregate_ems_within_runs(
    production_df: pd.DataFrame,
    ems_df: pd.DataFrame,
    ems_index: Optional[EMSIntervalIndex] = None,
) -> pd.DataFrame:
    prod = production_df.copy()
    if ems_df.empty:
        return prod

    start_col = first_existing_column(prod, ["prod_start_time", "begin_time", "start_time"])
    end_col = first_existing_column(prod, ["prod_end_time", "end_time", "stop_time"])
    ems_time_col = first_existing_column(ems_df, ["date_time", "timestamp", "_timestamp", "tm1_timestamp"])
    if not start_col or not end_col or not ems_time_col:
        return prod

    ems_index = ems_index or EMSIntervalIndex(ems_df)

    prod["energy_kwh"] = np.nan
    prod["energy_match_rows"] = 0
    if ems_index.power_count_prefix[-1] > 0:
        prod["avg_kw"] = np.nan

    for idx, row in prod.iterrows():
        if (
            pd.isna(row[start_col])
            or pd.isna(row[end_col])
            or row[end_col] < row[start_col]
        ):
            continue
        metrics = ems_index.measure(row[start_col], row[end_col])
        prod.at[idx, "energy_match_rows"] = metrics["match_rows"]
        prod.at[idx, "energy_kwh"] = metrics["energy_kwh"]
        if ems_index.power_count_prefix[-1]:
            prod.at[idx, "avg_kw"] = metrics["avg_kw"]
    return prod


def aggregate_ems_within_downtime(
    downtime_df: pd.DataFrame,
    ems_df: pd.DataFrame,
    ems_index: Optional[EMSIntervalIndex] = None,
) -> pd.DataFrame:
    down = downtime_df.copy()
    if down.empty or ems_df.empty:
        return down

    start_col = first_existing_column(
        down, ["begin_time", "downtime_start_time", "start_time", "event_start_time"]
    )
    end_col = first_existing_column(
        down, ["end_time", "downtime_end_time", "stop_time", "event_end_time"]
    )
    ems_time_col = first_existing_column(
        ems_df, ["date_time", "timestamp", "_timestamp", "tm1_timestamp"]
    )
    down["ems_match_rows"] = 0
    down["ems_energy_kwh"] = np.nan
    down["ems_avg_kw"] = np.nan
    if not start_col or not end_col or not ems_time_col:
        return down

    ems_index = ems_index or EMSIntervalIndex(ems_df)
    for idx, row in down.iterrows():
        if (
            pd.isna(row[start_col])
            or pd.isna(row[end_col])
            or row[end_col] < row[start_col]
        ):
            continue
        metrics = ems_index.measure(row[start_col], row[end_col])
        down.at[idx, "ems_match_rows"] = metrics["match_rows"]
        down.at[idx, "ems_energy_kwh"] = metrics["energy_kwh"]
        down.at[idx, "ems_avg_kw"] = metrics["avg_kw"]
    return down


def aggregate_ppc_within_runs(production_df: pd.DataFrame, ppc_df: pd.DataFrame) -> pd.DataFrame:
    prod = production_df.copy()
    if ppc_df.empty:
        return prod

    start_col = first_existing_column(prod, ["prod_start_time", "begin_time", "start_time"])
    end_col = first_existing_column(prod, ["prod_end_time", "end_time", "stop_time"])
    ppc_time_col = first_existing_column(ppc_df, ["tm1_timestamp", "timestamp", "date_time", "_timestamp"])
    if not start_col or not end_col or not ppc_time_col:
        return prod

    prod_machine_col = first_existing_column(prod, ["eq_id", "eq_code", "equipment_id"])
    ppc_machine_col = first_existing_column(ppc_df, ["equipment_id", "eq_id", "eq_code", "meter_id"])
    total_col = first_existing_column(ppc_df, ["tm1_total_qty", "total_qty", "pq_qty"])
    scrap_col = first_existing_column(ppc_df, ["tm1_scrap_qty", "scrap_qty"])

    prod["ppc_total_qty_delta"] = np.nan
    prod["ppc_scrap_qty_delta"] = np.nan

    for idx, row in prod.iterrows():
        if pd.isna(row[start_col]) or pd.isna(row[end_col]):
            continue
        mask = (ppc_df[ppc_time_col] >= row[start_col]) & (ppc_df[ppc_time_col] <= row[end_col])
        if prod_machine_col and ppc_machine_col and pd.notna(row[prod_machine_col]):
            mask &= machine_id_match(ppc_df[ppc_machine_col], row[prod_machine_col])
        ppc_slice = ppc_df.loc[mask]
        if ppc_slice.empty:
            continue
        if total_col:
            total_vals = pd.to_numeric(ppc_slice[total_col], errors="coerce").dropna()
            if not total_vals.empty:
                prod.at[idx, "ppc_total_qty_delta"] = max(total_vals.iloc[-1] - total_vals.iloc[0], 0)
        if scrap_col:
            scrap_vals = pd.to_numeric(ppc_slice[scrap_col], errors="coerce").dropna()
            if not scrap_vals.empty:
                prod.at[idx, "ppc_scrap_qty_delta"] = max(scrap_vals.iloc[-1] - scrap_vals.iloc[0], 0)
    return prod


def minmax_high(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty or np.isclose(valid.max(), valid.min()):
        return pd.Series(0.5, index=series.index)
    return (series - valid.min()) / (valid.max() - valid.min())


def minmax_low(series: pd.Series) -> pd.Series:
    return 1.0 - minmax_high(series)


def compute_machine_efficiency(
    production_df: pd.DataFrame,
    downtime_df: Optional[pd.DataFrame] = None,
    production_duration_unit: Optional[str] = None,
    downtime_duration_unit: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prod = production_df.copy()
    machine_col = first_existing_column(prod, ["eq_code", "eq_name", "eq_id", "equipment_id"])
    if not machine_col:
        raise ValueError("Could not identify machine column in production data.")
    prod["machine"] = prod[machine_col].astype(str)

    pq_col = first_existing_column(prod, ["pq_qty", "total_qty", "tm1_total_qty"])
    scrap_col = first_existing_column(prod, ["scrap_qty", "tm1_scrap_qty"])
    run_col = first_existing_column(prod, ["run_time"])
    net_col = first_existing_column(prod, ["net_run_time"])
    start_col = first_existing_column(prod, ["prod_start_time", "begin_time"])
    end_col = first_existing_column(prod, ["prod_end_time", "end_time"])

    prod["pq_qty_clean"] = pd.to_numeric(prod[pq_col], errors="coerce") if pq_col else np.nan
    prod["scrap_qty_clean"] = pd.to_numeric(prod[scrap_col], errors="coerce") if scrap_col else np.nan

    if net_col:
        prod["net_run_hours"] = duration_to_hours(prod[net_col], production_duration_unit)
    else:
        prod["net_run_hours"] = np.nan

    if run_col:
        prod["run_hours"] = duration_to_hours(prod[run_col], production_duration_unit)
    else:
        prod["run_hours"] = np.nan

    if start_col and end_col:
        elapsed = (prod[end_col] - prod[start_col]).dt.total_seconds() / 3600.0
        prod["run_hours"] = prod["run_hours"].fillna(elapsed)

    if "energy_kwh" not in prod.columns:
        prod["energy_kwh"] = np.nan
    if "avg_kw" not in prod.columns:
        prod["avg_kw"] = np.nan
    if "energy_match_rows" not in prod.columns:
        prod["energy_match_rows"] = 0
    for speed_col in ("target_speed", "actual_speed"):
        if speed_col not in prod.columns:
            prod[speed_col] = np.nan
        prod[speed_col] = pd.to_numeric(prod[speed_col], errors="coerce")
    prod["speed_attainment_pct"] = (
        prod["actual_speed"] / prod["target_speed"].where(prod["target_speed"] > 0) * 100
    )

    machine_summary = (
        prod.groupby("machine", dropna=False)
        .agg(
            run_count=("machine", "size"),
            produced_qty=("pq_qty_clean", lambda values: values.sum(min_count=1)),
            scrap_qty=("scrap_qty_clean", lambda values: values.sum(min_count=1)),
            run_hours=("run_hours", lambda values: values.sum(min_count=1)),
            net_run_hours=("net_run_hours", lambda values: values.sum(min_count=1)),
            energy_kwh=("energy_kwh", lambda values: values.sum(min_count=1)),
            avg_kw=("avg_kw", "mean"),
            target_speed=("target_speed", "mean"),
            actual_speed=("actual_speed", "mean"),
            speed_attainment_pct=("speed_attainment_pct", "mean"),
            energy_matched_runs=("energy_kwh", "count"),
            ems_runs_with_readings=(
                "energy_match_rows",
                lambda values: (pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum(),
            ),
            matched_ems_rows=("energy_match_rows", "sum"),
            runs_with_power=("avg_kw", "count"),
        )
        .reset_index()
    )
    machine_summary["energy_coverage_pct"] = (
        machine_summary["energy_matched_runs"] / machine_summary["run_count"] * 100
    )
    machine_summary["output_per_hour"] = machine_summary["produced_qty"] / machine_summary[
        "net_run_hours"
    ].where(machine_summary["net_run_hours"] > 0)
    total_qty = machine_summary["produced_qty"] + machine_summary["scrap_qty"]
    machine_summary["scrap_rate"] = machine_summary["scrap_qty"] / total_qty.where(total_qty > 0)
    machine_summary["availability"] = machine_summary["net_run_hours"] / machine_summary["run_hours"]
    machine_summary.loc[
        ~machine_summary["availability"].between(0, 1, inclusive="both"), "availability"
    ] = np.nan
    machine_summary["energy_per_unit"] = machine_summary["energy_kwh"] / machine_summary[
        "produced_qty"
    ].where(machine_summary["produced_qty"] > 0)

    if downtime_df is not None and not downtime_df.empty:
        down = downtime_df.copy()
        down_machine_col = first_existing_column(down, ["eq_code", "eq_name", "eq_id", "equipment_id"])
        prod_run_id_col = first_existing_column(prod, ["product_ref_id", "transaction_id"])
        down_run_id_col = first_existing_column(down, ["product_ref_id", "transaction_id"])
        duration_col = first_existing_column(down, ["duration"])
        if duration_col:
            down["downtime_hours"] = duration_to_hours(
                down[duration_col], downtime_duration_unit
            )

            if prod_run_id_col and down_run_id_col:
                prod_run_ids = set(prod[prod_run_id_col].dropna())
                run_to_machine = prod[[prod_run_id_col, "machine"]].dropna(
                    subset=[prod_run_id_col]
                ).drop_duplicates(subset=[prod_run_id_col])
                with_run_id = down[down[down_run_id_col].notna()]
                run_linked = with_run_id.merge(
                    run_to_machine,
                    left_on=down_run_id_col,
                    right_on=prod_run_id_col,
                    how="inner",
                )
                downtime_parts = [
                    run_linked[["machine", "downtime_hours"]]
                    .groupby("machine", dropna=False)["downtime_hours"]
                    .agg(lambda values: values.sum(min_count=1))
                    .reset_index()
                ]
                if down_machine_col:
                    fallback = down[
                        down[down_run_id_col].isna()
                        | ~down[down_run_id_col].isin(prod_run_ids)
                    ].copy()
                    fallback["machine"] = fallback[down_machine_col].astype(str)
                    downtime_parts.append(
                        fallback.groupby("machine", dropna=False)["downtime_hours"]
                        .agg(lambda values: values.sum(min_count=1))
                        .reset_index()
                    )
                downtime_summary = pd.concat(downtime_parts, ignore_index=True)
                downtime_summary = (
                    downtime_summary.groupby("machine", dropna=False)["downtime_hours"]
                    .agg(lambda values: values.sum(min_count=1))
                    .reset_index()
                )
            elif down_machine_col:
                down["machine"] = down[down_machine_col].astype(str)
                downtime_summary = (
                    down.groupby("machine", dropna=False)["downtime_hours"]
                    .agg(lambda values: values.sum(min_count=1))
                    .reset_index()
                )
            else:
                downtime_summary = pd.DataFrame(columns=["machine", "downtime_hours"])

            machine_summary = machine_summary.merge(downtime_summary, on="machine", how="left")
    if "downtime_hours" not in machine_summary.columns:
        machine_summary["downtime_hours"] = np.nan
    machine_summary["downtime_rate"] = machine_summary["downtime_hours"] / machine_summary[
        "run_hours"
    ].where(machine_summary["run_hours"] > 0)

    machine_summary = machine_summary.replace([np.inf, -np.inf], np.nan)

    machine_summary["efficiency_score"] = np.nan
    machine_summary["efficiency_band"] = "not_comparable"
    score_metrics = [
        "output_per_hour",
        "scrap_rate",
        "availability",
        "energy_per_unit",
        "downtime_rate",
    ]
    complete_metrics = machine_summary[score_metrics].notna().all(axis=1)
    comparable_count = int(complete_metrics.sum())

    if comparable_count >= 2:
        machine_summary["score_output"] = minmax_high(machine_summary["output_per_hour"])
        machine_summary["score_quality"] = minmax_low(machine_summary["scrap_rate"])
        machine_summary["score_availability"] = minmax_high(machine_summary["availability"])
        machine_summary["score_energy"] = minmax_low(machine_summary["energy_per_unit"])
        machine_summary["score_downtime"] = minmax_low(machine_summary["downtime_rate"])

        weighted_score = (
            0.30 * machine_summary["score_output"]
            + 0.20 * machine_summary["score_quality"]
            + 0.20 * machine_summary["score_availability"]
            + 0.20 * machine_summary["score_energy"]
            + 0.10 * machine_summary["score_downtime"]
        )
        machine_summary.loc[complete_metrics, "efficiency_score"] = weighted_score.loc[
            complete_metrics
        ]

    if comparable_count >= 4:
        valid_scores = machine_summary.loc[complete_metrics, "efficiency_score"]
        q_low = valid_scores.quantile(0.25)
        q_high = valid_scores.quantile(0.75)
        if q_low < q_high:
            machine_summary.loc[complete_metrics, "efficiency_band"] = np.where(
                valid_scores >= q_high,
                "efficient",
                np.where(valid_scores <= q_low, "inefficient", "mid"),
            )

    machine_summary = machine_summary.sort_values(
        "efficiency_score", ascending=False, na_position="last"
    ).reset_index(drop=True)

    return prod, machine_summary


def create_daily_summary(
    production_runs: pd.DataFrame,
    downtime_df: Optional[pd.DataFrame],
    downtime_duration_unit: Optional[str] = None,
) -> pd.DataFrame:
    daily_summary = pd.DataFrame()
    production_start_col = first_existing_column(
        production_runs, ["prod_start_time", "begin_time", "start_time"]
    )
    if production_start_col:
        run_daily = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    production_runs[production_start_col], errors="coerce"
                ).dt.normalize(),
                "production_runs": 1,
            }
        )
        energy_col = first_existing_column(production_runs, ["energy_kwh"])
        if energy_col:
            run_daily["energy_kwh"] = pd.to_numeric(
                production_runs[energy_col], errors="coerce"
            )
        run_hours_col = first_existing_column(production_runs, ["run_hours"])
        if run_hours_col:
            run_daily["run_hours"] = pd.to_numeric(
                production_runs[run_hours_col], errors="coerce"
            )
        run_daily = run_daily.dropna(subset=["date"])
        if not run_daily.empty:
            aggregations = {"production_runs": ("production_runs", "sum")}
            if energy_col:
                aggregations["energy_kwh"] = (
                    "energy_kwh",
                    lambda values: values.sum(min_count=1),
                )
            if run_hours_col:
                aggregations["run_hours"] = (
                    "run_hours",
                    lambda values: values.sum(min_count=1),
                )
            daily_summary = (
                run_daily.groupby("date", as_index=False)
                .agg(**aggregations)
            )

    if downtime_df is not None and not downtime_df.empty:
        downtime_start_col = first_existing_column(
            downtime_df,
            ["begin_time", "downtime_start_time", "start_time", "event_start_time"],
        )
        duration_col = first_existing_column(downtime_df, ["duration"])
        if downtime_start_col and duration_col:
            down_daily = pd.DataFrame(
                {
                    "date": pd.to_datetime(
                        downtime_df[downtime_start_col], errors="coerce"
                    ).dt.normalize(),
                    "downtime_hours": duration_to_hours(
                        downtime_df[duration_col], downtime_duration_unit
                    ),
                }
            ).dropna(subset=["date"])
            if not down_daily.empty:
                down_daily = (
                    down_daily.groupby("date", as_index=False)
                    .agg(
                        downtime_hours=(
                            "downtime_hours",
                            lambda values: values.sum(min_count=1),
                        )
                    )
                )
                daily_summary = (
                    down_daily
                    if daily_summary.empty
                    else daily_summary.merge(down_daily, on="date", how="outer")
                )

    if daily_summary.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "production_runs",
                "run_hours",
                "energy_kwh",
                "downtime_hours",
            ]
        )
    if "production_runs" in daily_summary:
        daily_summary["production_runs"] = (
            daily_summary["production_runs"].fillna(0).astype(int)
        )
    return daily_summary.sort_values("date").reset_index(drop=True)


def summarize_process_tags_for_dates(
    production_runs: pd.DataFrame,
    process_df: Optional[pd.DataFrame],
    dates: pd.Series,
) -> pd.DataFrame:
    if process_df is None or process_df.empty or dates.empty:
        return pd.DataFrame()

    run_id_col = first_existing_column(
        production_runs, ["product_ref_id", "transaction_id"]
    )
    run_start_col = first_existing_column(
        production_runs, ["prod_start_time", "begin_time", "start_time"]
    )
    process_run_id_col = first_existing_column(
        process_df, ["transaction_id", "product_ref_id"]
    )
    tag_col = first_existing_column(process_df, ["_name", "name"])
    value_col = first_existing_column(process_df, ["_value", "value"])
    if not all(
        (run_id_col, run_start_col, process_run_id_col, tag_col, value_col)
    ):
        return pd.DataFrame()

    selected_dates = pd.to_datetime(dates, errors="coerce").dropna().dt.normalize()
    if selected_dates.empty:
        return pd.DataFrame()

    run_dates = pd.to_datetime(
        production_runs[run_start_col], errors="coerce"
    ).dt.normalize()
    selected_run_ids = production_runs.loc[
        run_dates.isin(selected_dates), run_id_col
    ].dropna().astype("string")
    if selected_run_ids.empty:
        return pd.DataFrame()

    tagged = (
        process_df[[process_run_id_col, tag_col, value_col]]
        .rename(
            columns={
                process_run_id_col: "run_id",
                tag_col: "tag",
                value_col: "tag_value",
            }
        )
    )
    tagged["run_id"] = tagged["run_id"].astype("string")
    tagged["tag_value"] = pd.to_numeric(tagged["tag_value"], errors="coerce")
    tagged = tagged.loc[
        tagged["run_id"].isin(selected_run_ids)
        & tagged["tag"].notna()
        & tagged["tag_value"].notna()
    ]
    if tagged.empty:
        return pd.DataFrame()

    per_run = (
        tagged.groupby(["run_id", "tag"], dropna=False)
        .agg(run_mean=("tag_value", "mean"))
        .reset_index()
    )
    summary = (
        per_run.groupby("tag", dropna=False)
        .agg(
            linked_run_count=("run_id", "nunique"),
            mean_run_value=("run_mean", "mean"),
            max_run_value=("run_mean", "max"),
        )
        .reset_index()
    )
    tag_names = summary["tag"].astype("string")
    summary["category"] = np.select(
        [
            tag_names.str.contains(r"power|kw|kwh|watt", case=False, regex=True, na=False),
            tag_names.str.contains(r"coolant|flow|pressure", case=False, regex=True, na=False),
            tag_names.str.contains(r"temp|temperature", case=False, regex=True, na=False),
        ],
        ["Power", "Coolant", "Temperature"],
        default="Other",
    )
    summary = summary.loc[summary["category"] != "Other"]
    return summary.sort_values(
        ["category", "mean_run_value", "linked_run_count"],
        ascending=[True, False, False],
        na_position="last",
    ).head(8).reset_index(drop=True)


def create_plots(
    machine_summary: pd.DataFrame,
    downtime_df: Optional[pd.DataFrame],
    output_dir: Path,
    downtime_duration_unit: Optional[str] = None,
    daily_summary: Optional[pd.DataFrame] = None,
) -> None:
    sns.set_theme(style="whitegrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = [
        "efficiency_score_by_machine.png",
        "output_vs_energy.png",
        "machine_metric_correlation_heatmap.png",
        "top_downtime_reasons.png",
        "process_parameter_gap.png",
        "daily_downtime_vs_energy.png",
    ]
    for filename in chart_paths:
        chart_path = output_dir / filename
        if chart_path.exists():
            chart_path.unlink()

    scored = machine_summary.dropna(subset=["efficiency_score"])
    if len(scored) >= 2 and scored["efficiency_score"].nunique() > 1:
        ordered = scored.sort_values("efficiency_score", ascending=False)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=ordered, x="machine", y="efficiency_score", hue="efficiency_band", dodge=False)
        plt.xticks(rotation=45, ha="right")
        plt.title("Relative Machine Efficiency Score")
        plt.tight_layout()
        plt.savefig(output_dir / "efficiency_score_by_machine.png", dpi=150)
        plt.close()

    if {"output_per_hour", "energy_per_unit", "efficiency_score"}.issubset(machine_summary.columns):
        plot_data = machine_summary.dropna(
            subset=["output_per_hour", "energy_per_unit"]
        )
        if len(plot_data) >= 2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(
                data=plot_data,
                x="output_per_hour",
                y="energy_per_unit",
                hue="efficiency_band",
            )
            plt.title("Energy Intensity vs Output Rate")
            plt.tight_layout()
            plt.savefig(output_dir / "output_vs_energy.png", dpi=150)
            plt.close()

    corr_cols = [
        "produced_qty",
        "scrap_qty",
        "run_hours",
        "net_run_hours",
        "output_per_hour",
        "scrap_rate",
        "availability",
        "energy_per_unit",
        "downtime_hours",
        "efficiency_score",
    ]
    corr_cols = [c for c in corr_cols if c in machine_summary.columns]
    corr = machine_summary[corr_cols].corr(numeric_only=True)
    off_diagonal = corr.where(~np.eye(len(corr), dtype=bool))
    if len(machine_summary) >= 3 and off_diagonal.notna().any().any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
        plt.title("Machine-level Metric Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / "machine_metric_correlation_heatmap.png", dpi=150)
        plt.close()

    if downtime_df is not None and not downtime_df.empty:
        down = downtime_df.copy()
        loss_col = first_existing_column(down, ["main_loss_name", "sub_loss_name"])
        duration_col = first_existing_column(down, ["duration"])
        machine_col = first_existing_column(down, ["eq_code", "eq_name", "eq_id", "equipment_id"])
        if loss_col and duration_col and machine_col:
            down["machine"] = down[machine_col].astype(str)
            down["downtime_hours"] = duration_to_hours(
                down[duration_col], downtime_duration_unit
            )
            if "ems_energy_kwh" not in down:
                down["ems_energy_kwh"] = np.nan
            down["ems_energy_kwh"] = pd.to_numeric(
                down["ems_energy_kwh"], errors="coerce"
            )
            totals = (
                down.groupby(loss_col, dropna=False)
                .agg(
                    downtime_hours=(
                        "downtime_hours",
                        lambda values: values.sum(min_count=1),
                    ),
                    ems_energy_kwh=(
                        "ems_energy_kwh",
                        lambda values: values.sum(min_count=1),
                    ),
                )
                .sort_values("downtime_hours", ascending=False)
                .head(10)
                .reset_index()
            )
            totals = totals.dropna(subset=["downtime_hours"])
            if not totals.empty:
                x = np.arange(len(totals))
                width = 0.38
                fig, hours_axis = plt.subplots(figsize=(12, 6))
                hours_axis.bar(
                    x - width / 2,
                    totals["downtime_hours"],
                    width,
                    label="Downtime (hours)",
                    color="#9b59b6",
                )
                hours_axis.set_ylabel("Downtime (hours)")
                hours_axis.set_xticks(x)
                hours_axis.set_xticklabels(totals[loss_col], rotation=45, ha="right")

                energy_available = totals["ems_energy_kwh"].notna()
                if energy_available.any():
                    energy_axis = hours_axis.twinx()
                    energy_axis.bar(
                        x[energy_available] + width / 2,
                        totals.loc[energy_available, "ems_energy_kwh"],
                        width,
                        label="Matched EMS energy (kWh)",
                        color="#e67e22",
                        alpha=0.85,
                    )
                    energy_axis.set_ylabel("Matched EMS energy (kWh)")
                    energy_axis.set_ylim(bottom=0)
                    energy_limit = energy_axis.get_ylim()[1]
                    for missing_x in x[~energy_available]:
                        energy_axis.text(
                            missing_x + width / 2,
                            energy_limit * 0.02,
                            "N/A",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="#7f8c8d",
                        )
                    handles, labels = hours_axis.get_legend_handles_labels()
                    energy_handles, energy_labels = energy_axis.get_legend_handles_labels()
                    hours_axis.legend(
                        handles + energy_handles,
                        labels + energy_labels,
                        loc="upper right",
                    )
                    title = "Downtime and Matched EMS Energy by Loss Reason"
                    footnote = (
                        "EMS energy is time-associated; overlapping downtime windows may count "
                        "the same EMS energy more than once."
                    )
                else:
                    title = "Downtime by Loss Reason (hours)"
                    footnote = (
                        "Matched EMS energy is unavailable; missing energy is not treated as zero."
                    )

                hours_axis.set_title(title)
                fig.text(0.5, 0.01, footnote, ha="center", fontsize=8, color="#555555")
                fig.tight_layout(rect=(0, 0.06, 1, 1))
                plt.savefig(output_dir / "top_downtime_reasons.png", dpi=150)
                plt.close()

    if daily_summary is not None and not daily_summary.empty:
        has_downtime = (
            "downtime_hours" in daily_summary
            and daily_summary["downtime_hours"].notna().any()
        )
        has_energy = (
            "energy_kwh" in daily_summary
            and daily_summary["energy_kwh"].notna().any()
        )
        if has_downtime and has_energy:
            dates = pd.to_datetime(daily_summary["date"])
            date_numbers = mdates.date2num(dates.dt.to_pydatetime())
            bar_width = 0.35
            fig, downtime_axis = plt.subplots(figsize=(12, 6))
            downtime_bars = downtime_axis.bar(
                date_numbers - bar_width / 2,
                daily_summary["downtime_hours"],
                width=bar_width,
                color="#9b59b6",
                label="Downtime (hours)",
            )
            downtime_axis.set_ylabel("Downtime (hours)")
            downtime_axis.grid(axis="y", alpha=0.3)
            downtime_axis.set_axisbelow(True)

            energy_axis = downtime_axis.twinx()
            energy_bars = energy_axis.bar(
                date_numbers + bar_width / 2,
                daily_summary["energy_kwh"] / 1000,
                width=bar_width,
                color="#e67e22",
                label="Matched production energy (MWh)",
            )
            energy_axis.set_ylabel("Matched production energy (MWh)")
            energy_axis.set_ylim(bottom=0)

            downtime_axis.legend(
                [downtime_bars, energy_bars],
                ["Downtime (hours)", "Matched production energy (MWh)"],
                loc="upper right",
            )
            downtime_axis.set_title("Daily Downtime vs Matched Production Energy")
            downtime_axis.set_xlabel("Date")
            downtime_axis.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
            )
            downtime_axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            downtime_axis.set_xlim(
                date_numbers.min() - 0.5,
                date_numbers.max() + 0.5,
            )
            fig.tight_layout()
            plt.savefig(output_dir / "daily_downtime_vs_energy.png", dpi=150)
            plt.close()


def calculate_prewindow_features(
    timestamps: pd.Series,
    values: pd.Series,
    window_end: pd.Timestamp,
    tag_normal_low: float,
    tag_normal_high: float,
    tag_normal_std: float,
) -> dict[str, float | int | bool]:
    timestamps = pd.to_datetime(timestamps, errors="coerce")
    values = pd.to_numeric(values, errors="coerce")
    valid = timestamps.notna() & values.notna()
    timestamps = timestamps.loc[valid].sort_values()
    values = values.loc[valid].reindex(timestamps.index)

    if values.empty:
        return {
            "prediction_cutoff": window_end,
            "first_reading_lead_minutes": np.nan,
            "last_reading_lead_minutes": np.nan,
            "post_prediction_readings": 0,
            "causal_evidence": "insufficient_pre_stoppage_data",
            "reading_count": 0,
            "mean_value": np.nan,
            "min_value": np.nan,
            "max_value": np.nan,
            "fluctuation": np.nan,
            "start_to_end_change": np.nan,
            "sudden_changes": 0,
            "outside_normal_count": 0,
            "outside_normal_pct": np.nan,
            "missing_reading": True,
            "stale_reading": True,
        }

    ordered_values = values.to_numpy(dtype=float)
    diffs = np.diff(ordered_values)
    sudden_threshold = max(float(tag_normal_std) * 2.0, np.finfo(float).eps)
    sudden_changes = int(np.count_nonzero(np.abs(diffs) > sudden_threshold))
    outside_mask = (ordered_values < tag_normal_low) | (ordered_values > tag_normal_high)
    latest = timestamps.max()
    earliest = timestamps.min()
    stale = pd.isna(latest) or (window_end - latest).total_seconds() > 300
    first_lead = (window_end - earliest).total_seconds() / 60.0
    last_lead = (window_end - latest).total_seconds() / 60.0
    post_prediction_readings = int((timestamps >= window_end).sum())
    if post_prediction_readings:
        causal_evidence = "contaminated_by_post_prediction_readings"
    elif stale:
        causal_evidence = "pre_stoppage_candidate_but_stale_reading"
    else:
        causal_evidence = "pre_stoppage_candidate"

    return {
        "prediction_cutoff": window_end,
        "first_reading_lead_minutes": float(first_lead),
        "last_reading_lead_minutes": float(last_lead),
        "post_prediction_readings": post_prediction_readings,
        "causal_evidence": causal_evidence,
        "reading_count": int(len(ordered_values)),
        "mean_value": float(np.mean(ordered_values)),
        "min_value": float(np.min(ordered_values)),
        "max_value": float(np.max(ordered_values)),
        "fluctuation": float(np.std(ordered_values, ddof=1)) if len(ordered_values) > 1 else 0.0,
        "start_to_end_change": float(ordered_values[-1] - ordered_values[0]),
        "sudden_changes": sudden_changes,
        "outside_normal_count": int(outside_mask.sum()),
        "outside_normal_pct": float(outside_mask.mean() * 100.0),
        "missing_reading": False,
        "stale_reading": bool(stale),
    }


def classify_downtime_event(reason: object) -> tuple[str, bool]:
    text = str(reason or "").strip().lower()
    if not text:
        return "unclassified", True

    planned_patterns = (
        r"\bbreakfast\b", r"\blunch\b", r"\bsupper\b", r"\bdinner\b",
        r"\bmeal\b", r"\bchangeover\b", r"\bmodel change\b",
        r"\bscheduled\b", r"\bmaintenance\b", r"\btool changeover\b",
    )
    if any(re.search(pattern, text) for pattern in planned_patterns):
        return "planned_stoppage", False

    adjustment_patterns = (
        r"\bbutting\b", r"\boffset\b", r"\balignment\b",
        r"\bcorrection\b", r"\badjustment\b", r"\btuning\b",
    )
    if any(re.search(pattern, text) for pattern in adjustment_patterns):
        return "process_adjustment", False

    return "unplanned_stoppage", True


def create_downtime_pattern_dataset(
    production_runs: pd.DataFrame,
    downtime_df: Optional[pd.DataFrame],
    process_df: Optional[pd.DataFrame],
    output_dir: Path,
    windows: tuple[int, ...] = (5, 10, 15),
    max_tags: int = 20,
    max_workers: int = 4,
    max_events: int = 0,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "downtime_pattern_features.csv"
    summary_path = output_dir / "downtime_pattern_summary.csv"
    exclusion_path = output_dir / "downtime_pattern_exclusions.csv"
    plot_path = output_dir / "downtime_process_patterns.png"
    if plot_path.exists():
        plot_path.unlink()
    for path in (feature_path, summary_path, exclusion_path):
        if path.exists():
            path.unlink()

    empty_features = pd.DataFrame()
    empty_summary = pd.DataFrame()
    empty_exclusions = pd.DataFrame()

    if downtime_df is None or downtime_df.empty:
        empty_exclusions = pd.DataFrame(
            [{"reason": "No downtime data supplied", "excluded_events": 0, "data_quality": "No downtime rows"}]
        )
        empty_exclusions.to_csv(exclusion_path, index=False)
        return {
            "features": empty_features,
            "summary": empty_summary,
            "exclusions": empty_exclusions,
        }

    if process_df is None or process_df.empty:
        empty_exclusions = pd.DataFrame(
            [{"reason": "No process parameter data supplied", "excluded_events": int(len(downtime_df)), "data_quality": "No PLC/PLC-tag rows"}]
        )
        empty_exclusions.to_csv(exclusion_path, index=False)
        return {
            "features": empty_features,
            "summary": empty_summary,
            "exclusions": empty_exclusions,
        }

    down = downtime_df.copy()
    proc = process_df.copy()

    down_start_col = first_existing_column(down, ["begin_time", "downtime_start_time", "start_time", "event_start_time"])
    down_end_col = first_existing_column(down, ["end_time", "downtime_end_time", "end_time"])
    reason_col = first_existing_column(down, ["sub_loss_name", "main_loss_name", "reason", "downtime_reason"])
    down_run_id_col = first_existing_column(down, ["product_ref_id", "transaction_id"])
    down_machine_col = first_existing_column(down, ["eq_code", "eq_name", "eq_id", "equipment_id"])

    proc_time_col = first_existing_column(proc, ["_timestamp", "timestamp", "date_time", "begin_time", "tm1_timestamp"])
    proc_tag_col = first_existing_column(proc, ["_name", "name", "tag", "parameter"])
    proc_value_col = first_existing_column(proc, ["_value", "value", "tag_value"])
    proc_run_id_col = first_existing_column(proc, ["transaction_id", "product_ref_id", "route_card_no"])
    proc_machine_col = first_existing_column(proc, ["equipment_id", "eq_code", "eq_id", "eq_name"])

    if not all((down_start_col, proc_time_col, proc_tag_col, proc_value_col)):
        empty_exclusions = pd.DataFrame(
            [{
                "reason": "Required timestamp or process-tag columns unavailable",
                "excluded_events": int(len(down)),
                "data_quality": "Missing downtime start, PLC timestamp, tag name, or tag value",
            }]
        )
        empty_exclusions.to_csv(exclusion_path, index=False)
        return {"features": empty_features, "summary": empty_summary, "exclusions": empty_exclusions}

    down[down_start_col] = pd.to_datetime(down[down_start_col], errors="coerce")
    if down_end_col:
        down[down_end_col] = pd.to_datetime(down[down_end_col], errors="coerce")
    proc[proc_time_col] = pd.to_datetime(proc[proc_time_col], errors="coerce")
    proc[proc_value_col] = pd.to_numeric(proc[proc_value_col], errors="coerce")

    proc = proc.loc[
        proc[proc_time_col].notna()
        & proc[proc_tag_col].notna()
        & proc[proc_value_col].notna()
    ].copy()
    proc = proc.sort_values(proc_time_col)
    proc_by_tag: dict[str, pd.DataFrame] = {}
    for tag_name, tag_frame in proc.groupby(proc_tag_col, sort=False, dropna=False):
        proc_by_tag[str(tag_name)] = tag_frame

    run_start_col = first_existing_column(production_runs, ["prod_start_time", "begin_time", "start_time"])
    run_end_col = first_existing_column(production_runs, ["prod_end_time", "end_time"])
    run_id_col = first_existing_column(production_runs, ["product_ref_id", "transaction_id"])
    run_machine_col = first_existing_column(production_runs, ["eq_code", "eq_name", "eq_id", "equipment_id"])
    od_col = first_existing_column(production_runs, ["rolling_od", "od"])
    thickness_col = first_existing_column(production_runs, ["rolling_thickness", "thickness"])
    line_speed_col = first_existing_column(production_runs, ["actual_speed", "target_speed"])

    if run_start_col:
        production_runs = production_runs.copy()
        production_runs[run_start_col] = pd.to_datetime(production_runs[run_start_col], errors="coerce")
    if run_end_col:
        production_runs[run_end_col] = pd.to_datetime(production_runs[run_end_col], errors="coerce")

    tag_reading_counts = (
        proc.groupby(proc_tag_col, dropna=False)[proc_value_col]
        .count()
        .sort_values(ascending=False)
    )
    all_tags = [
        str(tag)
        for tag in tag_reading_counts.head(max(1, max_tags)).index
        if pd.notna(tag)
    ]
    print(
        f"[Patterns] evaluating top {len(all_tags)} of "
        f"{len(tag_reading_counts)} process tags by reading count",
        flush=True,
    )
    exclusions: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    valid_down = down[down_start_col].notna()
    if reason_col:
        valid_down &= down[reason_col].notna()
    down_events = down.loc[valid_down].copy()
    if reason_col:
        down_events["event_class"] = down_events[reason_col].apply(
            lambda reason: classify_downtime_event(reason)[0]
        )
        event_is_predictive = down_events[reason_col].apply(
            lambda reason: classify_downtime_event(reason)[1]
        )
        excluded_planned = down_events.loc[~event_is_predictive].copy()
        planned_counts = (
            excluded_planned.groupby(["event_class"], dropna=False)
            .size()
            .reset_index(name="excluded_events")
        )
        for row in planned_counts.to_dict("records"):
            exclusions.append({
                "reason": f"Excluded {row['event_class']}",
                "excluded_events": int(row["excluded_events"]),
                "data_quality": "Known planned or adjustment downtime; excluded from unplanned pattern prediction",
            })
        down_events = down_events.loc[event_is_predictive].copy()

    if down_events.empty:
        empty_exclusions = pd.DataFrame(
            [{"reason": "No downtime events with valid start timestamps and reasons", "excluded_events": int(len(down)), "data_quality": "Insufficient downtime timestamps"}]
        )
        empty_exclusions.to_csv(exclusion_path, index=False)
        return {"features": empty_features, "summary": empty_summary, "exclusions": empty_exclusions}

    ordered_events = down_events.sort_values(down_start_col).reset_index(drop=True)
    if max_events > 0:
        ordered_events = ordered_events.head(max_events).copy()
        print(
            f"[Patterns] fast mode limited to first {len(ordered_events)} downtime events",
            flush=True,
        )
    split_index = max(1, int(len(ordered_events) * 0.70))
    healthy_runs_cache: dict[tuple[str, float, float, float], pd.DataFrame] = {}
    normal_baseline_cache: dict[
        tuple[str, float, float, float, int, str],
        tuple[float, float, float, pd.Series],
    ] = {}

    for event_index, event in ordered_events.iterrows():
        if event_index and event_index % 10 == 0:
            print(
                f"[Patterns] processed {event_index}/{len(ordered_events)} downtime events",
                flush=True,
            )
        stoppage_start = event[down_start_col]
        stoppage_end = event[down_end_col] if down_end_col and pd.notna(event[down_end_col]) else stoppage_start
        reason = str(event[reason_col]).strip() if reason_col and pd.notna(event[reason_col]) else "Unspecified"
        event_class = classify_downtime_event(reason)[0]
        machine = str(event[down_machine_col]).strip() if down_machine_col and pd.notna(event[down_machine_col]) else ""
        run_id = str(event[down_run_id_col]).strip() if down_run_id_col and pd.notna(event[down_run_id_col]) else ""

        exclusion_reasons: list[str] = []
        if not all_tags:
            exclusion_reasons.append("No process parameter readings")

        event_specs: dict[str, float] = {}
        if run_id and run_id_col:
            matching_runs = production_runs[
                production_runs[run_id_col].astype("string").fillna("").str.strip() == run_id
            ]
            if not matching_runs.empty:
                row = matching_runs.iloc[0]
                if od_col:
                    event_specs["od"] = pd.to_numeric(pd.Series([row[od_col]]), errors="coerce").iloc[0]
                if thickness_col:
                    event_specs["thickness"] = pd.to_numeric(pd.Series([row[thickness_col]]), errors="coerce").iloc[0]
                if line_speed_col:
                    event_specs["line_speed"] = pd.to_numeric(pd.Series([row[line_speed_col]]), errors="coerce").iloc[0]

        od_key = float(event_specs["od"]) if pd.notna(event_specs.get("od")) else -1.0
        thickness_key = float(event_specs["thickness"]) if pd.notna(event_specs.get("thickness")) else -1.0
        line_speed_key = float(event_specs["line_speed"]) if pd.notna(event_specs.get("line_speed")) else -1.0
        healthy_key = (
            machine,
            round(od_key, 3),
            round(thickness_key, 3),
            round(line_speed_key, 3),
        )
        if healthy_key not in healthy_runs_cache:
            healthy_runs = production_runs
            if run_machine_col and machine:
                healthy_runs = healthy_runs[
                    machine_id_match(healthy_runs[run_machine_col], machine)
                ]
            if od_col and pd.notna(event_specs.get("od")):
                spec_od = pd.to_numeric(healthy_runs[od_col], errors="coerce")
                healthy_runs = healthy_runs[np.isclose(spec_od, event_specs["od"], rtol=0.01, atol=0.01)]
            if thickness_col and pd.notna(event_specs.get("thickness")):
                spec_thickness = pd.to_numeric(healthy_runs[thickness_col], errors="coerce")
                healthy_runs = healthy_runs[np.isclose(spec_thickness, event_specs["thickness"], rtol=0.02, atol=0.02)]
            if line_speed_col and pd.notna(event_specs.get("line_speed")):
                spec_line_speed = pd.to_numeric(healthy_runs[line_speed_col], errors="coerce")
                healthy_runs = healthy_runs[np.isclose(
                    spec_line_speed,
                    event_specs["line_speed"],
                    rtol=0.05,
                    atol=0.5,
                )]
            healthy_runs_cache[healthy_key] = healthy_runs
        healthy_runs = healthy_runs_cache[healthy_key]

        for window_minutes in windows:
            window_end = stoppage_start
            window_start = stoppage_start - pd.Timedelta(minutes=window_minutes)

            for tag in all_tags:
                tag_frame = proc_by_tag.get(tag)
                if tag_frame is None:
                    continue
                proc_window = tag_frame.loc[
                    tag_frame[proc_time_col].ge(window_start)
                    & tag_frame[proc_time_col].lt(window_end)
                ].copy()

                if proc_machine_col and machine:
                    machine_number = pd.to_numeric(pd.Series([machine]), errors="coerce").iloc[0]
                    if pd.notna(machine_number):
                        proc_window = proc_window[
                            machine_id_match(proc_window[proc_machine_col], machine)
                        ]
                if proc_run_id_col and run_id:
                    linked_window = proc_window[
                        proc_window[proc_run_id_col].astype("string").fillna("").str.strip() == run_id
                    ]
                    if not linked_window.empty:
                        proc_window = linked_window
                if proc_window.empty:
                    continue

                healthy_event_runs = healthy_runs
                if run_start_col and run_end_col:
                    overlap_mask = (
                        healthy_event_runs[run_start_col].le(stoppage_end)
                        & healthy_event_runs[run_end_col].ge(stoppage_start)
                    )
                    healthy_event_runs = healthy_event_runs.loc[
                        healthy_event_runs[run_start_col].notna() & ~overlap_mask
                    ]

                normal_endpoints = []
                if run_start_col:
                    normal_endpoints = pd.to_datetime(
                        healthy_event_runs[run_start_col], errors="coerce"
                    ).dropna().tolist()
                    if len(normal_endpoints) > 100:
                        normal_endpoints = [
                            normal_endpoints[index]
                            for index in np.linspace(
                                0, len(normal_endpoints) - 1, 100, dtype=int
                            )
                        ]

                normal_cache_key = (
                    machine,
                    round(od_key, 3),
                    round(thickness_key, 3),
                    round(line_speed_key, 3),
                    window_minutes,
                    tag,
                )
                if normal_cache_key not in normal_baseline_cache:
                    normal_values: list[pd.Series] = []
                    def collect_normal_values(normal_end: pd.Timestamp) -> pd.Series:
                        normal_start = normal_end - pd.Timedelta(minutes=window_minutes)
                        normal_proc = tag_frame.loc[
                            tag_frame[proc_time_col].ge(normal_start)
                            & tag_frame[proc_time_col].lt(normal_end)
                        ].copy()
                        if proc_machine_col and machine:
                            machine_number = pd.to_numeric(pd.Series([machine]), errors="coerce").iloc[0]
                            if pd.notna(machine_number):
                                normal_proc = normal_proc[
                                    machine_id_match(normal_proc[proc_machine_col], machine)
                                ]
                        if not normal_proc.empty:
                            return normal_proc[proc_value_col]
                        return pd.Series(dtype=float)

                    workers = max(1, max_workers)
                    if len(normal_endpoints) > 1 and workers > 1:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            normal_values = list(executor.map(
                                collect_normal_values,
                                normal_endpoints,
                            ))
                    else:
                        normal_values = [
                            collect_normal_values(normal_end)
                            for normal_end in normal_endpoints
                        ]
                    normal_value_list = (
                        pd.concat(normal_values, ignore_index=True)
                        if normal_values
                        else pd.Series(dtype=float)
                    )
                    normal_valid = pd.to_numeric(normal_value_list, errors="coerce").dropna()
                    normal_baseline_cache[normal_cache_key] = (
                        float(normal_valid.mean()) if not normal_valid.empty else np.nan,
                        float(normal_valid.std(ddof=1)) if len(normal_valid) > 1 else 0.0,
                        0.0,
                        normal_valid,
                    )

                normal_baseline = normal_baseline_cache[normal_cache_key]
                normal_std = normal_baseline[1]
                normal_valid = normal_baseline[3]
                if len(normal_valid) >= 8:
                    normal_low = float(normal_valid.quantile(0.05))
                    normal_high = float(normal_valid.quantile(0.95))
                    normal_std = float(normal_valid.std(ddof=1)) if len(normal_valid) > 1 else 0.0
                elif not normal_valid.empty:
                    normal_low = float(normal_valid.min())
                    normal_high = float(normal_valid.max())
                    normal_std = float(normal_valid.std(ddof=1)) if len(normal_valid) > 1 else 0.0
                else:
                    global_valid = pd.to_numeric(tag_frame[proc_value_col], errors="coerce").dropna()
                    normal_low = float(global_valid.quantile(0.05)) if len(global_valid) >= 8 else (float(global_valid.min()) if not global_valid.empty else float("-inf"))
                    normal_high = float(global_valid.quantile(0.95)) if len(global_valid) >= 8 else (float(global_valid.max()) if not global_valid.empty else float("inf"))
                    normal_std = float(global_valid.std(ddof=1)) if len(global_valid) > 1 else 0.0

                features = calculate_prewindow_features(
                    proc_window[proc_time_col],
                    proc_window[proc_value_col],
                    window_end,
                    normal_low,
                    normal_high,
                    normal_std,
                )
                base_od = event_specs.get("od", np.nan)
                base_thickness = event_specs.get("thickness", np.nan)
                base_line_speed = event_specs.get("line_speed", np.nan)
                normal_outside_mask = (
                    (normal_valid.to_numpy(dtype=float) < normal_low)
                    | (normal_valid.to_numpy(dtype=float) > normal_high)
                ) if not normal_valid.empty else np.array([], dtype=bool)
                feature_rows.append({
                    "event_id": event_index,
                    "is_stoppage": True,
                    "validation_segment": "validation" if event_index >= split_index else "discovery",
                    "downtime_start": stoppage_start,
                    "downtime_end": stoppage_end,
                    "machine": machine,
                    "run_id": run_id,
                    "downtime_reason": reason,
                    "event_class": event_class,
                    "od": base_od,
                    "thickness": base_thickness,
                    "line_speed": base_line_speed,
                    "window_minutes": window_minutes,
                    "lead_time_minutes": window_minutes,
                    "process_tag": tag,
                    **features,
                    "normal_mean": float(normal_valid.mean()) if not normal_valid.empty else np.nan,
                    "normal_std": float(normal_std),
                    "normal_outside_pct": float(normal_outside_mask.mean() * 100.0) if normal_outside_mask.size else np.nan,
                    "healthy_run_count": len(normal_endpoints),
                })

    features = pd.DataFrame(feature_rows)
    if not features.empty:
        features = features.sort_values(["downtime_start", "downtime_reason", "window_minutes", "process_tag"])
        features.to_csv(feature_path, index=False)

        summary_rows: list[dict[str, object]] = []
        for (reason, tag, window_minutes), group in features.groupby(
            ["downtime_reason", "process_tag", "window_minutes"], dropna=False
        ):
            valid = group.dropna(subset=["mean_value"])
            normal_means = valid["mean_value"]
            normal_change = valid["start_to_end_change"]
            occurrence_before = float(
                (valid["outside_normal_count"] > 0).mean() * 100.0
            ) if not valid.empty else 0.0
            std_value = float(valid["fluctuation"].mean()) if not valid.empty else 0.0

            if valid.empty:
                pattern = "Insufficient data"
            else:
                avg_change = float(normal_change.mean()) if normal_change.notna().any() else 0.0
                avg_fluctuation = float(valid["fluctuation"].mean()) if valid["fluctuation"].notna().any() else 0.0
                avg_outside = float(valid["outside_normal_pct"].mean()) if valid["outside_normal_pct"].notna().any() else 0.0
                if avg_change < -0.05:
                    pattern = "Repeated decrease"
                elif avg_change > 0.05:
                    pattern = "Repeated increase"
                elif avg_outside >= 20.0:
                    pattern = "Unusual fluctuation"
                elif avg_fluctuation > 0:
                    pattern = "Candidate variation"
                else:
                    pattern = "No clear pattern"

            discovery = valid.loc[valid["validation_segment"] == "discovery"]
            validation = valid.loc[valid["validation_segment"] == "validation"]
            discovery_pct = float((discovery["outside_normal_count"] > 0).mean() * 100.0) if not discovery.empty else np.nan
            validation_pct = float((validation["outside_normal_count"] > 0).mean() * 100.0) if not validation.empty else np.nan
            linked_event_count = int(valid["event_id"].nunique())
            normal_occurrence_pct = float(valid["normal_outside_pct"].mean()) if (
                "normal_outside_pct" in valid.columns
                and valid["normal_outside_pct"].notna().any()
            ) else np.nan
            false_warning_pct = normal_occurrence_pct
            separation_pct = (
                occurrence_before - normal_occurrence_pct
                if pd.notna(normal_occurrence_pct)
                else np.nan
            )

            if (
                linked_event_count >= 5
                and occurrence_before >= 70
                and pd.notna(validation_pct)
                and validation_pct >= 70
                and pd.notna(separation_pct)
                and separation_pct >= 20
            ):
                candidate_strength = "Strong candidate"
                evidence_level = "Investigate with engineering validation"
            elif (
                linked_event_count >= 3
                and occurrence_before >= 70
                and pd.notna(separation_pct)
                and separation_pct >= 15
            ):
                candidate_strength = "Moderate candidate"
                evidence_level = "Prioritize for review"
            elif linked_event_count >= 2:
                candidate_strength = "Weak candidate"
                evidence_level = "Collect more stoppage events"
            else:
                candidate_strength = "Insufficient evidence"
                evidence_level = "Do not operationalize"

            summary_rows.append({
                "downtime_reason": reason,
                "event_class": "unplanned_stoppage",
                "process_tag": tag,
                "window_minutes": int(window_minutes),
                "linked_events": int(valid["event_id"].nunique()),
                "causal_evidence": (
                    valid["causal_evidence"].mode().iloc[0]
                    if "causal_evidence" in valid.columns and not valid["causal_evidence"].mode().empty
                    else "insufficient_pre_stoppage_data"
                ),
                "mean_last_reading_lead_minutes": (
                    float(valid["last_reading_lead_minutes"].mean())
                    if "last_reading_lead_minutes" in valid.columns
                    and valid["last_reading_lead_minutes"].notna().any()
                    else np.nan
                ),
                "post_prediction_readings": (
                    int(valid["post_prediction_readings"].fillna(0).sum())
                    if "post_prediction_readings" in valid.columns
                    else 0
                ),
                "occurrence_before_stoppages_pct": occurrence_before,
                "occurrence_during_normal_operation_pct": normal_occurrence_pct,
                "false_warning_occurrence_pct": false_warning_pct,
                "separation_pct": separation_pct,
                "candidate_strength": candidate_strength,
                "evidence_level": evidence_level,
                "mean_before_stoppage": float(normal_means.mean()) if not normal_means.empty else np.nan,
                "mean_start_to_end_change": float(normal_change.mean()) if not normal_change.empty else np.nan,
                "mean_fluctuation": std_value,
                "discovery_occurrence_pct": discovery_pct,
                "validation_occurrence_pct": validation_pct,
                "pattern": pattern,
                "lead_time_minutes": int(window_minutes),
                "false_warning_occurrence": (
                    f"{false_warning_pct:.2f}%"
                    if pd.notna(false_warning_pct)
                    else "Unavailable"
                ),
            })

        summary = pd.DataFrame(summary_rows)
        if not summary.empty:
            summary["occurrence_before_stoppages_pct"] = pd.to_numeric(summary["occurrence_before_stoppages_pct"], errors="coerce").round(2)
            summary = summary.sort_values(
                ["occurrence_before_stoppages_pct", "downtime_reason", "process_tag"],
                ascending=[False, True, True],
            )
            summary.to_csv(summary_path, index=False)

            plot_data = summary.dropna(
                subset=["occurrence_before_stoppages_pct"]
            ).head(12).copy()
            plot_data["occurrence_during_normal_operation_pct"] = pd.to_numeric(
                plot_data["occurrence_during_normal_operation_pct"], errors="coerce"
            ).fillna(0)
            plot_data["label"] = (
                plot_data["process_tag"].astype(str).str.slice(0, 38)
                + " | "
                + plot_data["window_minutes"].astype(int).astype(str)
                + " min"
            )
            plot_data = plot_data.iloc[::-1]
            y = np.arange(len(plot_data))
            height = 0.38

            fig, axis = plt.subplots(figsize=(12, max(6, len(plot_data) * 0.48)))
            axis.barh(
                y - height / 2,
                plot_data["occurrence_before_stoppages_pct"],
                height,
                color="#2563eb",
                label="Before stoppages",
            )
            axis.barh(
                y + height / 2,
                plot_data["occurrence_during_normal_operation_pct"],
                height,
                color="#94a3b8",
                label="Normal operation",
            )
            axis.set_yticks(y)
            axis.set_yticklabels(plot_data["label"], fontsize=9)
            axis.set_xlim(0, 100)
            axis.set_xlabel("Occurrence outside normal range (%)")
            axis.set_title("Top Downtime Process Patterns")
            axis.legend(loc="lower right")
            axis.grid(axis="x", alpha=0.25)

            for index, row in enumerate(plot_data.itertuples(index=False)):
                before = pd.to_numeric(row.occurrence_before_stoppages_pct, errors="coerce")
                normal = pd.to_numeric(row.occurrence_during_normal_operation_pct, errors="coerce")
                if pd.notna(before):
                    axis.text(before + 1, index - height / 2, f"{before:.0f}%", va="center", fontsize=8)
                if pd.notna(normal):
                    axis.text(normal + 1, index + height / 2, f"{normal:.0f}%", va="center", fontsize=8)

            fig.text(
                0.5,
                0.01,
                "Candidate patterns only: pre-stoppage changes are not proof of cause.",
                ha="center",
                fontsize=8,
                color="#64748b",
            )
            fig.tight_layout(rect=(0, 0.05, 1, 1))
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
        else:
            summary = pd.DataFrame()
    else:
        summary = pd.DataFrame()

    if exclusion_reasons:
        pass
    exclusions.append({
        "reason": "No qualifying stoppage-level process pattern",
        "excluded_events": int(len(down_events)),
        "data_quality": "No valid process features generated",
    }) if features.empty else None
    exclusions_df = pd.DataFrame(exclusions)
    if exclusions_df.empty:
        exclusions_df = pd.DataFrame([{
            "reason": "No exclusions",
            "excluded_events": 0,
            "data_quality": "All qualifying downtime events evaluated",
        }])
    exclusions_df.to_csv(exclusion_path, index=False)

    return {"features": features, "summary": summary if not features.empty else empty_summary, "exclusions": exclusions_df}


def create_factor_tables(
    production_runs: pd.DataFrame,
    machine_summary: pd.DataFrame,
    process_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_csv = output_dir / "process_factor_gap_by_tag.csv"
    factor_plot = output_dir / "process_parameter_gap.png"
    profile_csv = output_dir / "process_parameter_summary.csv"
    for artifact in (factor_csv, factor_plot, profile_csv):
        if artifact.exists():
            artifact.unlink()

    machine_band = machine_summary[["machine", "efficiency_band"]].copy()

    run_machine_col = first_existing_column(production_runs, ["eq_code", "eq_name", "eq_id", "equipment_id"])
    run_id_col = first_existing_column(production_runs, ["product_ref_id", "transaction_id"])
    if not run_machine_col or not run_id_col:
        return

    run_band = (
        production_runs[[run_id_col, run_machine_col]]
        .rename(columns={run_machine_col: "machine", run_id_col: "run_id"})
        .merge(machine_band, on="machine", how="left")
    )
    run_band.to_csv(output_dir / "run_efficiency_band_mapping.csv", index=False)

    if process_df is None or process_df.empty:
        return

    proc = process_df.copy()
    tag_col = first_existing_column(proc, ["_name", "name"])
    value_col = first_existing_column(proc, ["_value", "value"])
    txn_col = first_existing_column(proc, ["transaction_id", "product_ref_id"])
    if not (tag_col and value_col and txn_col):
        return

    proc[value_col] = pd.to_numeric(proc[value_col], errors="coerce")
    tagged = (
        proc[[txn_col, tag_col, value_col]]
        .rename(columns={txn_col: "run_id", tag_col: "tag", value_col: "tag_value"})
        .merge(run_band[["run_id", "machine", "efficiency_band"]], on="run_id", how="left")
    )
    linked = tagged.dropna(subset=["run_id", "machine", "tag", "tag_value"])
    if not linked.empty:
        per_run_profile = (
            linked.groupby(["machine", "run_id", "tag"], dropna=False)
            .agg(
                run_mean=("tag_value", "mean"),
                reading_count=("tag_value", "count"),
            )
            .reset_index()
        )
        parameter_profile = (
            per_run_profile.groupby(["machine", "tag"], dropna=False)
            .agg(
                linked_run_count=("run_id", "nunique"),
                reading_count=("reading_count", "sum"),
                mean_run_value=("run_mean", "mean"),
                median_run_value=("run_mean", "median"),
                std_run_value=("run_mean", "std"),
                min_run_value=("run_mean", "min"),
                max_run_value=("run_mean", "max"),
            )
            .reset_index()
            .sort_values(["machine", "linked_run_count", "tag"], ascending=[True, False, True])
        )
        parameter_profile.to_csv(profile_csv, index=False)

    tagged = tagged[tagged["efficiency_band"].isin(["efficient", "inefficient"])]
    if tagged.empty:
        return
    per_run_tag = (
        tagged.groupby(["run_id", "tag", "efficiency_band"], dropna=False)["tag_value"]
        .mean()
        .reset_index()
    )
    grouped = per_run_tag.groupby(["tag", "efficiency_band"])["tag_value"].agg(["mean", "count"])
    comparison = grouped["mean"].unstack()
    run_counts = grouped["count"].unstack()
    if comparison.empty:
        return
    comparison = comparison.reindex(columns=["inefficient", "efficient"])
    run_counts = run_counts.reindex(columns=["inefficient", "efficient"])
    comparison = comparison.dropna(subset=["inefficient", "efficient"])
    run_counts = run_counts.reindex(index=comparison.index)
    if comparison.empty:
        return
    comparison["delta_inefficient_minus_efficient"] = (
        comparison["inefficient"] - comparison["efficient"]
    )
    comparison["runs_inefficient"] = run_counts["inefficient"]
    comparison["runs_efficient"] = run_counts["efficient"]
    within_tag_std = per_run_tag.groupby("tag")["tag_value"].std()
    comparison["standardized_delta"] = (
        comparison["delta_inefficient_minus_efficient"] / within_tag_std
    )
    comparison["abs_standardized_delta"] = comparison["standardized_delta"].abs()
    comparison = comparison.sort_values("abs_standardized_delta", ascending=False)
    comparison.to_csv(output_dir / "process_factor_gap_by_tag.csv")

    top = comparison.dropna(subset=["standardized_delta"]).head(12).reset_index()
    if top.empty:
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top, x="tag", y="standardized_delta", color="#e67e22")
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Standardized difference (within-tag standard deviations)")
    plt.title("Largest Process Parameter Differences by Run")
    plt.tight_layout()
    plt.savefig(output_dir / "process_parameter_gap.png", dpi=150)
    plt.close()

def create_html_report(
    production_runs: pd.DataFrame,
    machine_summary: pd.DataFrame,
    downtime_df: Optional[pd.DataFrame],
    output_dir: Path,
    production_duration_unit: Optional[str] = None,
    downtime_duration_unit: Optional[str] = None,
    ems_df: Optional[pd.DataFrame] = None,
    daily_summary: Optional[pd.DataFrame] = None,
    process_df: Optional[pd.DataFrame] = None,
) -> Path:
    if daily_summary is None:
        daily_summary = create_daily_summary(
            production_runs, downtime_df, downtime_duration_unit
        )

    def format_number(value: object, digits: int = 2) -> str:
        number = pd.to_numeric(value, errors="coerce")
        return "N/A" if pd.isna(number) else f"{number:,.{digits}f}"

    def table_html(
        frame: pd.DataFrame,
        columns: list[str],
        digits: int = 2,
        labels: Optional[dict[str, str]] = None,
    ) -> str:
        available = [column for column in columns if column in frame.columns]
        if frame.empty or not available:
            return "<p class='empty'>No applicable data available.</p>"
        display = frame[available].copy()
        formatters_by_source = {}
        for column in available:
            if not pd.api.types.is_numeric_dtype(display[column]):
                continue
            if column in {
                "linked_run_count",
                "reading_count",
                "production_runs",
                "runs_efficient",
                "runs_inefficient",
                "ems_row_matches",
                "events_with_ems_rows",
            }:
                formatters_by_source[column] = (
                    lambda value: format_number(value, 0)
                )
            else:
                formatters_by_source[column] = (
                    lambda value: format_number(value, digits)
                )
        display = display.rename(columns=labels or {})
        formatters = {
            (labels or {}).get(column, column): formatter
            for column, formatter in formatters_by_source.items()
        }
        return display.to_html(
            index=False,
            border=0,
            classes="data-table",
            escape=True,
            na_rep="N/A",
            formatters=formatters,
        )

    def image_html(filename: str, title: str) -> str:
        image_path = output_dir / filename
        if not image_path.is_file():
            return ""
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return (
            "<figure class='chart'>"
            f"<img src='data:image/png;base64,{encoded}' alt='{html.escape(title)}'>"
            f"<figcaption>{html.escape(title)}</figcaption>"
            "</figure>"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    scoreable = machine_summary.dropna(subset=["efficiency_score"])
    ranked = (
        scoreable.sort_values("efficiency_score", ascending=False)
        if not scoreable.empty
        else machine_summary.sort_values("machine")
    )
    ranking_available = (
        len(scoreable) >= 2 and scoreable["efficiency_score"].nunique() > 1
    )
    best = ranked.iloc[0] if ranking_available else None
    worst = ranked.iloc[-1] if ranking_available else None
    energy_ranked = machine_summary.dropna(subset=["energy_per_unit"]).sort_values(
        "energy_per_unit", ascending=False
    )
    energy_comparable = (
        len(energy_ranked) >= 2
        and energy_ranked["energy_per_unit"].nunique() > 1
    )
    highest_energy = energy_ranked.iloc[0] if energy_comparable else None
    generated_at = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

    finding_items = []
    if best is not None:
        finding_items.append(
            f"Highest composite score: <strong>{html.escape(str(best['machine']))}</strong> "
            f"({format_number(best['efficiency_score'])})."
        )
    if worst is not None:
        finding_items.append(
            f"Lowest composite score: <strong>{html.escape(str(worst['machine']))}</strong> "
            f"({format_number(worst['efficiency_score'])})."
        )
    if highest_energy is not None:
        finding_items.append(
            f"Highest observed energy intensity: <strong>{html.escape(str(highest_energy['machine']))}</strong> "
            f"({format_number(highest_energy['energy_per_unit'])} kWh/unit). Review comparable products and run conditions."
        )
    if len(machine_summary) < 2:
        finding_items.append(
            "Machine ranking is unavailable: only one machine was found. This report describes that machine; "
            "it cannot determine which machine is more efficient."
        )
    elif len(scoreable) < 2:
        finding_items.append(
            "Composite ranking is unavailable: fewer than two machines have complete, valid score inputs."
        )
    elif not ranking_available:
        finding_items.append(
            "Composite ranking is unavailable: complete machines have tied scores, so no winner or loser is distinguished."
        )
    elif not energy_comparable:
        finding_items.append(
            "Energy comparison is unavailable: fewer than two machines have distinct, valid energy-per-unit values."
        )
    findings_html = (
        "<ul>" + "".join(f"<li>{item}</li>" for item in finding_items) + "</ul>"
        if finding_items
        else "<p>No machine-level findings could be calculated.</p>"
    )

    efficiency_chart_html = image_html(
        "efficiency_score_by_machine.png",
        "Efficiency score by machine",
    )
    machine_correlation_chart_html = image_html(
        "machine_metric_correlation_heatmap.png",
        "Correlation between machine-level metrics",
    )
    output_energy_chart_html = image_html(
        "output_vs_energy.png",
        "Output versus energy intensity",
    )
    downtime_chart_html = image_html(
        "top_downtime_reasons.png",
        "Top downtime reasons",
    )
    process_gap_chart_html = image_html(
        "process_parameter_gap.png",
        "Process parameter gaps by efficiency band",
    )

    downtime_table = pd.DataFrame()
    if downtime_df is not None and not downtime_df.empty:
        loss_col = first_existing_column(downtime_df, ["main_loss_name", "sub_loss_name"])
        duration_col = first_existing_column(downtime_df, ["duration"])
        if loss_col and duration_col:
            downtime_table = downtime_df.copy()
            downtime_table["downtime_hours"] = duration_to_hours(
                downtime_table[duration_col], downtime_duration_unit
            )
            if "ems_match_rows" not in downtime_table:
                downtime_table["ems_match_rows"] = 0
            if "ems_energy_kwh" not in downtime_table:
                downtime_table["ems_energy_kwh"] = np.nan
            downtime_table = (
                downtime_table.groupby(loss_col, dropna=False)
                .agg(
                    downtime_hours=("downtime_hours", lambda values: values.sum(min_count=1)),
                    events_with_ems_rows=(
                        "ems_match_rows",
                        lambda values: (pd.to_numeric(values, errors="coerce").fillna(0) > 0).sum(),
                    ),
                    ems_row_matches=("ems_match_rows", "sum"),
                    ems_energy_kwh=("ems_energy_kwh", lambda values: values.sum(min_count=1)),
                )
                .sort_values("downtime_hours", ascending=False)
                .head(10)
                .reset_index()
                .rename(columns={loss_col: "loss_reason"})
            )

    business_summary_metrics = []

    def add_business_metric(label: str, value: object, digits: int = 2, suffix: str = "") -> None:
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number) or not np.isfinite(number):
            return
        business_summary_metrics.append(
            f"<div class='card'>{html.escape(label)}"
            f"<strong>{format_number(number, digits)}{html.escape(suffix)}</strong></div>"
        )

    def total_metric(column: str) -> object:
        if column not in machine_summary.columns:
            return np.nan
        values = pd.to_numeric(machine_summary[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return values.sum(min_count=1)

    if len(production_runs):
        add_business_metric("Production runs", len(production_runs), digits=0)
    if "production_runs" in daily_summary.columns:
        add_business_metric(
            "Days with production",
            daily_summary["production_runs"].gt(0).sum(),
            digits=0,
        )

    produced_qty = total_metric("produced_qty")
    scrap_qty = total_metric("scrap_qty")
    add_business_metric("Produced quantity", produced_qty)
    add_business_metric("Scrap quantity", scrap_qty)
    if pd.notna(produced_qty) and pd.notna(scrap_qty):
        total_output = produced_qty + scrap_qty
        if total_output > 0:
            add_business_metric("Scrap rate", scrap_qty / total_output * 100, suffix="%")

    if "speed_attainment_pct" in production_runs.columns:
        speed_values = pd.to_numeric(
            production_runs["speed_attainment_pct"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        if speed_values.notna().any():
            add_business_metric("Average speed vs target", speed_values.mean(), suffix="%")

    if {"energy_kwh", "pq_qty_clean"}.issubset(production_runs.columns):
        run_energy = pd.to_numeric(production_runs["energy_kwh"], errors="coerce")
        run_output = pd.to_numeric(production_runs["pq_qty_clean"], errors="coerce")
        matched_runs = run_energy.notna() & run_output.notna() & (run_output > 0)
        if matched_runs.any():
            matched_output = run_output.loc[matched_runs].sum()
            if matched_output > 0:
                add_business_metric(
                    "Matched energy per unit",
                    run_energy.loc[matched_runs].sum() / matched_output,
                    suffix=" kWh/unit",
                )

    add_business_metric("Recorded downtime", total_metric("downtime_hours"), suffix=" hours")
    if not downtime_table.empty:
        top_downtime = downtime_table.iloc[0]
        reason = top_downtime.get("loss_reason")
        hours = pd.to_numeric(top_downtime.get("downtime_hours"), errors="coerce")
        if pd.notna(reason) and str(reason).strip() and pd.notna(hours) and np.isfinite(hours):
            business_summary_metrics.append(
                f"<div class='card'>Largest downtime reason"
                f"<strong>{html.escape(str(reason))} ({format_number(hours)} hours)</strong></div>"
            )
    business_summary_html = (
        "<div class='cards'>" + "".join(business_summary_metrics) + "</div>"
        if business_summary_metrics
        else "<p class='empty'>No business metrics are available from the supplied data.</p>"
    )
    daily_report = daily_summary.copy()
    if not daily_report.empty and "date" in daily_report:
        daily_report["date"] = pd.to_datetime(
            daily_report["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        daily_columns = ["date"]
        daily_labels = {
            "date": "Date",
            "production_runs": "Production runs",
            "run_hours": "Run time (hours)",
            "downtime_hours": "Downtime (hours)",
            "energy_kwh": "Matched production energy (kWh)",
        }
        for column in (
            "production_runs",
            "run_hours",
            "downtime_hours",
            "energy_kwh",
        ):
            if column in daily_report and (
                column == "production_runs" or daily_report[column].notna().any()
            ):
                daily_columns.append(column)
        daily_table_html = table_html(
            daily_report, daily_columns, labels=daily_labels
        )
    else:
        daily_table_html = (
            "<p class='empty'>No production or downtime start dates are available.</p>"
        )
    daily_chart_html = image_html(
        "daily_downtime_vs_energy.png",
        "Daily downtime vs matched production energy",
    )
    if not daily_chart_html:
        daily_chart_html = (
            "<p class='empty'>The daily chart requires both downtime durations and "
            "matched production energy data.</p>"
        )
    daily_signal_html = (
        "<p class='empty'>Daily signals require both valid energy and downtime values.</p>"
    )
    process_tag_html = ""
    if (
        not daily_summary.empty
        and {"energy_kwh", "downtime_hours"}.issubset(daily_summary.columns)
    ):
        daily_values = daily_summary[["energy_kwh", "downtime_hours"]].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        comparable_days = daily_values.loc[
            daily_values.notna().all(axis=1) & (daily_values >= 0).all(axis=1)
        ]
        if not comparable_days.empty:
            energy_median = comparable_days["energy_kwh"].median()
            downtime_median = comparable_days["downtime_hours"].median()
            high_energy = comparable_days["energy_kwh"] > energy_median
            high_downtime = comparable_days["downtime_hours"] > downtime_median
            signal_rows = [
                {
                    "pattern": "Higher energy, lower downtime",
                    "days": int((high_energy & ~high_downtime).sum()),
                    "interpretation": "Needs attention: review energy use",
                },
                {
                    "pattern": "Lower energy, higher downtime",
                    "days": int((~high_energy & high_downtime).sum()),
                    "interpretation": "Low energy may be caused by downtime; investigate",
                },
                {
                    "pattern": "Higher energy, higher downtime",
                    "days": int((high_energy & high_downtime).sum()),
                    "interpretation": "Investigate both measures",
                },
                {
                    "pattern": "Lower energy, lower downtime",
                    "days": int((~high_energy & ~high_downtime).sum()),
                    "interpretation": "Favorable operating signal",
                },
            ]
            signal_table_html = table_html(
                pd.DataFrame(signal_rows),
                ["pattern", "days", "interpretation"],
                digits=0,
                labels={
                    "pattern": "Daily pattern",
                    "days": "Days",
                    "interpretation": "Business interpretation",
                },
            )
            selected_dates = daily_summary.loc[
                comparable_days.index[high_energy & ~high_downtime], "date"
            ]
            process_tag_summary = summarize_process_tags_for_dates(
                production_runs,
                process_df,
                selected_dates,
            )
            if process_tag_summary.empty:
                process_tag_html = (
                    "<p class='empty'>No process tags are linked to production runs "
                    "starting on the higher-energy, lower-downtime dates.</p>"
                )
            else:
                process_tag_html = (
                    "<p class='muted'>Mean process-tag reading for production runs "
                    "starting on the higher-energy, lower-downtime dates. This ranks "
                    "process tags within compatible categories, not meter kWh. Values use "
                    "each tag's source units and should not be compared across categories.</p>"
                    "<div class='table-wrap'>"
                    + table_html(
                        process_tag_summary,
                        [
                            "category",
                            "tag",
                            "linked_run_count",
                            "mean_run_value",
                            "max_run_value",
                        ],
                        digits=3,
                        labels={
                            "category": "Category",
                            "tag": "Process tag",
                            "linked_run_count": "Linked runs",
                            "mean_run_value": "Mean of run averages",
                            "max_run_value": "Highest run average",
                        },
                    )
                    + "</div>"
                )
            daily_signal_html = (
                f"<p>Out of {len(daily_summary)} dates in the table, "
                f"{len(comparable_days)} had both metrics available. Higher means above "
                f"the comparable-day median ({format_number(energy_median / 1000)} MWh "
                f"energy, {format_number(downtime_median)} downtime hours); lower means "
                "at or below the median. These are relative signals, not fixed targets; "
                "energy is not adjusted for production volume.</p>"
                f"<div class='table-wrap'>{signal_table_html}</div>"
                "<h4>Process tags for higher-energy, lower-downtime dates</h4>"
                f"{process_tag_html}"
            )

    factor_path = output_dir / "process_factor_gap_by_tag.csv"
    factor_table = pd.read_csv(factor_path).head(12) if factor_path.is_file() else pd.DataFrame()
    profile_path = output_dir / "process_parameter_summary.csv"
    parameter_profile = (
        pd.read_csv(profile_path)
        if profile_path.is_file()
        else pd.DataFrame()
    )
    if not parameter_profile.empty:
        parameter_profile = parameter_profile.sort_values(
            ["linked_run_count", "std_run_value"],
            ascending=[False, False],
            na_position="last",
        ).head(40)
    pattern_summary_path = output_dir / "downtime_pattern_summary.csv"
    pattern_summary = (
        pd.read_csv(pattern_summary_path).head(40)
        if pattern_summary_path.is_file()
        else pd.DataFrame()
    )
    pattern_exclusion_path = output_dir / "downtime_pattern_exclusions.csv"
    pattern_exclusions = (
        pd.read_csv(pattern_exclusion_path).head(20)
        if pattern_exclusion_path.is_file()
        else pd.DataFrame()
    )
    report_machines = machine_summary.copy()
    report_machines["scrap_rate_pct"] = report_machines["scrap_rate"] * 100
    report_machines["availability_pct"] = report_machines["availability"] * 100
    report_machines["downtime_rate_pct"] = report_machines["downtime_rate"] * 100
    glance_days = (
        int(daily_summary["production_runs"].gt(0).sum())
        if "production_runs" in daily_summary.columns
        else 0
    )
    glance_output = pd.to_numeric(
        machine_summary["produced_qty"], errors="coerce"
    ).sum(min_count=1) if "produced_qty" in machine_summary.columns else np.nan
    glance_scrap = pd.to_numeric(
        machine_summary["scrap_qty"], errors="coerce"
    ).sum(min_count=1) if "scrap_qty" in machine_summary.columns else np.nan
    glance_total = (
        glance_output + glance_scrap
        if pd.notna(glance_output) and pd.notna(glance_scrap)
        else np.nan
    )
    glance_scrap_rate = (
        glance_scrap / glance_total * 100
        if pd.notna(glance_total) and glance_total > 0
        else np.nan
    )
    glance_downtime = pd.to_numeric(
        machine_summary["downtime_hours"], errors="coerce"
    ).sum(min_count=1) if "downtime_hours" in machine_summary.columns else np.nan
    glance_availability = pd.to_numeric(
        machine_summary["availability"], errors="coerce"
    ).mean() if "availability" in machine_summary.columns else np.nan
    glance_output_per_hour = pd.to_numeric(
        machine_summary["output_per_hour"], errors="coerce"
    ).mean() if "output_per_hour" in machine_summary.columns else np.nan
    glance_speed = pd.to_numeric(
        production_runs["speed_attainment_pct"], errors="coerce"
    ).mean() if "speed_attainment_pct" in production_runs.columns else np.nan
    glance_energy = pd.to_numeric(
        machine_summary["energy_kwh"], errors="coerce"
    ).sum(min_count=1) if "energy_kwh" in machine_summary.columns else np.nan
    glance_energy_per_unit = (
        glance_energy / glance_output
        if pd.notna(glance_energy) and pd.notna(glance_output) and glance_output > 0
        else np.nan
    )
    glance_energy_coverage = pd.to_numeric(
        machine_summary["energy_coverage_pct"], errors="coerce"
    ).mean() if "energy_coverage_pct" in machine_summary.columns else np.nan
    machine_columns = [
        "machine",
        "run_count",
        "efficiency_score",
        "efficiency_band",
        "produced_qty",
        "scrap_qty",
        "scrap_rate_pct",
        "output_per_hour",
        "availability_pct",
        "target_speed",
        "actual_speed",
        "speed_attainment_pct",
        "energy_kwh",
        "energy_per_unit",
        "energy_coverage_pct",
        "ems_runs_with_readings",
        "matched_ems_rows",
        "runs_with_power",
        "downtime_hours",
        "downtime_rate_pct",
        "avg_kw",
    ]
    high_energy_columns = [
        "machine",
        "energy_per_unit",
        "energy_coverage_pct",
        "produced_qty",
        "output_per_hour",
        "efficiency_band",
    ]
    factor_columns = [
        "tag",
        "efficient",
        "inefficient",
        "delta_inefficient_minus_efficient",
        "standardized_delta",
        "runs_efficient",
        "runs_inefficient",
    ]
    profile_columns = [
        "machine",
        "tag",
        "linked_run_count",
        "reading_count",
        "mean_run_value",
        "median_run_value",
        "std_run_value",
        "min_run_value",
        "max_run_value",
    ]
    if not factor_table.empty and "tag" not in factor_table.columns:
        factor_table = factor_table.rename(columns={factor_table.columns[0]: "tag"})
    production_duration_numeric = any(
        col in production_runs
        and pd.to_numeric(production_runs[col], errors="coerce").notna().any()
        for col in ("run_time", "net_run_time")
    )
    downtime_duration_numeric = (
        downtime_df is not None
        and "duration" in downtime_df
        and pd.to_numeric(downtime_df["duration"], errors="coerce").notna().any()
    )

    def duration_note(is_numeric: bool, unit: Optional[str], option: str) -> str:
        if is_numeric and unit is None:
            return (
                "Numeric duration values were found but their unit is unknown; duration-based metrics are N/A. "
                f"After confirming the source unit, rerun with {option} seconds, minutes, hours, or days."
            )
        if is_numeric and unit:
            return f"Numeric duration values were interpreted as {html.escape(unit)}."
        return "Durations were parsed from their explicit time format."

    production_duration_note = duration_note(
        production_duration_numeric,
        production_duration_unit,
        "--production-duration-unit",
    )
    downtime_duration_note = duration_note(
        downtime_duration_numeric,
        downtime_duration_unit,
        "--downtime-duration-unit",
    )
    if energy_ranked.empty:
        energy_note = (
            "No valid run-level energy-per-unit values were available. Check EMS timestamp coverage, machine matching, "
            "and the energy-counter field mapping. Missing energy is not treated as zero."
        )
    else:
        energy_note = (
            f"Energy coverage is reported as the percentage of production runs with an energy value "
            f"(range {format_number(energy_ranked['energy_coverage_pct'].min())}% to "
            f"{format_number(energy_ranked['energy_coverage_pct'].max())}%)."
        )

    ems_overview = pd.DataFrame()
    ems_linkage = pd.DataFrame()
    ems_downtime_linkage = pd.DataFrame()
    ems_downtime_events = pd.DataFrame()
    ems_downtime_join_description = "No downtime EMS matching was attempted."
    ems_mapping_description = "No EMS dataset was supplied."
    ems_join_description = "No EMS-to-production run matching was attempted."
    ems_calculation_note = "No EMS energy calculation was available."
    if ems_df is not None and not ems_df.empty:
        ems_time_col = first_existing_column(
            ems_df, ["date_time", "timestamp", "_timestamp", "tm1_timestamp"]
        )
        ems_machine_col = first_existing_column(
            ems_df, ["equipment_id", "eq_id", "meter_id", "eq_code"]
        )
        machine_kwh_col = first_existing_column(ems_df, ["machine_kwh", "kwh"])
        master_kwh_col = first_existing_column(ems_df, ["master_kwh"])
        power_col = first_existing_column(ems_df, ["kw", "watts"])
        field_mapping = {
            "timestamp": ems_time_col,
            "meter/equipment ID": ems_machine_col,
            "machine KWH counter": machine_kwh_col,
            "master KWH": master_kwh_col,
            "power": power_col,
        }
        ems_mapping_description = "Detected EMS fields: " + "; ".join(
            f"{label}: {column or 'not detected'}" for label, column in field_mapping.items()
        )
        if machine_kwh_col and master_kwh_col:
            ems_calculation_note = (
                "For each EMS reading, consumed kWh = Machine KWH - Master KWH. "
                "Run kWh is the change between the latest valid values at or before run start and run end; "
                "cumulative readings are not added together."
            )
        elif machine_kwh_col:
            ems_calculation_note = (
                "Master KWH was not detected. Run energy uses consecutive Machine KWH counter differences "
                "as an approximate fallback, not the preferred Machine KWH - Master KWH calculation."
            )
        else:
            ems_calculation_note = (
                "No recognized Machine KWH counter was found. Matched EMS timestamps alone are not energy values."
            )

        ems_for_report = ems_df.copy()
        ems_for_report["_report_meter"] = (
            ems_for_report[ems_machine_col].astype("string").fillna("Unknown meter")
            if ems_machine_col
            else "All EMS readings"
        )
        overview_rows = []
        for meter_name, meter_rows in ems_for_report.groupby(
            "_report_meter", dropna=False, sort=False
        ):
            if ems_time_col:
                meter_rows = meter_rows.sort_values(ems_time_col)
                first_time = meter_rows[ems_time_col].min()
                last_time = meter_rows[ems_time_col].max()
            else:
                first_time = pd.NaT
                last_time = pd.NaT

            machine_values = (
                pd.to_numeric(meter_rows[machine_kwh_col], errors="coerce").dropna()
                if machine_kwh_col
                else pd.Series(dtype=float)
            )
            master_values = (
                pd.to_numeric(meter_rows[master_kwh_col], errors="coerce").dropna()
                if master_kwh_col
                else pd.Series(dtype=float)
            )
            shift_values = (
                pd.to_numeric(meter_rows["shift_energy_kwh"], errors="coerce").dropna()
                if "shift_energy_kwh" in meter_rows
                else pd.Series(dtype=float)
            )
            power_values = (
                pd.to_numeric(meter_rows[power_col], errors="coerce")
                if power_col
                else pd.Series(dtype=float)
            )
            if power_col == "watts":
                power_values = power_values / 1000
            interval_energy = (
                pd.to_numeric(meter_rows["energy_kwh"], errors="coerce")
                if "energy_kwh" in meter_rows
                else pd.Series(dtype=float)
            )
            overview_rows.append(
                {
                    "meter": str(meter_name),
                    "reading_count": len(meter_rows),
                    "first_timestamp": first_time,
                    "last_timestamp": last_time,
                    "machine_kwh_first": machine_values.iloc[0] if not machine_values.empty else np.nan,
                    "machine_kwh_last": machine_values.iloc[-1] if not machine_values.empty else np.nan,
                    "master_kwh_first": master_values.iloc[0] if not master_values.empty else np.nan,
                    "master_kwh_last": master_values.iloc[-1] if not master_values.empty else np.nan,
                    "shift_energy_first": shift_values.iloc[0] if not shift_values.empty else np.nan,
                    "shift_energy_last": shift_values.iloc[-1] if not shift_values.empty else np.nan,
                    "derived_interval_kwh": interval_energy.sum(min_count=1),
                    "mean_power_kw": power_values.mean(),
                }
            )
        ems_overview = pd.DataFrame(overview_rows)

        if not production_runs.empty:
            run_diagnostics = production_runs.copy()
            if "machine" not in run_diagnostics:
                run_machine_col = first_existing_column(
                    run_diagnostics, ["eq_code", "eq_name", "eq_id", "equipment_id"]
                )
                if run_machine_col:
                    run_diagnostics["machine"] = run_diagnostics[run_machine_col].astype(str)
            for col in ("energy_match_rows", "energy_kwh", "avg_kw"):
                if col not in run_diagnostics:
                    run_diagnostics[col] = np.nan if col != "energy_match_rows" else 0
            if "machine" in run_diagnostics:
                ems_linkage = (
                    run_diagnostics.groupby("machine", dropna=False)
                    .agg(
                        production_runs=("machine", "size"),
                        runs_with_ems_rows=(
                            "energy_match_rows",
                            lambda values: (
                                pd.to_numeric(values, errors="coerce").fillna(0) > 0
                            ).sum(),
                        ),
                        matched_ems_rows=("energy_match_rows", "sum"),
                        runs_with_energy=("energy_kwh", "count"),
                        runs_with_power=("avg_kw", "count"),
                    )
                    .reset_index()
                )

        if downtime_df is not None and not downtime_df.empty:
            downtime_start_col = first_existing_column(
                downtime_df,
                ["begin_time", "downtime_start_time", "start_time", "event_start_time"],
            )
            downtime_end_col = first_existing_column(
                downtime_df,
                ["end_time", "downtime_end_time", "stop_time", "event_end_time"],
            )
            if downtime_start_col and downtime_end_col and ems_time_col:
                downtime_diagnostics = downtime_df.copy()
                down_machine_col = first_existing_column(
                    downtime_diagnostics,
                    ["eq_code", "eq_name", "eq_id", "equipment_id"],
                )
                if down_machine_col:
                    downtime_diagnostics["machine"] = (
                        downtime_diagnostics[down_machine_col]
                        .astype("string")
                        .fillna("Unknown machine")
                    )
                else:
                    downtime_diagnostics["machine"] = "Unknown machine"
                for col in ("ems_match_rows", "ems_energy_kwh", "ems_avg_kw"):
                    if col not in downtime_diagnostics:
                        downtime_diagnostics[col] = 0 if col == "ems_match_rows" else np.nan
                ems_downtime_linkage = (
                    downtime_diagnostics.groupby("machine", dropna=False)
                    .agg(
                        downtime_events=("machine", "size"),
                        events_with_ems_rows=(
                            "ems_match_rows",
                            lambda values: (
                                pd.to_numeric(values, errors="coerce").fillna(0) > 0
                            ).sum(),
                        ),
                        matched_ems_rows=("ems_match_rows", "sum"),
                        events_with_energy=("ems_energy_kwh", "count"),
                        events_with_power=("ems_avg_kw", "count"),
                        matched_energy_kwh=(
                            "ems_energy_kwh",
                            lambda values: values.sum(min_count=1),
                        ),
                    )
                    .reset_index()
                )
                loss_col = first_existing_column(
                    downtime_diagnostics, ["main_loss_name", "sub_loss_name"]
                )
                downtime_diagnostics["event_start"] = downtime_diagnostics[downtime_start_col]
                downtime_diagnostics["event_end"] = downtime_diagnostics[downtime_end_col]
                downtime_diagnostics["loss_reason"] = (
                    downtime_diagnostics[loss_col].astype("string")
                    if loss_col
                    else "Unknown loss"
                )
                event_columns = [
                    "machine",
                    "event_start",
                    "event_end",
                    "loss_reason",
                    "ems_match_rows",
                    "ems_energy_kwh",
                    "ems_avg_kw",
                ]
                ems_downtime_events = (
                    downtime_diagnostics.loc[
                        pd.to_numeric(
                            downtime_diagnostics["ems_match_rows"], errors="coerce"
                        ).fillna(0)
                        .gt(0),
                        event_columns,
                    ]
                    .sort_values("ems_energy_kwh", ascending=False, na_position="last")
                    .head(25)
                    .copy()
                )
                ems_downtime_join_description = (
                    f"EMS rows are matched to downtime using inclusive timestamp windows "
                    f"({downtime_start_col} through {downtime_end_col}); meter IDs are not assumed "
                    "to identify production machines. Energy totals may include multiple EMS meters."
                )
            elif not ems_time_col:
                ems_downtime_join_description = (
                    "EMS timestamp column was not recognized; downtime time-window matching was unavailable."
                )
            else:
                ems_downtime_join_description = (
                    "Downtime start/end timestamps were not both recognized; downtime time-window matching was unavailable."
                )

        production_start_col = first_existing_column(
            production_runs, ["prod_start_time", "begin_time", "start_time"]
        )
        production_end_col = first_existing_column(
            production_runs, ["prod_end_time", "end_time", "stop_time"]
        )
        if not ems_time_col:
            ems_join_description = "EMS timestamp column was not recognized; run time-window matching was unavailable."
        elif not production_start_col or not production_end_col:
            ems_join_description = (
                "Production start/end timestamps were not both recognized; run time-window matching was unavailable."
            )
        elif machine_kwh_col and master_kwh_col:
            ems_join_description = (
                "EMS rows are matched to production run time windows by timestamp only; meter IDs are not "
                "assumed to map to production machine IDs. "
                "Each EMS reading's shift consumption is Machine KWH minus Master KWH. Run kWh is the change "
                "between the latest valid shift-consumption readings at or before run start and run end."
            )
        elif machine_kwh_col:
            ems_join_description = (
                "EMS rows are matched to production run time windows by timestamp only; meter IDs are not "
                "assumed to map to production machine IDs. "
                "No Master KWH baseline was recognized, so run energy uses consecutive Machine KWH counter "
                "differences as an approximate fallback."
            )
        else:
            ems_join_description = (
                "EMS rows are matched to production run time windows by timestamp only; meter IDs are not "
                "assumed to map to production machine IDs, "
                "but no recognized machine KWH counter was found to calculate run-level energy."
            )

    ems_overview_columns = [
        "meter",
        "reading_count",
        "first_timestamp",
        "last_timestamp",
        "machine_kwh_first",
        "machine_kwh_last",
        "master_kwh_first",
        "master_kwh_last",
        "shift_energy_first",
        "shift_energy_last",
        "derived_interval_kwh",
        "mean_power_kw",
    ]
    ems_linkage_columns = [
        "machine",
        "production_runs",
        "runs_with_ems_rows",
        "matched_ems_rows",
        "runs_with_energy",
        "runs_with_power",
    ]
    ems_downtime_linkage_columns = [
        "machine",
        "downtime_events",
        "events_with_ems_rows",
        "matched_ems_rows",
        "events_with_energy",
        "events_with_power",
        "matched_energy_kwh",
    ]
    ems_downtime_event_columns = [
        "machine",
        "event_start",
        "event_end",
        "loss_reason",
        "ems_match_rows",
        "ems_energy_kwh",
        "ems_avg_kw",
    ]
    ems_overview_labels = {
        "meter": "Meter / equipment",
        "reading_count": "EMS readings",
        "first_timestamp": "First timestamp",
        "last_timestamp": "Last timestamp",
        "machine_kwh_first": "First machine KWH",
        "machine_kwh_last": "Last machine KWH",
        "master_kwh_first": "First master KWH",
        "master_kwh_last": "Last master KWH",
        "shift_energy_first": "First Machine - Master (kWh)",
        "shift_energy_last": "Last Machine - Master (kWh)",
        "derived_interval_kwh": "Fallback counter-difference total (kWh)",
        "mean_power_kw": "Mean power (kW)",
    }
    ems_linkage_labels = {
        "machine": "Production machine",
        "production_runs": "Production runs",
        "runs_with_ems_rows": "Runs with matched EMS timestamps",
        "matched_ems_rows": "EMS row matches across runs",
        "runs_with_energy": "Runs with derived kWh",
        "runs_with_power": "Runs with recognized power",
    }
    ems_downtime_linkage_labels = {
        "machine": "Downtime machine",
        "downtime_events": "Downtime events",
        "events_with_ems_rows": "Events with matched EMS timestamps",
        "matched_ems_rows": "EMS row matches across events",
        "events_with_energy": "Events with derived kWh",
        "events_with_power": "Events with recognized power",
        "matched_energy_kwh": "Matched EMS energy (kWh)",
    }
    ems_downtime_event_labels = {
        "machine": "Downtime machine",
        "event_start": "Downtime start",
        "event_end": "Downtime end",
        "loss_reason": "Loss reason",
        "ems_match_rows": "Matched EMS rows",
        "ems_energy_kwh": "Matched EMS energy (kWh)",
        "ems_avg_kw": "Mean power during event (kW)",
    }
    ems_overview_html = table_html(
        ems_overview,
        ems_overview_columns,
        labels=ems_overview_labels,
    )
    ems_linkage_html = table_html(
        ems_linkage,
        ems_linkage_columns,
        labels=ems_linkage_labels,
    )
    ems_downtime_linkage_html = table_html(
        ems_downtime_linkage,
        ems_downtime_linkage_columns,
        labels=ems_downtime_linkage_labels,
    )
    ems_downtime_events_html = table_html(
        ems_downtime_events,
        ems_downtime_event_columns,
        labels=ems_downtime_event_labels,
    )
    bands_available = bool(
        machine_summary["efficiency_band"].isin(["efficient", "mid", "inefficient"]).any()
    )
    efficient_count = (
        str(int((machine_summary["efficiency_band"] == "efficient").sum()))
        if bands_available
        else "N/A"
    )
    inefficient_count = (
        str(int((machine_summary["efficiency_band"] == "inefficient").sum()))
        if bands_available
        else "N/A"
    )
    machine_labels = {
        "machine": "Machine",
        "run_count": "Production runs",
        "efficiency_score": "Relative score (0-1)",
        "efficiency_band": "Relative band",
        "produced_qty": "Produced quantity",
        "scrap_qty": "Scrap quantity",
        "scrap_rate_pct": "Scrap rate (%)",
        "output_per_hour": "Output per net run hour",
        "availability_pct": "Availability (%)",
        "target_speed": "Mean target speed (m/min)",
        "actual_speed": "Mean actual speed (m/min)",
        "speed_attainment_pct": "Mean speed attainment (%)",
        "energy_kwh": "Matched energy (kWh)",
        "energy_per_unit": "Energy per produced unit (kWh/unit)",
        "energy_coverage_pct": "Runs with matched energy (%)",
        "ems_runs_with_readings": "Runs with EMS readings",
        "matched_ems_rows": "EMS rows matched across runs",
        "runs_with_power": "Runs with power readings",
        "downtime_hours": "Matched downtime (hours)",
        "downtime_rate_pct": "Matched downtime / run time (%)",
        "avg_kw": "Mean power (kW)",
    }
    machine_table_html = table_html(
        ranked,
        machine_columns,
        labels=machine_labels,
    )
    energy_table_html = (
        table_html(
            energy_ranked,
            high_energy_columns,
            labels=machine_labels,
        )
        if not energy_ranked.empty
        else "<p class='empty'>No run-level energy-per-unit table is available; see the EMS diagnostics below.</p>"
    )
    downtime_table_html = table_html(
        downtime_table,
        [
            "loss_reason",
            "downtime_hours",
            "events_with_ems_rows",
            "ems_row_matches",
            "ems_energy_kwh",
        ],
        labels={
            "loss_reason": "Loss reason",
            "downtime_hours": "Downtime (hours)",
            "events_with_ems_rows": "Events with EMS readings",
            "ems_row_matches": "EMS row matches",
            "ems_energy_kwh": "Matched EMS energy (kWh)",
        },
    )
    factor_table_html = (
        table_html(
            factor_table,
            factor_columns,
            digits=3,
            labels={
                "tag": "Process tag",
                "efficient": "Mean: efficient band",
                "inefficient": "Mean: inefficient band",
                "delta_inefficient_minus_efficient": "Inefficient - efficient",
                "standardized_delta": "Standardized difference",
                "runs_efficient": "Runs in efficient band",
                "runs_inefficient": "Runs in inefficient band",
            },
        )
        if not factor_table.empty
        else (
            "<p class='empty'>No group comparison is available. It requires valid process-to-run links "
            "and distinguishable efficient and inefficient machine groups.</p>"
        )
    )
    if not parameter_profile.empty:
        if len(machine_summary) == 1:
            profile_description = (
                "PLC readings linked to this machine's production runs. The script averages each tag within each run, "
                "then summarizes those run averages so runs with more change-triggered readings do not dominate. "
                "Values are in the tag's source units. The HTML shows up to 40 tags with the most linked runs; "
                "the full profile is in process_parameter_summary.csv."
            )
        else:
            profile_description = (
                "Per-machine PLC parameter profile from production-linked readings. Each run contributes one average "
                "per tag. The HTML shows up to 40 tags with the most linked runs; the full profile is in "
                "process_parameter_summary.csv."
            )
        profile_table_html = table_html(
            parameter_profile,
            profile_columns,
            digits=3,
            labels={
                "machine": "Machine",
                "tag": "Process tag",
                "linked_run_count": "Linked runs",
                "reading_count": "PLC readings",
                "mean_run_value": "Mean of run averages",
                "median_run_value": "Median of run averages",
                "std_run_value": "Std. dev. across runs",
                "min_run_value": "Lowest run average",
                "max_run_value": "Highest run average",
            },
        )
    else:
        profile_description = (
            "No process readings were linked to production runs by transaction_id. Check the process and production "
            "run IDs; readings without a run link are not included in this run-level profile."
        )
        profile_table_html = "<p class='empty'>No production-linked process parameter profile is available.</p>"
    if not pattern_summary.empty:
        rank_order = {"Strong candidate": 0, "Moderate candidate": 1, "Weak candidate": 2, "Insufficient evidence": 3}
        business_view = pattern_summary.copy()
        business_view["_rank"] = business_view["candidate_strength"].map(rank_order).fillna(99)
        business_view["_before"] = pd.to_numeric(
            business_view["occurrence_before_stoppages_pct"], errors="coerce"
        ).fillna(0)
        business_view = business_view.sort_values(
            ["_rank", "_before", "linked_events"],
            ascending=[True, False, False],
        ).head(6)
        business_cards_html = ""
        for row in business_view.to_dict("records"):
            strength = str(row.get("candidate_strength", "Insufficient evidence"))
            reason = str(row.get("downtime_reason", "Unspecified reason"))
            tag = str(row.get("process_tag", "Unknown process tag"))
            before = row.get("occurrence_before_stoppages_pct")
            normal = row.get("occurrence_during_normal_operation_pct")
            lead = row.get("lead_time_minutes")
            status_class = (
                "status-good"
                if strength == "Strong candidate"
                else "status-watch"
                if strength in ("Moderate candidate", "Weak candidate")
                else ""
            )
            business_cards_html += (
                f"<div class='card {status_class}'>"
                f"{html.escape(reason)}"
                f"<strong>{html.escape(tag)}</strong>"
                f"<span>{html.escape(str(row.get('pattern', 'Pattern unavailable')))} · "
                f"{html.escape(str(before))}% before stoppage · "
                f"{html.escape(str(normal))}% during normal operation · "
                f"{html.escape(str(lead))}-minute window</span>"
                f"</div>"
            )
        if not business_cards_html:
            business_cards_html = "<p class='empty'>No business-level candidate is available.</p>"
        pattern_table_html = table_html(
            pattern_summary,
            [
                "downtime_reason",
                "event_class",
                "process_tag",
                "window_minutes",
                "linked_events",
                "occurrence_before_stoppages_pct",
                "occurrence_during_normal_operation_pct",
                "mean_before_stoppage",
                "mean_start_to_end_change",
                "mean_fluctuation",
                "discovery_occurrence_pct",
                "validation_occurrence_pct",
                "causal_evidence",
                "mean_last_reading_lead_minutes",
                "post_prediction_readings",
                "candidate_strength",
                "evidence_level",
                "separation_pct",
                "false_warning_occurrence_pct",
                "pattern",
                "lead_time_minutes",
            ],
            digits=3,
            labels={
                "downtime_reason": "Downtime reason",
                "event_class": "Event classification",
                "process_tag": "Process tag",
                "window_minutes": "Window (minutes)",
                "linked_events": "Linked events",
                "occurrence_before_stoppages_pct": "Before stoppages (%)",
                "occurrence_during_normal_operation_pct": "Normal operation (%)",
                "mean_before_stoppage": "Mean before stoppage",
                "mean_start_to_end_change": "Start-to-end change",
                "mean_fluctuation": "Mean fluctuation",
                "discovery_occurrence_pct": "Discovery occurrence (%)",
                "validation_occurrence_pct": "Later-period validation (%)",
                "causal_evidence": "Causal evidence status",
                "mean_last_reading_lead_minutes": "Mean last-reading lead (minutes)",
                "post_prediction_readings": "Post-prediction readings",
                "candidate_strength": "Business confidence",
                "evidence_level": "Recommended action",
                "separation_pct": "Stoppage-minus-normal gap (%)",
                "false_warning_occurrence_pct": "False-warning occurrence (%)",
                "pattern": "Pattern",
                "lead_time_minutes": "Lead time (minutes)",
            },
        )
    else:
        pattern_table_html = (
            "<p class='empty'>No stoppage/process pattern is available. "
            "Supply downtime start timestamps and process readings with timestamps and tag values.</p>"
        )
    pattern_chart_html = image_html(
        "downtime_process_patterns.png",
        "Top downtime process patterns",
    )
    if not pattern_exclusions.empty:
        pattern_exclusion_html = table_html(
            pattern_exclusions,
            ["reason", "excluded_events", "data_quality"],
            digits=0,
            labels={
                "reason": "Exclusion reason",
                "excluded_events": "Excluded events",
                "data_quality": "Data quality",
            },
        )
    else:
        pattern_exclusion_html = "<p class='empty'>No pattern exclusions were recorded.</p>"

    logo_path = Path(__file__).resolve().with_name("pi_systems_logo.png")
    if logo_path.is_file():
        logo_data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        company_logo_html = (
            f'<img class="company-logo" src="data:image/png;base64,{logo_data}" '
            'alt="Pi Systems">'
        )
        company_mark_html = (
            f'<img class="brand-logo" src="data:image/png;base64,{logo_data}" '
            'alt="Pi Systems">'
        )
    else:
        company_logo_html = '<span class="company-logo-text">PI SYSTEMS</span>'
        company_mark_html = '<div class="brand-mark">PI</div>'

    report_path = output_dir / "eda_report.html"
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tube Mill EDA Report</title>
  <style>
    :root {{ color-scheme: light; font-family: "Segoe UI", Arial, sans-serif; color: #172033; background: #eef3f8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #eef3f8; }}
    .dashboard {{ min-height: 100vh; display: grid; grid-template-columns: 248px minmax(0, 1fr); }}
    .sidebar {{ position: sticky; top: 0; height: 100vh; overflow-y: auto; background: linear-gradient(180deg, #0a1f33 0%, #102b46 100%); color: #fff; padding: 20px 14px; border-right: 1px solid rgba(255,255,255,.06); }}
    .brand {{ display: flex; align-items: center; gap: 8px; padding: 2px 6px 16px; border-bottom: 1px solid rgba(255,255,255,.12); margin-bottom: 14px; }}
    .brand-logo {{ width: 82px; height: 82px; object-fit: contain; border-radius: 8px; background: #08182a; }}
    .brand-mark {{ width: 82px; height: 82px; border-radius: 8px; display: grid; place-items: center; background: #f59e0b; color: #102b46; font-weight: 800; }}
    .brand strong {{ display: block; font-size: .88rem; letter-spacing: .01em; }}
    .brand span {{ display: block; margin-top: 3px; color: #b9c9d8; font-size: .72rem; }}
    .nav-label {{ color: #91a8bb; text-transform: uppercase; letter-spacing: .08em; font-size: .68rem; font-weight: 700; padding: 8px 10px 6px; }}
    .side-nav {{ display: grid; gap: 3px; }}
    .nav-button {{ width: 100%; border: 0; border-radius: 8px; background: transparent; color: #dce7f0; padding: 9px 10px; text-align: left; cursor: pointer; font: inherit; font-size: .86rem; transition: .15s ease; }}
    .nav-button:hover {{ background: rgba(255,255,255,.08); color: #fff; }}
    .nav-button.active {{ background: #f59e0b; color: #102b46; font-weight: 700; }}
    .content {{ min-width: 0; }}
    header {{ background: #fff; border-bottom: 1px solid #dde5ed; padding: 16px 28px; box-shadow: 0 2px 12px rgba(16,43,70,.03); }}
    header h1 {{ margin: 0 0 4px; font-size: 1.45rem; color: #102b46; }}
    header div {{ color: #667085; font-size: .84rem; }}
    .header-inner {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; width: 100%; }}
    .header-copy {{ min-width: 0; }}
    .company-logo {{ height: 54px; width: auto; object-fit: contain; flex: 0 0 auto; }}
    .company-logo-text {{ color: #102b46; font-size: .78rem; font-weight: 800; letter-spacing: .08em; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px 28px 40px; }}
    main section {{ display: none; background: #fff; border: 1px solid #dde5ed; border-radius: 12px; padding: 22px; box-shadow: 0 5px 18px rgba(16,43,70,.05); }}
    main section.active {{ display: block; animation: panel-in .18s ease-out; }}
    @keyframes panel-in {{ from {{ opacity: 0; transform: translateY(4px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    h1 {{ margin: 0 0 6px; }} h2 {{ margin: 0 0 16px; color: #102b46; }}
    .muted, .empty, figcaption {{ color: #667085; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .card {{ background: #f4f8fb; border: 1px solid #e5edf4; border-radius: 10px; padding: 16px; transition: transform .15s ease, box-shadow .15s ease; }}
    .card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 18px rgba(16,43,70,.08); }}
    .card strong {{ display: block; font-size: 1.45rem; margin-top: 6px; color: #102b46; }}
    .card.status-good {{ background: #edf8f1; border-color: #bce2c8; }}
    .card.status-good strong {{ color: #18794e; }}
    .card.status-watch {{ background: #fff8e6; border-color: #efdba5; }}
    .card.status-watch strong {{ color: #9a6700; }}
    .status-banner {{ display: flex; align-items: flex-start; gap: 10px; margin-top: 16px; padding: 12px 14px; border-radius: 9px; background: #eef4fa; border-left: 4px solid #4a7fa6; color: #40566b; }}
    .status-banner strong {{ color: #102b46; }}
    .subtabs {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 14px; padding-bottom: 12px; border-bottom: 1px solid #e5ebf2; }}
    .subtab-button {{ border: 1px solid #dbe4ec; background: #fff; color: #40566b; border-radius: 8px; padding: 8px 12px; cursor: pointer; font: inherit; font-size: .84rem; }}
    .subtab-button.active {{ background: #102b46; border-color: #102b46; color: #fff; }}
    .subpanel {{ display: none; }}
    .subpanel.active {{ display: block; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #e5e7eb; padding: 9px 10px; text-align: left; }}
    .data-table th {{ background: #f4f8fb; position: sticky; top: 0; z-index: 1; }}
    .table-wrap {{ overflow-x: auto; max-height: 70vh; border: 1px solid #edf1f5; border-radius: 8px; }}
    .table-wrap > .data-table {{ margin: 0; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    figure.chart {{ margin: 0; }} figure.chart img {{ width: 100%; height: auto; }}
    figcaption {{ margin-top: 6px; text-align: center; }}
    .note {{ border-left: 4px solid #d99a19; padding: 10px 14px; background: #fff8e6; }}
    .menu-toggle {{ display: none; }}
    @media (max-width: 820px) {{
      .dashboard {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: fixed; z-index: 20; width: 248px; transform: translateX(-100%); transition: transform .2s ease; box-shadow: 12px 0 30px rgba(16,43,70,.18); }}
      .sidebar.open {{ transform: translateX(0); }}
      .menu-toggle {{ display: inline-grid; place-items: center; width: 38px; height: 38px; border: 1px solid #dde5ed; border-radius: 8px; background: #fff; color: #102b46; cursor: pointer; margin-right: 10px; }}
      header {{ display: flex; align-items: center; }}
      header > div {{ flex: 1; }}
      header h1 {{ font-size: 1.18rem; }}
      .company-logo {{ height: 42px; }}
      .header-inner {{ gap: 8px; }}
      main {{ padding: 16px; }}
    }}
    @media print {{
      .dashboard {{ display: block; }} .sidebar {{ display: none; }} header {{ display: block; }}
      main section {{ display: block !important; break-inside: avoid; margin-bottom: 18px; }}
      .table-wrap {{ max-height: none; }}
    }}
    /* Polished report shell */
    :root {{
      --slate-50: #f8fafc; --slate-100: #f1f5f9; --slate-200: #e2e8f0;
      --slate-300: #cbd5e1; --slate-400: #94a3b8; --slate-500: #64748b;
      --slate-600: #475569; --slate-700: #334155; --slate-800: #1e293b;
      --slate-900: #0f172a; --blue-600: #2563eb; --amber-400: #fbbf24;
      --amber-500: #f59e0b; --amber-700: #b45309; --green-500: #22c55e;
      --green-700: #15803d; --shadow: 0 1px 3px rgb(15 23 42 / .07), 0 6px 16px rgb(15 23 42 / .05);
    }}
    :root {{ font-family: "Inter", "Segoe UI", Arial, sans-serif; }}
    body {{ background: #f1f5f9; }}
    .dashboard {{ grid-template-columns: 280px minmax(0, 1fr); }}
    .sidebar {{
      z-index: 30; gap: 14px; padding: 20px 14px 16px;
      background: linear-gradient(180deg, #0f172a 0%, #0b2440 100%);
      color: #cbd5e1;
    }}
    .brand {{ gap: 12px; padding: 0 6px 16px; border-bottom-color: rgba(255,255,255,.10); }}
    .brand-logo, .brand-mark {{ width: 56px; height: 56px; border-radius: 8px; }}
    .brand strong {{ color: #fff; font-size: .95rem; }}
    .brand span {{ color: #94a3b8; font-size: .75rem; }}
    .nav-label {{ color: #64748b; }}
    .side-nav {{ gap: 2px; }}
    .nav-button {{
      position: relative; display: grid; grid-template-columns: 28px minmax(0, 1fr);
      align-items: center; gap: 12px; min-height: 44px; padding: 8px 10px;
      border-radius: 8px; color: #cbd5e1; font-size: .875rem; line-height: 1.3;
    }}
    .nav-button:hover {{ background: rgba(255,255,255,.06); color: #fff; }}
    .nav-button.active {{ background: rgba(245,158,11,.14); color: #fff; font-weight: 600; }}
    .nav-button .step-dot {{
      display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%;
      border: 1.5px solid #475569; background: #0f172a; color: #cbd5e1;
      font-size: .75rem; font-weight: 700;
    }}
    .nav-button.active .step-dot {{ background: #f59e0b; border-color: #f59e0b; color: #0f172a; box-shadow: 0 0 0 4px rgba(245,158,11,.22); }}
    .nav-button.is-done .step-dot {{ background: #16a34a; border-color: #16a34a; color: #fff; }}
    .nav-button.is-done .step-dot::before {{ content: "✓"; }}
    .nav-button.is-done .step-dot span {{ display: none; }}
    .review {{ margin: 0 0 4px; padding: 12px; border: 1px solid rgba(255,255,255,.07); border-radius: 8px; background: rgba(255,255,255,.05); }}
    .review-row {{ display: flex; justify-content: space-between; color: #cbd5e1; font-size: .75rem; font-weight: 600; }}
    .review-row b {{ color: #fff; }}
    .review-track {{ margin-top: 8px; height: 6px; overflow: hidden; border-radius: 99px; background: rgba(255,255,255,.10); }}
    .review-track span {{ display: block; width: 9.09%; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #fbbf24, #22c55e); transition: width .3s ease; }}
    .sidebar-foot {{ margin-top: auto; padding: 12px 8px 0; border-top: 1px solid rgba(255,255,255,.08); color: #94a3b8; font-size: .75rem; }}
    kbd {{ display: inline-grid; min-width: 22px; height: 22px; place-items: center; margin-right: 4px; border: 1px solid rgba(255,255,255,.18); border-radius: 5px; color: #e2e8f0; }}
    .content {{ background: #f1f5f9; }}
    header {{
      position: sticky; top: 0; z-index: 20; padding: 14px 32px;
      background: rgba(255,255,255,.92); backdrop-filter: blur(10px); border-bottom: 1px solid #e2e8f0;
      box-shadow: none;
    }}
    .header-inner {{ max-width: none; }}
    header h1 {{ margin: 2px 0 6px; font-size: 1.375rem; line-height: 1.25; color: #0f172a; }}
    .header-copy > div {{ display: flex; flex-wrap: wrap; gap: 6px; color: #64748b; }}
    .header-copy > div {{ font-size: .75rem; }}
    .eyebrow {{ margin: 0; color: #1d4ed8; font-size: .75rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; }}
    .company-logo {{ height: 48px; }}
    .print-btn {{ min-height: 38px; padding: 8px 14px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; color: #475569; font-weight: 600; cursor: pointer; box-shadow: 0 1px 2px rgb(15 23 42 / .06); }}
    .header-actions {{ display: flex; align-items: center; gap: 14px; }}
    main {{ max-width: 1240px; padding: 28px 32px 48px; }}
    main section {{
      border: 1px solid #e2e8f0; border-radius: 12px; padding: 28px;
      box-shadow: var(--shadow);
    }}
    main section h2 {{ margin: 4px 0 0; font-size: 1.375rem; color: #0f172a; }}
    main section > h2 {{ margin: 0 0 16px; }}
    main section .section-head {{ display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 18px; padding-bottom: 18px; border-bottom: 1px solid #f1f5f9; }}
    main section .section-head h2 {{ margin: 4px 0 0; }}
    main p {{ color: #64748b; line-height: 1.65; }}
    .cards {{ grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 14px; }}
    .card {{
      position: relative; padding: 16px 18px; border: 1px solid #e2e8f0; border-radius: 12px;
      background: #fff; color: #64748b; box-shadow: 0 1px 2px rgb(15 23 42 / .06);
      overflow: hidden;
    }}
    .card::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 3px; background: #cbd5e1; }}
    .card strong {{ margin-top: 8px; color: #0f172a; font-size: clamp(1.2rem, 1.1rem + .5vw, 1.5rem); line-height: 1.2; }}
    .card.status-good::before {{ background: #22c55e; }}
    .card.status-good strong {{ color: #15803d; }}
    .card.status-watch::before {{ background: #f59e0b; }}
    .card.status-watch strong {{ color: #b45309; }}
    .status-banner {{ margin-top: 18px; border: 1px solid #bfdbfe; border-left: 4px solid #3b82f6; border-radius: 8px; background: #eff6ff; }}
    .subtabs {{ display: inline-flex; gap: 4px; padding: 4px; border: 1px solid #e2e8f0; border-radius: 10px; background: #f1f5f9; }}
    .subtab-button {{ border: 0; border-radius: 7px; background: transparent; color: #475569; }}
    .subtab-button.active {{ background: #fff; color: #0f172a; box-shadow: 0 1px 2px rgb(15 23 42 / .06), 0 0 0 1px #e2e8f0; }}
    .table-wrap {{ border: 1px solid #e2e8f0; border-radius: 10px; }}
    .data-table {{ border-collapse: separate; border-spacing: 0; }}
    .data-table th {{ top: 0; background: #f8fafc; color: #475569; border-bottom: 1px solid #e2e8f0; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #f1f5f9; }}
    .charts {{ grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); }}
    figure.chart {{ padding: 14px; border: 1px solid #e2e8f0; border-radius: 12px; background: #fff; box-shadow: 0 1px 2px rgb(15 23 42 / .06); }}
    .section-foot {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 28px; padding-top: 20px; border-top: 1px solid #f1f5f9; }}
    .section-foot .btn {{ min-height: 38px; padding: 8px 14px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; color: #334155; font-weight: 600; cursor: pointer; }}
    .section-foot .btn-primary {{ border-color: #2563eb; background: #2563eb; color: #fff; }}
    @media (max-width: 1023.98px) {{
      .dashboard {{ grid-template-columns: 84px minmax(0,1fr); }}
      .sidebar {{ width: 84px; padding: 16px 12px; }}
      .brand {{ justify-content: center; }}
      .brand > div, .review, .nav-label, .sidebar-foot {{ display: none; }}
      .nav-button {{ grid-template-columns: 1fr; justify-items: center; padding: 8px 0; }}
      .nav-button .step-label {{ display: none; }}
      header {{ padding: 12px 24px; }}
      main {{ padding: 24px; }}
    }}
    @media (max-width: 767.98px) {{
      .dashboard {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: fixed; inset: 0 auto 0 0; width: 280px; transform: translateX(-100%); transition: transform .22s ease; }}
      .sidebar.open {{ transform: translateX(0); }}
      .sidebar .brand > div, .sidebar .review, .sidebar .nav-label, .sidebar .sidebar-foot {{ display: block; }}
      .nav-button {{ grid-template-columns: 28px minmax(0,1fr); justify-items: initial; }}
      .nav-button .step-label {{ display: block; }}
      header {{ padding: 12px 16px; }}
      .company-logo, .print-btn {{ display: none; }}
      main {{ padding: 16px; }}
      main section {{ padding: 18px; }}
      .cards {{ grid-template-columns: 1fr; }}
      .charts {{ grid-template-columns: 1fr; }}
    }}
    @media print {{
      .sidebar, .print-btn, .menu-toggle, .section-foot {{ display: none !important; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <div class="dashboard">
    <aside class="sidebar" id="sidebar">
      <div class="brand">
        {company_mark_html}
        <div><strong>Tube Mill Analytics</strong><span>Enterprise EDA Dashboard</span></div>
      </div>
      <div class="review">
        <div class="review-row"><span>Sections reviewed</span><b id="reviewCount">1 / 11</b></div>
        <div class="review-track"><span id="reviewBar"></span></div>
      </div>
      <div class="nav-label">Report sections</div>
      <nav class="side-nav" id="sideNav" aria-label="Report sections"></nav>
      <div class="sidebar-foot"><kbd>←</kbd><kbd>→</kbd> move between sections</div>
    </aside>
    <div class="content">
      <header>
        <button class="menu-toggle" id="menuToggle" type="button" aria-label="Toggle menu" aria-controls="sidebar" aria-expanded="false">☰</button>
        <div class="header-inner">
          <div class="header-copy">
            <p class="eyebrow">{len(machine_summary)} machine(s) · Descriptive EDA</p>
            <h1>Tube Mill EDA Report</h1>
            <div>Generated {html.escape(generated_at)} | Descriptive comparison across {len(machine_summary)} machines and {len(production_runs):,} production runs</div>
          </div>
          <div class="header-actions">
            <button class="print-btn" id="printBtn" type="button">Print / PDF</button>
            {company_logo_html}
          </div>
        </div>
      </header>
      <main>
    <section>
      <h2>At a glance</h2>
      <div class="cards">
        <div class="card">Production runs<strong>{len(production_runs):,}</strong></div>
        <div class="card">Produced quantity<strong>{format_number(glance_output)}</strong></div>
        <div class="card status-watch">Scrap rate<strong>{format_number(glance_scrap_rate)}%</strong></div>
        <div class="card status-watch">Recorded downtime<strong>{format_number(glance_downtime)} hours</strong></div>
        <div class="card status-good">Availability<strong>{format_number(glance_availability * 100)}%</strong></div>
        <div class="card">Matched energy<strong>{format_number(glance_energy)} kWh</strong></div>
        <div class="card">Energy per produced unit<strong>{format_number(glance_energy_per_unit)} kWh/unit</strong></div>
        <div class="card status-watch">Average speed vs target<strong>{format_number(glance_speed)}%</strong></div>
      </div>
      <div class="status-banner"><strong>Report scope:</strong><span>{len(machine_summary):,} machine, {glance_days:,} production days, {format_number(glance_energy_coverage)}% energy coverage. Machine ranking is unavailable because only one machine was found.</span></div>
    </section>
    <section>
      <h2>Business summary</h2>
      {business_summary_html}
    </section>
    <section>
      <h2>Daily operating trends</h2>
      <p class="muted">Plant-wide daily totals. Run duration and matched energy are assigned to each production run's start date; downtime duration is assigned to each event's start date. Energy is matched by time window, not by meter-to-machine ID. Overlapping production windows may count energy more than once.</p>
      <div class="subtabs" data-subtabs>
        <button class="subtab-button active" type="button" data-subpanel="daily-table">Daily data</button>
        <button class="subtab-button" type="button" data-subpanel="daily-chart">Chart</button>
        <button class="subtab-button" type="button" data-subpanel="daily-signals">Signal summary</button>
        <button class="subtab-button" type="button" data-subpanel="daily-process">Process tags</button>
      </div>
      <div class="subpanel active" id="daily-table">
        <div class="table-wrap">{daily_table_html}</div>
      </div>
      <div class="subpanel" id="daily-chart">{daily_chart_html}</div>
      <div class="subpanel" id="daily-signals">{daily_signal_html.split('<h4>Process tags for higher-energy, lower-downtime dates</h4>')[0]}</div>
      <div class="subpanel" id="daily-process"><h3>Process tags for higher-energy, lower-downtime dates</h3>{process_tag_html}</div>
    </section>
    <section>
      <h2>Machine metrics</h2>
      <p class="muted">Machine-level descriptive metrics. A relative score is shown only when at least two machines have complete score inputs; efficiency bands require at least four.</p>
      <div class="table-wrap">{machine_table_html}</div>
      <div class="charts">{efficiency_chart_html}{machine_correlation_chart_html}</div>
    </section>
    <section>
      <h2>Energy intensity</h2>
      <p>{energy_note} A single machine's energy intensity is a measurement, not a machine-to-machine ranking. Compare similar products and run conditions.</p>
      <div class="table-wrap">{energy_table_html}</div>
      <div class="charts">{output_energy_chart_html}</div>
    </section>
    <section>
      <h2>EMS meter readings and run linkage</h2>
      <p>{html.escape(ems_mapping_description)}</p>
      <p>{html.escape(ems_join_description)}</p>
      <p>{html.escape(ems_calculation_note)} The meter overview covers the EMS file's full time range,
      which may include periods outside production runs.</p>
      <h3>Meter overview</h3>
      <div class="table-wrap">{ems_overview_html}</div>
      <h3>Production-run coverage</h3>
      <div class="table-wrap">{ems_linkage_html}</div>
      <p class="muted">EMS rows matched across runs may be counted more than once if production time windows overlap.</p>
      <h3>Downtime-event coverage</h3>
      <p>{html.escape(ems_downtime_join_description)}</p>
      <div class="table-wrap">{ems_downtime_linkage_html}</div>
      <h3>Matched downtime events</h3>
      <p class="muted">Showing up to 25 downtime events with at least one EMS timestamp match; see downtime_events_enriched.csv for all events.</p>
      <div class="table-wrap">{ems_downtime_events_html}</div>
    </section>
    <section>
      <h2>Downtime reasons</h2>
      <p>{downtime_duration_note} Durations that cannot be interpreted are displayed as N/A, not zero.</p>
      <div class="table-wrap">{downtime_table_html}</div>
      <div class="charts">{downtime_chart_html}</div>
    </section>
    <section>
      <h2>Downtime process patterns</h2>
      <p class="muted">Business view: which process stage or sensor changed before downtime, how often it also appeared during normal operation, and whether enough repeated events support investigation. Planned meal breaks, tool/changeover events, scheduled maintenance, and adjustment reasons such as butting-offset correction are excluded from unplanned-failure candidates. A candidate is not proof of cause. Readings after the stoppage timestamp are excluded from pre-stoppage evidence.</p>
      <h3>Business candidates</h3>
      <div class="cards">{business_cards_html}</div>
      <div class="status-banner"><strong>How to use this:</strong><span>Strong candidate = at least 5 linked events, repeated pre-stoppage behavior, and later-period validation. Moderate candidate = at least 3 events with a meaningful difference from normal operation. Weak candidate = only limited evidence. Insufficient evidence = do not operationalize.</span></div>
      <h3>Pattern summary</h3>
      <div class="table-wrap">{pattern_table_html}</div>
      <h3>Graphical comparison</h3>
      <div class="charts">{pattern_chart_html}</div>
      <h3>Exclusions and data quality</h3>
      <div class="table-wrap">{pattern_exclusion_html}</div>
    </section>
    <section>
      <h2>Speed attainment</h2>
      <p>Mean actual speed divided by mean target speed at each production run, averaged by machine. This is descriptive and is not included in the composite score. Compare within similar product specifications.</p>
      <div class="table-wrap">{table_html(report_machines, ["machine", "target_speed", "actual_speed", "speed_attainment_pct"], labels=machine_labels)}</div>
    </section>
    <section>
      <h2>Process parameter profile</h2>
      <p class="muted">{profile_description}</p>
      <div class="table-wrap">{profile_table_html}</div>
    </section>
    <section>
      <h2>Process parameter differences</h2>
      <p class="muted">Each tag is first averaged within each linked production run, so runs with more change-triggered readings do not automatically get more weight. Standardized differences help compare tags with different units. These are associations, not proof of cause.</p>
      <div class="table-wrap">{factor_table_html}</div>
      <div class="charts">{process_gap_chart_html}</div>
    </section>
    <section>
      <h2>How to interpret these results</h2>
      <div class="note">
        Efficiency bands are relative quartiles of the machines in this dataset, not fixed engineering standards.
        Scores require complete values for output/hour (30%), scrap rate (20%), availability (20%), energy/unit (20%),
        and downtime rate (10%). At least two complete machines are required for scores; at least four with
        distinguishable scores are required for bands. Product mix and operating conditions can affect the ranking.
        Availability and downtime rate both reflect production loss, so the score partly weights related measures twice.
        Production: {production_duration_note} Downtime: {downtime_duration_note}
        PPC counter changes are in the run-level enriched CSV but are not score inputs.
        Treat findings as investigation leads, not causal conclusions or operating instructions.
      </div>
    </section>
      </main>
    </div>
  </div>
  <script>
    (function () {{
      const sections = Array.from(document.querySelectorAll("main > section"));
      const nav = document.getElementById("sideNav");
      const sidebar = document.getElementById("sidebar");
      const menuToggle = document.getElementById("menuToggle");

      document.querySelectorAll("[data-subtabs]").forEach(tabGroup => {{
        tabGroup.addEventListener("click", event => {{
          const button = event.target.closest(".subtab-button");
          if (!button) return;
          const panelId = button.dataset.subpanel;
          const section = tabGroup.closest("section");
          tabGroup.querySelectorAll(".subtab-button").forEach(tab => {{
            tab.classList.toggle("active", tab === button);
          }});
          section.querySelectorAll(":scope > .subpanel").forEach(panel => {{
            panel.classList.toggle("active", panel.id === panelId);
          }});
        }});
      }});

      sections.forEach((section, index) => {{
        const heading = section.querySelector("h2");
        if (!heading) return;
        const title = heading.textContent.trim();
        const panelId = "panel-" + (index + 1);
        section.id = panelId;
        section.setAttribute("role", "tabpanel");

        const head = document.createElement("div");
        head.className = "section-head";
        head.innerHTML = `<div><p class="eyebrow">Step ${{index + 1}} of ${{sections.length}}</p></div><button class="print-btn" type="button" data-copy-link="${{panelId}}" aria-label="Copy link to ${{title}}">Copy link</button>`;
        heading.replaceWith(head);
        head.querySelector("div").appendChild(heading);

        const foot = document.createElement("div");
        foot.className = "section-foot";
        if (index > 0) {{
          foot.innerHTML += `<button class="btn" type="button" data-go="${{index - 1}}">← Previous</button>`;
        }}
        foot.innerHTML += `<span></span>`;
        if (index < sections.length - 1) {{
          foot.innerHTML += `<button class="btn btn-primary" type="button" data-go="${{index + 1}}">Next →</button>`;
        }}
        section.appendChild(foot);

        const button = document.createElement("button");
        button.type = "button";
        button.className = "nav-button";
        button.innerHTML = `<span class="step-dot"><span>${{index + 1}}</span></span><span class="step-label">${{title}}</span>`;
        button.dataset.target = panelId;
        button.setAttribute("role", "tab");
        button.setAttribute("aria-controls", panelId);
        nav.appendChild(button);
      }});

      function showPanel(panelId) {{
        sections.forEach(section => {{
          const active = section.id === panelId;
          section.classList.toggle("active", active);
          section.setAttribute("aria-hidden", active ? "false" : "true");
        }});
        Array.from(nav.querySelectorAll(".nav-button")).forEach(button => {{
          const activeIndex = Array.from(nav.querySelectorAll(".nav-button")).indexOf(button);
          const currentIndex = sections.findIndex(section => section.id === panelId);
          const active = button.dataset.target === panelId;
          button.classList.toggle("active", active);
          button.classList.toggle("is-done", activeIndex < currentIndex);
          button.setAttribute("aria-selected", active ? "true" : "false");
        }});
        const currentIndex = sections.findIndex(section => section.id === panelId);
        const reviewCount = document.getElementById("reviewCount");
        const reviewBar = document.getElementById("reviewBar");
        if (reviewCount) reviewCount.textContent = `${{currentIndex + 1}} / ${{sections.length}}`;
        if (reviewBar) reviewBar.style.width = `${{((currentIndex + 1) / sections.length) * 100}}%`;
        sidebar.classList.remove("open");
        menuToggle.setAttribute("aria-expanded", "false");
        if (window.location.hash !== "#" + panelId) {{
          history.replaceState(null, "", "#" + panelId);
        }}
        window.scrollTo({{ top: 0, behavior: "smooth" }});
      }}

      nav.addEventListener("click", event => {{
        const button = event.target.closest(".nav-button");
        if (button) showPanel(button.dataset.target);
      }});

      document.addEventListener("click", event => {{
        const go = event.target.closest("[data-go]");
        if (go) showPanel(sections[Number(go.dataset.go)].id);
        const copy = event.target.closest("[data-copy-link]");
        if (copy) {{
          const url = window.location.href.split("#")[0] + "#" + copy.dataset.copyLink;
          navigator.clipboard.writeText(url).catch(() => {{}});
        }}
      }});

      document.addEventListener("keydown", event => {{
        if (event.target.closest("input, textarea")) return;
        const currentIndex = sections.findIndex(section => section.classList.contains("active"));
        if (event.key === "ArrowRight" && currentIndex < sections.length - 1) {{
          showPanel(sections[currentIndex + 1].id);
        }} else if (event.key === "ArrowLeft" && currentIndex > 0) {{
          showPanel(sections[currentIndex - 1].id);
        }}
      }});

      const printButton = document.getElementById("printBtn");
      if (printButton) printButton.addEventListener("click", () => window.print());

      menuToggle.addEventListener("click", () => {{
        const open = sidebar.classList.toggle("open");
        menuToggle.setAttribute("aria-expanded", open ? "true" : "false");
      }});

      const initial = window.location.hash;
      const initialPanel = initial ? document.querySelector(initial) : null;
      showPanel(initialPanel && sections.includes(initialPanel) ? initial.slice(1) : "panel-1");
    }})();
  </script>
</body>
</html>
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tube mill EDA script: preprocesses TSV/SQL/Excel datasets, links production-downtime-process-EMS data, "
            "and identifies efficient vs inefficient machines."
        )
    )
    parser.add_argument("--production-path", required=True, help="Path to Production Data file.")
    parser.add_argument("--production-table", default=None, help="Table name when production input is SQL/SQLite.")
    parser.add_argument("--production-sheet", default=None, help="Sheet name when production input is Excel.")

    parser.add_argument("--downtime-path", default=None, help="Path to Downtime Data file.")
    parser.add_argument("--downtime-table", default=None, help="Table name when downtime input is SQL/SQLite.")
    parser.add_argument("--downtime-sheet", default=None, help="Sheet name when downtime input is Excel.")

    parser.add_argument("--process-path", default=None, help="Path to Process Parameter file.")
    parser.add_argument("--process-table", default=None, help="Table name when process input is SQL/SQLite.")
    parser.add_argument("--process-sheet", default=None, help="Sheet name when process input is Excel.")

    parser.add_argument("--ems-path", default=None, help="Path to Energy Management file.")
    parser.add_argument("--ems-table", default=None, help="Table name when EMS input is SQL/SQLite.")
    parser.add_argument("--ems-sheet", default=None, help="Sheet name when EMS input is Excel.")

    parser.add_argument("--ppc-path", default=None, help="Path to Tube Mill PPC file (optional).")
    parser.add_argument("--ppc-table", default=None, help="Table name when PPC input is SQL/SQLite.")
    parser.add_argument("--ppc-sheet", default=None, help="Sheet name when PPC input is Excel.")
    parser.add_argument(
        "--pattern-max-tags",
        type=int,
        default=20,
        help="Maximum number of highest-volume process tags evaluated for downtime patterns.",
    )
    parser.add_argument(
        "--pattern-max-workers",
        type=int,
        default=4,
        help="Maximum concurrent workers used while collecting normal-operation PLC windows.",
    )
    parser.add_argument(
        "--pattern-max-events",
        type=int,
        default=0,
        help="Maximum downtime events to analyze. Use 0 to analyze every event.",
    )

    parser.add_argument("--output-dir", default="eda_outputs", help="Directory where outputs are written.")
    parser.add_argument(
        "--production-duration-unit",
        choices=["seconds", "minutes", "hours", "days"],
        default=None,
        help=(
            "Unit used by numeric Run_Time and Net_Run_Time values. "
            "Leave unset if production durations are formatted time strings."
        ),
    )
    parser.add_argument(
        "--downtime-duration-unit",
        choices=["seconds", "minutes", "hours", "days"],
        default=None,
        help=(
            "Unit used by numeric downtime Duration values. "
            "Leave unset if downtime durations are formatted time strings."
        ),
    )
    return parser.parse_args()


def maybe_load_dataset(
    path: Optional[str],
    table_name: Optional[str],
    sheet_name: Optional[str],
    dataset_name: str,
) -> pd.DataFrame:
    if not path:
        print(f"[{dataset_name}] skipped (no path supplied).")
        return pd.DataFrame()
    raw = read_input_to_dataframe(path, table_name=table_name, sheet_name=sheet_name)
    return preprocess_dataset(raw, dataset_name=dataset_name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    production_df = maybe_load_dataset(
        args.production_path, args.production_table, args.production_sheet, "production"
    )
    downtime_df = maybe_load_dataset(args.downtime_path, args.downtime_table, args.downtime_sheet, "downtime")
    process_df = maybe_load_dataset(args.process_path, args.process_table, args.process_sheet, "process")
    ems_df = maybe_load_dataset(args.ems_path, args.ems_table, args.ems_sheet, "ems")
    ppc_df = maybe_load_dataset(args.ppc_path, args.ppc_table, args.ppc_sheet, "ppc")

    if production_df.empty:
        raise ValueError("Production dataset is empty after preprocessing.")

    if not ems_df.empty:
        print("[EMS] indexing timestamps and matching production/downtime windows...", flush=True)
        ems_df = add_ems_energy_columns(ems_df)
        ems_index = EMSIntervalIndex(ems_df)
        production_df = aggregate_ems_within_runs(production_df, ems_df, ems_index)
        if not downtime_df.empty:
            downtime_df = aggregate_ems_within_downtime(
                downtime_df, ems_df, ems_index
            )
    if not ppc_df.empty:
        print("[PPC] matching production windows...", flush=True)
        production_df = aggregate_ppc_within_runs(production_df, ppc_df)

    print("[Metrics] calculating machine summaries...", flush=True)
    production_runs, machine_summary = compute_machine_efficiency(
        production_df=production_df,
        downtime_df=downtime_df if not downtime_df.empty else None,
        production_duration_unit=args.production_duration_unit,
        downtime_duration_unit=args.downtime_duration_unit,
    )
    daily_summary = create_daily_summary(
        production_runs,
        downtime_df if not downtime_df.empty else None,
        args.downtime_duration_unit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    machine_summary.to_csv(output_dir / "machine_efficiency_summary.csv", index=False)
    production_runs.to_csv(output_dir / "production_runs_enriched.csv", index=False)
    if not downtime_df.empty:
        downtime_df.to_csv(output_dir / "downtime_events_enriched.csv", index=False)

    print("[Process] building process-parameter summaries...", flush=True)
    create_factor_tables(
        production_runs=production_runs,
        machine_summary=machine_summary,
        process_df=process_df if not process_df.empty else None,
        output_dir=output_dir,
    )
    print("[Patterns] building downtime/process pattern dataset...", flush=True)
    pattern_result = create_downtime_pattern_dataset(
        production_runs=production_runs,
        downtime_df=downtime_df if not downtime_df.empty else None,
        process_df=process_df if not process_df.empty else None,
        output_dir=output_dir,
        max_tags=args.pattern_max_tags,
        max_workers=args.pattern_max_workers,
        max_events=args.pattern_max_events,
    )
    print(
        f"[Patterns] complete: {len(pattern_result['features'])} event/window/tag rows, "
        f"{len(pattern_result['summary'])} pattern rows",
        flush=True,
    )
    print("[Plots] generating charts...", flush=True)
    create_plots(
        machine_summary=machine_summary,
        downtime_df=downtime_df if not downtime_df.empty else None,
        output_dir=output_dir,
        downtime_duration_unit=args.downtime_duration_unit,
        daily_summary=daily_summary,
    )
    print("[Report] generating HTML...", flush=True)
    report_path = create_html_report(
        production_runs=production_runs,
        machine_summary=machine_summary,
        downtime_df=downtime_df if not downtime_df.empty else None,
        output_dir=output_dir,
        production_duration_unit=args.production_duration_unit,
        downtime_duration_unit=args.downtime_duration_unit,
        ems_df=ems_df if not ems_df.empty else None,
        daily_summary=daily_summary,
        process_df=process_df if not process_df.empty else None,
    )

    print("\nEDA complete.")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print(f"HTML report: {report_path.resolve()}")
    scored_machines = machine_summary.dropna(subset=["efficiency_score"])
    if len(scored_machines) >= 2 and scored_machines["efficiency_score"].nunique() > 1:
        print("Highest relative scores:")
        print(scored_machines[["machine", "efficiency_score", "efficiency_band"]].head(5).to_string(index=False))
        print("\nLowest relative scores:")
        print(scored_machines[["machine", "efficiency_score", "efficiency_band"]].tail(5).to_string(index=False))
    else:
        print("No composite machine ranking was produced; see the HTML report for unavailable inputs.")


if __name__ == "__main__":
    main()
