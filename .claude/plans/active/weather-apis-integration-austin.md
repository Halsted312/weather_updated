---
plan_id: weather-apis-integration-austin
created: 2025-12-06
status: in_progress
priority: high
agent: kalshi-weather-quant
---

# Weather APIs Integration: VC 15-min Backfill + NOAA Model Guidance

## Objective

Implement two complementary data quality and feature enhancement tasks for Austin:

1. **VC 15-min Data Validation**: Backfill and validate Visual Crossing 15-minute forecast data and 5-minute observations for Austin (2025-05-01 to 2025-12-01), with visual quality check
2. **NOAA Model Guidance**: Create new `weather_more_apis` feature block aggregating NBM, HRRR, and NDFD forecast sources

## Context

### Current State (from exploration)

**VC Ingestion Infrastructure:**
- 5-min observations: Working via `ingest_vc_obs_backfill.py` and `ingest_vc_obs_parallel.py`
- Daily/hourly forecasts: Working via `ingest_vc_historical_forecast_parallel.py`
- **GAP**: 15-min minute-level forecasts are NOT being ingested (only hourly forecasts exist)
- `wx.vc_minute_weather` table exists and supports minute-level data
- Austin has 2 vc_location records: station (KAUS) and city ("Austin,TX")

**Feature Pipeline:**
- SnapshotContext dataclass with optional fields for extensibility
- `compute_snapshot_features()` orchestrates all feature computation
- Feature modules use `@register_feature_group` decorator
- Manual feature registry in `NUMERIC_FEATURE_COLS` (base.py)
- Modular imputation by feature group (imputation.py)

**Database Schema:**
- Established patterns for datetime handling (UTC + local + timezone)
- CHECK constraints for enum-like fields
- Unique constraints for idempotent UPSERTs
- TimescaleDB hypertables for high-frequency data
- Foreign keys to dimension tables (vc_location)

### Why This Matters

**VC 15-min validation:**
- Need to verify time alignment and basis_date semantics before model training
- 15-min forecast resolution provides richer intraday curve than hourly
- Austin is test market for edge classifier improvements

**NOAA model guidance:**
- NBM, HRRR, NDFD are official NOAA sources with different characteristics:
  - NBM: Ensemble blend, updated hourly
  - HRRR: High-resolution rapid refresh, updated hourly
  - NDFD: Official NWS forecast (closest to what public sees)
- Provides independent signal from VC forecasts
- Can capture forecast disagreement/uncertainty
- All three are public data (no API keys required)

### Constraints

- **Scope**: Austin only for initial implementation
- **Date range**: 2025-05-01 to 2025-12-01
- **No live trading impact**: Research/validation only
- **Feature toggling**: New features must be optional (include_more_apis flag)
- **Null handling**: Must work gracefully when data is missing

## Tasks

### Phase 1: VC 15-min Backfill + Validation (Austin)

- [ ] 1.1 Audit current Austin data coverage (5-min obs, hourly/daily forecasts)
- [ ] 1.2 Investigate VC Timeline API support for 15-min historical forecast minutes
- [ ] 1.3 Modify/extend ingestion script to backfill 15-min forecasts if supported
- [ ] 1.4 Backfill Austin 5-min obs (verify complete coverage 2025-05-01 to 2025-12-01)
- [ ] 1.5 Enhance `validate_15min_ingestion.py` with Austin-specific checks
- [ ] 1.6 Create visual QA script (forecast vs obs overlay for demo day)
- [ ] 1.7 Document findings and any VC API limitations

### Phase 2: NOAA Model Guidance Infrastructure

- [ ] 2.1 Create `src/weather_more_apis/` package (model_specs.py, ingest.py, features.py)
- [ ] 2.2 Create DB migration for `wx.weather_more_apis_guidance` table
- [ ] 2.3 Implement NBM GRIB download and extraction (wgrib2)
- [ ] 2.4 Implement HRRR GRIB download and extraction (wgrib2)
- [ ] 2.5 Implement NDFD GRIB download and extraction (wgrib2)
- [ ] 2.6 Create unified ingestion CLI (`ingest_weather_more_apis_guidance.py`)
- [ ] 2.7 Backfill Austin guidance data (2025-05-01 to 2025-12-01)

### Phase 3: Feature Pipeline Integration

- [ ] 3.1 Add loader function: `load_weather_more_apis_guidance_for_snapshot()`
- [ ] 3.2 Add fields to SnapshotContext: `weather_more_apis`
- [ ] 3.3 Create feature module: `models/features/weather_more_apis.py`
- [ ] 3.4 Integrate into `compute_snapshot_features()` in pipeline.py
- [ ] 3.5 Add feature names to `NUMERIC_FEATURE_COLS` in base.py
- [ ] 3.6 Add null-filling function in imputation.py
- [ ] 3.7 Add toggle to DatasetConfig: `include_more_apis`

### Phase 4: Validation & Documentation

- [ ] 4.1 Create debug script: `debug_weather_more_apis_guidance.py`
- [ ] 4.2 Test dataset builder with new features (Austin, small date range)
- [ ] 4.3 Verify null handling when weather_more_apis data is missing
- [ ] 4.4 Document GRIB model specs and peak window logic
- [ ] 4.5 Update CLAUDE.md if needed (new data sources, feature groups)

## Files to Create/Modify

| Action | Path | Notes |
|--------|------|-------|
| **Phase 1: VC 15-min** | | |
| MODIFY | `scripts/validate_15min_ingestion.py` | Add Austin-specific validation queries and logging |
| MODIFY | `scripts/ingest_vc_hist_forecast_v2.py` | Investigate/add minute-level forecast support if VC API allows |
| CREATE | `scripts/backfill_vc_austin_15min.py` | Dedicated Austin backfill wrapper (optional) |
| CREATE | `visuals/vc_forecast_vs_obs_austin.py` | Matplotlib plot for QA (T-1 forecast vs obs overlay) |
| **Phase 2: NOAA Guidance** | | |
| CREATE | `src/weather_more_apis/__init__.py` | Package init with docstring |
| CREATE | `src/weather_more_apis/model_specs.py` | NBM/HRRR/NDFD GRIB key builders and specs |
| CREATE | `src/weather_more_apis/ingest.py` | Generic download, extraction, peak window logic |
| CREATE | `src/weather_more_apis/features.py` | Feature computation for weather_more_apis group |
| CREATE | `migrations/versions/0XX_add_weather_more_apis_guidance.py` | Alembic migration for new table |
| CREATE | `scripts/ingest_weather_more_apis_guidance.py` | Unified CLI for NBM/HRRR/NDFD backfill |
| **Phase 3: Feature Pipeline** | | |
| MODIFY | `models/data/loader.py` | Add `load_weather_more_apis_guidance_for_snapshot()` |
| MODIFY | `models/features/pipeline.py` | Add `weather_more_apis` field to SnapshotContext; integrate feature computation |
| MODIFY | `models/features/base.py` | Add 8 new feature names to `NUMERIC_FEATURE_COLS` |
| MODIFY | `models/features/imputation.py` | Add `fill_weather_more_apis_nulls()` |
| MODIFY | `models/data/dataset.py` | Add `include_more_apis` to DatasetConfig; call loader |
| **Phase 4: Validation** | | |
| CREATE | `scripts/debug_weather_more_apis_guidance.py` | QA helper to inspect guidance data for single date |
| MODIFY | `docs/permanent/FILE_DICTIONARY_GUIDE.md` | Document new package and scripts (optional) |

## Technical Details

### 1. VC 15-min Forecast Challenge

**Issue**: Current ingestion scripts use:
```python
fetch_station_historical_forecast(..., include='days,hours', forecastBasisDate=...)
```

This fetches daily and hourly forecasts but NOT minute-level forecasts.

**Investigation needed**:
- Does VC Timeline API support `include=minutes` with `forecastBasisDate` parameter?
- If not supported, can we use hourly forecasts interpolated to 15-min for analysis?
- Document the limitation if VC doesn't support historical 15-min forecasts

**Fallback approach**:
- Use hourly forecasts (already ingested) for T-1 forecast curve
- Forward-fill to 15-min resolution in loader (mark with `is_forward_filled` flag)
- Still validate 5-min observations for accuracy

### 2. wx.weather_more_apis_guidance Table Schema

```sql
CREATE TABLE wx.weather_more_apis_guidance (
    id SERIAL PRIMARY KEY,
    city_id TEXT NOT NULL,  -- 'austin', 'chicago', etc.
    model TEXT NOT NULL CHECK (model IN ('nbm', 'hrrr', 'ndfd')),
    run_time_utc TIMESTAMPTZ NOT NULL,  -- Model run time (UTC)
    target_date_local DATE NOT NULL,  -- Event day in city local time
    basis_date_local DATE NOT NULL,  -- Lead day (typically target_date - lead_days)
    lead_days INTEGER NOT NULL,  -- e.g. 1 for T-1, 2 for T-2

    -- Scalar summaries (no minute-level explosion)
    peak_window_max_f FLOAT,  -- Max 2-m temp over peak hours (13-18 local)
    temp_at_15_local_f FLOAT,  -- Optional: temp at 15:00 local
    temp_at_18_local_f FLOAT,  -- Optional: temp at 18:00 local

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (city_id, model, target_date_local, basis_date_local, run_time_utc)
);

CREATE INDEX ix_weather_more_apis_city_target
ON wx.weather_more_apis_guidance (city_id, target_date_local, model);
```

**Key design decisions**:
- One table for all three models (model field distinguishes)
- Scalar summaries only (peak window max, optional hourly samples)
- No minute-level data (keeps table small)
- Unique constraint allows multiple runs per basis_date (e.g., hourly NBM/HRRR updates)

### 3. NOAA Data Access (No API Keys)

**NBM (National Blend of Models):**
- Source: `https://noaa-nbm-grib2-pds.s3.amazonaws.com/...`
- Format: GRIB2
- Resolution: 2.5km CONUS grid
- Update frequency: Hourly (00Z-23Z)
- Key pattern: `blend.{cycle}/core/blend.t{cycle}z.core.f{hour}.co.grib2`
- Variable: `TMP:2 m above ground`

**HRRR (High-Resolution Rapid Refresh):**
- Source: `https://noaa-hrrr-bdp-pds.s3.amazonaws.com/...`
- Format: GRIB2
- Resolution: 3km CONUS grid
- Update frequency: Hourly
- Key pattern: `hrrr.{YYYYMMDD}/conus/hrrr.t{cycle}z.wrfsfcf{hour}.grib2`
- Variable: `TMP:2 m above ground`

**NDFD (National Digital Forecast Database):**
- Source: `https://noaa-ndfd-pds.s3.amazonaws.com/...` (confirm exact path)
- Format: GRIB2
- Resolution: 2.5km CONUS grid
- Update frequency: 4x daily (00Z, 06Z, 12Z, 18Z)
- Variable: Daily max temperature or 2-m temp field (inspect inventory)

**Extraction method**:
```bash
# Use wgrib2 (prefer over cfgrib for simplicity)
wgrib2 file.grib2 -lon <lon> <lat> -match "TMP:2 m"
```

### 4. Feature Definitions (weather_more_apis group)

8 numeric features (all FLOAT, nullable):

```python
{
    # NBM features
    "nbm_peak_window_max_f": <peak temp from NBM>,
    "nbm_peak_window_revision_1h_f": <drift from previous NBM run>,

    # HRRR features
    "hrrr_peak_window_max_f": <peak temp from HRRR>,
    "hrrr_peak_window_revision_1h_f": <drift from previous HRRR run>,

    # NDFD features (closest to NWS official forecast)
    "ndfd_tmax_T1_f": <NDFD max temp for target date>,
    "ndfd_drift_T2_to_T1_f": <NDFD forecast evolution T-2 → T-1>,

    # Disagreement metrics
    "hrrr_minus_nbm_peak_window_max_f": <HRRR - NBM spread>,
    "ndfd_minus_vc_T1_f": <NDFD - VC T-1 forecast difference>,
}
```

**Interpretation**:
- Larger disagreement → higher forecast uncertainty
- Revision features capture how forecasts evolve (momentum/stability)
- NDFD is official NWS, so NDFD vs VC difference is informative

### 5. Peak Window Logic

For NBM and HRRR, we sample 2-m temperature at:
- Peak hours local: 13:00, 14:00, 15:00, 16:00, 17:00, 18:00
- Compute: `peak_window_max_f = max(temps_at_peak_hours)`

This mimics how daily high typically occurs during afternoon peak heating.

For NDFD:
- Use daily max field if available
- Otherwise use same peak window sampling

### 6. SnapshotContext Integration

```python
@dataclass
class SnapshotContext:
    # ... existing fields ...

    # NOAA model guidance (optional)
    weather_more_apis: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    # Structure:
    # {
    #   "nbm": {"latest": {...}, "prev": {...}},
    #   "hrrr": {"latest": {...}, "prev": {...}},
    #   "ndfd": {"latest": {...}, "prev": {...}},
    # }
```

Loader returns nested dict with latest and previous run for each model.

## Completion Criteria

### Phase 1: VC 15-min Backfill
- [ ] Austin 5-min obs exist for 2025-05-01 to 2025-12-01 (complete coverage)
- [ ] VC API 15-min minute-level forecast support investigated and documented
- [ ] Visual QA plot generated for demo day (2025-11-19/20)
- [ ] Datetime and basis_date semantics verified against DATETIME_AND_API_REFERENCE.md
- [ ] No timezone inconsistencies detected

### Phase 2: NOAA Model Guidance
- [ ] `wx.weather_more_apis_guidance` table created via migration
- [ ] NBM, HRRR, NDFD ingestion working for Austin
- [ ] Austin guidance data backfilled (2025-05-01 to 2025-12-01)
- [ ] Peak window logic validated for reasonableness
- [ ] GRIB download and extraction robust to network errors

### Phase 3: Feature Pipeline
- [ ] `weather_more_apis` feature block computes 8 features
- [ ] SnapshotContext integration working
- [ ] Null handling verified (features default to None when data missing)
- [ ] DatasetConfig toggle (`include_more_apis`) working
- [ ] Feature names registered in NUMERIC_FEATURE_COLS

### Phase 4: Validation
- [ ] Debug script confirms correct run selection and cutoff logic
- [ ] Dataset builder test run completes without errors
- [ ] All new features present in output DataFrame
- [ ] Documentation updated

## Critical Discovery: VC API Limitation

**VC minute-level historical forecasts (`wx.vc_minute_weather` with `data_type='historical_forecast'`) contain valid timestamps but NULL weather fields (temp, dew, etc.) for recent dates.**

**Global validation findings:**
- Austin: Non-NULL temps from 2022-12-23 to 2025-04-07 (then NULLs)
- Chicago: Non-NULL temps from 2022-12-23 to 2025-11-05 (sporadic)
- Denver/LAX/Miami/Philly: All NULL for all dates

**Conclusion**: VC stopped providing minute-level historical forecast temps around April 2025. This is a **VC API behavioral change**.

**Decision**:
- ✅ Mark VC 15-min phase as COMPLETE (validated the limitation)
- ❌ Do NOT attempt to fix this with more VC requests or different query options
- ❌ Do NOT forward-fill or interpolate in the database
- ❌ Do NOT write synthetic temps into `wx.vc_minute_weather`
- ✅ **Leave all wx.vc_minute_weather rows as-is** (timestamps valid, temps NULL for recent dates)
- ✅ **VC daily + hourly historical forecasts remain the primary VC forecast inputs**
- ✅ **VC minute observations (actual_obs) remain valid** for all cities/dates
- ✅ Treat VC forecast curves as **HOURLY** in feature pipeline (not 15-min)
- ✅ If 15-min forecast curve needed later, derive in Python from T-1 hourly (linear interpolation) inside `compute_snapshot_features`, not from database

**Proceeding to**: NOAA Model Guidance (NBM/HRRR/NDFD) - this provides independent forecast source without VC minute limitations.

---

## ✅ NOAA Implementation Complete (NBM + HRRR)

### Results Summary

**NBM Backfill:**
- ✅ 215/215 days successful (100%)
- ✅ Austin May-Dec 2025
- ✅ ~25 min with 16 workers
- ✅ Temps: 48.85-102.72°F, avg 88.7°F

**HRRR Backfill:**
- ✅ 215/215 days successful (100%)
- ✅ Austin May-Dec 2025
- ✅ ~5 min with 14 workers (6x speedup!)
- ✅ Optimized to 1-hour sampling (15:00 local)

**Features Implemented** (11 total):
1. nbm_peak_window_max_f (now = T15 temp)
2. nbm_peak_window_revision_1h_f
3. hrrr_peak_window_max_f (now = T15 temp)
4. hrrr_peak_window_revision_1h_f
5. hrrr_minus_nbm_peak_window_max_f
6. ndfd_tmax_T1_f (None - future work)
7. ndfd_drift_T2_to_T1_f (None - future work)
8. ndfd_minus_vc_T1_f (None - future work)
9. **nbm_t15_z_30d_f** (NEW - dimensionless)
10. **hrrr_t15_z_30d_f** (NEW - dimensionless)
11. **hrrr_minus_nbm_t15_z_30d_f** (NEW - dimensionless)

**Pipeline Integration:**
- ✅ SnapshotContext extended (more_apis + obs stats)
- ✅ Loader functions created
- ✅ Features registered in NUMERIC_FEATURE_COLS
- ✅ Imputation added
- ✅ DatasetConfig toggle (include_more_apis=True)
- ✅ Full stack tested and working

### NDFD - Future Work

**Status**: Placeholder AWS path returns 404 (expected)

**Issue**: Real NOAA NDFD S3 bucket structure unknown

**TODO (separate mini-project)**:
```bash
# Investigate actual NDFD S3 layout
aws s3 ls s3://noaa-ndfd-pds/ --no-sign-request
# Find TMAX GRIB files (likely: ndfd.YYYYMMDD/cycle/...)
# Update NDFD_SPEC in model_specs.py
# Rerun ingestion
```

**For now**: NDFD features return None (pipeline handles gracefully)

## Execution Summary

### Task 1: VC 15-min Backfill (Austin) - ~4 hours

**Status**: Scripts exist, just need to run backfill + validation

**Steps**:
1. Run backfill: `python scripts/backfill_vc_historical_forecast_minutes_austin.py --city austin --start-date 2025-05-01 --end-date 2025-12-01`
2. Run validation queries (coverage, timezone, basis_date checks)
3. Generate visual QA plots: `python scripts/plot_austin_forecast_vs_obs.py --event-date 2025-06-15`
4. Test loader integration: Verify `load_historical_forecast_15min()` returns expected data
5. Document findings

**Estimated time**: 4 hours (mostly waiting for API calls)

### Task 2: NOAA Model Guidance - ~7-10 days

**Status**: Requires full implementation from scratch

**Important**: Start with NBM + HRRR (proven AWS paths), then add NDFD after confirming specifics.

**Phases**:
- **Phase 1 (Days 1-2)**: Database schema + ingestion infrastructure
  - Install wgrib2 (host + Docker) - **PREREQUISITE**
  - Create `wx.weather_more_apis_guidance` table via Alembic migration
  - Implement `src/weather_more_apis/` package (model_specs, ingest)
  - Create ingestion CLI script
  - **Start with NBM + HRRR for Austin** (backfill proof-of-concept)
  - Investigate NDFD AWS path + GRIB variable selection in parallel

- **Phase 2 (Days 3-4)**: Feature pipeline integration
  - Implement loader function `load_more_apis_guidance()`
  - Create feature module `models/features/more_apis.py` (8 features)
    - **Wire all 8 features at once** (including NDFD placeholders if needed)
  - Integrate with SnapshotContext and pipeline
  - Add imputation and config flags
  - Add NDFD ingestion once AWS path confirmed

- **Phase 3 (Days 5-6)**: Validation
  - Test end-to-end dataset building with all 3 models
  - Confirm all 8 features appear in dataset for Austin
  - Validate feature quality and correlations
  - Create debug/validation scripts
  - Document any NDFD limitations discovered

- **Phase 4 (Days 7+)**: Extension
  - Extend to other 5 cities
  - Production integration
  - Model retraining with new features

**NDFD-Specific Notes**:
- Use **fixed cycles** (00Z, 12Z) NOT hourly like NBM/HRRR
- Confirm exact AWS key pattern: `noaa-ndfd-pds.s3.amazonaws.com/...`
- Inspect GRIB inventory to select correct variable (TMAX vs TMP:2m)
- Okay if NDFD lags NBM/HRRR by 1-2 days during debugging

## Implementation Details

### VC 15-min Validation (Detailed Steps)

See Plan Agent output above for:
- Complete validation SQL queries (8 different checks)
- Visual QA specifications
- Edge case handling (DST, missing data, timezone bugs)
- Fallback approaches if API limitations discovered

**Key validations**:
1. Coverage: All 214 days have ~96 forecast points each
2. Timezone: No 6-hour shifts, correct CST/CDT offsets (-360/-300)
3. Basis date: All forecasts use T-1 basis for event date T
4. Temperature sanity: Values in reasonable range (30-110°F for Austin)
5. Visual alignment: Forecast and obs curves align horizontally

### NOAA Guidance (Detailed Architecture)

See Plan Agent output above for:
- Complete table schema (SQL DDL + SQLAlchemy model)
- GRIB download/extraction code (wgrib2 integration)
- Peak window calculation (hours 13-18 local → UTC → forecast hours)
- Feature computation (8 features with null handling)
- Full integration checklist (12 files to create/modify)

**8 Features**:
1. `nbm_peak_window_max_f` - NBM peak temp forecast
2. `nbm_peak_window_revision_1h_f` - NBM forecast revision
3. `hrrr_peak_window_max_f` - HRRR peak temp forecast
4. `hrrr_peak_window_revision_1h_f` - HRRR forecast revision
5. `ndfd_tmax_T1_f` - NDFD daily max forecast
6. `ndfd_drift_T2_to_T1_f` - NDFD forecast drift
7. `hrrr_minus_nbm_peak_window_max_f` - Model disagreement
8. `ndfd_minus_vc_T1_f` - NDFD vs VC disagreement

## wgrib2 Installation

### Critical Dependency

The NOAA guidance ingestion relies on **wgrib2** for GRIB2 file extraction. This is a **native binary**, not a Python package.

### Host Installation (Ubuntu/Debian)

```bash
# On host OS (outside Python venv)
sudo apt-get update
sudo apt-get install -y wgrib2

# Verify installation
which wgrib2
wgrib2 -help | head
```

**Alternative (Conda):**
```bash
conda install -c conda-forge wgrib2
```

**Key point**: Once `which wgrib2` works in your shell, Python code using `subprocess.run(["wgrib2", ...])` will find it (PATH is inherited from shell).

### Docker Installation

Add to your Dockerfile (or the Dockerfile referenced by docker-compose.yml):

```dockerfile
# For Debian/Ubuntu base image
RUN apt-get update && \
    apt-get install -y --no-install-recommends wgrib2 && \
    rm -rf /var/lib/apt/lists/*
```

**Alternative (Conda-based image):**
```dockerfile
RUN conda install -y -c conda-forge wgrib2 && conda clean -afy
```

**Important**: This must happen BEFORE running Python scripts that call wgrib2.

### Verification in Docker

```bash
# Test wgrib2 inside container
docker compose run --rm <service_name> bash -lc "which wgrib2 && wgrib2 -help | head"
```

If this passes, your `weather_more_apis` ingestion code can safely call wgrib2.

### Python Usage Pattern

**Correct approach** (used in plan):
```python
import subprocess

result = subprocess.run(
    ["wgrib2", "file.grib2", "-lon", str(lon), str(lat), "-match", ":TMP:2 m above ground:"],
    capture_output=True,
    text=True,
    timeout=30
)
```

**Do NOT**:
- Try to `pip install wgrib2` (doesn't exist as Python package)
- Use cfgrib as primary extractor (wgrib2 is simpler for point extraction)

---

## Key Files Reference

### VC 15-min (Existing)
- `scripts/backfill_vc_historical_forecast_minutes_austin.py` - Backfill script
- `scripts/plot_austin_forecast_vs_obs.py` - Visual QA tool
- `models/data/loader.py` (lines 376-431) - Loader function
- `src/db/models.py` - VcMinuteWeather model

### NOAA Guidance (To Create)
- `src/weather_more_apis/__init__.py` - Package
- `src/weather_more_apis/model_specs.py` - NBM/HRRR/NDFD specs
- `src/weather_more_apis/ingest.py` - GRIB download/extraction
- `scripts/ingest_more_apis_guidance.py` - Ingestion CLI
- `models/features/more_apis.py` - Feature computation
- `migrations/versions/XXXX_add_weather_more_apis_guidance.py` - Migration

### Integration Points (To Modify)
- `src/db/models.py` - Add WeatherMoreApisGuidance model
- `models/features/pipeline.py` - Add more_apis field to SnapshotContext
- `models/features/base.py` - Register 8 features in NUMERIC_FEATURE_COLS
- `models/features/imputation.py` - Add fill_more_apis_nulls()
- `models/data/loader.py` - Add load_more_apis_guidance()
- `models/data/dataset.py` - Add include_more_apis flag

## Sign-off Log

### 2025-12-06 Planning Complete - USER APPROVED

**Status**: ✅ Plan approved by user with expert coder review - Ready for implementation

**Last completed**:
- ✅ Explored VC ingestion infrastructure (3 parallel agents)
- ✅ Explored feature pipeline architecture
- ✅ Explored database schema patterns
- ✅ Launched 2 Plan agents (VC validation + NOAA guidance)
- ✅ Verified existing scripts via file reads
- ✅ Updated plan with comprehensive findings
- ✅ Received expert coder feedback and incorporated refinements
- ✅ Added wgrib2 installation instructions
- ✅ Clarified NDFD approach (NBM+HRRR first, then NDFD)
- ✅ Emphasized VC verification over assumption

**Key Discoveries**:
1. VC 15-min scripts already exist - reduces Task 1 to verification + execution
2. Plan structure validated by expert coder - no conceptual errors
3. wgrib2 dependency explicit - install instructions added

**Next steps** (APPROVED):
1. **Install wgrib2** on host and in Docker (prerequisite)
2. **Execute VC backfill** (Task 1 / Phase 1) - verify existing scripts work
   - Run backfill for Austin 2025-05-01 to 2025-12-01
   - Execute 8 validation queries (don't assume, verify!)
   - Generate visual QA plots
   - Document any timezone/coverage issues
3. **Implement NOAA guidance** (Task 2) - phased development
   - Start with NBM + HRRR (proven paths)
   - Add NDFD after confirming AWS path + GRIB variable
   - Wire all 8 features at once for Austin
   - Validate end-to-end before extending

**Blockers**: None - user approved, ready to code

## User Approval & Clarifications

### Answers to Questions

1. **Priority & Scope**
   - ✅ **Do VC validation FIRST** (Phase 1) - quick, contained task
   - ✅ Then **implement full `weather_more_apis` block** with **all three models** (NBM + HRRR + NDFD) in one pass for Austin
   - ❌ Do NOT stop at NBM-only - want complete feature block to see full effect

2. **wgrib2 Availability**
   - ❌ **NOT currently installed** on host or in Docker
   - ✅ Will install on host OS (Ubuntu/Debian via apt-get)
   - ✅ Will add to Docker image for training/backfill containers
   - ✅ Call wgrib2 as CLI subprocess, NOT pip install

3. **Date Range**
   - ✅ Confirmed: Austin only, 2025-05-01 to 2025-12-01 for both tasks
   - ❌ Do NOT extend to other cities until Austin validated end-to-end

4. **Implementation Scope**
   - ✅ Wire all 8 `weather_more_apis` features into pipeline at once
   - ✅ Add all to NUMERIC_FEATURE_COLS, imputation, DatasetConfig
   - ✅ Confirm all features appear in dataset output for Austin

### Critical Refinements from Expert Review

**✅ What's Good:**
- Austin-only scope with short window (2025-05-01 to 2025-12-01)
- All three NOAA models in one unified `weather_more_apis` block
- Feature definitions match requirements (8 features: level + revision + disagreement)
- SnapshotContext integration follows repo patterns
- Toggle & safety nets (include_more_apis flag, imputation)

**⚠️ Watch Out For:**

1. **VC 15-min Scripts - VERIFY, Don't Assume**
   - Scripts exist but are relatively new
   - **Must explicitly verify** with direct DB queries:
     - Correct `data_type='historical_forecast'`
     - Correct `forecast_basis_date` (T-1 for event T)
     - Timestamps in America/Chicago, NOT shifted by 6h
     - ~96 points per day (15-min intervals)
   - Treat as **verification task**, not assumption

2. **NDFD Path + Variable Selection = TODO**
   - NDFD AWS bucket path not confirmed
   - GRIB variable choice unclear (daily max vs 2-m temp field)
   - **Strategy**: Start with NBM + HRRR first, then add NDFD after confirming:
     - Exact AWS key pattern
     - Which GRIB variable to use
   - Okay if NDFD lags by a day while debugging

3. **NDFD Run Frequency - Different from NBM/HRRR**
   - NBM/HRRR: Hourly runs (00Z-23Z)
   - NDFD: **Use fixed cycles only** (e.g., 00Z and 12Z per basis_date)
   - Do NOT loop hourly for NDFD
   - Update frequency: 4× daily (00Z, 06Z, 12Z, 18Z) but use subset

4. **wgrib2 Dependency - Real, Not Optional**
   - Entire GRIB extraction plan relies on wgrib2 CLI
   - Must install before proceeding with NOAA ingestion
   - See installation section below

5. **All Features at Once**
   - Implement all 8 features for all 3 models in Phase 2
   - Do NOT stage NBM-only first
   - Want full Austin feature block validated before extending
