# Data sources and licences

The code in this repository is MIT-licensed. The data is not: each file keeps its source licence, listed below. Everything here is redistributed under the terms of those licences, with attribution.

| File | Source | Licence | Notes |
|---|---|---|---|
| `raw_spatial/keCellTowers.csv` | [OpenCellID](https://opencellid.org) cell registry, Kenya (MCC 639) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Crowdsourced, approximate coordinates; a 2019 snapshot with no 5G (NR) cells. County, sub-county and ward columns were added by a spatial join to Kenya's administrative boundaries. |
| `raw_ookla/ookla_mobile_kenya_all_quarters.csv` | [Ookla Open Data](https://github.com/teamookla/ookla-open-data), mobile performance tiles, Q1 2022 to Q1 2026 | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Filtered to a bounding box around Kenya (lat −4.9 to 5.1, lon 33.8 to 42.0). Non-commercial use only. |
| `processed/ca_quarterly_telemetry.csv` | [Communications Authority of Kenya](https://www.ca.go.ke) quarterly sector statistics reports | Public statistics, compiled by hand | Q2 to Q4 2022/23 data volumes are missing in the source and are imputed in notebook 02. |

**Required attribution for Ookla data:** Speedtest® by Ookla® Global Fixed and Mobile Network Performance Maps. Based on analysis by Ookla of Speedtest Intelligence® data for Q1 2022 to Q1 2026. Provided by Ookla and accessed October 2026.

**Share-alike.** Because the Ookla and OpenCellID data are share-alike, the derived tables, figures and models in `Outputs/` are shared under CC BY-NC-SA 4.0 as well.

## Re-downloading the Ookla tiles

The original quarterly files are public on AWS (no account needed):

```
https://ookla-open-data.s3.amazonaws.com/parquet/performance/type=mobile/year=2022/quarter=1/2022-01-01_performance_mobile_tiles.parquet
```

Change the year, quarter and start date in the path for other quarters, then filter to the bounding box above. Column meanings are in the [Ookla Open Data repository](https://github.com/teamookla/ookla-open-data).

## Files created when you run the notebooks

Notebooks 01 to 03 write their intermediate datasets (`cells_clean.parquet`, `sites_all_technologies.csv`, `ookla_kenya_clean.parquet`, `subcounty_panel.parquet`, and others) into `processed/`. These are ignored by git because they can be regenerated in under two minutes.
