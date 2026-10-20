> **Reading Tip:** Open this file in VS Code's Markdown Preview (`Cmd + Shift + V`) or view it directly on GitHub for better readability.

# 02: Data Acquisition and Understanding

## 1. Purpose of This Stage

The purpose of this stage is to understand the dataset before writing the data-preprocessing and modelling code.

This stage investigates:

- The source and format of the dataset.
- The meaning of each row and column.
- The number of records and features.
- The data types and units.
- The temporal structure of the observations.
- Data-quality issues and potential limitations.
- The suitability of the dataset for Tasks A, B, and C.


## 2. Dataset Identification

| Property                      | Preliminary information                       |
| ----------------------------- | --------------------------------------------- |
| Dataset name                  | 5G Core Traffic and Energy Management Dataset |
| File format                   | CSV                                           |
| Number of records             | Approximately 15,000                          |
| Number of columns             | 49                                            |
| Time coverage                 | January 1 to February 22, 2024                |
| Sampling interval             | Five minutes                                  |
| Original source               | Kaggle                                        |


## 3. Data Source and Provenance

The dataset's source, ownership, licence, and collection method must be identified from reliable documentation wherever possible.

The dataset could contain real measurements, simulated observations, or synthetically generated records. Its exact origin must not be assumed from the column names alone.

The following information should be documented if available:

- Original dataset title and provider.
- Source URL or citation.
- Download date and dataset version.
- Licence or permitted use.
- Data-generation or collection method.
- Whether the values are real, simulated, or synthetic.
- Known limitations and intended use.

If this information is unavailable, explicitly document it as unknown. If the dataset is synthetic or simulated, it may still be useful for demonstrating machine-learning techniques, but successful results would not prove that the same performance can be achieved in a live telecom network.

## 4. Understanding What Each Row Represents

The preliminary interpretation is that each row represents a network observation associated with a timestamp and a set of traffic, resource, and energy measurements.

However, the precise unit of observation must be established from the dataset documentation and its structure.

A row might describe:

- An aggregate network observation.
- A particular network slice.
- A network function.
- A deployment zone.
- Another defined entity or aggregation.

It must not be assumed that every row represents a different physical network.

To establish the meaning of a row, inspect:

1. The dataset documentation and feature descriptions.
2. Whether timestamps repeat across multiple rows.
3. Whether fields identify services, slices, network functions, zones, or nodes.
4. Whether repeated timestamps represent different entities or duplicate records.
5. Whether an entity identifier is available to distinguish separate time series.


## 5. Initial Feature Groups

The following groups are based on candidate column names discussed during preliminary analysis. Confirm the existence and meaning of each column in the actual CSV before using it.

### 5.1 Time-Related Features

Possible examples:

- `Record_ID`
- `Timestamp`
- `Hour`
- `Day_of_Week`
- `Month`

**Purpose:** Represent when an observation was recorded and help investigate daily, weekly, or longer-term patterns.

`Record_ID` is generally an identifier rather than a meaningful predictive feature. `Timestamp` should be parsed into an appropriate datetime type.

### 5.2 Network and Deployment Context

Possible examples:

- `Service_Class`
- `Network_Slice`
- `Network_Function`
- `Deployment_Zone`

**Purpose:** Describe the service, slice, function, or deployment context associated with an observation.

These fields may be categorical and could help explain differences in resource usage and energy consumption.

### 5.3 Traffic and Demand

Possible examples:

- `Traffic_Volume_Mbps`
- `Active_User_Count`
- `Packet_Arrival_Rate_Kpps`
- `Future_Traffic_Demand_Mbps`

**Purpose:** Represent network workload and demand.

Fields labelled `Future` require special attention. A future-demand value may be used only if it would genuinely be available at the time the prediction is made, for example as the output of a separate forecast already available to the system.

### 5.4 Quality of Service and Congestion

Possible examples:

- `Latency_ms`
- `Jitter_ms`
- `Packet_Loss_Rate_Pct`
- `Congestion_Index`
- `Future_Congestion_Index`

**Purpose:** Describe network performance and congestion.

These features primarily relate to the separate Congestion Model. They should only be used in the Energy Model if their meaning, timing, availability, and relevance are established and their use does not cause data leakage.

### 5.5 Computing and Network Resources

Possible examples:

- `CPU_Utilization_Pct`
- `Memory_Utilization_Pct`
- Storage-utilization fields, if present.
- `VNF_Instance_Count`
- `Active_Core_Nodes`
- Bandwidth or edge/cloud usage fields, if present.

**Purpose:** Describe resource utilization and infrastructure configuration that may be associated with power and energy use.

Check the units, expected ranges, and whether each value represents a measurement, configuration, or generated value.

### 5.6 Energy, Efficiency, Cost, and Carbon

Possible examples:

- `Total_Power_Watt`
- `Energy_Consumption_kWh`
- `Energy_Efficiency_Mbps_per_kWh`
- `Carbon_Emission_kg`
- `Operational_Cost_USD`
- `Resource_Utilization_Efficiency_Pct`

**Purpose:** Describe power, energy, and related operational outcomes.

These fields may be mathematically related. Their definitions and calculation formulas must be checked before deciding which fields can safely be used as model inputs.

### 5.7 Precomputed Predictions and Management Outputs

Possible examples:

- `Predicted_Energy_Requirement_kWh`
- `Energy_Management_Status`

**Purpose:** These fields may represent existing model predictions, management decisions, or status labels.

Their definitions and generation methods must be established. A precomputed prediction is not automatically a ground-truth target, and an existing status field is not automatically a valid label for anomaly detection.

## 6. Expected Data Types

The actual data types must be inspected from the CSV.

| Data category | Examples                               | Checks requiredrequired                                    |              |
| ------------- | -------------------------------------- | ---------------------------------------------------------- |
| Datetime      | `Timestamp`                            | Valid parsing, date range, ordering                        |
| Numeric       | Traffic, utilization, power, energy    | Numeric conversion, units, missing values, valid ranges    |
| Categorical   | Service, slice, function, zone, status | Category counts, inconsistent spelling, missing categories |
| Identifier    | `Record_ID` or entity identifier       | Uniqueness where expected and suitability for modelling    |

Some numeric columns may be stored as text. These must be investigated before preprocessing. Values outside expected ranges should be checked against the data definitions instead of being removed automatically.

## 7. Candidate Targets for the Three Tasks

### 7.1 Task A — Current Power Estimation

- **Candidate target:** `Total_Power_Watt`.
- **Problem type:** Regression.
- **Expected unit:** Watts (W).

Before using this field, confirm that it represents the intended power measurement and check for missing, invalid, or target-derived values.

### 7.2 Task B — Next-Interval Energy Forecasting

- **Candidate actual-energy field:** `Energy_Consumption_kWh`.
- **Problem type:** Time-series regression or forecasting.
- **Expected output:** Energy in kWh for the next interval.

If the field represents actual energy consumption for a defined interval, the future target can potentially be created by aligning each observation with the actual energy value for the next interval.

Before constructing this target:

1. Parse and sort timestamps correctly.
2. Confirm the meaning of one row.
3. Identify separate entities or time series, if applicable.
4. Check timestamp gaps and sampling consistency.
5. Shift the actual-energy field only within the correct time series.

`Predicted_Energy_Requirement_kWh` should not automatically be used as the target because it may already contain a prediction rather than the actual energy outcome.

### 7.3 Task C — Unusual Energy Consumption Detection

- **Output:** An anomaly score or flag.
- **Problem type:** Anomaly detection, or classification if reliable labels exist.

First determine whether the dataset contains a trustworthy label identifying abnormal energy use.

If it does not, a documented baseline or anomaly-detection method must be developed. Any threshold-based flag should be described as a project-defined rule until independently validated.

## 8. Initial Data-Quality Assessment

The following checks must be performed on the actual CSV:

1. Count the total number of records and columns.
2. List all column names.
3. Inspect inferred data types.
4. Count missing values per column.
5. Identify exact duplicate rows.
6. Check uniqueness of identifier fields where appropriate.
7. Parse timestamps and determine the minimum and maximum dates.
8. Inspect timestamp differences and sampling-interval consistency.
9. Investigate repeated timestamps and identify possible entity keys.
10. Examine numeric ranges, invalid values, outliers, and inconsistent units.
11. Inspect category counts and unexpected category values.
12. Examine relationships among power, energy, efficiency, cost, and carbon fields.
13. Identify precomputed outputs and features containing future information.

No conclusion such as “there are no missing values” or “all timestamps are unique” should be recorded until the checks have been run. Repeated timestamps can be valid when several entities are observed at the same time.

## 9. Potential Data Leakage and Redundancy

Several feature groups require special investigation before modelling.

### 9.1 Precomputed Predictions

A column containing an existing prediction may encode information related to the target. Its source and availability must be established before using it as an input.

### 9.2 Future-Related Fields

Columns describing future traffic, congestion, or energy can leak future information if they would not be available at prediction time.

### 9.3 Derived Energy Indicators

Energy efficiency, carbon emissions, and operational cost may be calculated from energy consumption or power. If so, including them may reveal the target rather than help the model learn a legitimate predictive relationship.

### 9.4 Relationship Between Power and Energy

Power and energy are related, but the conversion depends on the measurement interval and the meaning of the power value.

If power is constant or represents average power over five minutes:

`Energy_kWh = Power_W × 5 / 60,000`

This relationship should not be assumed if the power value is instantaneous or measured over a different interval.

### 9.5 Time-Series Splitting

Randomly splitting time-series observations can result in training on later observations and testing on earlier ones. Forecasting evaluation should instead preserve chronological order.

### 9.6 Incorrect Target Alignment

Shifting rows without grouping by entity can create incorrect future targets. The row meaning and entity identifiers must be understood before target construction.

These risks should be documented and addressed during preprocessing and feature engineering.

## 10. Potential Uses of the Dataset

If the dataset's definitions, quality, and temporal structure support the proposed tasks, it may be used to:

- Estimate power consumption from network workload and resource information.
- Forecast energy consumption for the next interval using past and currently available information.
- Identify observations whose energy consumption differs from an expected baseline.

Suitability depends on the availability of valid target columns, reliable timestamps, a clear unit of observation, and enough usable observations.

If the necessary information is absent, the affected task should be revised rather than relying on invented labels or unsupported assumptions.

## 11. Conclusion

second step establishes the initial understanding of the dataset and identifies the checks needed before model development.

The preliminary dataset size, column names, date range, and sampling interval must be reconciled with the actual CSV. Once the schema, row meaning, target feasibility, and basic data quality have been verified, the project can proceed to Stage 3: Data Preparation.
