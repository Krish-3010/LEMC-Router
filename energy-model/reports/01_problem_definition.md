> **Reading Tip:** Open this file in VS Code's Markdown Preview (`Cmd + Shift + V`) or view it directly on GitHub for better readability.

# 01: Problem Definition and Use Case

## 1. Project Overview

**Project Name:** Energy Model for 5G Core Traffic and Energy Management

The Energy Model aims to use network traffic, resource-utilization, and historical energy information to estimate current power consumption, forecast future energy requirements, and identify potentially unusual energy consumption patterns.

This project is one component of a larger project that also includes a Congestion Model. The Energy Model focuses on power and energy outcomes, while the Congestion Model primarily addresses network congestion and quality of service (QoS).

## 2. Problem Statement

Operating a 5G core network requires energy to run computing infrastructure and network functions. Energy demand can vary with traffic volume, active users, service types, virtual network function (VNF) instances, active core nodes, and resource utilization.

If operators can estimate current power consumption and predict future energy requirements, they can make better-informed decisions about capacity planning and resource allocation. They can also identify periods when actual energy consumption is unusually high compared with an appropriate baseline.

The Energy Model will investigate three tasks:

1. **Task A — Current Power Estimation:** Estimate current power consumption using information available for the same time interval.
2. **Task B — Future Energy Forecasting:** Predict energy consumption for the next time interval using information available at the time of prediction.
3. **Task C — Unusual Energy Consumption Detection:** Identify observations where energy consumption is unexpectedly high relative to workload and resource configuration.

The feasibility and exact targets of these tasks will be confirmed during dataset understanding and data preparation.

## 3. Project Objectives

### 3.1 Task A: Current Power Estimation

Develop a regression model to estimate current power consumption from relevant network and resource features.

- **Candidate target:** `Total_Power_Watt`
- **Expected output:** Estimated power consumption in watts (W).
- **Purpose:** Understand the relationship between network workload, resource utilization, and power consumption.

The target column must be verified against the actual dataset before model development.

### 3.2 Task B: Future Energy Forecasting

Develop a forecasting model to predict energy consumption for the next time interval using current and historical information.

- **Candidate actual-energy column:** `Energy_Consumption_kWh`
- **Expected output:** Predicted energy consumption for the next interval in kilowatt-hours (kWh).
- **Purpose:** Help anticipate future energy requirements before the next interval occurs.

If the dataset contains actual energy measurements, the future target may be constructed by aligning each observation with the actual energy value in the next interval. This requires correctly ordered timestamps and, where applicable, separate time series for different entities.

### 3.3 Task C: Unusual Energy Consumption Detection

Identify observations where actual energy consumption is higher than expected for the associated workload and resource configuration.

- **Expected output:** An anomaly score or a clearly defined flag.
- **Purpose:** Highlight observations that may deserve investigation for possible inefficiency.

A documented baseline, defensible threshold, or trustworthy anomaly labels must be established. Arbitrary labels must not be presented as verified ground truth.

## 4. Intended Stakeholders

The intended stakeholders are:

- **Network Operations Teams:** Understand expected energy requirements under different network conditions.
- **Energy and Sustainability Managers:** Analyse energy consumption and investigate potential improvement opportunities.
- **Capacity-Planning Teams:** Anticipate energy requirements when planning computing and network capacity.
- **Infrastructure and Orchestration Teams:** Use energy forecasts as an input when considering resource allocation.
- **Researchers and Project Evaluators:** Assess model performance, reliability, and practical usefulness.

The model provides information to support decisions. It does not automatically reconfigure a network or guarantee energy savings.

## 5. Proposed Inputs and Outputs

The final features depend on the actual dataset schema, column definitions, and whether each value would be available when a prediction is made.

| Task | Candidate inputs | Expected output |
|---|---|---|
| Task A: Current power estimation | Time features, service or slice information, traffic, active users, CPU and memory utilization, VNF count, active nodes | Estimated power in watts |
| Task B: Next-interval energy forecasting | Current and recent traffic, resource utilization, historical actual energy, and other information available at prediction time | Forecast energy in kWh for the next interval |
| Task C: Unusual energy detection | Workload and resource information, historical data, and expected-energy baseline | Anomaly score or flag |

Potential feature names discussed during preliminary analysis include `Timestamp`, `Traffic_Volume_Mbps`, `Active_User_Count`, `CPU_Utilization_Pct`, `Memory_Utilization_Pct`, `VNF_Instance_Count`, `Active_Core_Nodes`, `Total_Power_Watt`, and `Energy_Consumption_kWh`.

These names must be checked against the supplied CSV before being used in the implementation.

## 6. Project Scope

### Included in the project

- Understanding and validating the dataset.
- Identifying relevant features and target variables.
- Data cleaning and preparation.
- Exploratory Data Analysis (EDA).
- Feature engineering and feature selection.
- Establishing baseline models.
- Training and comparing suitable machine-learning models.
- Hyperparameter tuning.
- Model validation and evaluation.
- Documenting limitations and potential real-world applications.

### Outside the current scope

- Directly controlling or reconfiguring a live 5G network.
- Claiming measured energy or carbon savings without real-world validation.
- Treating predictions as guaranteed outcomes.

## 7. Constraints and Modelling Assumptions

### 7.1 Prevention of Data Leakage

Only information genuinely available at prediction time should be used as model input. Future measurements and target-derived fields must not reveal the answer to the model.

### 7.2 Time-Aware Evaluation

For Task B, the dataset must be split chronologically into training, validation, and testing periods. Randomly mixing past and future observations may produce an unrealistic estimate of forecasting performance.

### 7.3 Target Validation

The meaning, units, and measurement interval of every target must be confirmed before training.

### 7.4 Derived and Precomputed Features

Fields related to energy efficiency, carbon emissions, operational cost, existing predictions, and future demand must be investigated. Some may be calculated from the target or contain information unavailable at prediction time.

### 7.5 Difference Between Power and Energy

Power is measured in watts (W), whereas energy is measured in kilowatt-hours (kWh). Their relationship depends on the duration of energy consumption and the meaning of the power measurement. An instantaneous power reading should not automatically be treated as average power over an entire interval.

### 7.6 Dataset Representativeness

The dataset's source, collection method, and representativeness must be established where possible. If the dataset is synthetic or simulated, model performance on it will not, by itself, establish real-world effectiveness.

## 8. Expected Benefits and Impact

If the models are sufficiently accurate and the dataset represents the intended application, the Energy Model could help operators:

- Anticipate energy requirements instead of reacting only after demand changes.
- Support network capacity and resource planning.
- Compare observed energy consumption with expected consumption.
- Identify intervals that merit investigation for potential inefficiency.
- Support future studies of electricity costs and carbon emissions.

These are potential benefits, not demonstrated results. Actual energy savings and environmental impact would need to be measured in an appropriate operational setting.

## 9. Success Criteria

The project will aim to:

1. Define each modelling task and its target unambiguously.
2. Document the dataset and its quality.
3. Prevent data leakage and respect prediction-time availability.
4. Establish baseline models before trying more advanced approaches.
5. Evaluate models on appropriate held-out data.
6. Use metrics suitable for regression, forecasting, or anomaly detection.
7. Explain model errors, limitations, and practical interpretation.
8. Maintain reproducible code and documentation.

Numerical performance thresholds will be selected after examining the dataset and baseline results.

## 10. Conclusion

first step is establishes the problem, stakeholders, objectives, proposed inputs and outputs, scope, constraints, and expected impact of the Energy Model.

The next stage will examine the actual dataset to verify its structure, understand its features, assess data quality, and determine whether it supports all three proposed tasks.
