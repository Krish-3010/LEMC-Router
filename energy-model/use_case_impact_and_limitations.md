# Use Case, Impact & Limitations of this Model

## 1. Real-World Use Cases: Turning Predictions into Decisions

The Energy Model is designed to transform network telemetry into actionable insights. By combining current power estimation, short-term energy forecasting, and excess-energy flagging, the pipeline can help telecom operators make more informed decisions about energy management.

### 👨‍💻 Network Engineers — Detect What Doesn't Look Right

Task C acts as a statistical screening system that highlights network slots where observed power consumption is unusually high relative to the model's estimate for the corresponding workload.

Instead of manually inspecting every network observation, engineers can prioritize flagged slots for further investigation. These observations may indicate unusual resource configurations, inefficient operating conditions, or other issues worth examining.

**Practical value:** Focus engineering attention on suspicious energy-use patterns rather than investigating every slot equally.

### ⚡ Infrastructure & Operations Teams — Prepare for What's Coming

Task B forecasts energy requirements for the next five-minute time slot. These forecasts could help infrastructure teams anticipate changes in computing demand and plan supporting resources, such as cooling capacity and battery backup readiness.

With sufficiently reliable forecasts and appropriate operational integration, teams could move toward proactive planning instead of responding only after demand increases.

**Practical value:** Make resource-planning decisions with advance information about expected energy demand.

### 🌱 Sustainability & Energy Managers — Measure and Understand Consumption

Task A estimates current power consumption, providing a model-based reference for analysing energy-use patterns across network conditions.

Combined with validated energy measurements and appropriate emissions factors, the model's outputs could support energy-efficiency analysis and carbon-reporting workflows.

**Practical value:** Build a clearer picture of expected power consumption and identify areas that warrant an energy-efficiency review.

*Important distinction: estimated power alone is not a verified carbon-footprint measurement. Carbon reporting requires suitable energy data, emissions factors, and documented calculation methods.*

---

## 2. Potential Practical Impact

The value of the Energy Model lies not only in prediction accuracy, but also in how its outputs can guide further investigation and operational planning.

### 💰 Targeted Energy-Waste Investigation

Task C can help narrow a large set of network observations to those with unusually high positive power-estimation residuals.

For example, if approximately 1–2% of slots are flagged under the selected threshold, operators could use those observations as a focused investigation queue.

The percentage is an illustrative target from the proposed use case, not a verified finding unless the final evaluation confirms it. A flagged slot does not automatically represent wasted electricity or a hardware fault.

### 🔋 Proactive Energy and Capacity Planning

Task B provides a forecast of the energy required in the next five-minute interval. If forecasts are accurate enough for the operational setting, teams could use them to plan capacity and supporting infrastructure ahead of demand changes.

This may support better-informed cooling and backup-power planning. Actual cooling-energy savings would depend on the facility, cooling-control strategy, forecast accuracy, and the way predictions are integrated into operations.

### 📊 A More Consistent Energy Baseline

Task A establishes an estimated relationship between network workload, resource configuration, and power consumption.

That baseline can help teams compare expected and observed power use under different conditions. Repeated deviations may reveal patterns worth investigating, although additional evidence is required to determine their causes.

### Overall Potential Value

- **Operational:** Prioritize observations for investigation.
- **Planning:** Anticipate near-term energy requirements.
- **Analytical:** Compare observed power consumption with expected consumption.
- **Environmental:** Support future energy and emissions analysis when validated measurements are available.

These benefits represent potential applications. Quantified savings or improvements must be demonstrated through appropriate operational evaluation.

---

## 3. Risks and Limitations

A reliable machine-learning system must communicate not only what it can do, but also where its predictions may fail.

### 🌪️ Unpredictable Network Traffic

Network traffic can change abruptly because of sudden demand, unusual events, or shifts in user activity. Historical patterns may not contain enough information to anticipate every spike.

Consequently, Task B's forecast accuracy is limited by both the predictability of network demand and the quality of available features. A five-minute forecast should not be treated as a guarantee of future energy consumption.

**Mitigation:** Evaluate errors during high-demand periods, compare against simple forecasting baselines, and monitor performance as workload patterns change.

### 🚨 Anomaly Does Not Mean Hardware Failure

Task C identifies statistical deviations from expected power consumption. Without verified labels for hardware failures or confirmed inefficiency events, the system cannot reliably distinguish a faulty device from every other possible explanation for a high residual.

A flagged observation may reflect an unusual workload, a configuration difference, measurement noise, an omitted feature, or a genuine efficiency issue.

**Mitigation:** Treat flags as investigation signals and validate them using engineering checks, operational logs, or expert review.

### 🔗 Correlated and Redundant Features

Some network measurements, such as throughput and traffic volume, may describe closely related aspects of workload. Redundant or highly correlated features can complicate interpretation and make a model sensitive to changes in data pipelines.

Correlation alone does not necessarily invalidate a model, but it can make feature importance harder to interpret. More importantly, a feature that indirectly reveals the target or contains future information can cause data leakage.

**Mitigation:** Inspect feature relationships, verify feature definitions, remove unjustified leakage-prone inputs, and ensure that all prediction features are available at the intended prediction time.

### 🧪 Dataset Representativeness

Model performance depends on the quality and representativeness of the dataset used for development. If the dataset is synthetic or simulated, good test performance does not establish that the same accuracy will hold across real telecom infrastructure.

Differences in hardware, deployment configurations, traffic patterns, and measurement systems may affect generalization.

**Mitigation:** Document dataset provenance and limitations, then validate the models on independent data representative of the intended deployment.

### 📉 Model Error and Operational Risk

Every model makes errors. Underestimating future energy demand could lead to insufficient preparation, while overestimating it could result in unnecessary resource planning. Incorrect anomaly flags could also create extra investigation work.

**Mitigation:** Evaluate both underprediction and overprediction, establish suitable operating thresholds, and keep human review involved in consequential decisions.

---

## 4. Future Improvements

The current pipeline provides a foundation that can be extended as more data and operational requirements become available.

### 🔮 Advanced Time-Series Models

Explore sequence-based approaches such as Long Short-Term Memory (LSTM) networks or Transformers to determine whether they capture useful temporal dependencies beyond the current feature-based models.

These approaches should be compared against strong baselines using the same chronological evaluation strategy. Greater model complexity does not automatically guarantee better forecasting performance.

### 🎚️ Load-Aware Anomaly Thresholds

A single global residual threshold may behave differently under low-load and high-load conditions.

Future versions could investigate thresholds that vary by workload regime, active-node configuration, or expected-power range. Any adaptive threshold must be developed without using information from the test period.

The goal is to reduce inappropriate flags while retaining sensitivity to genuinely unusual energy consumption.

### 🌡️ Incorporating Physical and Environmental Telemetry

Additional measurements could help explain power consumption more accurately, including:

- Ambient server-room temperature.
- Cooling-system power and fan speeds.
- Hardware type and configuration.
- CPU frequency and operating state.
- Server or network-function health indicators.

These features may help distinguish changes caused by workload from changes caused by the surrounding environment or hardware operating conditions.

Their value would need to be verified using suitable data.

### 🧠 Explainable Predictions

Feature-attribution methods could help engineers understand which inputs contribute most to a power estimate or forecast.

For Task C, explanations could make flagged observations easier to investigate by showing how actual power differs from expected power and which workload or resource characteristics were associated with the estimate.

Explanations should be interpreted as model-based evidence, not proof of physical causation.

### 📡 Operational Monitoring and Deployment

A future deployment could integrate the models into a dashboard that displays:

- Current estimated power.
- Next-interval energy forecasts.
- Forecast errors and confidence or prediction intervals, where validated.
- Flagged observations and their residual magnitudes.
- Historical trends and model-performance monitoring.

Before operational use, the system would need testing for reliability, latency, data quality, model drift, and safe handling of missing or abnormal inputs.

---

## 5. Final Conclusion

The Energy Model demonstrates how machine learning can support three complementary aspects of network energy management: estimating current power consumption, forecasting near-term energy requirements, and identifying observations with unexpectedly high power use.

Its greatest practical value is not simply producing predictions, but turning those predictions into **better-informed planning and more targeted engineering investigations**.

However, reliable real-world impact depends on more than model metrics. Dataset representativeness, leakage prevention, chronological evaluation, anomaly validation, and operational testing are all essential before the system can be trusted to guide production decisions.

**The guiding principle is simple: predict energy demand, investigate unusual consumption, and validate every operational benefit with evidence.**