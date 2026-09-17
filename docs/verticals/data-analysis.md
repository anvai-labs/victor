# Data Analysis Vertical

The Data Analysis vertical provides comprehensive data exploration, statistical analysis, visualization, and machine learning capabilities. It is designed to compete with ChatGPT Data Analysis, Claude Artifacts, and Jupyter AI.

## Overview

The Data Analysis vertical (`verticals/victor-dataanalysis/victor_dataanalysis/`) enables end-to-end data science workflows from data loading through visualization and machine learning. It integrates with pandas, matplotlib, seaborn, plotly, scipy, and scikit-learn for comprehensive analysis capabilities.

### Key Use Cases

- **Exploratory Data Analysis (EDA)**: Profile datasets, identify patterns, and understand distributions
- **Data Cleaning**: Handle missing values, outliers, type conversions, and normalization
- **Statistical Analysis**: Hypothesis testing, correlation analysis, and regression
- **Visualization**: Create charts, dashboards, and interactive plots
- **Machine Learning**: Classification, regression, clustering, and model evaluation
- **Report Generation**: Automated insights and documentation

## Available Tools

The Data Analysis vertical uses the following tools from `victor.tools.tool_names`:

| Tool | Description |
|------|-------------|
| `read` | Read data files (CSV, Excel, JSON, Parquet) |
| `write` | Write results and reports |
| `edit` | Modify existing files |
| `ls` | List directory contents |
| `shell` | Execute Python scripts for analysis |
| `grep` | Search through data files |
| `code_search` | Semantic search for analysis patterns |
| `overview` | Understand project structure |
| `graph` | Dependency analysis |
| `web_search` | Find datasets and documentation |
| `web_fetch` | Fetch external data |

## Available Workflows

### 1. EDA Pipeline (`eda_pipeline.yaml`)

Comprehensive exploratory data analysis:

```yaml
workflows:
  eda_pipeline:
    nodes:
      - load_data           # Load CSV/Parquet/JSON
      - validate            # Check data quality
      - basic_stats         # Descriptive statistics
      - data_cleanup        # Handle missing/invalid data (Agent)
      - parallel_analysis   # Correlations + distributions + anomalies
      - deep_dive           # Investigate patterns (Agent)
      - visualizations      # Generate charts (parallel)
      - human_review        # HITL approval gate
      - report              # Generate insights report (Agent)
```

**Key Features**:
- Parallel execution of correlation, distribution, and anomaly analysis
- Human-in-the-loop review before final report
- Retry loop for data cleanup until quality threshold met
- Configurable quality thresholds and visualization formats

**Configuration**:
```yaml
quality_threshold: 0.8        # Data quality score to skip cleanup
cleanup_attempts_max: 3       # Retry limit for cleanup
correlation_method: pearson   # pearson, spearman, kendall
histogram_bins: 50            # Distribution binning
anomaly_contamination: 0.05   # Expected outlier ratio
output_format: png            # png, svg, pdf, html
```

### 2. ML Pipeline (`ml_pipeline.yaml`)

End-to-end machine learning workflow:

```yaml
workflows:
  ml_pipeline:
    nodes:
      - load_train_data      # Load training data
      - validate_data        # Quality checks
      - feature_engineering  # Parallel feature processing
      - feature_selection    # Select best features
      - parallel_training    # Train multiple models
      - evaluate_models      # Compare model performance
      - analyze_results      # Interpret metrics (Agent)
      - deploy_preparation   # Save model and API spec
```

**Key Features**:
- Parallel feature engineering (numeric, categorical, text)
- Multiple model training (RandomForest, XGBoost, LightGBM, Neural Network)
- Cross-validation with early stopping
- Model comparison and selection
- ONNX export for deployment

**Configuration**:
```yaml
cv_folds: 5                       # Cross-validation folds
early_stopping_patience: 10       # Epochs without improvement
best_model_threshold: 0.85        # Auto-deploy threshold
model_output_format: pickle,onnx  # Serialization formats
max_training_time: 900s           # Per-model timeout
```

### 3. Statistical Analysis (`statistical_analysis.yaml`)

Hypothesis testing and statistical modeling:

```yaml
workflows:
  statistical_analysis:
    nodes:
      - load_data             # Load dataset
      - normality_tests       # Shapiro-Wilk, D'Agostino
      - hypothesis_tests      # t-test, ANOVA, chi-square
      - regression_analysis   # OLS, logistic, polynomial
      - confidence_intervals  # Compute CIs
      - effect_sizes          # Cohen's d, eta-squared
      - report_results        # Statistical report (Agent)
```

### 4. Data Cleaning (`data_cleaning.yaml`)

Automated data preparation:

```yaml
workflows:
  data_cleaning:
    nodes:
      - profile_data          # Identify data issues
      - handle_missing        # Imputation strategies
      - fix_types             # Type conversion
      - remove_duplicates     # Deduplication
      - handle_outliers       # Outlier treatment
      - normalize             # Scaling and normalization
      - validate_output       # Quality verification
```

### 5. AutoML Pipeline (`automl_pipeline.yaml`)

Automated machine learning:

```yaml
workflows:
  automl_pipeline:
    nodes:
      - auto_feature_engineering  # Automatic feature creation
      - model_search             # Hyperparameter optimization
      - ensemble_creation        # Model ensembling
      - final_evaluation         # Performance assessment
```

## Stage Definitions

The Data Analysis vertical progresses through these stages:

| Stage | Description | Primary Tools |
|-------|-------------|---------------|
| `INITIAL` | Understanding data and goals | `read`, `ls`, `overview` |
| `DATA_LOADING` | Loading and validating data | `read`, `shell`, `write` |
| `EXPLORATION` | Profiling and statistics | `shell`, `read`, `write` |
| `CLEANING` | Data transformation | `shell`, `write`, `edit` |
| `ANALYSIS` | Statistical analysis and modeling | `shell`, `write`, `read` |
| `VISUALIZATION` | Creating charts | `shell`, `write` |
| `REPORTING` | Generating insights | `write`, `edit`, `read` |
| `COMPLETION` | Finalizing deliverables | `write`, `read` |

## Key Features

### Data Format Support

Comprehensive data format handling:

| Format | Library | Notes |
|--------|---------|-------|
| CSV | pandas | Auto-detect delimiters |
| Excel | openpyxl | Multiple sheets |
| JSON | pandas | Nested structures |
| Parquet | pyarrow | Columnar storage |
| SQL | sqlalchemy | Multiple databases |
| Feather | pyarrow | Fast I/O |

### Visualization Capabilities

Multiple visualization libraries:

```python
# matplotlib - Publication-quality static plots
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(data['x'], data['y'])

# seaborn - Statistical visualizations
import seaborn as sns
sns.heatmap(correlation_matrix, annot=True)

# plotly - Interactive plots
import plotly.express as px
fig = px.scatter(df, x='col1', y='col2', color='category')
```

### Statistical Methods

Built-in statistical analysis:

- **Descriptive**: Mean, median, std, quartiles, skewness, kurtosis
- **Correlation**: Pearson, Spearman, Kendall
- **Hypothesis Testing**: t-test, ANOVA, chi-square, Mann-Whitney
- **Regression**: OLS, logistic, polynomial, ridge, lasso
- **Distribution**: Normality tests, KDE, histograms

### Machine Learning Integration

scikit-learn compatible workflow:

```python
# Supported models
- RandomForestClassifier/Regressor
- XGBClassifier/Regressor
- LGBMClassifier/Regressor
- LogisticRegression
- SVM, KNN, Neural Networks

# Evaluation metrics
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Regression: MSE, RMSE, MAE, R-squared
- Clustering: silhouette, Davies-Bouldin
```

### Capability Providers

The Data Analysis vertical provides these capabilities:

| Capability | Description |
|------------|-------------|
| `data_quality` | Data quality rules and validation |
| `visualization_style` | Chart styling and configuration |
| `statistical_analysis` | Statistical method configuration |
| `ml_pipeline` | ML training and evaluation settings |
| `data_privacy` | Anonymization and PII detection |

## Configuration Options

### Vertical Configuration

```python
from victor_dataanalysis.assistant import DataAnalysisAssistant

# Get system prompt
prompt = DataAnalysisAssistant.get_system_prompt()

# Get tiered tools
tiered_tools = DataAnalysisAssistant.get_tiered_tool_config()

# Access capability provider
capabilities = DataAnalysisAssistant.get_capability_provider()
```

### Analysis Configuration

```yaml
# Data quality settings
quality:
  missing_threshold: 0.1     # Max 10% missing per column
  duplicate_threshold: 0.01  # Max 1% duplicates
  type_consistency: true     # Enforce consistent types

# Visualization settings
visualization:
  style: seaborn-whitegrid
  color_palette: viridis
  figure_size: [10, 6]
  dpi: 150
  output_format: png

# ML settings
ml:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
  scoring: f1_weighted
```

### Workflow Parameters

```yaml
# Common workflow settings
llm_config:
  temperature: 0.2      # Precise analysis
  max_tokens: 4000

tool_budget: 30         # Tool calls per node
timeout: 300            # Node timeout
```

## Example Usage

Use the public factory with the `dataanalysis` vertical. Install the corresponding
`victor-dataanalysis` package if it is not already available in your environment.

```python
import asyncio
from victor.framework import Agent

async def main():
    async with await Agent.create(
        vertical="dataanalysis", provider="ollama", model="llama3.1:8b"
    ) as agent:
        result = await agent.run("Summarize the columns and missing values in data.csv")
        print(result.content)

asyncio.run(main())
```

For a named workflow supplied by the installed vertical, call
`await agent.run_workflow(workflow_name, context={...})` on the configured agent.
Use the installed package's workflow catalog to select a name and its expected input
keys. A bare workflow compiler does not create the agent runtime or provider.
See [Python API](../reference/api/python-api.md) and
[workflow execution](../tutorials/create-workflow.md).

## Integration with Other Verticals

The Data Analysis vertical integrates with:

- **RAG**: Build knowledge bases from analysis reports
- **Research**: Statistical analysis for research findings
- **Coding**: Generate analysis scripts and notebooks

## File Structure

```
verticals/victor-dataanalysis/victor_dataanalysis/
├── assistant.py          # DataAnalysisAssistant definition
├── capabilities.py       # Capability providers
├── mode_config.py        # Mode configurations
├── prompts.py            # Prompt templates
├── safety.py             # Safety checks for data ops
├── tool_dependencies.py  # Tool dependency configuration
├── workflows/
│   ├── eda_pipeline.yaml       # EDA workflow
│   ├── ml_pipeline.yaml        # ML training workflow
│   ├── statistical_analysis.yaml  # Stats workflow
│   ├── data_cleaning.yaml      # Cleaning workflow
│   └── automl_pipeline.yaml    # AutoML workflow
├── handlers.py           # Compute handlers
├── escape_hatches.py     # Complex condition logic
├── rl.py                 # Reinforcement learning config
└── teams.py              # Multi-agent team specs
```

## Best Practices

1. **Profile first**: Always run data profiling before analysis
2. **Handle missing data**: Explicitly document imputation strategies
3. **Validate assumptions**: Check statistical assumptions before tests
4. **Visualize distributions**: Understand data before modeling
5. **Use cross-validation**: Never evaluate on training data
6. **Document findings**: Include methodology and limitations
7. **Protect PII**: Anonymize sensitive data before analysis

## Privacy and Ethics

The Data Analysis vertical includes safeguards:

- **PII Detection**: Automatic detection of personally identifiable information
- **Anonymization**: Built-in column anonymization
- **Bias Detection**: Flag potential biases in data
- **Transparency**: Document all transformations and assumptions
- **Limitations**: Note statistical and methodological limitations

## Code Standards

When generating analysis code:

```python
# Always use pandas for data manipulation
import pandas as pd
df = pd.read_csv("data.csv")

# Include comments explaining methodology
# Calculate Pearson correlation between features
correlation = df.corr(method='pearson')

# Handle missing data explicitly
df['column'] = df['column'].fillna(df['column'].median())

# Use descriptive variable names
monthly_revenue_by_region = df.groupby('region')['revenue'].sum()

# Save intermediate results for reproducibility
df.to_parquet("cleaned_data.parquet")
```
