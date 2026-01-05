# Trading Risk System — Project Context  
### Personal PM Toolkit: Strategy Design, Position Sizing & Portfolio Construction

This document defines the architecture, data model, and development blueprint for a personal portfolio‑management toolkit.  
It is designed to support discretionary macro/commodity trading at a hedge fund, where firm‑level risk (VaR, drawdown, stress tests) is already provided.

PyCharm AI should use this file as the authoritative reference when generating or modifying code.

---

# 1. Project Purpose

This system is a decision‑support toolkit for a discretionary PM.  
It focuses on:

- Strategy design & structuring  
- Conviction‑based position sizing  
- Portfolio construction & diversification  
- Regime‑aware allocation  
- Correlation‑aware sizing  
- Excel front‑end for inputs and dashboards  

Not included (handled by firm risk systems):
- VaR calculation  
- Drawdown calculation  
- Stress testing  
- Regulatory risk reporting  

---

# 2. Core Components

## 2.1 Strategy Library  
A structured database of all active and candidate strategies.

Each strategy includes:

- StrategyName  
- Theme / Catalyst  
- Bucket (Directional, Spread, RV, Volatility, Systematic, etc.)  
- SignalType (weather, macro, flow, seasonal, etc.)  
- Conviction (Low, Medium, High, Exceptional)  
- ExpectedReturn (subjective PM estimate)  
- ExpectedRisk (volatility or uncertainty score)  
- RegimeFit (Calm, Normal, Event)  
- TimeHorizon  
- Notes  

---

## 2.2 Position Sizing Engine  
A Python module that converts strategy inputs into recommended position sizes.

Inputs:
- Conviction tier  
- Expected return  
- Expected risk  
- Regime  
- Correlation to existing book  
- PM‑defined sizing rules  
- Optional: fractional Kelly sizing  

Outputs:
- Recommended position size  
- Recommended capital allocation  
- Recommended risk contribution  
- Hedge ratios (if applicable)  

---

## 2.3 Portfolio Construction Engine  
Combines all active strategies into a coherent book.

Responsibilities:
- Aggregate recommended sizes  
- Apply diversification rules  
- Apply bucket caps  
- Apply regime adjustments  
- Apply correlation adjustments  
- Produce a portfolio heat map  
- Produce bucket allocation summary  
- Produce conviction distribution  

---

## 2.4 Excel Front‑End  
A user‑friendly interface for:

- Entering strategy inputs  
- Viewing recommended sizes  
- Viewing portfolio construction outputs  
- Viewing bucket allocations  
- Viewing correlation heat maps  
- Viewing scenario adjustments  

Python modules feed data into Excel and read inputs from it.

---

# 3. Data Model

## 3.1 strategies.csv  
```
StrategyName
Bucket
SignalType
Conviction
ExpectedReturn
ExpectedRisk
RegimeFit
TimeHorizon
Notes
```

## 3.2 portfolio.csv  
```
StrategyName
RecommendedSize
CapitalAllocation
RiskContribution
CorrelationAdjustment
RegimeAdjustment
FinalSize
```

## 3.3 correlations.csv  
Matrix of pairwise correlations between strategies or instruments.

## 3.4 sizing_rules.csv  
Defines PM‑specific rules:

```
ConvictionTier, BaseSizeMultiplier
Regime, RegimeAdjustment
Bucket, MaxAllocationPct
KellyFraction
```

---

# 4. Python Modules

## 4.1 strategy_engine.py  
Responsibilities:
- Load strategy library  
- Validate inputs  
- Apply regime filters  
- Score strategies (optional)  
- Output structured strategy objects  

## 4.2 sizing_engine.py  
Responsibilities:
- Apply conviction → size mapping  
- Apply expected return/risk adjustments  
- Apply regime adjustments  
- Apply correlation adjustments  
- Apply Kelly fraction (optional)  
- Output recommended sizes  

## 4.3 portfolio_optimizer.py  
Responsibilities:
- Combine all recommended sizes  
- Apply bucket caps  
- Apply diversification rules  
- Apply correlation matrix  
- Compute final portfolio weights  
- Produce allocation summaries  

## 4.4 excel_interface.py  
Responsibilities:
- Read strategy inputs from Excel  
- Write recommended sizes back to Excel  
- Update dashboard sheets  

---

# 5. main.py Workflow

The main script orchestrates the daily workflow:

1. Load strategy inputs (Excel or CSV)  
2. Run strategy_engine  
3. Run sizing_engine  
4. Run portfolio_optimizer  
5. Export results to Excel dashboard  
6. Print a clean summary in terminal  

---

# 6. Excel Dashboard Structure

## Sheet: Strategy Input
- Editable fields for strategy design  
- Dropdowns for conviction, regime, bucket  

## Sheet: Sizing Output
- Recommended sizes  
- Capital allocation  
- Risk contribution  

## Sheet: Portfolio Dashboard
- Bucket allocation chart  
- Conviction distribution  
- Correlation heat map  
- Regime exposure  

## Sheet: Trade Planner
- Select strategies  
- Generate trade tickets (optional)  

---

# 7. Future Extensions

- Strategy scoring model  
- Machine‑learning‑based conviction calibration  
- Automated trade ticket generation  
- Integration with OMS export  
- Quarterly strategy performance analytics

---

# 9. Expected PnL Engine (Regime → Environment → Expected Return → Expected PnL)

This section defines the logic for estimating expected PnL based on market conditions, portfolio structure, and bucket-level expected returns. PyCharm AI should use this as the authoritative reference when implementing the expected PnL engine in Python.

---

## 9.1 Purpose

The Expected PnL Engine provides a forward-looking estimate of what the portfolio *should* earn per day/week given:

- current market regime  
- environment factors  
- bucket-level capital allocation  
- bucket-level expected returns  

This allows the PM to compare **actual vs expected performance**, identify underperformance, and scale risk appropriately.

---

## 9.2 Regime Classification Logic

Regimes are determined using observable market metrics:

- Volatility index (e.g., VIX, OVX, MOVE, or custom commodity vol index)  
- Cross-sectional dispersion of returns  
- Optional: correlation regime  

Define three regimes:

- **Calm**  
- **Normal**  
- **Event**

### Regime rules (initial version)

Let:

- \( V_z \) = z-score of volatility index  
- \( D_z \) = z-score of cross-sectional dispersion  

Rules:

- **Calm:**  
  - \( V_z < 0 \) AND \( D_z < 0 \)

- **Event:**  
  - \( V_z > 1 \) OR \( D_z > 1 \)

- **Normal:**  
  - All other cases

These thresholds can be refined later.

---

## 9.3 Environment Factor Model

Each bucket behaves differently in each regime.  
Environment factors adjust expected returns up or down depending on the regime.

### Environment Factor Table

| Regime   | Bucket        | EnvFactor |
|----------|---------------|-----------|
| Calm     | Discretionary | 0.7       |
| Calm     | Spread        | 1.1       |
| Calm     | RV            | 1.3       |
| Calm     | Volatility    | 0.6       |
| Calm     | Systematic    | 1.2       |
| Normal   | Discretionary | 1.0       |
| Normal   | Spread        | 1.0       |
| Normal   | RV            | 1.0       |
| Normal   | Volatility    | 1.0       |
| Normal   | Systematic    | 1.0       |
| Event    | Discretionary | 1.4       |
| Event    | Spread        | 1.1       |
| Event    | RV            | 0.9       |
| Event    | Volatility    | 1.5       |
| Event    | Systematic    | 0.8       |

These values are configurable and should be stored in a CSV or Python dictionary.

---

## 9.4 Bucket-Level Base Expected Returns

Each bucket has a base expected return per week (as a % of capital), independent of regime.

Initial values:

| Bucket        | BaseExpReturnWeekly |
|---------------|----------------------|
| Discretionary | 0.0035              |
| Spread        | 0.0020              |
| RV            | 0.0025              |
| Volatility    | 0.0030              |
| Systematic    | 0.0020              |

These values will be refined over time.

---

## 9.5 Regime-Adjusted Expected Return

For each bucket:

\[
\mathbb{E}[r_{\text{bucket}}] = \mathbb{E}[r_{\text{bucket}}]_{\text{base}} \times \text{EnvFactor}_{\text{regime,bucket}}
\]

This produces the expected return for the current environment.

---

## 9.6 Expected PnL Calculation

Given:

- \( C_{\text{bucket}} \) = capital allocated to the bucket  
- \( \mathbb{E}[r_{\text{bucket}}] \) = regime-adjusted expected return  

Expected weekly PnL per bucket:

\[
\mathbb{E}[\text{PnL}_{\text{bucket}}] = C_{\text{bucket}} \times \mathbb{E}[r_{\text{bucket}}]
\]

Total expected weekly PnL:

\[
\mathbb{E}[\text{PnL}_{\text{portfolio}}] = \sum_{\text{buckets}} \mathbb{E}[\text{PnL}_{\text{bucket}}]
\]

---

## 9.7 Python Implementation Requirements

PyCharm AI should implement a Python module named:

```
python_modules/expected_pnl_engine.py
```

This module must include:

### 1. Regime classification
- Compute z-scores for volatility and dispersion  
- Apply regime rules  
- Return "Calm", "Normal", or "Event"

### 2. Environment factor lookup
- Load environment factor table  
- Return the correct multiplier for (regime, bucket)

### 3. Bucket expected return calculation
- Load base expected returns  
- Multiply by environment factor  
- Return expected return per bucket

### 4. Expected PnL calculation
- Accept a dictionary of capital by bucket  
- Compute expected PnL per bucket  
- Compute total expected PnL  
- Return a DataFrame with results

### 5. Optional Excel integration
- Write expected PnL results to a sheet named "ExpectedPnL"  
- Update dashboard cells with regime and expected PnL

---

## 9.8 Example Python API

The module should expose functions:

```python
def classify_regime(vol_series, dispersion_series) -> str:
    ...

def expected_return(bucket: str, regime: str) -> float:
    ...

def expected_pnl(capital_by_bucket: dict, regime: str) -> pd.DataFrame:
    ...

def total_expected_pnl(df: pd.DataFrame) -> float:
    ...
```

---

## 9.9 Integration With Portfolio Construction

The Expected PnL Engine will be used to:

- Set weekly PnL targets  
- Compare actual vs expected performance  
- Adjust bucket allocations  
- Adjust discretionary scaling  
- Identify underperforming strategies  
- Support risk-on / risk-off decisions  

This module is a core part of the PM toolkit.

---

# End of Expected PnL Engine Section
# End of project_context.md