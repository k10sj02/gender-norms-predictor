# Predicting Gender Norm Internalization
### An Analysis of the WNYC / FiveThirtyEight Masculinity Survey

This project builds binary classifiers on the [WNYC/FiveThirtyEight masculinity survey](https://fivethirtyeight.com/features/what-do-men-think-it-means-to-be-a-man/) (n=1,615 American men, 2018), exploring what behavioral and attitudinal patterns predict how men relate to traditional gender norms.

---

## The Question

Most masculinity research focuses on self-perception — *how masculine do you feel?* — which is circular and heavily skewed. This project instead predicts two more grounded targets:

**Primary — Q17: Do you feel expected to make the first move in romantic relationships?**  
A concrete, behaviorally-grounded expression of internalized courtship norms. Not circular — it asks about a *felt obligation*, which can be meaningfully predicted from lifestyle behaviors, relationship patterns, and demographics.

**Secondary — Q5: Does society put unhealthy pressure on men?**  
A social/political belief about structural gender pressure. Harder to predict (~0.63 AUC vs ~0.66 for Q17) because abstract beliefs are noisier targets than enacted behaviors.

---

## Results

| Target | Best Model | Accuracy | AUC |
|---|---|---|---|
| Q17 — first move | Random Forest | 0.670 | 0.667 |
| Q5 — society pressure | Logistic Regression | 0.614 | 0.632 |

Both targets are above majority-class baselines (Q17: 64%, Q5: 60%). The convergence across model types (LR, RF, GBM) suggests the ~0.65–0.67 AUC ceiling is real — these are genuinely noisy attitudinal targets, and further tuning is unlikely to help without leakage.

### Top predictors for Q17 (feels expected to initiate)
- **Tries to pay on dates** — strongest signal; both reflect the same traditional courtship script
- **Feels lonely/isolated frequently** — internalizing pressure to be the active pursuer
- **Learned masculinity norms from father** — direct transmission of traditional role expectations
- **Perceives men earn more at work** — provider-role mindset extends to romantic context
- **Contacted past partner to check on consent** → *less* likely to feel expected (actively questioning norms)
- **Non-straight orientation** → *less* likely (outside the heteronormative courtship script)

### Top predictors for Q5 (society pressures men)
- **Feels lonely/isolated, sees therapist** — emotional burden as evidence of systemic pressure
- **Worries about finances and sexual performance** — experiencing the specific pressures the question asks about
- **Absorbed masculinity norms from pop culture** — scripted norms may increase awareness of their weight
- **Works out frequently** → *less* likely (comfortable performing traditional masculinity)
- **Non-straight orientation** → *less* likely (less subject to heteronormative pressure)

---

## Project Structure

```
.
├── data/
│   ├── raw-responses.csv                  # Original survey responses (1,615 × 98)
│   ├── cleaned-responses.csv              # Preprocessed binary/ordinal encodings
│   ├── quantized-data.csv                 # Quantized encodings from original analysis
│   ├── masculinity-survey.csv             # Aggregate summary table
│   └── masculinity-survey-questions.pdf   # Full survey instrument
│
├── notebooks/
│   ├── 01_eda.ipynb                       # Data exploration, distributions, demographics
│   ├── 02_modeling_q17.ipynb              # Primary target: first-move expectation
│   └── 03_modeling_q05.ipynb              # Secondary target: society pressure belief
│
├── figures/
│   ├── choropleth_map.png
│   └── choropleth_map_with_horizontal_colorbar.png
│
├── archive/                               # Prior iteration of the project
│   ├── WNYC_Masculinity_Survey.ipynb      # Original notebook (target: Q1 self-perception)
│   ├── correlation.ipynb
│   └── data_cleaner.py
│
└── README.md
```

---

## Running the Notebooks

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook
```

Open notebooks in order: `01_eda` → `02_modeling_q17` → `03_modeling_q05`. Each notebook reads from `../data/` and is self-contained.

---

## Data Source

Survey conducted by WNYC and FiveThirtyEight, fielded May 2018. Full methodology and original analysis: [fivethirtyeight.com](https://fivethirtyeight.com/features/what-do-men-think-it-means-to-be-a-man/). Raw data via the [FiveThirtyEight GitHub](https://github.com/fivethirtyeight/data/tree/master/masculinity-survey).
