"""
Masculinity Survey — Self-Assessment App
Predicts whether you feel expected to make the first move in romantic relationships
based on your answers to the 2018 WNYC/FiveThirtyEight masculinity survey.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Masculinity & Norms — Self-Assessment",
    page_icon="🧭",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 760px; }
    h1 { font-size: 1.9rem !important; }
    h2 { font-size: 1.2rem !important; color: #444; }
    .stSlider > label { font-weight: 500; }
    .result-box {
        border-radius: 10px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .result-yes { background: #e3f2fd; border-left: 5px solid #1976D2; }
    .result-no  { background: #fce4ec; border-left: 5px solid #c62828; }
    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin: 2rem 0 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Data & model (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on survey data…")
def load_model():
    raw      = pd.read_csv("data/raw-responses.csv")
    df_clean = pd.read_csv("data/cleaned-responses.csv")

    q7_map  = {"Often": 4, "Sometimes": 3, "Rarely": 2,
               "Never, but open to it": 1, "Never, and not open to it": 0,
               "No answer": np.nan}
    q7_cols = [c for c in raw.columns if c.startswith("q0007")]
    q7_df   = raw[q7_cols].replace(q7_map)

    q18_map = {"Always": 5, "Often": 4, "Sometimes": 3,
               "Rarely": 2, "Never": 1, "No answer": np.nan}

    q4_cols  = [c for c in df_clean.columns if c.startswith("q0004")]
    q8_cols  = [c for c in df_clean.columns if c.startswith("q0008")]
    q10_cols = [c for c in df_clean.columns if c.startswith("q0010")]
    q11_cols = [c for c in df_clean.columns if c.startswith("q0011")]
    q12_cols = [c for c in df_clean.columns if c.startswith("q0012")]
    q19_cols = [c for c in df_clean.columns if c.startswith("q0019")]
    q20_cols = [c for c in df_clean.columns if c.startswith("q0020")]
    q21_cols = [c for c in df_clean.columns if c.startswith("q0021")]

    feat = pd.concat([
        df_clean[q4_cols],
        q7_df.reset_index(drop=True),
        df_clean[q8_cols],
        df_clean["q0009"].map({"employed": 1, "not_employed": 0}).rename("q0009"),
        df_clean[q10_cols], df_clean[q11_cols], df_clean[q12_cols],
        df_clean["auto_q0014"].replace({0: np.nan}).rename("q0014"),
        raw["q0018"].map(q18_map).reset_index(drop=True).rename("q0018"),
        df_clean[q19_cols], df_clean[q20_cols], df_clean[q21_cols],
        df_clean["auto_q0022"].map({0: 0, 1: 1, 2: np.nan}).rename("q0022"),
        df_clean["auto_q0024"].rename("marital"),
        df_clean["auto_orientation"].rename("orientation"),
        df_clean["auto_race2"].rename("race"),
        df_clean["auto_educ4"].rename("educ"),
        df_clean["auto_age3"].rename("age"),
        df_clean["auto_kids"].map({0: 0, 1: 1, 2: np.nan}).rename("kids"),
    ], axis=1)

    y    = raw["q0017"].map({"Yes": 1, "No": 0}).reset_index(drop=True)
    mask = y.notna()
    X_raw = feat[mask].copy()
    y     = y[mask].copy()

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, imputer, list(X.columns)


model, imputer, feature_cols = load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
FREQ_OPTIONS  = ["Never, and not open to it", "Never, but open to it",
                 "Rarely", "Sometimes", "Often"]
FREQ_MAP      = {v: i for i, v in enumerate(FREQ_OPTIONS)}

PAY_OPTIONS   = ["Never", "Rarely", "Sometimes", "Often", "Always"]
PAY_MAP       = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5}

WORRY_OPTIONS = ["Yes", "No"]

def make_feature_row(answers: dict) -> pd.DataFrame:
    """Convert user answers dict into a single-row DataFrame matching training features."""
    row = {col: np.nan for col in feature_cols}

    # Q4 — source of masculinity ideas
    row["q0004_0001"] = 1 if "Father / father figure"    in answers.get("q4", []) else 0
    row["q0004_0002"] = 1 if "Mother / mother figure"    in answers.get("q4", []) else 0
    row["q0004_0003"] = 1 if "Other family members"      in answers.get("q4", []) else 0
    row["q0004_0004"] = 1 if "Pop culture"               in answers.get("q4", []) else 0
    row["q0004_0005"] = 1 if "Friends"                   in answers.get("q4", []) else 0
    row["q0004_0006"] = 1 if "Other"                     in answers.get("q4", []) else 0

    # Q7 — lifestyle frequency (ordinal 0–4)
    row["q0007_0001"] = FREQ_MAP[answers["q7_ask_pro"]]
    row["q0007_0002"] = FREQ_MAP[answers["q7_ask_personal"]]
    row["q0007_0003"] = FREQ_MAP[answers["q7_physical_affection"]]
    row["q0007_0004"] = FREQ_MAP[answers["q7_cry"]]
    row["q0007_0005"] = FREQ_MAP[answers["q7_sex_women"]]
    row["q0007_0006"] = FREQ_MAP[answers["q7_sex_men"]]
    row["q0007_0007"] = FREQ_MAP[answers["q7_sports"]]
    row["q0007_0008"] = FREQ_MAP[answers["q7_workout"]]
    row["q0007_0009"] = FREQ_MAP[answers["q7_therapist"]]
    row["q0007_0010"] = FREQ_MAP[answers["q7_therapist"]]   # duplicate col in data
    row["q0007_0011"] = FREQ_MAP[answers["q7_lonely"]]

    # Q8 — daily worries
    row["q0008_0001"] = 1 if "Height"               in answers.get("q8", []) else 0
    row["q0008_0002"] = 1 if "Weight"               in answers.get("q8", []) else 0
    row["q0008_0003"] = 1 if "Hair / hairline"      in answers.get("q8", []) else 0
    row["q0008_0004"] = 1 if "Physique"             in answers.get("q8", []) else 0
    row["q0008_0005"] = 1 if "Genitalia appearance" in answers.get("q8", []) else 0
    row["q0008_0006"] = 1 if "Clothing / style"     in answers.get("q8", []) else 0
    row["q0008_0007"] = 1 if "Sexual performance"   in answers.get("q8", []) else 0
    row["q0008_0008"] = 1 if "Mental health"        in answers.get("q8", []) else 0
    row["q0008_0009"] = 1 if "Physical health"      in answers.get("q8", []) else 0
    row["q0008_0010"] = 1 if "Finances / income"    in answers.get("q8", []) else 0
    row["q0008_0011"] = 1 if "Ability to provide"   in answers.get("q8", []) else 0
    row["q0008_0012"] = 1 if "None of the above"    in answers.get("q8", []) else 0

    # Q9 — employment
    row["q0009"] = 1 if answers.get("q9") == "Employed" else 0

    # Q10 — work advantages (only if employed)
    row["q0010_0001"] = 1 if "Men make more money"                        in answers.get("q10", []) else 0
    row["q0010_0002"] = 1 if "Men are taken more seriously"               in answers.get("q10", []) else 0
    row["q0010_0003"] = 1 if "Men have more choice"                       in answers.get("q10", []) else 0
    row["q0010_0004"] = 1 if "Men have more promotion opportunities"      in answers.get("q10", []) else 0
    row["q0010_0005"] = 1 if "Men are explicitly praised more"            in answers.get("q10", []) else 0
    row["q0010_0006"] = 1 if "Men have more support from managers"        in answers.get("q10", []) else 0
    row["q0010_0007"] = 1 if "Other"                                      in answers.get("q10", []) else 0
    row["q0010_0008"] = 1 if "None of the above"                          in answers.get("q10", []) else 0

    # Q11 — work disadvantages
    row["q0011_0001"] = 1 if "Managers prefer to hire/promote women"      in answers.get("q11", []) else 0
    row["q0011_0002"] = 1 if "Greater risk of harassment accusation"      in answers.get("q11", []) else 0
    row["q0011_0003"] = 1 if "Greater risk of sexism/racism accusation"   in answers.get("q11", []) else 0
    row["q0011_0004"] = 1 if "Other"                                      in answers.get("q11", []) else 0
    row["q0011_0005"] = 1 if "None of the above"                          in answers.get("q11", []) else 0

    # Q12 — harassment response
    row["q0012_0001"] = 1 if "Confronted the accused"              in answers.get("q12", []) else 0
    row["q0012_0002"] = 1 if "Contacted HR"                        in answers.get("q12", []) else 0
    row["q0012_0003"] = 1 if "Contacted accused's manager"         in answers.get("q12", []) else 0
    row["q0012_0004"] = 1 if "Reached out to support the victim"   in answers.get("q12", []) else 0
    row["q0012_0005"] = 1 if "Did not respond"                     in answers.get("q12", []) else 0
    row["q0012_0006"] = 1 if "Never witnessed harassment"          in answers.get("q12", []) else 0
    row["q0012_0007"] = 0

    # Q14 — heard about MeToo
    metoo_map = {"A lot": 1, "Some": 2, "Only a little": 3, "Nothing at all": 4}
    row["q0014"] = metoo_map.get(answers.get("q14", "Some"), 2)

    # Q18 — pays on dates
    row["q0018"] = PAY_MAP[answers["q18"]]

    # Q19 — reasons for paying
    row["q0019_0001"] = 1 if "It's the right thing to do"                        in answers.get("q19", []) else 0
    row["q0019_0002"] = 1 if "I make more money than my date"                    in answers.get("q19", []) else 0
    row["q0019_0003"] = 1 if "I feel good being the one who pays"                in answers.get("q19", []) else 0
    row["q0019_0004"] = 1 if "Societal expectation"                              in answers.get("q19", []) else 0
    row["q0019_0005"] = 1 if "I asked them out so I feel obligated"              in answers.get("q19", []) else 0
    row["q0019_0006"] = 1 if "To see if they offer to share the cost"            in answers.get("q19", []) else 0
    row["q0019_0007"] = 0

    # Q20 — gauging consent
    row["q0020_0001"] = 1 if "Read their body language"          in answers.get("q20", []) else 0
    row["q0020_0002"] = 1 if "Ask for verbal confirmation"       in answers.get("q20", []) else 0
    row["q0020_0003"] = 1 if "Make a physical move and see"      in answers.get("q20", []) else 0
    row["q0020_0004"] = 1 if "Every situation is different"      in answers.get("q20", []) else 0
    row["q0020_0005"] = 1 if "It isn't always clear"             in answers.get("q20", []) else 0
    row["q0020_0006"] = 0

    # Q21 — sexual boundary reflection
    row["q0021_0001"] = 1 if "Wondered if I pushed too far"                               in answers.get("q21", []) else 0
    row["q0021_0002"] = 1 if "Talked with friends about whether I pushed too far"         in answers.get("q21", []) else 0
    row["q0021_0003"] = 1 if "Contacted a past partner to ask if I went too far"          in answers.get("q21", []) else 0
    row["q0021_0004"] = 1 if "None of the above"                                          in answers.get("q21", []) else 0

    # Q22 — changed behavior post MeToo
    row["q0022"] = 1 if answers.get("q22") == "Yes" else 0

    # Demographics
    marital_map = {"Married": 2, "Never married": 0, "Divorced": 3,
                   "Widowed": 1, "Separated": 4}
    row["marital"] = marital_map.get(answers.get("marital", "Never married"), 0)

    row["orientation"] = 1 if answers.get("orientation") == "Straight" else 0
    row["race"]        = 1 if answers.get("race") == "White" else 0

    educ_map = {"High school or less": 0, "Some college / Associate's": 1,
                "College graduate": 2, "Post-graduate degree": 3}
    row["educ"] = educ_map.get(answers.get("educ", "Some college / Associate's"), 1)

    age_map = {"18–34": 0, "35–64": 1, "65+": 2}
    row["age"]  = age_map.get(answers.get("age", "18–34"), 0)
    row["kids"] = 1 if answers.get("kids") == "Yes" else 0

    df_row = pd.DataFrame([row])[feature_cols]
    df_imp = pd.DataFrame(imputer.transform(df_row), columns=feature_cols)
    return df_imp


def top_drivers(row_df: pd.DataFrame, n: int = 6):
    """Return the top features pushing this prediction toward Yes or No."""
    feat_imp = model.feature_importances_
    user_vals = row_df.iloc[0]

    # Compare user value to training median (stored in imputer)
    medians = imputer.statistics_
    median_series = pd.Series(medians, index=feature_cols)

    drivers = []
    for feat, imp in zip(feature_cols, feat_imp):
        delta = float(user_vals[feat]) - float(median_series[feat])
        drivers.append((feat, imp, delta))

    drivers.sort(key=lambda x: abs(x[1] * x[2]), reverse=True)
    return drivers[:n]


FEAT_LABELS = {
    "q0018":      "How often you try to pay on dates",
    "q0007_0005": "Frequency of sex with women",
    "q0007_0006": "Frequency of sex with men",
    "q0007_0011": "How often you feel lonely/isolated",
    "q0007_0003": "Physical affection with male friends",
    "q0007_0001": "Asking friends for professional advice",
    "q0007_0008": "How often you work out",
    "q0007_0009": "How often you see a therapist",
    "q0004_0001": "Getting masculinity ideas from father",
    "q0004_0005": "Getting masculinity ideas from friends",
    "q0008_0010": "Worrying about finances daily",
    "q0008_0008": "Worrying about mental health daily",
    "q0008_0002": "Worrying about weight daily",
    "q0008_0005": "Worrying about genitalia appearance",
    "q0011_0001": "Thinking managers prefer to hire women",
    "q0011_0002": "Worrying about harassment accusation risk",
    "q0010_0001": "Thinking men earn more at work",
    "q0010_0005": "Thinking men are praised more at work",
    "q0012_0005": "Not responding to witnessed harassment",
    "q0021_0001": "Wondering if you pushed a partner too far",
    "q0021_0003": "Contacting a past partner re: consent",
    "q0019_0005": "Paying because you asked them out",
    "q0019_0002": "Paying because you earn more",
    "q0014":      "How much you've heard about #MeToo",
    "q0022":      "Changing behavior post-#MeToo",
    "orientation":"Sexual orientation",
    "race":       "Race",
    "educ":       "Education level",
    "age":        "Age group",
    "kids":       "Having children",
    "marital":    "Marital status",
    "q0009":      "Employment status",
    "q0008_0004": "Worrying about physique daily",
    "q0020_0002": "Asking for verbal consent",
    "q0020_0003": "Making a physical move to test interest",
    "q0012_0004": "Reaching out to support a harassment victim",
    "q0004_0003": "Getting masculinity ideas from other family",
    "q0004_0004": "Getting masculinity ideas from pop culture",
}


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🧭 Do You Feel Expected to Make the First Move?")
st.markdown("""
Answer a short version of the 2018 **WNYC / FiveThirtyEight Masculinity Survey**.
A Random Forest model trained on 1,584 responses will predict whether your answers
match those of men who feel expected to initiate in romantic relationships — and show
you which of your answers drove the result.

*This is a research tool, not a personality test. All questions are optional context
for the model — the prediction reflects statistical patterns, not a judgment.*
""")

st.divider()

answers = {}

# ── Section 1: Lifestyle ──────────────────────────────────────────────────────
st.markdown('<p class="section-label">Section 1 of 4 — Lifestyle</p>', unsafe_allow_html=True)
st.markdown("**How often would you say you do each of the following?**")

col1, col2 = st.columns(2)
with col1:
    answers["q7_ask_pro"]          = st.select_slider("Ask a friend for professional advice", FREQ_OPTIONS, value="Sometimes")
    answers["q7_ask_personal"]     = st.select_slider("Ask a friend for personal advice",     FREQ_OPTIONS, value="Sometimes")
    answers["q7_physical_affection"]= st.select_slider("Express physical affection to male friends (hugging, etc.)", FREQ_OPTIONS, value="Rarely")
    answers["q7_cry"]              = st.select_slider("Cry",                                  FREQ_OPTIONS, value="Rarely")
    answers["q7_sex_women"]        = st.select_slider("Have sexual relations with women",      FREQ_OPTIONS, value="Sometimes")
    answers["q7_sex_men"]          = st.select_slider("Have sexual relations with men",        FREQ_OPTIONS, value="Never, and not open to it")

with col2:
    answers["q7_sports"]           = st.select_slider("Watch sports",                         FREQ_OPTIONS, value="Sometimes")
    answers["q7_workout"]          = st.select_slider("Work out",                             FREQ_OPTIONS, value="Sometimes")
    answers["q7_therapist"]        = st.select_slider("See a therapist",                      FREQ_OPTIONS, value="Never, but open to it")
    answers["q7_lonely"]           = st.select_slider("Feel lonely or isolated",              FREQ_OPTIONS, value="Rarely")

st.markdown("**Which of the following do you worry about on a daily or near-daily basis?** *(select all that apply)*")
answers["q8"] = st.multiselect(
    label="Daily worries",
    options=["Height", "Weight", "Hair / hairline", "Physique", "Genitalia appearance",
             "Clothing / style", "Sexual performance", "Mental health", "Physical health",
             "Finances / income", "Ability to provide", "None of the above"],
    default=[],
    label_visibility="collapsed",
)

st.divider()

# ── Section 2: Work ───────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Section 2 of 4 — Work</p>', unsafe_allow_html=True)

answers["q9"] = st.radio("Employment status", ["Employed", "Not employed"], horizontal=True)

if answers["q9"] == "Employed":
    st.markdown("**In which ways is it an *advantage* to be a man at your workplace?** *(select all that apply)*")
    answers["q10"] = st.multiselect(
        "Work advantages",
        options=["Men make more money", "Men are taken more seriously", "Men have more choice",
                 "Men have more promotion opportunities", "Men are explicitly praised more",
                 "Men have more support from managers", "Other", "None of the above"],
        default=[],
        label_visibility="collapsed",
    )

    st.markdown("**In which ways is it a *disadvantage* to be a man at your workplace?** *(select all that apply)*")
    answers["q11"] = st.multiselect(
        "Work disadvantages",
        options=["Managers prefer to hire/promote women", "Greater risk of harassment accusation",
                 "Greater risk of sexism/racism accusation", "Other", "None of the above"],
        default=[],
        label_visibility="collapsed",
    )

    st.markdown("**If you've seen sexual harassment at work, how did you respond?** *(select all that apply)*")
    answers["q12"] = st.multiselect(
        "Harassment response",
        options=["Confronted the accused", "Contacted HR", "Contacted accused's manager",
                 "Reached out to support the victim", "Did not respond",
                 "Never witnessed harassment"],
        default=["Never witnessed harassment"],
        label_visibility="collapsed",
    )

    answers["q14"] = st.selectbox(
        "How much have you heard about the #MeToo movement?",
        ["A lot", "Some", "Only a little", "Nothing at all"], index=1
    )
else:
    answers["q10"] = []
    answers["q11"] = []
    answers["q12"] = ["Never witnessed harassment"]
    answers["q14"] = "Some"

st.divider()

# ── Section 3: Relationships ──────────────────────────────────────────────────
st.markdown('<p class="section-label">Section 3 of 4 — Relationships</p>', unsafe_allow_html=True)

answers["q18"] = st.select_slider(
    "How often do you try to be the one who pays when on a date?",
    PAY_OPTIONS, value="Often"
)

if answers["q18"] in ["Always", "Often"]:
    st.markdown("**Why do you try to pay?** *(select all that apply)*")
    answers["q19"] = st.multiselect(
        "Reasons for paying",
        options=["It's the right thing to do", "I make more money than my date",
                 "I feel good being the one who pays", "Societal expectation",
                 "I asked them out so I feel obligated", "To see if they offer to share the cost"],
        default=[],
        label_visibility="collapsed",
    )
else:
    answers["q19"] = []

st.markdown("**When you want to be physically intimate with someone, how do you gauge their interest?** *(select all that apply)*")
answers["q20"] = st.multiselect(
    "Gauging consent",
    options=["Read their body language", "Ask for verbal confirmation",
             "Make a physical move and see", "Every situation is different",
             "It isn't always clear"],
    default=["Read their body language"],
    label_visibility="collapsed",
)

st.markdown("**Over the past 12 months, which of the following have you done?** *(select all that apply)*")
answers["q21"] = st.multiselect(
    "Sexual boundary reflection",
    options=["Wondered if I pushed too far",
             "Talked with friends about whether I pushed too far",
             "Contacted a past partner to ask if I went too far",
             "None of the above"],
    default=["None of the above"],
    label_visibility="collapsed",
)

answers["q22"] = st.radio(
    "Have you changed your behavior in romantic relationships in the wake of #MeToo?",
    ["Yes", "No"], index=1, horizontal=True
)

st.divider()

# ── Section 4: Demographics ───────────────────────────────────────────────────
st.markdown('<p class="section-label">Section 4 of 4 — Demographics</p>', unsafe_allow_html=True)
st.markdown("*Used as context by the model — not used to judge your result.*")

col3, col4 = st.columns(2)
with col3:
    answers["q4"] = st.multiselect(
        "Where did you get your ideas about what it means to be a good man?",
        options=["Father / father figure", "Mother / mother figure", "Other family members",
                 "Pop culture", "Friends", "Other"],
        default=["Father / father figure"],
    )
    answers["orientation"] = st.selectbox("Sexual orientation", ["Straight", "Gay", "Bisexual", "Other"])
    answers["marital"]     = st.selectbox("Marital status", ["Never married", "Married", "Divorced", "Separated", "Widowed"])

with col4:
    answers["age"]  = st.selectbox("Age group",         ["18–34", "35–64", "65+"])
    answers["educ"] = st.selectbox("Education",         ["High school or less", "Some college / Associate's",
                                                          "College graduate", "Post-graduate degree"])
    answers["race"] = st.selectbox("Race",              ["White", "Non-white"])
    answers["kids"] = st.radio("Do you have children?", ["Yes", "No"], horizontal=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 See my result", type="primary", use_container_width=True):

    row_df = make_feature_row(answers)
    proba  = model.predict_proba(row_df)[0]
    pred   = model.predict(row_df)[0]
    prob_yes = proba[1]

    st.markdown("## Your Result")

    # Probability gauge
    fig, ax = plt.subplots(figsize=(7, 0.6))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh([0], [1],    color="#f0f0f0", height=0.5)
    ax.barh([0], [prob_yes], color="#1976D2" if pred == 1 else "#c62828", height=0.5)
    ax.axvline(0.5, color="#aaa", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.set_xlabel("Model confidence → 'Yes, feels expected'", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(prob_yes, 0, f" {prob_yes*100:.0f}%",
            va="center", ha="left" if prob_yes < 0.85 else "right",
            fontsize=11, fontweight="bold",
            color="#1976D2" if pred == 1 else "#c62828")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    if pred == 1:
        st.markdown(f"""
<div class="result-box result-yes">
<strong>The model predicts: Yes — your answers match men who feel expected to initiate.</strong><br><br>
With {prob_yes*100:.0f}% confidence, your responses align with men in the 2018 survey who said
they typically feel it's expected of them to make the first move in romantic relationships.
The model is trained on 1,584 men — this reflects a statistical pattern, not a certainty about you.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="result-box result-no">
<strong>The model predicts: No — your answers match men who don't feel this expectation.</strong><br><br>
With {(1-prob_yes)*100:.0f}% confidence, your responses align with men in the 2018 survey who said
they do <em>not</em> typically feel expected to make the first move.
The model is trained on 1,584 men — this reflects a statistical pattern, not a certainty about you.
</div>""", unsafe_allow_html=True)

    # Top drivers chart
    st.markdown("### What drove this prediction?")
    st.markdown(
        "The bars below show which of your answers had the most influence. "
        "**Blue** pushed toward *Yes (feels expected)*, **red** toward *No*."
    )

    drivers = top_drivers(row_df, n=8)
    labels, values, colors = [], [], []
    for feat, imp, delta in drivers:
        label = FEAT_LABELS.get(feat, feat)
        score = imp * delta
        labels.append(label)
        values.append(score)
        colors.append("#1976D2" if score > 0 else "#c62828")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")
    y_pos = range(len(labels))
    ax2.barh(list(y_pos), values, color=colors, edgecolor="white", height=0.6)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels[::-1] if False else labels, fontsize=9)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Influence on prediction (relative)", fontsize=9)
    ax2.set_title("Top factors from your answers", fontsize=11, fontweight="bold")
    yes_patch = mpatches.Patch(color="#1976D2", label="Pushes toward Yes")
    no_patch  = mpatches.Patch(color="#c62828", label="Pushes toward No")
    ax2.legend(handles=[yes_patch, no_patch], fontsize=8, loc="lower right")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # Context
    st.markdown("### How does this compare to the survey population?")
    col_a, col_b = st.columns(2)
    col_a.metric("Men who said Yes (survey)", "64%")
    col_b.metric("Model confidence (you)", f"{prob_yes*100:.0f}%")

    st.markdown("""
---
**About the model:** Random Forest (300 trees) trained on 1,584 responses to the 2018
WNYC / FiveThirtyEight masculinity survey. Cross-validated AUC: 0.667. Features include
lifestyle behaviors, workplace attitudes, relationship patterns, and demographics.
The model predicts the majority class (Yes) at baseline — a result near 50% means
your answers are genuinely ambiguous relative to the training data.

[View the full analysis on GitHub](https://github.com/k10sj02/us_views_of_masculinity)
""")
