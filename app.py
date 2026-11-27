import joblib
import numpy as np
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Credit Card Fraud Detection System")
st.write(
    "This production-ready application uses a **SMOTE + Random Forest** pipeline"
    " to predict fraudulent transactions in real-time."
)


# 2. Load the trained production pipeline
@st.cache_resource
def load_model():
  return joblib.load("fraud_detection_final_smote_rf.pkl")


model = load_model()

# 3. Initialize Session State
DEFAULTS = {
    "time_val": 50000.0,
    "amount_val": 75.50,
    "v_seed": 42,
    "v_scale": 0.5,
}


def generate_base_v(seed: int, scale: float) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return np.round(rng.standard_normal(28) * scale, 4)


for k, v in DEFAULTS.items():
  if k not in st.session_state:
    st.session_state[k] = v

if "v_vals" not in st.session_state:
  st.session_state.v_vals = generate_base_v(
      st.session_state.v_seed, st.session_state.v_scale
  )


def reset_to_defaults():
  st.session_state.time_val = DEFAULTS["time_val"]
  st.session_state.amount_val = DEFAULTS["amount_val"]
  st.session_state.v_seed = DEFAULTS["v_seed"]
  st.session_state.v_scale = DEFAULTS["v_scale"]
  st.session_state.v_vals = generate_base_v(
      DEFAULTS["v_seed"], DEFAULTS["v_scale"]
  )
  for i in range(28):
    st.session_state[f"v_{i + 1}"] = float(st.session_state.v_vals[i])


# 4. Sidebar Inputs
st.sidebar.header("Transaction Parameters")
st.sidebar.write("Adjust transaction values or use default test values.")

st.sidebar.button(
    "🔄 Reset to Defaults",
    on_click=reset_to_defaults,
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.markdown("**Time Range:** `0.0` to `172,800.0` sec (~2 days)")
time_val = st.sidebar.number_input(
    "Time (Seconds elapsed)",
    min_value=0.0,
    max_value=172800.0,
    step=100.0,
    key="time_val",
)

st.sidebar.markdown("**Amount Range:** `0.0` to `25,691.16` $ (Dataset max)")
amount_val = st.sidebar.number_input(
    "Amount ($)",
    min_value=0.0,
    max_value=25691.16,
    step=10.0,
    key="amount_val",
)

st.sidebar.subheader("PCA Features (V1 - V28)")
v_scale = st.sidebar.slider(
    "Outlier Spread Scale",
    min_value=0.1,
    max_value=6.0,
    step=0.1,
    key="v_scale",
)
if st.sidebar.button("🎲 Resample V1–V28", use_container_width=True):
  st.session_state.v_seed = int(np.random.default_rng().integers(1, 100000))
  st.session_state.v_vals = generate_base_v(st.session_state.v_seed, v_scale)
  for i in range(28):
    st.session_state[f"v_{i + 1}"] = float(st.session_state.v_vals[i])

v_features = []
with st.sidebar.expander("Configure Individual V1–V28 Components"):
  for i in range(28):
    key_name = f"v_{i + 1}"
    if key_name not in st.session_state:
      st.session_state[key_name] = float(st.session_state.v_vals[i])
    val = st.number_input(f"V{i + 1}", format="%.4f", key=key_name)
    v_features.append(val)

# 5. Main Interface & Prediction Trigger
if st.button("Analyze Transaction", type="primary"):
  input_data = np.hstack(
      [[time_val], np.array(v_features, dtype=float), [amount_val]]
  ).reshape(1, -1)

  fraud_prob = float(model.predict_proba(input_data)[:, 1][0])
  prediction = int(model.predict(input_data)[0])

  st.divider()
  st.subheader("Analysis Results")

  col1, col2 = st.columns(2)
  with col1:
    st.metric(label="Fraud Probability", value=f"{fraud_prob * 100:.2f}%")
  with col2:
    if prediction == 1 or fraud_prob > 0.5:
      st.error("🚨 FRAUD DETECTED (Suspicious Transaction)")
    else:
      st.success("✅ NORMAL (Safe Transaction)")

  st.progress(min(max(fraud_prob, 0.0), 1.0))