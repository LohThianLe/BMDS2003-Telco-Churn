import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

# ---- Model options ----
model_options = {
    "Random Forest": "outputs/random_forest_model.pkl",
    "Decision Tree": "outputs/decision_tree_model.pkl",
    "XGBoost": "outputs/xgboost_model.pkl",
    "Logistic Regression": "outputs/logistic_regression_model.pkl",
}

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = list(model_options.keys())[0]

# ---- Sidebar ----
with st.sidebar:
    st.title("📶 Churn Predictor")
    st.caption("A prototype comparing 4 models trained on the Telco Customer Churn dataset.")

    st.write("")
    st.divider()
    st.write("")

    st.header("Choose a Model")
    st.write("")

    model_icons = {
        "Random Forest": "🌲",
        "Decision Tree": "🌳",
        "XGBoost": "⚡",
        "Logistic Regression": "📈",
    }

    for name in model_options:
        is_selected = (st.session_state.selected_model_name == name)
        label = f"{model_icons[name]}  {name}"
        if st.button(label, use_container_width=True, type="primary" if is_selected else "secondary"):
            st.session_state.selected_model_name = name
            st.rerun()

    selected_model_name = st.session_state.selected_model_name

    st.write("")
    st.divider()
    st.write("")

    st.header("Risk Levels")
    st.write("")
    st.markdown("🟢 **Low** — under 35%")
    st.markdown("🟠 **Moderate** — 35–65%")
    st.markdown("🔴 **High** — over 65%")

model_path = model_options[selected_model_name]

if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}. Make sure it's been trained and saved.")
    st.stop()

model = joblib.load(model_path)

# ---- Main page ----
st.title("Telco Customer Churn Predictor")
st.subheader("Enter a customer's details to predict whether they are likely to churn.")

st.info(f"{model_icons[selected_model_name]} Currently predicting with **{selected_model_name}**")

st.header("Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    with st.container(border=True, height="stretch"):
        st.subheader("👤 Profile")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        def sync_from_number():
            st.session_state.tenure_slider = st.session_state.tenure_number

        def sync_from_slider():
            st.session_state.tenure_number = st.session_state.tenure_slider

        if "tenure_number" not in st.session_state:
            st.session_state.tenure_number = 12
            st.session_state.tenure_slider = 12

        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72,
            key="tenure_number", on_change=sync_from_number,
        )
        st.slider(
            "Drag to adjust", min_value=0, max_value=72,
            key="tenure_slider", on_change=sync_from_slider,
            label_visibility="collapsed",
        )

with col2:
    with st.container(border=True, height="stretch"):
        st.subheader("📡 Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

with col3:
    with st.container(border=True, height="stretch"):
        st.subheader("💳 Billing & Contract")
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Bank transfer (automatic)", "Credit card (automatic)",
            "Electronic check", "Mailed check"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

# ---- Predict button ----
if st.button("Predict Churn", use_container_width=True, type="primary"):

    input_dict = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "tenure": tenure,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 1 if multiple_lines == "Yes" else 0,
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "OnlineBackup": 1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies": 1 if streaming_movies == "Yes" else 0,
        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    if selected_model_name == "Logistic Regression":
        scaler = joblib.load("outputs/scaler.pkl")
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.header("Result")
    st.caption(f"Model used: {selected_model_name}")
    if prediction == 1:
        st.error(f"This customer is likely to CHURN. (Probability: {probability:.1%})")
    else:
        st.success(f"This customer is likely to STAY. (Probability of churn: {probability:.1%})")

    # ---- Charts & Insights: everything in one tabbed navigator ----
    st.header("Charts & Insights")

    cm_files = {
        "Random Forest": "outputs/charts/random_forest_cm.png",
        "Decision Tree": "outputs/charts/decision_tree_cm.png",
        "XGBoost": "outputs/charts/xgboost_confusion_matrix.png",
        "Logistic Regression": "outputs/charts/logistic_regression_cm.png",
    }

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Top Features",
        "🎯 Confusion Matrix",
        "📉 Churn Split",
        "📶 Distributions",
        "💰 Charges vs Churn",
        "🗂️ By Category",
        "🔥 Correlations",
        "⏳ Tenure Cohorts",
    ])

    with tab1:
        has_importance = hasattr(model, "feature_importances_")
        if has_importance:
            # Feature names come from the model itself (feature_names_in_), not from
            # input_df.columns -- so the names are guaranteed to line up with
            # feature_importances_ instead of relying on two separately-maintained
            # orderings happening to stay in sync.
            importance_df = pd.DataFrame({
                "Feature": model.feature_names_in_,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(10)

        left, right = st.columns(2)
        with left:
            if has_importance:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.barh(importance_df["Feature"], importance_df["Importance"], color="#d95f02")
                ax.set_xlabel("Importance")
                st.pyplot(fig)
            else:
                st.caption(f"{selected_model_name} doesn't expose feature importance the way tree-based models do.")
        with right:
            st.title("What This Shows")
            if has_importance:
                top_feature = importance_df.iloc[-1]["Feature"]
                second_feature = importance_df.iloc[-2]["Feature"]
                st.subheader(
                    f"This ranks which features **{selected_model_name}** relies on most, across "
                    "all the predictions it makes — not just the customer you just entered."
                )
                st.subheader(
                    f"Across the whole dataset, **{top_feature}** and **{second_feature}** carry the "
                    f"most weight for {selected_model_name}. Longer bars mean the model leans on that "
                    "feature more heavily when deciding whether someone is likely to churn."
                )
            else:
                st.subheader(
                    "Logistic Regression doesn't rank features the same way tree-based models do — "
                    "it uses coefficients instead, which work on a different scale. Switch to "
                    "Random Forest, Decision Tree, or XGBoost in the sidebar to see this breakdown."
                )

    with tab2:
        left, right = st.columns(2)
        with left:
            cm_path = cm_files.get(selected_model_name)
            if cm_path and os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.caption("Confusion matrix image not found.")
        with right:
            st.title("What This Shows")
            st.subheader("Each cell counts how the model's guess compared to reality on the test set:")
            st.markdown("#### • **Top-left** — correctly predicted *no churn*")
            st.markdown("#### • **Top-right** — wrongly predicted *churn* (false alarm)")
            st.markdown("#### • **Bottom-left** — wrongly predicted *no churn* (missed a real churner)")
            st.markdown("#### • **Bottom-right** — correctly predicted *churn*")

            comparison_path = "outputs/model_comparison.csv"
            if os.path.exists(comparison_path):
                comparison = pd.read_csv(comparison_path)
                if selected_model_name in comparison.columns:
                    st.subheader(f"{selected_model_name}'s overall test-set performance:")
                    for _, row in comparison.iterrows():
                        st.markdown(f"#### • {row['Metric']}: **{row[selected_model_name]:.2f}%**")
                else:
                    st.caption(
                        f"⚠️ `{comparison_path}` has no column for {selected_model_name}, so its "
                        "metrics can't be shown. The file is stale — re-run "
                        "`python update_comparison.py` to rebuild it from all four saved models."
                    )

    with tab3:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/01_churn_count.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Out of 7,043 customers in the dataset, **1,869 (26.5%) churned** and the rest stayed. "
                "This roughly 3-to-1 imbalance matters because a model can score well on accuracy "
                "just by predicting \"no churn\" for almost everyone — while missing the customers "
                "who actually matter."
            )
            st.subheader(
                "Three of the four models correct for this: **Decision Tree** and **Random Forest** "
                "use `class_weight='balanced'`, and **XGBoost** uses its equivalent, `scale_pos_weight`. "
                "**Logistic Regression** is left unweighted as a plain baseline — which is likely why it "
                "posts the highest accuracy (81.16%) but the lowest recall (56.79%) of the four: without "
                "rebalancing it leans toward the safe \"no churn\" answer more often, so it catches fewer "
                "of the customers who really did leave. Worth keeping in mind when comparing the numbers "
                "on the Confusion Matrix tab."
            )

    with tab4:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/02_distributions.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Tenure splits into two groups: a large spike of brand-new customers (under a few "
                "months in) and a second, smaller spike of long-term customers near the 70-month mark "
                "— relatively few people sit in between."
            )
            st.subheader(
                "Monthly charges show a similar shape: a spike around \\$20 (customers on a single "
                "basic service) and a broader spread from \\$70–100 (customers bundling multiple services)."
            )

    with tab5:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/03_monthly_charges_boxplot.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Churned customers have a noticeably **higher** median monthly charge (\\$79.65) "
                "than retained customers (\\$64.43) — consistent with the Correlations tab, where "
                "MonthlyCharges shows a positive (+0.19) relationship with churn."
            )
            st.subheader(
                "Retained customers show a much wider spread (roughly \\$25 to \\$88), including "
                "many low-cost, long-tenure accounts, while churned customers cluster more tightly "
                "in the $56–94 range. High charges alone don't guarantee churn, but they're clearly "
                "part of the picture."
            )

    with tab6:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/04_categorical_churn.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Contract type shows the clearest split: **month-to-month customers churn at 42.7%**, "
                "versus 11.3% for one-year contracts and just 2.8% for two-year contracts."
            )
            st.subheader(
                "Fiber optic internet users churn more (41.9%) than DSL (19.0%) or customers with no "
                "internet (7.4%), and customers paying by **electronic check churn the most** of any "
                "payment method, at 45.3%."
            )

    with tab7:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/05_correlation_heatmap.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Among the numeric features, **tenure has the strongest relationship with churn** "
                "(-0.35) — longer-tenured customers are noticeably less likely to leave. "
                "TotalCharges follows a similar pattern (-0.20)."
            )
            st.subheader(
                "MonthlyCharges (+0.19) and SeniorCitizen (+0.15) point the other way, but more weakly. "
                "Tenure and TotalCharges are also strongly correlated with **each other** (+0.83), which "
                "makes sense — customers who stay longer naturally accumulate higher total bills."
            )

    with tab8:
        left, right = st.columns(2)
        with left:
            st.image("outputs/charts/06_additional_pattern_tenure_groups.png", use_container_width=True)
        with right:
            st.title("What This Shows")
            st.subheader(
                "Breaking tenure into cohorts makes the pattern even clearer: churn rate drops steadily "
                "from **47.4%** in a customer's first year, to 28.7% in year two, 20.4% in years 2–4, "
                "and just **9.5%** for customers who've stayed 4–6 years."
            )
            st.subheader(
                "This is the single strongest pattern in the whole dataset — the longer someone stays, "
                "the more unlikely they become to leave."
            )