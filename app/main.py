import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import torch
from models.bnn import BayesianNN
from sklearn.preprocessing import MinMaxScaler
import shap
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import datetime
from io import BytesIO
from streamlit_lottie import st_lottie
import requests
import json

# ------------------------
# Custom CSS for Professional Styling
# ------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fa;
    }
    .main {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 6px;
        padding: 12px 24px;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    .stTextInput, .stSelectbox, .stSlider, .stNumberInput {
        background-color: #f1f3f5;
        border-radius: 6px;
        padding: 8px;
        border: 1px solid #e0e0e0;
    }
    .stSidebar {
        background-color: #e9ecef;
        border-right: 1px solid #dee2e6;
        padding: 16px;
    }
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        text-align: center;
        margin-bottom: 16px;
    }
    .stAlert {
        border-radius: 6px;
        padding: 12px;
    }
    .section-divider {
        border-top: 1px solid #e0e0e0;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------
# Helper Function for Lottie Animations
# ------------------------
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning(f"Animation file {filepath} not found.")
        return None

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="Churn Predictor", layout="wide", initial_sidebar_state="expanded")

# Header with Logo
st.image("app/assets/logo.png", width=150)  # Replace with your logo path or URL
st.title("Churn Predictor")
st.markdown("Predict customer churn with confidence scores and actionable insights using a Bayesian Neural Network.")

# Load Lottie Animations
loading_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_3pg6fL.json")  # Loading spinner
prediction_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_UJNc2t.json")  # Prediction animation

# ------------------------
# Load Model and Metadata
# ------------------------
with st.container():
    try:
        checkpoint = torch.load("app/bnn_model.pth")
        training_columns = torch.load("app/columns.pth")
        input_dim = checkpoint['input_dim']

        model = BayesianNN(input_dim=input_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        scaler = MinMaxScaler()

    except FileNotFoundError:
        st.error("Model or columns file not found. Please ensure 'bnn_model.pth' and 'columns.pth' are in the 'app/' directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading the model: {e}")
        st.stop()

# ------------------------
# Tabs for Single and Batch Prediction
# ------------------------
tab1, tab2 = st.tabs(["Single Customer Prediction", "Batch Prediction"])

# ------------------------
# Single Customer Prediction
# ------------------------
with tab1:
    with st.container():
        st.header("Predict Churn for a Single Customer")
        st.markdown("Enter customer details to predict their likelihood of churning.")

        with st.form("single_predict_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender")
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Is the customer a senior citizen?")
                partner = st.selectbox("Has Partner?", ["Yes", "No"], help="Does the customer have a partner?")
                dependents = st.selectbox("Has Dependents?", ["Yes", "No"], help="Does the customer have dependents?")
                tenure = st.slider("Tenure (months)", 0, 72, 12, help="Months with the company")
                phone = st.selectbox("Phone Service", ["Yes", "No"], help="Does the customer have phone service?")
                multi = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], help="Does the customer have multiple lines?")
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], help="Type of internet service")
                online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"], help="Does the customer have online security?")
            with col2:
                online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], help="Does the customer have online backup?")
                device_protect = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], help="Does the customer have device protection?")
                tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], help="Does the customer have tech support?")
                streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], help="Does the customer have streaming TV?")
                streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], help="Does the customer have streaming movies?")
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], help="Type of contract")
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"], help="Does the customer use paperless billing?")
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ], help="Customer's payment method")
                monthly = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.1, help="Monthly charges")
                total = st.number_input("Total Charges ($)", min_value=0.0, value=500.0, step=0.1, help="Total charges to date")

            st.markdown("**Email Report**")
            send_email = st.checkbox("Send report via email", help="Check to email the prediction report")
            email_address = st.text_input("Recipient Email", placeholder="example@gmail.com", help="Enter email for report delivery")

            submitted = st.form_submit_button("Predict Churn")

        if submitted:
            if prediction_animation:
                st_lottie(prediction_animation, height=100, key="prediction_anim")
            with st.spinner("Calculating churn probability..."):
                input_dict = {
                    "gender": gender,
                    "SeniorCitizen": senior,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "PhoneService": phone,
                    "MultipleLines": multi,
                    "InternetService": internet,
                    "OnlineSecurity": online_sec,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protect,
                    "TechSupport": tech_support,
                    "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies,
                    "Contract": contract,
                    "PaperlessBilling": paperless,
                    "PaymentMethod": payment_method,
                    "MonthlyCharges": monthly,
                    "TotalCharges": total
                }

                input_df = pd.DataFrame([input_dict])
                input_df_encoded = pd.get_dummies(input_df)
                input_df_encoded = input_df_encoded.reindex(columns=training_columns, fill_value=0)

                try:
                    if 'TotalCharges' in input_df_encoded.columns and input_df_encoded['TotalCharges'].sum() == 0:
                        st.warning("Total Charges is 0, which may affect prediction accuracy.")
                    
                    temp_df_for_scaler = pd.DataFrame(0, index=[0], columns=training_columns)
                    temp_df_for_scaler.update(input_df_encoded)
                    
                    scaler.fit(temp_df_for_scaler)
                    scaled = scaler.transform(input_df_encoded)
                    
                    x = torch.tensor(scaled, dtype=torch.float32)

                    with torch.no_grad():
                        preds = torch.stack([model(x) for _ in range(100)])
                        prob = preds.mean(0).item()
                        stddev = preds.std(0).item()

                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Churn Probability", f"{prob:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Model Uncertainty", f"±{stddev:.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)

                    with st.expander("Generate Report", expanded=False):
                        if st.button("Download PDF / Send Email", key="report_button"):
                            pdf = FPDF()
                            pdf.add_page()
                            pdf.set_font("Arial", size=12)
                            pdf.cell(200, 10, txt="Churn Prediction Report", ln=True, align="C")
                            pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

                            for key, val in input_dict.items():
                                pdf.cell(200, 8, txt=f"{key}: {val}", ln=True)

                            pdf.cell(200, 10, txt=f"Churn Probability: {prob:.2f}", ln=True)
                            pdf.cell(200, 10, txt=f"Model Uncertainty: ±{stddev:.2f}", ln=True)

                            pdf_buffer = BytesIO()
                            pdf.output(pdf_buffer)
                            pdf_buffer.seek(0)

                            st.download_button(
                                label="Download Report as PDF",
                                data=pdf_buffer,
                                file_name="churn_prediction_report.pdf",
                                mime="application/pdf"
                            )

                            if send_email and email_address:
                                try:
                                    sender_email = "your_gmail@gmail.com"
                                    app_password = "your_app_password"

                                    msg = MIMEMultipart()
                                    msg['From'] = sender_email
                                    msg['To'] = email_address
                                    msg['Subject'] = "Churn Prediction Report"
                                    msg.attach(MIMEText("Attached is your churn prediction report.", "plain"))

                                    part = MIMEApplication(pdf_buffer.getvalue(), Name="churn_report.pdf")
                                    part['Content-Disposition'] = 'attachment; filename="churn_report.pdf"'
                                    msg.attach(part)

                                    server = smtplib.SMTP("smtp.gmail.com", 587)
                                    server.starttls()
                                    server.login(sender_email, app_password)
                                    server.send_message(msg)
                                    server.quit()

                                    st.success(f"Report emailed to {email_address}")
                                except Exception as e:
                                    st.error(f"Failed to send email: {e}")
                            elif send_email and not email_address:
                                st.warning("Please enter a recipient email address.")

                    with st.expander("Feature Importance (SHAP)", expanded=False):
                        try:
                            def model_predict(x_val):
                                with torch.no_grad():
                                    return model(torch.tensor(x_val, dtype=torch.float32)).numpy()
                            
                            scaled_df = pd.DataFrame(scaled, columns=training_columns)
                            explainer = shap.Explainer(model_predict, scaled_df)
                            shap_values = explainer(scaled_df)

                            st.subheader("Why did the model predict this?")
                            fig, ax = plt.subplots()
                            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                            st.pyplot(fig, bbox_inches='tight')
                            plt.close(fig)
                            
                        except Exception as e:
                            st.warning(f"Could not generate SHAP explanation: {e}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# ------------------------
# Batch Prediction
# ------------------------
with tab2:
    with st.container():
        st.header("Batch Prediction from CSV")
        st.markdown("Upload a CSV file to predict churn for multiple customers.")
        uploaded_file = st.file_uploader("Upload a customer CSV file", type=['csv'], help="CSV should contain customer data matching the model features.")

        if uploaded_file:
            if loading_animation:
                st_lottie(loading_animation, height=100, key="batch_anim")
            with st.spinner("Processing batch predictions..."):
                df = pd.read_csv(uploaded_file)

                if 'customerID' in df.columns:
                    df = df.drop(columns=['customerID'])

                df_encoded = pd.get_dummies(df)
                df_encoded = df_encoded.reindex(columns=training_columns, fill_value=0)

                try:
                    X_scaled = scaler.fit_transform(df_encoded)
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

                    with torch.no_grad():
                        preds = torch.stack([model(X_tensor) for _ in range(100)])
                        mean = preds.mean(0).numpy().flatten()
                        std = preds.std(0).numpy().flatten()

                    df['Churn Probability'] = mean
                    df['Uncertainty'] = std
                    st.success("Predictions generated successfully!")
                    st.dataframe(df, use_container_width=True)

                    st.header("Dashboard: Insights & Visuals")
                    
                    with st.expander("Feature Importance (SHAP)", expanded=False):
                        if st.checkbox("Show SHAP Summary Plot for all customers"):
                            with st.spinner("Generating SHAP summary plot..."):
                                try:
                                    def model_predict(x_val):
                                        with torch.no_grad():
                                            return model(torch.tensor(x_val, dtype=torch.float32)).numpy()
                                    
                                    X_scaled_df = pd.DataFrame(X_scaled, columns=training_columns)
                                    explainer = shap.Explainer(model_predict, X_scaled_df)
                                    shap_values = explainer(X_scaled_df)

                                    st.subheader("Feature Importance Across Customers")
                                    fig, ax = plt.subplots()
                                    shap.summary_plot(shap_values.values, X_scaled_df, plot_type="bar", show=False)
                                    st.pyplot(fig, bbox_inches='tight')
                                    plt.close(fig)
                                    
                                except Exception as e:
                                    st.error(f"Could not generate SHAP summary plot: {e}")

                    st.sidebar.header("Filter Data")
                    if 'gender' in df.columns:
                        gender_filter = st.sidebar.multiselect("Filter by Gender", options=df["gender"].unique(), default=df["gender"].unique(), help="Select genders to filter")
                    else:
                        gender_filter = df["gender"].unique()
                        st.sidebar.info("Gender column not found in CSV.")

                    if 'Contract' in df.columns:
                        contract_filter = st.sidebar.multiselect("Filter by Contract", options=df["Contract"].unique(), default=df["Contract"].unique(), help="Select contract types to filter")
                    else:
                        contract_filter = df["Contract"].unique()
                        st.sidebar.info("Contract column not found in CSV.")
                        
                    tenure_min, tenure_max = st.sidebar.slider("Tenure Range", 0, 72, (0, 72), help="Select tenure range in months")

                    filtered_df = df[
                        (df.get("gender", "").isin(gender_filter)) & 
                        (df.get("Contract", "").isin(contract_filter)) &
                        (df["tenure"].between(tenure_min, tenure_max))
                    ]

                    st.subheader("Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Total Customers", len(filtered_df))
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Avg Churn Probability", f"{filtered_df['Churn Probability'].mean():.2f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("High Risk (p > 0.7)", f"{(filtered_df['Churn Probability'] > 0.7).mean() * 100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Uncertain (σ > 0.1)", f"{(filtered_df['Uncertainty'] > 0.1).mean() * 100:.1f}%")
                        st.markdown('</div>', unsafe_allow_html=True)

                    st.subheader("Visualizations")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = px.histogram(filtered_df, x="Churn Probability", nbins=20, color_discrete_sequence=["#1e90ff"], title="Churn Probability Distribution")
                        fig1.update_layout(bargap=0.1, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig1, use_container_width=True)

                    with col2:
                        fig2 = px.histogram(filtered_df, x="Uncertainty", nbins=20, color_discrete_sequence=["#ff851b"], title="Prediction Uncertainty Distribution")
                        fig2.update_layout(bargap=0.1, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)

                    if 'Contract' in filtered_df.columns:
                        fig3 = px.bar(filtered_df, x="Contract", y="Churn Probability", color="Contract", color_discrete_sequence=px.colors.sequential.Blues, title="Avg Churn Probability by Contract")
                        fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.info("Cannot display 'Avg Churn Probability by Contract' as 'Contract' column is missing.")

                    if 'Contract' in filtered_df.columns:
                        contract_counts = filtered_df["Contract"].value_counts()
                        fig4 = go.Figure(data=[go.Pie(labels=contract_counts.index, values=contract_counts.values, marker_colors=px.colors.sequential.Blues)])
                        fig4.update_layout(title="Contract Type Distribution", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig4, use_container_width=True)
                    else:
                        st.info("Cannot display 'Contract Type Distribution' as 'Contract' column is missing.")

                    st.subheader("Top 10 Most Uncertain Customers")
                    st.dataframe(df.sort_values(by='Uncertainty', ascending=False).head(10), use_container_width=True)

                    csv_download = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_download,
                        file_name='churn_predictions_with_uncertainty.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"An error occurred during batch prediction: {e}")