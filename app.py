import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration with premium tab title and icon
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern premium UI styling (glassmorphism, clean fonts, sleek colors)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background gradient */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(98, 114, 244, 0.05) 0%, rgba(18, 20, 26, 0.02) 90%);
    }
    
    /* Header card with glassmorphism style */
    .header-box {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        color: #ffffff;
    }
    
    .header-box h1 {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-box p {
        font-size: 1.1rem;
        color: #9ca3af;
        margin: 0;
    }
    
    /* Elegant card container for predictions */
    .prediction-card {
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .salary-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(5, 150, 105, 0.05));
        border-left: 5px solid #10b981;
    }
    
    .salary-low {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(217, 119, 6, 0.05));
        border-left: 5px solid #f59e0b;
    }
    
    .card-title {
        font-weight: 600;
        font-size: 1.2rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .salary-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .salary-value.high {
        color: #10b981;
    }
    
    .salary-value.low {
        color: #f59e0b;
    }
    
    /* Interactive card design pattern */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
        border-color: #6366f1;
    }
    
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load all artifacts
@st.cache_resource
def load_pipeline_artifacts():
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "best_model.pkl")
        scaler_path = os.path.join(base_dir, "scaler.pkl")
        encoders_path = os.path.join(base_dir, "encoders.pkl")
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        meta = joblib.load(encoders_path)
        encoders = meta["encoders"]
        feature_cols = meta["feature_cols"]
        return model, scaler, encoders, feature_cols, None
    except Exception as e:
        return None, None, None, None, e

# Load the trained model and helper artifacts
model, scaler, encoders, feature_cols, load_error = load_pipeline_artifacts()

# Main top banner design
st.markdown("""
    <div class="header-box">
        <h1>💼 Employee Salary Predictor</h1>
        <p>Predict whether an employee's salary class exceeds $50K/year using optimized Gradient Boosting Machine Learning.</p>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.error("❌ **Pipeline Artifacts Not Found!**")
    if load_error is not None:
        st.error(f"**Error Details:** {load_error}")
        st.exception(load_error)
        try:
            import sklearn
            st.info(f"ℹ️ **Deployed scikit-learn version:** `{sklearn.__version__}` (Local is `1.6.1`)")
        except Exception as sklearn_err:
            st.error(f"Could not import scikit-learn: {sklearn_err}")
    st.info("Please run `python train_pipeline.py` in your project folder first to train the model and dump `best_model.pkl`, `scaler.pkl`, and `encoders.pkl`.")
    st.stop()

# Layout of the main page - 3 Tabs for beautiful structure
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📂 Batch Processing", "📈 Model Insights"])

with tab1:
    st.markdown("### Input Employee Profile Details")
    st.write("Modify the values below and click **Predict Salary Class** to classify the employee's income.")
    
    # Organize fields in elegant rows/columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", 17, 75, 30, help="Employee age (restricted to training set range 17-75)")
        workclass = st.selectbox("Workclass / Employer Category", list(encoders['workclass'].classes_), index=3)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=12285, max_value=1490400, value=189785, step=5000, help="Census weight factor representing demographic scale")
        education = st.selectbox("Education Level", ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc", "11th", "Assoc-acdm", "10th", "7th-8th", "Prof-school", "9th", "12th", "Doctorate"])
        
    with col2:
        # Convert selected education level into the exact educational-num representing it
        edu_num_map = {
            "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6, 
            "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11, 
            "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
        }
        educational_num = edu_num_map[education]
        st.info(f"🎓 **Education Number mapped to:** {educational_num}")
        
        marital_status = st.selectbox("Marital Status", list(encoders['marital-status'].classes_), index=4)
        occupation = st.selectbox("Occupation / Job Role", list(encoders['occupation'].classes_), index=3)
        relationship = st.selectbox("Relationship in Household", list(encoders['relationship'].classes_), index=1)
        
    with col3:
        race = st.selectbox("Race / Ethnicity", list(encoders['race'].classes_), index=4)
        gender = st.selectbox("Gender", list(encoders['gender'].classes_), index=1)
        capital_gain = st.number_input("Capital Gains ($)", min_value=0, max_value=99999, value=0, step=1000)
        capital_loss = st.number_input("Capital Losses ($)", min_value=0, max_value=4356, value=0, step=100)
        hours_per_week = st.slider("Work Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", list(encoders['native-country'].classes_), index=39)

    st.markdown("---")
    
    # Button to predict
    if st.button("🔮 Predict Salary Class", use_container_width=True):
        # 1. Map input text to label encoded numbers
        try:
            w_class_encoded = encoders['workclass'].transform([workclass])[0]
            marital_encoded = encoders['marital-status'].transform([marital_status])[0]
            occup_encoded = encoders['occupation'].transform([occupation])[0]
            rel_encoded = encoders['relationship'].transform([relationship])[0]
            race_encoded = encoders['race'].transform([race])[0]
            gender_encoded = encoders['gender'].transform([gender])[0]
            country_encoded = encoders['native-country'].transform([native_country])[0]
            
            # 2. Build X array corresponding to the features columns order
            input_data = pd.DataFrame([{
                'age': age,
                'workclass': w_class_encoded,
                'fnlwgt': fnlwgt,
                'educational-num': educational_num,
                'marital-status': marital_encoded,
                'occupation': occup_encoded,
                'relationship': rel_encoded,
                'race': race_encoded,
                'gender': gender_encoded,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': country_encoded
            }])
            
            # 3. Reorder feature columns to exact match training
            input_data = input_data[feature_cols]
            
            # 4. Scale inputs using MinMaxScaler
            input_scaled = scaler.transform(input_data)
            
            # 5. Predict using model
            prediction = model.predict(input_scaled)[0]
            probs = model.predict_proba(input_scaled)[0]
            
            # Display Prediction
            if prediction == '>50K':
                st.markdown(f"""
                    <div class="prediction-card salary-high">
                        <div class="card-title">Salary Prediction Result</div>
                        <div class="salary-value high">👑 High Income (&gt;50K)</div>
                        <p style="color: #047857; margin-bottom: 0;">Predicted probability: <b>{probs[1]*100:.2f}%</b> likelihood of earning more than $50,000 annually.</p>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                    <div class="prediction-card salary-low">
                        <div class="card-title">Salary Prediction Result</div>
                        <div class="salary-value low">💼 Standard Income (&le;50K)</div>
                        <p style="color: #b45309; margin-bottom: 0;">Predicted probability: <b>{probs[0]*100:.2f}%</b> likelihood of earning less than or equal to $50,000 annually.</p>
                    </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

with tab2:
    st.markdown("### 📂 Upload CSV for Batch Prediction")
    st.write("Upload a CSV file containing employee features to calculate batch classifications instantly.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load and display sample of uploaded file
            df_upload = pd.read_csv(uploaded_file)
            st.markdown("##### Uploaded Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            # Check for necessary feature columns
            missing_cols = []
            expected_input_cols = ['age', 'workclass', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
            for col in expected_input_cols:
                if col not in df_upload.columns:
                    # Let's check educational-num as alternative to education
                    if col == 'education' and 'educational-num' in df_upload.columns:
                        continue
                    missing_cols.append(col)
                    
            if len(missing_cols) > 0:
                st.error(f"The uploaded file is missing required feature columns: {missing_cols}")
                st.stop()
                
            # Perform exact data preprocessing on uploaded file
            df_cleaned = df_upload.copy()
            
            # Fill missing/NaN values if any
            df_cleaned = df_cleaned.fillna("?")
            
            # Keep copy of clean columns
            df_cleaned['occupation'] = df_cleaned['occupation'].astype(str).replace({"?": "others"})
            df_cleaned['workclass'] = df_cleaned['workclass'].astype(str).replace({"?": "NotListed"})
            
            # Map education to edu_num if needed
            if 'educational-num' not in df_cleaned.columns:
                df_cleaned['educational-num'] = df_cleaned['education'].map(edu_num_map).fillna(9).astype(int)
            
            # Drop education if it's there
            if 'education' in df_cleaned.columns:
                df_cleaned = df_cleaned.drop(columns=['education'])
                
            # Encode strings using LabelEncoders
            for col in encoders:
                encoder = encoders[col]
                # Handle unexpected categories elegantly by replacing them with the mode or first category
                default_class = encoder.classes_[0]
                df_cleaned[col] = df_cleaned[col].apply(lambda val: val if val in encoder.classes_ else default_class)
                df_cleaned[col] = encoder.transform(df_cleaned[col].astype(str))
            
            # Order columns exactly
            df_cleaned = df_cleaned[feature_cols]
            
            # Scale
            X_scaled = scaler.transform(df_cleaned)
            
            # Predict
            batch_predictions = model.predict(X_scaled)
            batch_probs = model.predict_proba(X_scaled)
            
            # Join predictions back to uploaded dataframe
            df_output = df_upload.copy()
            df_output['Predicted Salary Class'] = batch_predictions
            df_output['Probability (<=50K)'] = [f"{p[0]:.4f}" for p in batch_probs]
            df_output['Probability (>50K)'] = [f"{p[1]:.4f}" for p in batch_probs]
            
            st.markdown("---")
            st.success("🎉 **Batch Prediction Completed Successfully!**")
            
            # High-end metrics summary of batch
            high_count = sum(batch_predictions == '>50K')
            low_count = sum(batch_predictions == '<=50K')
            total = len(df_output)
            
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-card">
                        <div class="metric-num" style="color: #6366f1;">{total}</div>
                        <div class="metric-label">Total Records Processed</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-num" style="color: #10b981;">{high_count} ({high_count/total*100:.1f}%)</div>
                        <div class="metric-label">Predicted &gt;50K (High)</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-num" style="color: #f59e0b;">{low_count} ({low_count/total*100:.1f}%)</div>
                        <div class="metric-label">Predicted &le;50K (Standard)</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### Download predicted reports")
            st.dataframe(df_output.head(20), use_container_width=True)
            
            # Download button
            csv = df_output.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV Report",
                data=csv,
                file_name="employee_predictions_report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

with tab3:
    st.markdown("### 📈 Model Metrics and Core Features")
    st.write("Analyze the trained Gradient Boosting Classifier metrics and dataset insights.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 🤖 Classification Metrics
        The Gradient Boosting Classifier was trained on the Adult Census dataset and achieved the following local performance:
        
        * **Testing Set Accuracy**: `87.10%`
        * **Standard Baseline Accuracy**: `~75.00%` (Imbalanced class default)
        * **Algorithm Optimized**: Gradient Boosting Machine (GBM)
        
        GBM handles mixed numerical and categorical features extremely well, and maps complex interactions (like capital-gain vs educational-num) natively without requiring complex feature engineering.
        """)
        
    with col2:
        st.markdown("""
        #### 🌟 Strongest Salary Predictors
        Our dataset analysis highlights these core drivers of individual salary levels:
        
        1. 📈 **Capital Gain & Loss**: The strongest financial predictor. Higher investments significantly correlate with exceeding the $50K threshold.
        2. 🎓 **Educational Level**: Higher degrees (`educational-num` > 12) increase the likelihood of high income by over **2.5x**.
        3. ⏳ **Work Hours per Week**: Full-time and overtime workers have exponentially higher classification success rates compared to part-time positions.
        4. 🧠 **Age**: Earnings peak in the 40-50 age bracket, matching professional experience maturity.
        """)
        
    st.markdown("---")
    st.info("ℹ️ **AICTE – Edunet Foundation Internship Project** | Developed under AI & ML program.")
