import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="Housing Price Predictor", layout="wide")

st.title("🏡 California Housing Price Prediction")
st.markdown("""
This app uses a **Ridge Regression** model to predict the median house value in California districts. 
Adjust the sliders in the sidebar to simulate different housing conditions.
""")
@st.cache_resource
def build_model():
    raw_data = fetch_california_housing()
    X = pd.DataFrame(raw_data.data, columns=raw_data.feature_names)
    y = raw_data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    return model, mse, X, y

model, mse, X_data, y_data = build_model()

st.sidebar.header("Adjust Features")

def user_input_features():
    inputs = {}
    for col in X_data.columns:
        min_val = float(X_data[col].min())
        max_val = float(X_data[col].max())
        mean_val = float(X_data[col].mean())
        
        if col == 'Latitude':
            inputs[col] = st.sidebar.slider(col, min_val, max_val, mean_val, step=0.01)
        elif col == 'Longitude':
            inputs[col] = st.sidebar.slider(col, min_val, max_val, mean_val, step=0.01)
        else:
            inputs[col] = st.sidebar.slider(col, min_val, max_val, mean_val)
            
    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction")
    
    prediction = model.predict(input_df)
    
    price_fmt = f"${prediction[0] * 100000:,.2f}"
    
    st.metric(label="Estimated Median House Value", value=price_fmt)
    
    st.info(f"Model Mean Squared Error (on Test Set): {mse:.4f}")

    st.markdown("### Feature Inputs Summary")
    st.dataframe(input_df, hide_index=True)

with col2:
    st.subheader("Location Context")
    map_data = pd.DataFrame({
        'lat': [input_df['Latitude'][0]],
        'lon': [input_df['Longitude'][0]]
    })
    st.map(map_data, zoom=5)

st.divider()

st.subheader("Model Insights: Feature Importance")
st.write("Which features have the biggest impact on price? (Positive = Increases Price, Negative = Decreases Price)")

coefficients = model.named_steps['ridge'].coef_
feature_names = X_data.columns

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
# Color bars: Green for positive, Red for negative
colors = ['green' if x > 0 else 'red' for x in coef_df['Coefficient']]
sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette=colors, ax=ax)
ax.set_title("Ridge Regression Coefficients")
st.pyplot(fig)