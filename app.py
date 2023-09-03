import os
import requests
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data
    
def remove_alphabets(value):
    if isinstance(value, str):
        return re.sub(r'[a-zA-Z\s,]', '', value)
    return value

st.title('Linear Regression Model for Exchange Rate Prediction')

# File Upload
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    country_name = pd.read_excel(uploaded_file, sheet_name = "Quarterly").columns[0].split(":")[0]
    st.write(f'Country: {country_name}')
    df_bop = pd.read_excel(uploaded_file, header = 4, sheet_name = "Quarterly")
    df_bop = df_bop.set_index('Unnamed: 0').T
    df_bop.index = df_bop.index.astype('period[Q-DEC]')
    df_bop = df_bop.iloc[:,:-6]
    
    # Check for the existence of the file
    if not os.path.exists("BOP_Exchange.xlsx"):
        with st.spinner("Downloading Exchange Rate Data..."):
    
            # Specify the URL to download the file from
            url = "https://github.com/safi842/Exchange-Rate-Prediction/releases/download/v1/BOP_Exchange.xlsx"
    
            # Download the file and save it in the current directory
            response = requests.get(url)
            with open("BOP_Exchange.xlsx", "wb") as f:
                f.write(response.content)
            st.write("File downloaded successfully.")

    country_loc = list(pd.read_excel('BOP_Exchange.xlsx',sheet_name = 'Nominal', header = 3).columns).index(country_name)
    df_ex = pd.read_excel('BOP_Exchange.xlsx',sheet_name = 'Nominal', header = 4)
    df_ex.index = df_ex.set_index('Unnamed: 0').index.to_period('Q')
    df_ex = df_ex[~df_ex.index.duplicated(keep='last')]
    df_ex = df_ex[[df_ex.columns[country_loc]]]
    df_ex.columns = ['Exchange Rate']
    df = df_bop.merge(df_ex, left_index=True, right_index=True, how='left')
    df = df.drop(columns=df.columns[df.eq("...").any()])
    df = df.applymap(remove_alphabets).apply(pd.to_numeric)
    # Rename 'Credit' and 'Debit' columns by adding prefixes based on the preceding column
    new_columns = []
    prefix = None
    for col in df.columns:
        if 'Credit' in col:
            prefix = new_columns[-1]  # Take the last added column name as the prefix
            new_columns.append(f"{prefix}_Credit")
        elif 'Debit' in col:
            new_columns.append(f"{prefix}_Debit")
        else:
            new_columns.append(col)
    # Update the DataFrame with the new column names
    df.columns = new_columns
    df = df.rename_axis('Date').reset_index()
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df = df.set_index('Date')
    data = df
    selected_vars = st.multiselect("Select variables to include in the model", options=data.columns.tolist(), default=['Current account', 'Goods', 'Services', 'Exchange Rate'])
    data = data[selected_vars]
    #st.session_state.download_data = data
    st.write('Data Preview:')
    st.write(data.head())

    st.subheader('Lag Variables')
    create_lag = st.selectbox("Would you like to create lag variables?", ["Yes", "No"])
    if create_lag == "Yes":
        num_lags = st.number_input("How many lag variables?", min_value=1, max_value=20)

    st.subheader('Seasonal Decomposition')
    seasonal_decompose_option = st.checkbox("Would you like to perform seasonal decomposition?")
    if seasonal_decompose_option:
        variables_to_decompose = st.multiselect("Select variables to decompose", options=data.columns)
        include_trend = st.checkbox("Include Trend Component")
        include_seasonal = st.checkbox("Include Seasonal Component")
        include_residual = st.checkbox("Include Residual Component")

    st.subheader('Visualization Options')
    plot_seasonal = st.checkbox("Plot Seasonal Decomposition")
    plot_heatmap = st.checkbox("Plot Correlation Heatmap")

    st.subheader('Preprocessing Options')
    standardize_option = st.checkbox("Would you like to standardize the variables before modelling?")

    st.subheader('Backward Elimination Method')
    backward_elimination_option = st.checkbox("Would you like to use VIF (Variance Inflation Factor) based backward elimination before modelling")

    
    # Run Model Button
    if st.button("Run Model"):
        with st.spinner("Running the model..."):
            
            # Create lag variables
            if create_lag == "Yes":
                for col in data.columns:
                    for i in range(1, num_lags + 1):
                        data[f"{col}_Lag{i}"] = data[col].shift(i)
            
            # Seasonal Decomposition
            if seasonal_decompose_option:
                decomposed_df = pd.DataFrame(index=data.index)
                for var in variables_to_decompose:
                    decomposition = seasonal_decompose(data[var], period=4)
                    if include_trend:
                        decomposed_df[f"{var}_Trend"] = decomposition.trend.dropna()
                    if include_seasonal:
                        decomposed_df[f"{var}_Seasonal"] = decomposition.seasonal.dropna()
                    if include_residual:
                        decomposed_df[f"{var}_Residual"] = decomposition.resid.dropna()
            
                data = data.merge(decomposed_df, left_index=True, right_index=True, how='left').dropna()
                
                with st.expander("Seasonal Decomposition Plots"):
                    if plot_seasonal:
                        for var in variables_to_decompose:
                            fig = go.Figure()

                            fig.add_trace(go.Scatter(x=data.index, y=data[var], mode='lines', name='Original'))
                            fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df[f"{var}_Trend"], mode='lines', name='Trend'))
                            fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df[f"{var}_Seasonal"], mode='lines', name='Seasonal'))
                            fig.add_trace(go.Scatter(x=decomposed_df.index, y=decomposed_df[f"{var}_Residual"], mode='lines', name='Residual'))

                            fig.update_layout(title=f'Seasonal Decomposition of {var}',
                                            xaxis_title='Time',
                                            yaxis_title=f'{var} Value')
                
                            st.plotly_chart(fig)
            
            # Drop NaN due to lag and decomposition
            data.dropna(inplace=True)
            
            # Correlation Heatmap
            if plot_heatmap:
                st.write('Correlation Heatmap')
                corr = data.corr()
                fig = px.imshow(corr, x=corr.columns, y=corr.columns, color_continuous_scale='Viridis')
                fig.update_layout(
                    autosize=False,
                width=800,
                height=800,
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4
                    ),
                )
                st.plotly_chart(fig)
            
            # Prepare features and target
            X = data.drop('Exchange Rate', axis=1)
            y = data['Exchange Rate']
            
            # Standardize if the user opts for it
            if standardize_option:
                scaler = StandardScaler()
                X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            
            #VIF based Backward Elimination
            features_to_keep = X.columns.tolist()
            if backward_elimination_option:
                while True:
                    vif_df = calculate_vif(X[features_to_keep])
                    max_vif = vif_df['VIF'].max()
                    if max_vif <= 5:
                        break
                    remove = vif_df.sort_values('VIF', ascending=False).iloc[0]
                    features_to_keep.remove(remove['feature'])
            
            # p-value-Based Backward Elimination
            X = sm.add_constant(X[features_to_keep])  # adding a constant for the intercept
            model = sm.OLS(y, X).fit()
            while True:
                p_values = model.pvalues
                max_p_value = p_values.max()
                if max_p_value < 0.05:
                    break
                remove = p_values.idxmax()
                if remove == 'const':  # Don't remove the constant
                    break
                X.drop([remove], axis=1, inplace=True)
                model = sm.OLS(y, X).fit()
            st.success("Model Run Succesfully!")
            # Model Evaluation and Summary
            st.write(model.summary())

