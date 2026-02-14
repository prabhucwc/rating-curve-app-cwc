import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from io import BytesIO # Import BytesIO for Excel download

st.set_page_config(layout="wide")

st.title("Excel Data Uploader, Cleaner, Rating Curve Fitter and Predictor")
st.write("Upload an Excel file to see it cleaned, a rating curve fitted, and missing discharge values predicted.")

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])

# Define the Stage Discharge Equation function
def rating_curve(H, C, H0, n):
    effective_H = np.maximum(H - H0, 0)
    return C * (effective_H ** n)

if uploaded_file is not None:
    try:
        # Load the Excel file
        df = pd.read_excel(uploaded_file)

        # Initial cleaning: Drop the first two rows and 'Unnamed: 4' column
        df = df.drop(df.index[[0, 1]])
        df = df.drop(columns=['Unnamed: 4'])
        df = df.reset_index(drop=True)

        # Rename columns
        new_column_names = ['Station Code', 'Date', 'Water Level', 'Discharge']
        df.columns = new_column_names

        st.success("File uploaded and initial cleaning successful!")

        # Convert 'Water Level' and 'Discharge' to numeric, coercing errors to NaN
        df['Water Level'] = pd.to_numeric(df['Water Level'], errors='coerce')
        df['Discharge'] = pd.to_numeric(df['Discharge'], errors='coerce')

        # Create a boolean column to flag rows where 'Discharge' was originally NaN
        df['initial_discharge_missing'] = df['Discharge'].isna()

        st.subheader("Raw Data after Initial Cleaning and Type Conversion")
        st.dataframe(df.head())
        st.write(f"Total rows: {len(df)}")
        st.write(f"Rows with missing Discharge values: {df['initial_discharge_missing'].sum()}")

        st.markdown("--- ")
        st.subheader("Splitting Data for Curve Fitting and Prediction")

        # Split data into df_for_fitting and df_for_prediction
        df_for_fitting = df.dropna(subset=['Water Level', 'Discharge']).copy()
        df_for_prediction = df[df['Discharge'].isna() & df['Water Level'].notna()].copy()

        st.write("**Data for Curve Fitting (both Water Level and Discharge present):**")
        st.dataframe(df_for_fitting.head())
        st.write(f"Number of rows for fitting: {len(df_for_fitting)}")

        st.write("**Data for Predicting Missing Discharge (Water Level present, Discharge missing):**")
        st.dataframe(df_for_prediction.head())
        st.write(f"Number of rows for prediction: {len(df_for_prediction)}")

        # --- Rating Curve Fitting Section ---
        if not df_for_fitting.empty and len(df_for_fitting) >= 3:
            st.markdown("--- ")
            st.subheader("Fitting Stage Discharge Rating Curve")

            water_level_data = df_for_fitting['Water Level'].values
            discharge_data = df_for_fitting['Discharge'].values

            try:
                min_observed_H = water_level_data.min()
                H0_initial = min_observed_H - 0.1
                p0 = [1.0, H0_initial, 2.0]

                # Bounds for C, H0, n
                lower_bounds = [1e-5, min_observed_H - (water_level_data.max() - min_observed_H) * 2, 1.5]
                upper_bounds = [np.inf, min_observed_H + 1e-5, 2.5] # H0 upper bound adjusted for flexibility

                params, covariance = curve_fit(
                    rating_curve,
                    water_level_data,
                    discharge_data,
                    p0=p0,
                    bounds=(lower_bounds, upper_bounds),
                    maxfev=10000
                )

                C_opt, H0_opt, n_opt = params
                st.write(f"📌 Fitted Rating Curve Equation: Q = {C_opt:.3f} * (H - {H0_opt:.3f})^{n_opt:.3f}")

                # Calculate 'Discharge computed' for the ENTIRE original DataFrame
                df['Discharge computed'] = rating_curve(df['Water Level'], C_opt, H0_opt, n_opt)

                # Calculate 'Difference b/n obs and computed' only for rows with original Discharge
                df['Difference b/n obs and computed'] = np.where(
                    df['initial_discharge_missing'] == False, # Only where Discharge was observed
                    df['Discharge'] - df['Discharge computed'],
                    np.nan # Set to NaN for rows where Discharge was originally missing
                )

                # Calculate 'Relative difference (%)' for observed values
                df['Relative difference (%)'] = np.where(
                    (df['initial_discharge_missing'] == False) & (df['Discharge computed'] != 0),
                    (df['Difference b/n obs and computed'] / df['Discharge computed']) * 100,
                    np.nan # Set to NaN if Discharge was missing or computed Discharge is 0
                )

                # Calculate Overall Standard Error (RMSE) using only non-null differences
                valid_differences = df['Difference b/n obs and computed'].dropna()
                if not valid_differences.empty:
                    overall_standard_error = np.sqrt(np.mean(valid_differences**2))
                    st.write(f"Overall Standard Error (RMSE) of the fit: {overall_standard_error:.3f}")

                    if overall_standard_error >= 15:
                        st.warning("⚠️ Warning: The RMSE is high (>= 15), indicating a potentially poor fit. Consider checking data quality or the model suitability.")
                    else:
                        st.success("✅ Good fit: The RMSE is within acceptable limits.")
                else:
                    st.warning("Could not calculate RMSE as there are no observed discharge values for comparison.")

                # --- Status Flagging and Prediction Integration ---
                # Default for rows where Water Level is NaN (before other flags)
                df['Data Status'] = 'Discarded (Water Level Missing)'

                # Mark all valid water level rows as 'Not Applicable' initially, then refine
                df.loc[df['Water Level'].notna(), 'Data Status'] = 'Not Applicable'

                # Flag 'Observed (Used for Fitting)' for data used in fitting
                df.loc[~df['initial_discharge_missing'] & df['Water Level'].notna(), 'Data Status'] = 'Observed (Used for Fitting)'

                # Flag 'Discarded (High Relative Difference)' for observed values with high relative difference
                df.loc[
                    (df['Data Status'] == 'Observed (Used for Fitting)') &
                    (df['Relative difference (%)'].abs() > 10),
                    'Data Status'
                ] = 'Discarded (High Relative Difference)'

                # Flag 'Missing Discharge Predicted' for data where discharge was missing and now predicted
                df.loc[df['initial_discharge_missing'] & df['Water Level'].notna(), 'Data Status'] = 'Missing Discharge Predicted'

                # --- Plotting ---
                fig = px.scatter(df_for_fitting, x="Water Level", y="Discharge",
                                 title="Water Level vs. Discharge with Fitted Stage Discharge Equation",
                                 labels={
                                     "Water Level": "Water Level (units)",
                                     "Discharge": "Discharge (units)"
                                 },
                                 hover_name="Date",
                                 hover_data={'Water Level': ':.2f', 'Discharge': ':.3f', 'Station Code': True},
                                 width=900, height=600)

                # Add the fitted curve to the plot
                H_curve = np.linspace(water_level_data.min(), water_level_data.max(), 100)
                Q_curve = rating_curve(H_curve, C_opt, H0_opt, n_opt)
                fig.add_trace(go.Scatter(
                    x=H_curve,
                    y=Q_curve,
                    mode='lines',
                    name=f'Fitted Curve: Q = {C_opt:.3f} * (H - {H0_opt:.3f})^{n_opt:.3f}',
                    line=dict(color='red', width=3)
                ))

                # Add predicted points for visual clarity (only for those that were originally missing and are now predicted)
                df_predicted_display = df[df['Data Status'] == 'Missing Discharge Predicted']
                if not df_predicted_display.empty:
                    fig.add_trace(go.Scatter(
                        x=df_predicted_display['Water Level'],
                        y=df_predicted_display['Discharge computed'],
                        mode='markers',
                        name='Predicted Missing Discharge',
                        marker=dict(color='green', size=8, symbol='star', line=dict(width=0.5, color='DarkGreen')),
                        hovertext=df_predicted_display.apply(lambda row: f"Date: {row['Date']}<br>Water Level: {row['Water Level']:.2f}<br>Predicted Discharge: {row['Discharge computed']:.3f}", axis=1)
                    ))

                # Add discarded points to the plot
                df_discarded_display = df[df['Data Status'] == 'Discarded (High Relative Difference)']
                if not df_discarded_display.empty:
                    fig.add_trace(go.Scatter(
                        x=df_discarded_display['Water Level'],
                        y=df_discarded_display['Discharge'],
                        mode='markers',
                        name='Discarded (High Relative Diff.)',
                        marker=dict(color='orange', size=8, symbol='x', line=dict(width=0.5, color='DarkOrange')),
                        hovertext=df_discarded_display.apply(lambda row: f"Date: {row['Date']}<br>Water Level: {row['Water Level']:.2f}<br>Observed Discharge: {row['Discharge']:.3f}<br>Relative Diff: {row['Relative difference (%)']:.2f}%", axis=1)
                    ))

                fig.update_layout(
                    xaxis_title="Water Level",
                    yaxis_title="Discharge",
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("--- ")
                st.subheader("Comprehensive Processed Data")
                st.dataframe(df.head())
                st.dataframe(df.tail())

                # --- Excel Download ---
                # Create an Excel writer object
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Processed Data', index=False)
                    # Add summary statistics if RMSE was calculated
                    if 'overall_standard_error' in locals() and overall_standard_error is not None:
                        summary_df = pd.DataFrame({
                            'Metric': ['Fitted C', 'Fitted H0', 'Fitted n', 'Overall RMSE'],
                            'Value': [f'{C_opt:.3f}', f'{H0_opt:.3f}', f'{n_opt:.3f}', f'{overall_standard_error:.3f}']
                        })
                        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
                output.seek(0)
                st.download_button(
                    label="Download Processed Data as Excel",
                    data=output,
                    file_name="processed_stage_discharge_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"Error fitting rating curve or predicting: {e}")
                st.info("Ensure your 'Water Level' and 'Discharge' columns have sufficient numerical data (at least 3 points) and are positive for fitting.")

        else:
            st.warning("Not enough valid data points for curve fitting (need at least 3 rows with both Water Level and Discharge). No rating curve will be fitted.")

    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
else:
    st.info("Please upload an Excel file to get started.")

st.markdown("--- ")
st.write("**How to use:**")
st.write("1. Click 'Browse files' to upload your .xlsx or .xls file.")
st.write("2. The app will automatically clean the data, identify missing discharge values, and split the data.")
st.write("3. If sufficient data is available, it will fit a rating curve, predict missing discharge values, and evaluate the fit.")
