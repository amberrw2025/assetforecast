import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from data_loader import get_btc_data
from predictor import train_and_predict, generate_future_predictions
from utils import validate_date_range

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Bitcoin Price Prediction")
st.markdown("""
This website predicts Bitcoin prices using machine learning models. You can:
- Choose between Linear Regression and Random Forest models
- Get predictions for the next day and one month ahead
- Adjust model parametersValueError: all the input arrays must have same number of dimensions, but the array at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
- View 6-year price history
""")

# Sidebar for date selection
st.sidebar.header("Date Range Selection")
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.now() - timedelta(days=365*5)  # 5 years ago
)
end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.now()
)

# Validate date range
is_valid, message = validate_date_range(start_date, end_date)
if not is_valid:
    st.error(message)
    st.stop()

# Convert dates to datetime objects for data fetching
start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.min.time())

# Model selection and parameters
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["linear", "random_forest", "xgboost", "all"],
    format_func=lambda x: {
        "linear": "Linear Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "all": "All Models"
    }[x]
)

params = {
    'lookback': 7,
    # 'lookback': st.sidebar.slider(
    #     "Lookback Period (days)",
    #     min_value=1,
    #     max_value=30,
    #     value=7,
    #     help="Number of past days to consider for prediction"
    # )
}
st.sidebar.info("The model uses a fixed lookback period of 7 days, resulting in 10 input features.")

if model_type in ['random_forest', 'xgboost', 'all']:
    params['n_estimators'] = st.sidebar.slider(
        "Number of Trees",
        min_value=50,
        max_value=200,
        value=100,
        help="Number of trees in the model"
    )
    params['max_depth'] = st.sidebar.slider(
        "Max Tree Depth",
        min_value=5,
        max_value=20,
        value=10,
        help="Maximum depth of each tree"
    )

if model_type in ['xgboost', 'all']:
    params['learning_rate'] = st.sidebar.slider(
        "Learning Rate",
        min_value=0.01,
        max_value=0.3,
        value=0.1,
        step=0.01,
        help="Step size shrinkage used in update to prevent overfitting"
    )
    params['subsample'] = st.sidebar.slider(
        "Subsample Ratio",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Subsample ratio of the training instances"
    )
    params['colsample_bytree'] = st.sidebar.slider(
        "Column Sample Ratio",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.1,
        help="Subsample ratio of columns when constructing each tree"
    )

# Fetch data
df = get_btc_data(start_datetime, end_datetime)

if df is not None and not df.empty:
    # Defensive check for 'Close' column
    if 'Close' not in df.columns or df['Close'].dropna().empty:
        st.error("No valid 'Close' price data available.")
        st.stop()
    
    # Display current price metrics
    current_price = df['Close'].dropna().iloc[-1].item()
    price_change = df['Close'].dropna().iloc[-1].item() - df['Close'].dropna().iloc[-2].item()
    price_change_pct = (price_change / df['Close'].dropna().iloc[-2].item()) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:,.2f}")
    with col2:
        st.metric("24h Change", f"${price_change:,.2f}", f"{price_change_pct:.2f}%")
    with col3:
        st.metric("24h Volume", f"${df['Volume'].iloc[-1].item():,.0f}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Historical Data", "6-Year Price History", "Price Forecast"])
    
    # Calculate predictions
    models, predictions, confidence_intervals = train_and_predict(df, model_type=model_type, params=params)
    
    # Generate future predictions for visualization
    if models and predictions:
        future_dates, future_predictions, future_intervals = generate_future_predictions(
            df, models, predictions, confidence_intervals, days_ahead=180
        )
    
    with tab1:
        # Show prediction
        if predictions:
            pred_tab1, pred_tab2 = st.tabs(["Next Day Prediction", "One Month Prediction"])
            
            with pred_tab1:
                st.subheader("Next Day Price Prediction")
                
                if 'linear' in predictions:
                    st.markdown("### Linear Regression Model")
                    pred = predictions['linear']
                    lower, upper = confidence_intervals['linear']
                    pred_change = pred - current_price
                    pred_change_pct = (pred_change / current_price) * 100
                    st.metric(
                        "Predicted Price",
                        f"${pred:,.2f}",
                        f"{pred_change_pct:+.2f}%"
                    )
                    st.info(f"95% Confidence Interval: ${lower:,.2f} - ${upper:,.2f}")
                
                if 'random_forest' in predictions:
                    st.markdown("### Random Forest Model")
                    pred = predictions['random_forest']
                    lower, upper = confidence_intervals['random_forest']
                    pred_change = pred - current_price
                    pred_change_pct = (pred_change / current_price) * 100
                    st.metric(
                        "Predicted Price",
                        f"${pred:,.2f}",
                        f"{pred_change_pct:+.2f}%"
                    )
                    st.info(f"95% Prediction Interval: ${lower:,.2f} - ${upper:,.2f}")
                
                if 'xgboost' in predictions:
                    st.markdown("### XGBoost Model")
                    pred = predictions['xgboost']
                    lower, upper = confidence_intervals['xgboost']
                    pred_change = pred - current_price
                    pred_change_pct = (pred_change / current_price) * 100
                    st.metric(
                        "Predicted Price",
                        f"${pred:,.2f}",
                        f"{pred_change_pct:+.2f}%"
                    )
                    st.info(f"95% Prediction Interval: ${lower:,.2f} - ${upper:,.2f}")
                
                if 'combined' in predictions:
                    st.markdown("### Combined Model Prediction")
                    pred = predictions['combined']
                    lower, upper = confidence_intervals['combined']
                    pred_change = pred - current_price
                    pred_change_pct = (pred_change / current_price) * 100
                    st.metric(
                        "Predicted Price",
                        f"${pred:,.2f}",
                        f"{pred_change_pct:+.2f}%"
                    )
                    st.info(f"95% Combined Prediction Interval: ${lower:,.2f} - ${upper:,.2f}")
                    st.markdown("""
                    The combined prediction uses a weighted average of all selected models, where weights are based on the inverse of prediction intervals.
                    Models with narrower prediction intervals (higher confidence) are given more weight.
                    """)
            
            with pred_tab2:
                st.subheader("One Month Price Prediction")
                
                if 'linear' in predictions:
                    st.markdown("### Linear Regression Model")
                    daily_returns = df['Close'].pct_change().dropna()
                    monthly_volatility = daily_returns.std() * np.sqrt(30)
                    monthly_pred = current_price * (1 + (predictions['linear'] - current_price) / current_price * 30)
                    monthly_lower = monthly_pred * (1 - 1.96 * monthly_volatility.item())
                    monthly_upper = monthly_pred * (1 + 1.96 * monthly_volatility.item())
                    monthly_change = monthly_pred - current_price
                    monthly_change_pct = (monthly_change / current_price) * 100
                    st.metric(
                        "One Month Prediction",
                        f"${monthly_pred:,.2f}",
                        f"{monthly_change_pct:+.2f}%"
                    )
                    st.info(f"95% Confidence Interval: ${monthly_lower:,.2f} - ${monthly_upper:,.2f}")
                
                if 'random_forest' in predictions:
                    st.markdown("### Random Forest Model")
                    daily_returns = df['Close'].pct_change().dropna()
                    monthly_volatility = daily_returns.std() * np.sqrt(30)
                    monthly_pred = current_price * (1 + (predictions['random_forest'] - current_price) / current_price * 30)
                    monthly_lower = monthly_pred * (1 - 1.96 * monthly_volatility.item())
                    monthly_upper = monthly_pred * (1 + 1.96 * monthly_volatility.item())
                    monthly_change = monthly_pred - current_price
                    monthly_change_pct = (monthly_change / current_price) * 100
                    st.metric(
                        "One Month Prediction",
                        f"${monthly_pred:,.2f}",
                        f"{monthly_change_pct:+.2f}%"
                    )
                    st.info(f"95% Prediction Interval: ${monthly_lower:,.2f} - ${monthly_upper:,.2f}")
                
                if 'xgboost' in predictions:
                    st.markdown("### XGBoost Model")
                    daily_returns = df['Close'].pct_change().dropna()
                    monthly_volatility = daily_returns.std() * np.sqrt(30)
                    monthly_pred = current_price * (1 + (predictions['xgboost'] - current_price) / current_price * 30)
                    monthly_lower = monthly_pred * (1 - 1.96 * monthly_volatility.item())
                    monthly_upper = monthly_pred * (1 + 1.96 * monthly_volatility.item())
                    monthly_change = monthly_pred - current_price
                    monthly_change_pct = (monthly_change / current_price) * 100
                    st.metric(
                        "One Month Prediction",
                        f"${monthly_pred:,.2f}",
                        f"{monthly_change_pct:+.2f}%"
                    )
                    st.info(f"95% Prediction Interval: ${monthly_lower:,.2f} - ${monthly_upper:,.2f}")
                
                if 'combined' in predictions:
                    st.markdown("### Combined Model Prediction")
                    daily_returns = df['Close'].pct_change().dropna()
                    monthly_volatility = daily_returns.std() * np.sqrt(30)
                    monthly_pred = current_price * (1 + (predictions['combined'] - current_price) / current_price * 30)
                    monthly_lower = monthly_pred * (1 - 1.96 * monthly_volatility.item())
                    monthly_upper = monthly_pred * (1 + 1.96 * monthly_volatility.item())
                    monthly_change = monthly_pred - current_price
                    monthly_change_pct = (monthly_change / current_price) * 100
                    st.metric(
                        "One Month Prediction",
                        f"${monthly_pred:,.2f}",
                        f"{monthly_change_pct:+.2f}%"
                    )
                    st.info(f"95% Combined Prediction Interval: ${monthly_lower:,.2f} - ${monthly_upper:,.2f}")
                
                st.warning("""
                Note: Monthly predictions are based on historical volatility and should be interpreted with caution.
                The cryptocurrency market is highly volatile and past performance is not indicative of future results.
                """)
        else:
            st.info("No prediction available.")
    
    with tab2:
        # Display historical data
        st.subheader("Historical Price Data")
        st.dataframe(
            df.style.format({
                'Open': '${:,.2f}',
                'High': '${:,.2f}',
                'Low': '${:,.2f}',
                'Close': '${:,.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("Bitcoin Price History (6 Years) - Training/Testing Split")
        
        # Fetch 6-year historical data
        six_years_ago = datetime.now() - timedelta(days=365*6)
        
        try:
            # Use yfinance directly for simplicity
            import yfinance as yf
            btc_ticker = yf.Ticker("BTC-USD")
            hist_data = btc_ticker.history(start=six_years_ago, end=datetime.now())
            
            if not hist_data.empty:
                # Calculate 80/20 split
                total_points = len(hist_data)
                split_point = int(total_points * 0.8)
                
                # Split the data
                train_data = hist_data.iloc[:split_point]
                test_data = hist_data.iloc[split_point:]
                
                # Create chart with training and testing data
                fig = go.Figure()
                
                # Add training data (80%)
                fig.add_trace(go.Scatter(
                    x=train_data.index,
                    y=train_data['Close'],
                    mode='lines',
                    name='Training Data (80%)',
                    line=dict(color='#2E86AB', width=2)
                ))
                
                # Add testing data (20%)
                fig.add_trace(go.Scatter(
                    x=test_data.index,
                    y=test_data['Close'],
                    mode='lines',
                    name='Testing Data (20%)',
                    line=dict(color='#F24236', width=2)
                ))
                
                fig.update_layout(
                    title='Bitcoin Price - Training (80%) vs Testing (20%) Data',
                    xaxis_title='Date',
                    yaxis_title='Price (USD)',
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display split information
                st.info(f"📊 Data Split: {len(train_data):,} training points | {len(test_data):,} testing points")
                
                # Simple stats
                current_price = hist_data['Close'].iloc[-1]
                max_price = hist_data['High'].max()
                min_price = hist_data['Low'].min()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:,.2f}")
                with col2:
                    st.metric("6-Year High", f"${max_price:,.2f}")
                with col3:
                    st.metric("6-Year Low", f"${min_price:,.2f}")
            else:
                st.error("No data available")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Please check your internet connection and try again.")
    
    with tab4:
        st.subheader("Bitcoin Price Forecast")
        
        if models and predictions and 'combined' in predictions and 'combined' in future_predictions and future_predictions['combined'] is not None and len(future_predictions['combined']) > 0:
            # Create the figure
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#2E86AB', width=2)
            ))
            
            # Add future predictions
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions['combined'],
                mode='lines',
                name='Predicted Price',
                line=dict(color='#F24236', width=2)
            ))
            
            # Add vertical line for current date
            fig.add_vline(
                x=df.index[-1].to_pydatetime(),
                line_dash="dash",
                line_color="gray",
                annotation_text="Current Date",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title='Bitcoin Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price (USD)',
                height=600,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            ### Forecast Details
            - **Blue Line**: Historical Bitcoin prices
            - **Red Line**: Predicted prices for the next 6 months
            - **Vertical Line**: Current date
            """)
        else:
            st.info("Please select 'All Models' in the sidebar to see the forecast.")
    
    # Add disclaimer
    st.markdown("---")
    st.markdown("""
    ### Disclaimer
    This app is for educational purposes only. The predictions are based on historical data and machine learning models,
    and should not be used as financial advice. Cryptocurrency markets are highly volatile and unpredictable.
    Always do your own research before making any investment decisions.
    """)
else:
    st.error("Failed to fetch Bitcoin data. Please try again in a few minutes or try a different date range.") 