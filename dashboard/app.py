import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import os

# Page config
st.set_page_config(
    page_title="Network Congestion Dashboard",
    page_icon="ðŸ“Š",
    layout="wide"
)

# Load trained models
@st.cache_resource
def load_models():
    """Load pre-trained XGBoost models for each router"""
    try:
        models = {}
        current_dir = os.path.dirname(os.path.abspath(__file__))  # dashboard folder
        parent_dir = os.path.dirname(current_dir)  # project_folder
        models_dir = os.path.join(parent_dir, "models")
        
        for router in ['A', 'B', 'C']:
            model_path = os.path.join(models_dir, f"model{router}_p.pkl")
            models[router] = joblib.load(model_path)
        return models
    except FileNotFoundError as e:
        st.error(f"Model files not found at {models_dir}: {e}")
        return None

# Load label encoder for Impact column
@st.cache_resource
def load_label_encoder():
    """Load or create label encoder for Impact column"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        encoder_path = os.path.join(current_dir, "label_encoder_impact.pkl")
        return joblib.load(encoder_path)
    except FileNotFoundError:
        # Create a default encoder if file doesn't exist
        le = LabelEncoder()
        le.fit(['None', 'Low', 'Medium', 'High'])  # Default categories
        return le

def safe_datetime_arithmetic(timestamp, hours_offset, operation='subtract'):
    """
    Safely perform datetime arithmetic avoiding pandas Timestamp issues
    """
    # Convert input to standard datetime
    if hasattr(timestamp, 'to_pydatetime'):
        dt = timestamp.to_pydatetime()
    elif isinstance(timestamp, str):
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = timestamp
    
    # Perform operation
    if operation == 'subtract':
        result_dt = dt - timedelta(hours=hours_offset)
    else:  # add
        result_dt = dt + timedelta(hours=hours_offset)
    
    return result_dt

def create_training_samples_single(data, target_timestamp, hours=12):
    """
    Create features for a single prediction using last 12 hours of data
    """
    # Safe datetime arithmetic
    target_dt = safe_datetime_arithmetic(target_timestamp, 0, 'add')  # normalize
    window_start_dt = safe_datetime_arithmetic(target_dt, hours, 'subtract')
    
    # Convert to pandas timestamps for filtering
    window_start = pd.Timestamp(window_start_dt)
    window_end = pd.Timestamp(target_dt)
    
    window_data = data[(data['Timestamp'] >= window_start) & (data['Timestamp'] < window_end)]
    
    if window_data.empty:
        return None
    
    features = []
    time_indexed = sorted(window_data['Timestamp'].unique())
    
    for ts in time_indexed:
        ts_data = window_data[window_data['Timestamp'] == ts]
        for router in ['Router_A', 'Router_B', 'Router_C']:
            router_data = ts_data[ts_data['Device Name'] == router]
            if not router_data.empty:
                row = router_data.iloc[0]
                features.extend([
                    float(row['Traffic Volume (MB/s)']),
                    float(row['Latency (ms)']),
                    float(row['Bandwidth Used (MB/s)']),
                    float(row['Bandwidth Allocated (MB/s)']),
                    float(row['total_avg_app_traffic']),
                    float(row['total_peak_app_traffic']),
                    float(row['Impact_encoded']),
                    float(row['total_peak_user_usage']),
                    float(row['total_logins'])
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    return np.array(features).reshape(1, -1)

def predict_congestion_proba(df, models, target_timestamp):
    """
    Predict congestion probabilities for all routers at target timestamp
    """
    features = create_training_samples_single(df, target_timestamp)
    
    if features is None:
        return None
    
    proba_results = {}
    for router in ['Router_A', 'Router_B', 'Router_C']:
        model_key = router.split('_')[1]  # Convert Router_A to A
        if model_key in models:
            try:
                proba = models[model_key].predict_proba(features)[0, 1]
                proba_results[router] = float(proba)
            except Exception as e:
                st.warning(f"Error predicting for {router}: {e}")
                proba_results[router] = 0.0
        else:
            proba_results[router] = 0.0
    
    return proba_results

def bandwidth_recommendation(window_data, congestion_probs):
    """
    Generate bandwidth recommendations based on congestion probabilities and current usage
    """
    recommendations = {}
    
    for router in ['Router_A', 'Router_B', 'Router_C']:
        router_data = window_data[window_data['Device Name'] == router]
        
        if router_data.empty:
            recommendations[router] = {
                'action': 'monitor',
                'amount': 0.0,
                'reason': 'No data available in window',
                'utilization': 0.0,
                'congestion_prob': 0.0,
                'current_allocated': 0.0,
                'current_used': 0.0
            }
            continue
        
        congestion_prob = congestion_probs.get(router, 0.0)
        
        # Calculate key metrics
        current_allocated = float(router_data['Bandwidth Allocated (MB/s)'].mean())
        current_used = float(router_data['Bandwidth Used (MB/s)'].mean())
        avg_latency = float(router_data['Latency (ms)'].mean())
        
        # Calculate utilization percentage
        utilization = (current_used / current_allocated) if current_allocated > 0 else 0.0
        
        # Decision logic
        if congestion_prob >= 0.8:
            if utilization >= 0.9:
                amount = min(current_allocated * 0.4, 50.0)
                action = 'increase_bandwidth'
                reason = f'CRITICAL: High congestion probability ({congestion_prob:.2f}) with {utilization:.1%} utilization'
            else:
                amount = current_allocated * 0.25
                action = 'increase_bandwidth'
                reason = f'HIGH RISK: Congestion probability ({congestion_prob:.2f}) requires bandwidth increase'
        elif congestion_prob >= 0.6:
            if utilization >= 0.8:
                amount = current_allocated * 0.2
                action = 'increase_bandwidth'
                reason = f'MODERATE RISK: Congestion probability ({congestion_prob:.2f}) with high utilization ({utilization:.1%})'
            elif avg_latency > 60:
                amount = current_allocated * 0.15
                action = 'increase_bandwidth'
                reason = f'LATENCY CONCERN: High latency ({avg_latency:.1f}ms) with congestion risk ({congestion_prob:.2f})'
            else:
                amount = 0.0
                action = 'monitor_closely'
                reason = f'WATCH: Medium congestion risk ({congestion_prob:.2f}) - monitor for changes'
        elif congestion_prob >= 0.4:
            if utilization >= 0.85:
                amount = current_allocated * 0.1
                action = 'increase_bandwidth'
                reason = f'PREVENTIVE: High utilization ({utilization:.1%}) with moderate risk ({congestion_prob:.2f})'
            else:
                amount = 0.0
                action = 'monitor'
                reason = f'NORMAL: Moderate risk ({congestion_prob:.2f}) within acceptable range'
        elif congestion_prob <= 0.2:
            if utilization <= 0.4:
                amount = -min(current_allocated * 0.15, 20.0)
                action = 'decrease_bandwidth'
                reason = f'OPTIMIZE: Low utilization ({utilization:.1%}) and low risk ({congestion_prob:.2f})'
            elif utilization <= 0.6:
                amount = 0.0
                action = 'maintain'
                reason = f'EFFICIENT: Good utilization ({utilization:.1%}) with low risk ({congestion_prob:.2f})'
            else:
                amount = 0.0
                action = 'monitor'
                reason = f'STABLE: Acceptable utilization ({utilization:.1%}) with low risk'
        else:
            if utilization >= 0.8:
                amount = current_allocated * 0.1
                action = 'increase_bandwidth'
                reason = f'PREVENTIVE: High utilization ({utilization:.1%}) requires small increase'
            else:
                amount = 0.0
                action = 'maintain'
                reason = f'NORMAL: Balanced operation with {congestion_prob:.2f} risk and {utilization:.1%} utilization'
        
        amount = round(float(amount), 1)
        
        recommendations[router] = {
            'action': action,
            'amount': amount,
            'reason': reason,
            'utilization': float(utilization),
            'congestion_prob': float(congestion_prob),
            'current_allocated': float(current_allocated),
            'current_used': float(current_used)
        }
    
    return recommendations

def create_visualizations(df, target_time, congestion_probs, recommendations):
    """Create visualizations for the dashboard"""
    
    try:
        # Create a copy for visualization with proper datetime handling
        df_viz = df.copy()
        
        # Convert target_time to string for plotly
        if hasattr(target_time, 'strftime'):
            target_time_str = target_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            target_time_str = str(target_time)
        
        # 1. Traffic Volume Over Time
        fig_traffic = px.line(df_viz, x='Timestamp', y='Traffic Volume (MB/s)', 
                             color='Device Name', title='Traffic Volume Over Time')
        
        # Add vertical line safely
        fig_traffic.add_vline(x=target_time_str, line_dash="dash", line_color="red", 
                             annotation_text="Prediction Point")
        
        # 2. Congestion Probabilities
        router_names = list(congestion_probs.keys())
        prob_values = [float(congestion_probs[router]) * 100 for router in router_names]
        
        colors = []
        for p in prob_values:
            if p > 70:
                colors.append('red')
            elif p > 40:
                colors.append('orange') 
            else:
                colors.append('green')
        
        fig_prob = go.Figure(data=[
            go.Bar(x=router_names, y=prob_values, marker_color=colors)
        ])
        fig_prob.update_layout(title='Congestion Probability (%)', 
                              yaxis_title='Probability (%)')
        
        # 3. Bandwidth Utilization
        utilization_data = []
        for router, rec in recommendations.items():
            utilization_data.append({
                'Router': router,
                'Utilization (%)': float(rec['utilization']) * 100,
                'Allocated (MB/s)': float(rec['current_allocated']),
                'Used (MB/s)': float(rec['current_used'])
            })
        
        util_df = pd.DataFrame(utilization_data)
        fig_util = px.bar(util_df, x='Router', y='Utilization (%)', 
                         title='Current Bandwidth Utilization',
                         color='Utilization (%)',
                         color_continuous_scale=['green', 'yellow', 'red'])
        
        return fig_traffic, fig_prob, fig_util
    
    except Exception as e:
        st.error(f"Error creating visualizations: {e}")
        return None, None, None

# Main Dashboard
def main():
    st.title("ðŸ“Š Network Congestion Prediction Dashboard")
    st.markdown("Upload your network traffic data to get congestion predictions and bandwidth recommendations.")
    
    # Load models
    models = load_models()
    if models is None:
        st.stop()
    
    le_impact = load_label_encoder()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=['csv'],
        help="Upload CSV with columns: Timestamp, Device Name, Traffic Volume (MB/s), Latency (ms), Bandwidth Used (MB/s), Bandwidth Allocated (MB/s), total_avg_app_traffic, total_peak_app_traffic, Impact, total_peak_user_usage, total_logins"
    )
    
    if uploaded_file is not None:
        try:
            # Load and process data
            df = pd.read_csv(uploaded_file)
            
            # Ensure Timestamp column is properly converted
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # Remove any rows with invalid timestamps
            df = df.dropna(subset=['Timestamp'])
            
            if df.empty:
                st.error("âŒ No valid data found. Please check your timestamp format.")
                st.stop()
            
            # Handle Impact column encoding
            if 'Impact' in df.columns:
                df['Impact'] = df['Impact'].fillna('None')
                try:
                    df['Impact_encoded'] = le_impact.transform(df['Impact'])
                except ValueError:
                    # Handle unseen categories
                    df['Impact_encoded'] = 0
            else:
                df['Impact_encoded'] = 0
            
            # Get latest timestamp using safe arithmetic
            latest_time = df['Timestamp'].max()
            latest_dt = safe_datetime_arithmetic(latest_time, 0, 'add')  # normalize
            prediction_dt = safe_datetime_arithmetic(latest_dt, 1, 'add')  # add 1 hour
            
            st.success(f"âœ… Data loaded successfully! {len(df)} records")
            st.info(f"ðŸ“… Latest data: {latest_time}")
            st.info(f"ðŸ”® Predicting congestion for: {prediction_dt}")
            
            # Create columns for layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("ðŸ“ˆ Data Overview")
                st.dataframe(df.head())
                
                # Show data summary
                st.subheader("ðŸ“Š Data Summary")
                summary_stats = df.groupby('Device Name').agg({
                    'Traffic Volume (MB/s)': ['mean', 'max'],
                    'Latency (ms)': ['mean', 'max'],
                    'Bandwidth Used (MB/s)': 'mean'
                }).round(2)
                st.dataframe(summary_stats)
            
            with col2:
                st.subheader("âš™ï¸ Prediction Settings")
                st.write("**Prediction Window:** Next 1 hour")
                st.write("**Historical Window:** Last 12 hours")
                st.write("**Routers:** Router_A, Router_B, Router_C")
            
            if st.button("ðŸ”„ Run Prediction", type="primary"):
                with st.spinner("Running predictions..."):
                    # Make predictions
                    congestion_probs = predict_congestion_proba(df, models, latest_time)
                    
                    if congestion_probs is None:
                        st.error("âŒ Not enough data for prediction. Need at least 12 hours of historical data.")
                    else:
                        # Get window data for recommendations
                        window_start_dt = safe_datetime_arithmetic(latest_dt, 12, 'subtract')
                        window_start = pd.Timestamp(window_start_dt)
                        
                        window_data = df[(df['Timestamp'] >= window_start) & 
                                       (df['Timestamp'] <= latest_time)]
                        
                        # Get recommendations
                        recommendations = bandwidth_recommendation(window_data, congestion_probs)
                        
                        # Display results
                        st.subheader("ðŸŽ¯ Congestion Predictions")
                        
                        # Create metrics
                        prob_cols = st.columns(3)
                        for i, (router, prob) in enumerate(congestion_probs.items()):
                            with prob_cols[i]:
                                if prob > 0.7:
                                    color = "ðŸ”´"
                                elif prob > 0.4:
                                    color = "ðŸŸ¡"
                                else:
                                    color = "ðŸŸ¢"
                                
                                st.metric(
                                    label=f"{color} {router}",
                                    value=f"{prob:.1%}",
                                    delta=None
                                )
                        
                        # Recommendations
                        st.subheader("ðŸ’¡ Bandwidth Recommendations")
                        for router, rec in recommendations.items():
                            action_title = rec['action'].replace('_', ' ').title()
                            with st.expander(f"{router} - {action_title}"):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    if rec['amount'] > 0:
                                        st.success(f"ðŸ“ˆ Increase by {rec['amount']} MB/s")
                                    elif rec['amount'] < 0:
                                        st.info(f"ðŸ“‰ Decrease by {abs(rec['amount'])} MB/s")
                                    else:
                                        st.info("âž¡ï¸ Maintain current allocation")
                                    
                                    st.write(f"**Utilization:** {rec['utilization']:.1%}")
                                    st.write(f"**Congestion Risk:** {rec['congestion_prob']:.1%}")
                                
                                with col_b:
                                    st.write(f"**Current Allocated:** {rec['current_allocated']:.1f} MB/s")
                                    st.write(f"**Current Used:** {rec['current_used']:.1f} MB/s")
                                    st.write(f"**Reason:** {rec['reason']}")
                        
                        # Visualizations
                        st.subheader("ðŸ“Š Visualizations")
                        
                        try:
                            fig_traffic, fig_prob, fig_util = create_visualizations(
                                df, latest_time, congestion_probs, recommendations
                            )
                            
                            if fig_traffic is not None:
                                st.plotly_chart(fig_traffic, use_container_width=True)
                                
                                viz_col1, viz_col2 = st.columns(2)
                                with viz_col1:
                                    if fig_prob is not None:
                                        st.plotly_chart(fig_prob, use_container_width=True)
                                with viz_col2:
                                    if fig_util is not None:
                                        st.plotly_chart(fig_util, use_container_width=True)
                            else:
                                st.warning("âš ï¸ Charts could not be generated, but predictions completed successfully.")
                                
                        except Exception as viz_error:
                            st.warning(f"âš ï¸ Visualization error: {viz_error}")
                            st.info("ðŸ“Š Predictions completed successfully, but charts could not be displayed.")
                        
                        # Store in session state for history
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'prediction_for': prediction_dt,
                            'congestion_probs': congestion_probs,
                            'recommendations': recommendations
                        })
            
            # Prediction History
            if 'prediction_history' in st.session_state and st.session_state.prediction_history:
                st.subheader("ðŸ“œ Prediction History")
                history_data = []
                
                for entry in st.session_state.prediction_history[-10:]:  # Last 10 predictions
                    for router, prob in entry['congestion_probs'].items():
                        history_data.append({
                            'Timestamp': entry['timestamp'],
                            'Prediction For': entry['prediction_for'],
                            'Router': router,
                            'Congestion Probability': f"{prob:.1%}",
                            'Recommendation': entry['recommendations'][router]['action']
                        })
                
                if history_data:
                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)
                
                if st.button("ðŸ—‘ï¸ Clear History"):
                    st.session_state.prediction_history = []
                    st.rerun()
                        
        except Exception as e:
            st.error(f"âŒ Error processing file: {str(e)}")
            st.write("Please check that your CSV has the required columns:")
            st.code("""
Required columns:
- Timestamp
- Device Name (Router_A, Router_B, Router_C)
- Traffic Volume (MB/s)
- Latency (ms)  
- Bandwidth Used (MB/s)
- Bandwidth Allocated (MB/s)
- total_avg_app_traffic
- total_peak_app_traffic
- Impact
- total_peak_user_usage
- total_logins
""")

if __name__ == "__main__":
    main()