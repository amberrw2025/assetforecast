"""
Visualization utilities for the Forecast Accuracy Assessment Model Pipeline.
Provides functions for creating plots and charts using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from utils.logger import get_logger

logger = get_logger("visualization")

class DataVisualizer:
    """
    Provides visualization utilities for the forecast model data.
    """
    
    def __init__(self):
        self.default_colors = px.colors.qualitative.Set3
        
    def plot_time_series(self, df: pd.DataFrame, x_column: str = 'date', 
                        y_columns: List[str] = None, title: str = "Time Series Plot",
                        height: int = 600) -> go.Figure:
        """
        Create a time series plot for multiple columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            x_column (str): Column to use for x-axis (typically date)
            y_columns (List[str]): Columns to plot on y-axis
            title (str): Plot title
            height (int): Plot height
            
        Returns:
            go.Figure: Plotly figure object
        """
        if y_columns is None:
            # Auto-select numeric columns
            y_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            y_columns = [col for col in y_columns if col != x_column]
        
        # Limit to first 5 columns for readability
        y_columns = y_columns[:5]
        
        fig = go.Figure()
        
        for i, column in enumerate(y_columns):
            if column in df.columns:
                color = self.default_colors[i % len(self.default_colors)]
                fig.add_trace(
                    go.Scatter(
                        x=df[x_column],
                        y=df[column],
                        mode='lines',
                        name=column,
                        line=dict(color=color, width=2),
                        hovertemplate=f'{column}: %{{y}}<br>Date: %{{x}}<extra></extra>'
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title=x_column,
            yaxis_title="Value",
            height=height,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str] = None,
                              title: str = "Correlation Matrix") -> go.Figure:
        """
        Create a correlation matrix heatmap.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to include in correlation matrix
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        if columns is None:
            # Auto-select numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to first 15 columns for readability
        columns = columns[:15]
        
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=800,
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig
    
    def plot_missing_values(self, df: pd.DataFrame, title: str = "Missing Values Analysis") -> go.Figure:
        """
        Create a plot showing missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        # Create DataFrame for plotting
        missing_df = pd.DataFrame({
            'column': missing_data.index,
            'missing_count': missing_data.values,
            'missing_percentage': missing_pct.values
        }).sort_values('missing_percentage', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=missing_df['missing_percentage'],
            y=missing_df['column'],
            orientation='h',
            marker_color='lightcoral',
            text=missing_df['missing_percentage'].round(1),
            textposition='auto',
            hovertemplate='Column: %{y}<br>Missing: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Missing Percentage (%)",
            yaxis_title="Columns",
            height=max(400, len(missing_df) * 20),
            showlegend=False
        )
        
        return fig
    
    def plot_distribution(self, df: pd.DataFrame, columns: List[str] = None,
                         title: str = "Distribution Plots") -> go.Figure:
        """
        Create distribution plots for numeric columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            columns (List[str]): Columns to plot
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        if columns is None:
            # Auto-select numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to first 6 columns
        columns = columns[:6]
        
        # Calculate number of rows and columns for subplots
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=columns,
            specs=[[{"secondary_y": False}] * n_cols] * n_rows
        )
        
        for i, column in enumerate(columns):
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            # Remove NaN values
            data = df[column].dropna()
            
            if len(data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=data,
                        name=column,
                        nbinsx=30,
                        marker_color=self.default_colors[i % len(self.default_colors)]
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=title,
            height=300 * n_rows,
            showlegend=False
        )
        
        return fig
    
    def plot_forecast_vs_actual(self, actual: pd.Series, forecast: pd.Series,
                              dates: pd.Series = None, title: str = "Forecast vs Actual",
                              confidence_intervals: Tuple[pd.Series, pd.Series] = None) -> go.Figure:
        """
        Create a plot comparing forecast vs actual values.
        
        Args:
            actual (pd.Series): Actual values
            forecast (pd.Series): Forecasted values
            dates (pd.Series): Date series
            title (str): Plot title
            confidence_intervals (Tuple[pd.Series, pd.Series]): Lower and upper confidence intervals
            
        Returns:
            go.Figure: Plotly figure object
        """
        if dates is None:
            dates = pd.date_range(start='2023-01-01', periods=len(actual), freq='D')
        
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Plot forecast values
        fig.add_trace(go.Scatter(
            x=dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ))
        
        # Add confidence intervals if provided
        if confidence_intervals is not None:
            lower, upper = confidence_intervals
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)',
                name='Confidence Interval',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_model_performance(self, metrics: Dict[str, float],
                              title: str = "Model Performance Metrics") -> go.Figure:
        """
        Create a bar plot of model performance metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=self.default_colors[:len(metric_names)],
            text=[f'{val:.4f}' for val in metric_values],
            textposition='auto',
            hovertemplate='Metric: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Metrics",
            yaxis_title="Value",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_dashboard(self, df: pd.DataFrame, title: str = "Data Dashboard") -> go.Figure:
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            title (str): Dashboard title
            
        Returns:
            go.Figure: Plotly figure object with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Time Series", "Correlation Matrix", "Missing Values", "Distribution"),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # Time series plot (top left)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'date' in df.columns and len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig.add_trace(
                go.Scatter(x=df['date'], y=df[col], mode='lines', name=col),
                row=1, col=1
            )
        
        # Correlation matrix (top right)
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols[:5]].corr()
            fig.add_trace(
                go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns),
                row=1, col=2
            )
        
        # Missing values (bottom left)
        missing_data = df.isnull().sum()
        fig.add_trace(
            go.Bar(x=missing_data.index, y=missing_data.values),
            row=2, col=1
        )
        
        # Distribution (bottom right)
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            fig.add_trace(
                go.Histogram(x=df[col].dropna()),
                row=2, col=2
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def save_plot(self, fig: go.Figure, filename: str, format: str = "html"):
        """
        Save a plot to file.
        
        Args:
            fig (go.Figure): Plotly figure to save
            filename (str): Output filename
            format (str): Output format ('html', 'png', 'pdf')
        """
        if format == "html":
            fig.write_html(filename)
        elif format == "png":
            fig.write_image(filename)
        elif format == "pdf":
            fig.write_image(filename)
        else:
            logger.warning(f"Unsupported format: {format}. Saving as HTML.")
            fig.write_html(filename)
        
        logger.info(f"Plot saved to {filename}")

def main():
    """
    Main function to demonstrate visualization utilities.
    """
    logger.info("Data visualization module loaded successfully")
    logger.info("Use DataVisualizer class to create various plots and charts")

if __name__ == "__main__":
    main() 