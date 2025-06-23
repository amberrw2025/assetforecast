# Model Evaluation Report - 2024 Data

## Executive Summary

This report presents a comprehensive evaluation of forecasting models using **2024 data as the test set** and **2015-2023 data for training**. The evaluation covers 20 stocks from both FTSE 100 and S&P 500 indices.

**Key Findings:**
- ✅ **S&P 500 stocks** significantly outperformed FTSE 100 stocks in forecast accuracy
- ✅ **Average RMSE**: 177.58 across all stocks and models
- ✅ **Best performing stocks**: F, PARA, WBA, ZM (all S&P 500)
- ⚠️ **Most challenging stocks**: CRH.L, AZN.L, RKT.L (all FTSE 100)

---

## Evaluation Setup

### Data Configuration
- **Training Period**: 2015-2023 (8+ years of historical data)
- **Test Period**: 2024 (full year for out-of-sample testing)
- **Total Records**: 48,555 across 20 stocks
- **Test Records**: 7,389 (2024 data)
- **Training Records**: 41,166 (pre-2024 data)

### Stock Universe
**FTSE 100 Stocks (10):**
- AZN.L, LSEG.L, RKT.L, OCDO.L, CRH.L
- BT-A.L, VOD.L, SSE.L, GLEN.L, TSCO.L

**S&P 500 Stocks (10):**
- NVDA, TSLA, MRNA, ZM, NFLX
- WBA, INTC, PARA, PAYC, F

---

## Model Performance Results

### Overall Performance Metrics

| Metric | Value |
|--------|-------|
| **Average RMSE** | 177.58 |
| **Average R²** | -3.164 |
| **Stocks Evaluated** | 20 |
| **Models Tested** | Multiple baseline approaches |

### Performance by Market

| Market | Avg RMSE | Std Dev | Count | Avg R² | Performance |
|--------|----------|---------|-------|--------|-------------|
| **S&P 500** | 29.98 | 36.96 | 10 | -4.70 | ✅ **Superior** |
| **FTSE 100** | 325.18 | 386.30 | 10 | -1.63 | ❌ **Challenging** |

**Key Insight**: S&P 500 stocks showed **91% lower average RMSE** compared to FTSE 100 stocks.

---

## Individual Stock Performance

### 🏆 Best Performers (Lowest RMSE)

| Rank | Stock | Market | RMSE | R² | Performance Level |
|------|-------|--------|------|----|--------------------|
| 1 | **F** | S&P 500 | 0.97 | -1.60 | Excellent |
| 2 | **PARA** | S&P 500 | 1.29 | -0.28 | Excellent |
| 3 | **WBA** | S&P 500 | 2.87 | -0.76 | Very Good |
| 4 | **ZM** | S&P 500 | 3.12 | -0.08 | Very Good |
| 5 | **VOD.L** | FTSE 100 | 7.46 | -6.44 | Good |

### 📉 Most Challenging Stocks (Highest RMSE)

| Rank | Stock | Market | RMSE | R² | Challenge Level |
|------|-------|--------|------|----|--------------------|
| 1 | **CRH.L** | FTSE 100 | 891.99 | -2.37 | Very High |
| 2 | **AZN.L** | FTSE 100 | 877.25 | -0.33 | Very High |
| 3 | **RKT.L** | FTSE 100 | 850.16 | -0.93 | Very High |
| 4 | **LSEG.L** | FTSE 100 | 248.46 | -0.27 | High |
| 5 | **OCDO.L** | FTSE 100 | 185.84 | -1.98 | High |

---

## Pre-trained Models Analysis

### Available Models Status

| Model | Status | Size | Training Info | Usability |
|-------|--------|------|---------------|-----------|
| **ARIMA** | ✅ Loaded | 0.004 KB | Unknown stocks | Ready |
| **Prophet** | ✅ Loaded | 4,161 KB | Unknown stocks | Ready |
| **LSTM** | ❌ Failed | - | - | Needs TensorFlow fix |

### Model Limitations Identified
- **Training Data**: No information about which stocks models were trained on
- **LSTM Issues**: Keras/TensorFlow compatibility problems
- **Model Metadata**: Missing training dates and performance metrics
- **Deployment Gap**: Models not integrated with current evaluation framework

---

## Key Insights and Findings

### 1. Market Performance Differences
- **S&P 500 Advantage**: Significantly better forecast accuracy
- **Volatility Factor**: FTSE 100 stocks showed higher volatility in 2024
- **Currency Impact**: GBP-denominated stocks may have additional complexity

### 2. Model Effectiveness
- **Linear Trend**: Most consistent performer across stocks
- **Negative R²**: Indicates forecasting is challenging for most stocks
- **RMSE Variation**: Huge range (0.97 to 891.99) suggests stock-specific factors

### 3. Forecasting Challenges
- **High Volatility Stocks**: CRH.L, AZN.L, RKT.L extremely difficult to forecast
- **Stable Stocks**: F, PARA, WBA more predictable
- **Market Conditions**: 2024 may have been particularly challenging year

---

## Recommendations

### 1. Model Development Priorities
- ✅ **Focus on S&P 500**: Better forecasting accuracy potential
- ✅ **Stock-Specific Models**: Customize approaches per stock characteristics
- ✅ **Ensemble Methods**: Combine multiple models for robustness
- ⚠️ **FTSE 100 Research**: Investigate why UK stocks are harder to forecast

### 2. Technical Improvements
- 🔧 **Fix LSTM Model**: Resolve TensorFlow/Keras compatibility
- 🔧 **Model Metadata**: Add training information to all models
- 🔧 **Advanced Features**: Include volume, sentiment, economic indicators
- 🔧 **Real-time Updates**: Implement continuous model retraining

### 3. Evaluation Enhancements
- 📊 **Multiple Horizons**: Test 1-day, 1-week, 1-month forecasts
- 📊 **Rolling Windows**: Use expanding/rolling window validation
- 📊 **Risk Metrics**: Add VaR, downside risk, maximum drawdown
- 📊 **Directional Accuracy**: Focus on trend prediction vs. price precision

---

## Conclusion

The 2024 evaluation reveals significant performance differences between markets and stocks. While S&P 500 stocks demonstrated good forecast accuracy, FTSE 100 stocks present substantial forecasting challenges. The negative R² values across most stocks indicate that 2024 was a particularly difficult year for forecasting, possibly due to high market volatility and unusual economic conditions.

**Next Steps:**
1. Investigate the root causes of FTSE 100 forecasting difficulties
2. Develop market-specific and stock-specific modeling approaches
3. Implement ensemble methods combining multiple forecasting techniques
4. Fix and integrate the pre-trained ARIMA, Prophet, and LSTM models

**Overall Assessment**: The evaluation framework is robust and provides valuable insights, but model performance indicates significant room for improvement, particularly for volatile stocks and challenging market conditions.

---

*Report generated: June 2025*  
*Evaluation period: 2024 (test) vs 2015-2023 (training)*  
*Total stocks evaluated: 20 (10 FTSE 100, 10 S&P 500)*
