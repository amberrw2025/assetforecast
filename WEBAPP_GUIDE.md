# 🌐 FORECAST ASSESSMENT WEB APPLICATION

## 🚀 **Interactive Web Dashboard for Your Forecasting Models**

Your forecast accuracy assessment model is now available as a modern, interactive web application!

## 📋 **Features**

### 🎯 **Core Functionality**
- **📊 Interactive Dashboard**: Real-time data visualization and analytics
- **📁 File Upload**: Drag-and-drop CSV/Excel file upload
- **🔮 Forecast Generation**: Generate predictions with all trained models
- **📈 Model Comparison**: Compare performance across ARIMA, Prophet, and LSTM
- **📊 Performance Metrics**: RMSE, MAE, R², and MAPE evaluation
- **💾 Results Export**: Download forecasts and reports as CSV

### 🎨 **User Interface**
- **Modern Design**: Clean, responsive Bootstrap interface
- **Interactive Charts**: Plotly.js powered visualizations
- **Real-time Updates**: Live data refresh and status monitoring
- **Mobile Friendly**: Works on desktop, tablet, and mobile devices

## 🚀 **Quick Start**

### 1. **Start the Web Application**
```bash
python3 run_webapp.py
```

### 2. **Access the Dashboard**
Open your web browser and go to:
```
http://localhost:5003
```

### 3. **Upload Your Data**
- Click "Upload Data" or drag-and-drop your CSV/Excel file
- Supported formats: `.csv`, `.xlsx`, `.xls`
- Maximum file size: 50MB

### 4. **Generate Forecasts**
- Select models (ARIMA, Prophet, LSTM)
- Choose forecast period (default: 30 days)
- Click "Generate Forecast"

### 5. **View Results**
- Interactive charts showing actual vs predicted values
- Performance comparison tables
- Download results as CSV

## 📊 **Dashboard Sections**

### 🏠 **Home Page**
- **Quick Actions**: Upload data, generate forecasts, evaluate models
- **Model Status**: Real-time status of all trained models
- **Data Summary**: Key metrics and statistics
- **Recent Activity**: Log of recent operations

### 📈 **Analytics Dashboard**
- **Price Time Series**: Interactive chart with moving averages
- **Model Performance**: Bar charts comparing RMSE across models
- **Forecast Accuracy**: Pie charts showing accuracy percentages
- **System Status**: Real-time model and pipeline status

## 🔧 **API Endpoints**

### **Data Management**
- `POST /upload` - Upload data files
- `GET /api/data-summary` - Get data summary
- `GET /api/chart-data` - Get data for charts

### **Forecasting**
- `POST /forecast` - Generate forecasts
- `POST /evaluate` - Evaluate model performance
- `GET /download-forecast` - Download forecast results

### **Pages**
- `GET /` - Home dashboard
- `GET /dashboard` - Analytics dashboard

## 📁 **File Structure**

```
webapp/
├── __init__.py              # Package initialization
├── app.py                   # Main Flask application
└── templates/               # HTML templates
    ├── base.html           # Base template with styling
    ├── index.html          # Home page
    └── dashboard.html      # Analytics dashboard

uploads/                     # Uploaded data files
run_webapp.py               # Application runner
```

## 🎯 **Usage Examples**

### **Basic Workflow**
1. **Start the app**: `python3 run_webapp.py`
2. **Upload data**: Use the web interface to upload your CSV file
3. **Generate forecast**: Select models and generate predictions
4. **View results**: Analyze charts and download results

### **Advanced Usage**
```python
# Programmatic access to the web app
import requests

# Upload data
with open('your_data.csv', 'rb') as f:
    response = requests.post('http://localhost:5003/upload', 
                           files={'file': f})

# Generate forecast
forecast_data = {
    'steps': 30,
    'models': ['ARIMA', 'Prophet', 'LSTM']
}
response = requests.post('http://localhost:5003/forecast', 
                        json=forecast_data)

# Get results
results = response.json()
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Port already in use**
   ```bash
   # Change port in run_webapp.py
   app.run(port=5003)  # Use a different port like 5003
   ```

2. **Model loading errors**
   - Ensure models are trained: `python3 model_training_pipeline.py`
   - Check model files exist in `models/` directory

3. **File upload issues**
   - Check file format (CSV, Excel)
   - Ensure file size < 50MB
   - Verify file has date and price columns

4. **Chart not displaying**
   - Check browser console for JavaScript errors
   - Ensure internet connection (for CDN resources)

### **Performance Tips**
- Use smaller datasets for faster processing
- Close other applications to free memory
- Use modern browsers (Chrome, Firefox, Safari)

## 🔒 **Security Notes**

- **Development Mode**: The app runs in debug mode for development
- **File Uploads**: Files are stored temporarily in `uploads/` directory
- **No Authentication**: Add authentication for production use
- **Data Privacy**: Ensure sensitive data is handled appropriately

## 🚀 **Production Deployment**

### **For Production Use**
1. **Disable debug mode** in `app.py`
2. **Add authentication** and user management
3. **Use production WSGI server** (Gunicorn, uWSGI)
4. **Set up HTTPS** with SSL certificates
5. **Add monitoring** and logging
6. **Configure database** for persistent storage

### **Example Production Setup**
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5003 webapp.app:create_app()
```

## 📞 **Support**

### **Getting Help**
- Check the console output for error messages
- Review the browser's developer console
- Ensure all dependencies are installed
- Verify model files are present

### **Next Steps**
- Customize the interface for your specific needs
- Add additional models and features
- Integrate with external data sources
- Set up automated forecasting schedules

---

**🎉 Your forecast assessment model is now accessible through a beautiful web interface!**

**🌐 Start the web app**: `python3 run_webapp.py`
**📱 Access at**: http://localhost:5003 