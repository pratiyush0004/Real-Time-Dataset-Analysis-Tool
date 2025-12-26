from flask import Flask, render_template, jsonify, request, session, send_file
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import requests
import json
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production-12345'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CLEANED_FOLDER'] = 'cleaned'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Perplexity API Configuration
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY', '')  # Set via environment variable
PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLEANED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_dataframe():
    """Get the current active dataframe from session"""
    if 'current_file' not in session:
        raise ValueError('No dataset uploaded. Please upload a file first.')
    
    file_path = session['current_file']
    
    if not os.path.exists(file_path):
        raise ValueError(f'Dataset file not found: {file_path}. Please upload again.')
    
    # Load based on file extension
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path, engine='openpyxl')
        else:
            return pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f'Error reading file: {str(e)}')

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/clean-data')
def clean_data_page():
    """Data cleaning page - shown after upload"""
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('clean_data.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page - requires cleaned file"""
    if 'current_file' not in session:
        return render_template('upload.html')
    if 'cleaned_file' not in session:
        return render_template('clean_data.html')
    return render_template('dashboard.html')

@app.route('/visualization/pie-chart')
def pie_chart_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('pie_chart.html')

@app.route('/visualization/boxplot')
def boxplot_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('boxplot.html')

@app.route('/visualization/bar-chart')
def bar_chart_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('bar_chart.html')

@app.route('/visualization/line-chart')
def line_chart_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('line_chart.html')

@app.route('/visualization/scatter-plot')
def scatter_plot_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('scatter_plot.html')

@app.route('/visualization/heatmap')
def heatmap_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('heatmap.html')

@app.route('/visualization/eda')
def eda_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('eda.html')

@app.route('/data-table')
def data_table_page():
    if 'current_file' not in session:
        return render_template('upload.html')
    return render_template('data_table.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload CSV or Excel files only.'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Test if file can be loaded
        try:
            if filename.endswith('.csv'):
                test_df = pd.read_csv(filepath)
            else:
                test_df = pd.read_excel(filepath)
            
            # Store in session
            session['current_file'] = filepath
            session['original_filename'] = filename
            
            return jsonify({
                'success': True,
                'message': f'File "{filename}" uploaded successfully!',
                'filename': filename,
                'rows': len(test_df),
                'columns': len(test_df.columns)
            })
        except Exception as e:
            # Remove invalid file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_dataset():
    """Reset to default dataset"""
    try:
        if 'current_file' in session:
            session.pop('current_file')
        if 'original_filename' in session:
            session.pop('original_filename')
        
        return jsonify({
            'success': True,
            'message': 'Reset to default dataset'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/current-file')
def get_current_file():
    """Get current active file info"""
    try:
        # Get the actual current file path
        current_file_path = session.get('current_file', '')
        
        # Extract just the filename from the full path
        if current_file_path:
            filename = os.path.basename(current_file_path)
        else:
            filename = session.get('original_filename', 'No file loaded')
        
        df = get_current_dataframe()
        return jsonify({
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/overview')
def get_overview():
    """Get dataset overview statistics"""
    try:
        df = get_current_dataframe()
        total_firms = len(df)
        
        # Try to find specific columns or use generic counts
        cities_count = 0
        business_types = 0
        states_count = 0
        
        for col in df.columns:
            col_lower = col.lower()
            if 'city' in col_lower or 'location' in col_lower:
                cities_count = int(df[col].nunique())
            if any(keyword in col_lower for keyword in ['type', 'category', 'business', 'class']):
                business_types = int(df[col].nunique())
            if 'state' in col_lower or 'region' in col_lower:
                states_count = int(df[col].nunique())
        
        # Fallback to generic counts if specific columns not found
        if cities_count == 0:
            cities_count = int(df.select_dtypes(include=['object']).nunique().max()) if not df.select_dtypes(include=['object']).empty else 0
        if business_types == 0:
            business_types = int(df.select_dtypes(include=['object']).nunique().max()) if not df.select_dtypes(include=['object']).empty else 0
        if states_count == 0:
            states_count = int(df.select_dtypes(include=['object']).nunique().max()) if not df.select_dtypes(include=['object']).empty else 0
        
        missing_values = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        
        return jsonify({
            'total_firms': int(total_firms),
            'cities': int(cities_count),
            'business_types': int(business_types),
            'states': int(states_count),
            'missing_values': int(missing_values),
            'duplicate_rows': int(duplicate_rows),
            'columns': int(df.shape[1]),
            'rows': int(df.shape[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/top-cities')
def get_top_cities():
    """Get top 20 cities by number of records"""
    try:
        df = get_current_dataframe()
        
        # Try to find a city/location column
        city_col = None
        for col in df.columns:
            if 'city' in col.lower() or 'location' in col.lower():
                city_col = col
                break
        
        if not city_col:
            return jsonify({'error': 'No city column found in dataset'}), 400
        
        df_cleaned = df.dropna(subset=[city_col])
        city_counts = df_cleaned[city_col].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.barplot(x=city_counts.values, y=city_counts.index, ax=ax, palette='viridis')
        ax.set_title("Top 20 Cities by Number of Records", fontsize=16, fontweight='bold')
        ax.set_xlabel("Number of Records", fontsize=12)
        ax.set_ylabel("City", fontsize=12)
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'data': {
                'cities': city_counts.index.tolist(),
                'counts': city_counts.values.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/business-types')
def get_business_types():
    """Get business type distribution"""
    try:
        df = get_current_dataframe()
        
        # Try to find a business type/category column
        type_col = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['type', 'category', 'business', 'class']):
                type_col = col
                break
        
        if not type_col:
            return jsonify({'error': 'No business type column found in dataset'}), 400
        
        df_cleaned = df.dropna(subset=[type_col])
        business_type_counts = df_cleaned[type_col].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(business_type_counts)))
        ax.pie(
            business_type_counts.values,
            labels=business_type_counts.index,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            shadow=True
        )
        ax.set_title("Type Distribution", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'data': {
                'types': business_type_counts.index.tolist(),
                'counts': business_type_counts.values.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/registration-trends')
def get_registration_trends():
    """Get registration trends over time"""
    try:
        df = get_current_dataframe()
        df_temp = df.copy()
        df_temp.columns = df_temp.columns.str.strip()
        
        # Try to find a date column
        date_col = None
        for col in df_temp.columns:
            if any(keyword in col.lower() for keyword in ['date', 'registration', 'created', 'issued', 'start']):
                date_col = col
                break
        
        if not date_col:
            return jsonify({'error': 'No date column found in dataset'}), 400
        
        df_temp["Original Issue Date"] = pd.to_datetime(df_temp[date_col], errors="coerce")
        df_cleaned = df_temp.dropna(subset=["Original Issue Date"]).copy()
        df_cleaned["Registration Year"] = df_cleaned["Original Issue Date"].dt.year
        yearly_counts = df_cleaned["Registration Year"].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(yearly_counts.index, yearly_counts.values, color='skyblue', alpha=0.7, label='Registrations')
        ax.plot(yearly_counts.index, yearly_counts.values, color='darkblue', linewidth=2, marker='o', label='Trend')
        ax.set_title("Registration Trends Over Time", fontsize=16, fontweight='bold')
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Number of Registrations", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'data': {
                'years': yearly_counts.index.tolist(),
                'counts': yearly_counts.values.tolist()
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/lifespan-analysis')
def get_lifespan_analysis():
    """Get lifespan analysis"""
    try:
        df = get_current_dataframe()
        df_temp = df.copy()
        
        # Find date columns
        start_col = None
        end_col = None
        type_col = None
        
        for col in df_temp.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['registration', 'start', 'created', 'issued']) and not end_col:
                start_col = col
            if any(keyword in col_lower for keyword in ['expir', 'end', 'close']):
                end_col = col
            if any(keyword in col_lower for keyword in ['type', 'category', 'business', 'class']):
                type_col = col
        
        if not start_col or not end_col:
            return jsonify({'error': 'Date columns not found for lifespan analysis'}), 400
        
        df_temp['Registration.Date'] = pd.to_datetime(df_temp[start_col], errors='coerce')
        df_temp['Expires'] = pd.to_datetime(df_temp[end_col], errors='coerce')
        df_clean = df_temp.dropna(subset=['Registration.Date', 'Expires']).copy()
        df_clean['Lifespan'] = (df_clean['Expires'] - df_clean['Registration.Date']).dt.days / 365.25
        df_clean = df_clean[df_clean['Lifespan'] > 0]
        
        if not type_col or type_col not in df_clean.columns:
            # Just show overall distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(df_clean['Lifespan'], bins=30, color='skyblue', edgecolor='black')
            ax.set_title('Lifespan Distribution', fontsize=16, fontweight='bold')
            ax.set_xlabel('Lifespan (Years)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df_clean, x=type_col, y='Lifespan', ax=ax, palette='Set2')
            ax.set_title('Lifespan Distribution by Type', fontsize=16, fontweight='bold')
            ax.set_xlabel('Type', fontsize=12)
            ax.set_ylabel('Lifespan (Years)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/location-comparison')
def get_location_comparison():
    """Get in-state vs out-of-state comparison"""
    try:
        df = get_current_dataframe()
        df_temp = df.copy()
        
        # Find date and state columns
        date_col = None
        state_col = None
        
        for col in df_temp.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'registration', 'created']):
                date_col = col
            if 'state' in col_lower or 'region' in col_lower:
                state_col = col
        
        if not date_col or not state_col:
            return jsonify({'error': 'Required columns not found for location comparison'}), 400
        
        df_temp['Registration.Date'] = pd.to_datetime(df_temp[date_col], errors='coerce')
        df_clean = df_temp.dropna(subset=['Registration.Date']).copy()
        df_clean['Registration_Year'] = df_clean['Registration.Date'].dt.year
        
        # Determine the most common state
        main_state = df_clean[state_col].mode()[0] if not df_clean[state_col].empty else 'WA'
        df_clean['Location'] = np.where(df_clean[state_col] == main_state, 'In-State', 'Out-of-State')
        
        yearly_counts = df_clean.groupby(['Registration_Year', 'Location']).size().reset_index(name='Firm_Count')
        pivot_table = yearly_counts.pivot(index='Registration_Year', columns='Location', values='Firm_Count').fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in pivot_table.columns:
            ax.plot(pivot_table.index, pivot_table[col], marker='o', linewidth=2, label=col)
        ax.set_title('In-State vs Out-of-State Registrations Over Time', fontsize=16, fontweight='bold')
        ax.set_xlabel('Registration Year', fontsize=12)
        ax.set_ylabel('Number Registered', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Location')
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data-table')
def get_data_table():
    """Get sample data for table view"""
    try:
        df = get_current_dataframe()
        sample_data = df.head(50).fillna('N/A')
        columns = sample_data.columns.tolist()
        
        # Convert data to native Python types to avoid JSON serialization issues
        data = []
        for _, row in sample_data.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append('N/A')
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    row_data.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    row_data.append(float(val))
                else:
                    row_data.append(str(val))
            data.append(row_data)
        
        return jsonify({
            'columns': columns,
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/preview')
def get_preview():
    """Get preview of dataset with column info"""
    try:
        df = get_current_dataframe()
        limit = int(request.args.get('limit', 10))
        
        sample_data = df.head(limit).fillna('N/A')
        columns = sample_data.columns.tolist()
        
        # Convert data to native Python types to avoid JSON serialization issues
        data = []
        for _, row in sample_data.iterrows():
            row_data = []
            for val in row:
                if pd.isna(val):
                    row_data.append('N/A')
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    row_data.append(int(val))
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    row_data.append(float(val))
                else:
                    row_data.append(str(val))
            data.append(row_data)
        
        return jsonify({
            'columns': columns,
            'data': data,
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/numeric-columns')
def get_numeric_columns():
    """Get list of numeric columns only"""
    try:
        df = get_current_dataframe()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return jsonify({
            'columns': numeric_cols,
            'count': len(numeric_cols)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correlation-matrix')
def get_correlation_matrix():
    """Get correlation matrix for numeric columns"""
    try:
        df = get_current_dataframe()
        correlation_matrix = df.corr(numeric_only=True)
        
        if correlation_matrix.empty:
            return jsonify({'error': 'No numeric columns found for correlation analysis'}), 400
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                    linewidths=0.5, ax=ax, center=0)
        ax.set_title('Correlation Matrix of Numeric Columns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/pie-chart')
def get_pie_chart():
    """Generate pie chart for categorical data distribution"""
    try:
        df = get_current_dataframe()
        column = request.args.get('column')
        
        if not column or column not in df.columns:
            # Auto-select first categorical column
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) == 0:
                return jsonify({'error': 'No categorical columns found for pie chart'}), 400
            column = categorical_cols[0]
        
        # Get top categories
        value_counts = df[column].value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set3(range(len(value_counts)))
        wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index, 
                                           autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title(f'Distribution of {column}', fontsize=16, fontweight='bold', pad=20)
        
        # Improve text readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'column': column,
            'categories': int(len(value_counts)),
            'data': {str(k): int(v) for k, v in value_counts.to_dict().items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/boxplot')
def get_boxplot():
    """Generate boxplot for outlier detection"""
    try:
        df = get_current_dataframe()
        column = request.args.get('column')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return jsonify({'error': 'No numeric columns found for boxplot'}), 400
        
        if not column or column not in numeric_cols:
            column = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(df[column].dropna(), vert=True, patch_artist=True)
        
        # Customize colors
        for patch in bp['boxes']:
            patch.set_facecolor('#667eea')
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set(color='#764ba2', linewidth=2)
        for cap in bp['caps']:
            cap.set(color='#764ba2', linewidth=2)
        for median in bp['medians']:
            median.set(color='red', linewidth=2)
        
        ax.set_ylabel(column, fontsize=12)
        ax.set_title(f'Boxplot for Outlier Detection - {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        # Calculate outlier statistics
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)][column]
        
        return jsonify({
            'image': img_data,
            'column': column,
            'outliers_count': int(len(outliers)),
            'outliers_percentage': round(float(len(outliers)) / float(len(df)) * 100, 2),
            'stats': {
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'median': float(df[column].median())
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/bar-chart')
def get_bar_chart():
    """Generate bar chart for categorical data"""
    try:
        df = get_current_dataframe()
        column = request.args.get('column')
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            return jsonify({'error': 'No categorical columns found for bar chart'}), 400
        
        if not column or column not in categorical_cols:
            column = categorical_cols[0]
        
        # Get top 15 categories
        value_counts = df[column].value_counts().head(15)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(value_counts)), value_counts.values, color='#667eea', alpha=0.8)
        ax.set_xticks(range(len(value_counts)))
        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Top Categories - {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'column': column,
            'data': {str(k): int(v) for k, v in value_counts.to_dict().items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/line-chart')
def get_line_chart():
    """Generate line chart for trend analysis"""
    try:
        df = get_current_dataframe()
        column = request.args.get('column')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return jsonify({'error': 'No numeric columns found for line chart'}), 400
        
        if not column or column not in numeric_cols:
            column = numeric_cols[0]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df[column], color='#667eea', linewidth=2, marker='o', 
               markersize=4, markerfacecolor='#764ba2', alpha=0.7)
        ax.set_xlabel('Index', fontsize=12)
        ax.set_ylabel(column, fontsize=12)
        ax.set_title(f'Trend Analysis - {column}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'column': column,
            'trend': 'increasing' if df[column].iloc[-1] > df[column].iloc[0] else 'decreasing'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/scatter-plot')
def get_scatter_plot():
    """Generate scatter plot for correlation analysis"""
    try:
        df = get_current_dataframe()
        x_col = request.args.get('x_column')
        y_col = request.args.get('y_column')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return jsonify({'error': 'Need at least 2 numeric columns for scatter plot'}), 400
        
        if not x_col or x_col not in numeric_cols:
            x_col = numeric_cols[0]
        if not y_col or y_col not in numeric_cols:
            y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        
        # Remove rows with NaN values in either column
        plot_data = df[[x_col, y_col]].dropna()
        
        if len(plot_data) < 2:
            total_rows = len(df)
            x_nulls = df[x_col].isnull().sum()
            y_nulls = df[y_col].isnull().sum()
            return jsonify({
                'error': f'Not enough valid data points for scatter plot.\n\n'
                        f'Dataset has {total_rows} rows total.\n'
                        f'Column "{x_col}" has {x_nulls} missing values.\n'
                        f'Column "{y_col}" has {y_nulls} missing values.\n'
                        f'Only {len(plot_data)} rows have data in both columns.\n\n'
                        f'Please clean your data or select columns with fewer missing values.'
            }), 400
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, c='#667eea', 
                            edgecolors='#764ba2', linewidth=0.5, s=50)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'Scatter Plot - {x_col} vs {y_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        if len(plot_data) >= 2:
            z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(plot_data[x_col], p(plot_data[x_col]), "r--", alpha=0.8, linewidth=2, label='Trend line')
            ax.legend()
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        # Calculate correlation
        correlation = plot_data[x_col].corr(plot_data[y_col])
        
        # Handle NaN correlation (happens when column has no variance)
        if pd.isna(correlation):
            correlation = 0.0
        else:
            correlation = float(correlation)
        
        return jsonify({
            'image': img_data,
            'x_column': x_col,
            'y_column': y_col,
            'correlation': correlation,
            'correlation_strength': 'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/heatmap')
def get_heatmap():
    """Generate correlation heatmap"""
    try:
        df = get_current_dataframe()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return jsonify({'error': 'Need at least 2 numeric columns for heatmap'}), 400
        
        correlation_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        img_data = fig_to_base64(fig)
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                # Replace NaN with 0 for JSON serialization
                if pd.isna(corr_val):
                    corr_val = 0.0
                else:
                    corr_val = float(corr_val)
                correlations.append({
                    'var1': correlation_matrix.columns[i],
                    'var2': correlation_matrix.columns[j],
                    'correlation': corr_val
                })
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        return jsonify({
            'image': img_data,
            'top_correlations': correlations[:5]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/eda')
def get_eda():
    """Generate comprehensive Exploratory Data Analysis"""
    try:
        df = get_current_dataframe()
        
        # Create a comprehensive EDA report
        eda_report = {
            'basic_info': {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1]),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
                'duplicate_rows': int(df.duplicated().sum())
            },
            'column_info': [],
            'missing_data': {},
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Column information
        for col in df.columns:
            eda_report['column_info'].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'non_null': int(df[col].count()),
                'null': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique())
            })
        
        # Missing data analysis
        missing = df.isnull().sum()
        eda_report['missing_data'] = {
            col: {
                'count': int(count),
                'percentage': round(float(count) / float(len(df)) * 100, 2) if len(df) > 0 else 0.0
            }
            for col, count in missing.items() if count > 0
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                eda_report['numeric_summary'][col] = {
                    'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else 0.0,
                    'median': float(col_data.median()) if not pd.isna(col_data.median()) else 0.0,
                    'std': float(col_data.std()) if not pd.isna(col_data.std()) else 0.0,
                    'min': float(col_data.min()) if not pd.isna(col_data.min()) else 0.0,
                    'max': float(col_data.max()) if not pd.isna(col_data.max()) else 0.0,
                    'q1': float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else 0.0,
                    'q3': float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else 0.0
                }
            else:
                eda_report['numeric_summary'][col] = {
                    'mean': 0.0, 'median': 0.0, 'std': 0.0,
                    'min': 0.0, 'max': 0.0, 'q1': 0.0, 'q3': 0.0
                }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            top_values = df[col].value_counts().head(5)
            eda_report['categorical_summary'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': {str(k): int(v) for k, v in top_values.items()}
            }
        
        # Generate a summary visualization
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Missing values bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            ax1.barh(range(len(missing_data)), missing_data.values, color='#667eea')
            ax1.set_yticks(range(len(missing_data)))
            ax1.set_yticklabels(missing_data.index)
            ax1.set_xlabel('Missing Values Count')
            ax1.set_title('Missing Values by Column', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax1.set_title('Missing Values by Column', fontweight='bold')
        
        # 2. Data types distribution
        ax2 = fig.add_subplot(gs[0, 1])
        dtype_counts = df.dtypes.value_counts()
        ax2.pie(dtype_counts.values, labels=[str(x) for x in dtype_counts.index], 
               autopct='%1.1f%%', colors=plt.cm.Set3(range(len(dtype_counts))))
        ax2.set_title('Data Types Distribution', fontweight='bold')
        
        # 3. Numeric columns distribution
        if len(numeric_cols) > 0:
            ax3 = fig.add_subplot(gs[1, :])
            df[numeric_cols].hist(bins=20, figsize=(16, 4), ax=ax3, color='#667eea', alpha=0.7)
            ax3.set_title('Numeric Columns Distribution', fontweight='bold')
        
        # 4. Correlation heatmap (if enough numeric columns)
        if len(numeric_cols) >= 2:
            ax4 = fig.add_subplot(gs[2, :])
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       square=True, ax=ax4, cbar_kws={"shrink": 0.8})
            ax4.set_title('Correlation Matrix', fontweight='bold')
        
        plt.suptitle('Exploratory Data Analysis Summary', fontsize=18, fontweight='bold', y=0.995)
        img_data = fig_to_base64(fig)
        
        return jsonify({
            'image': img_data,
            'report': eda_report
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-cleaning', methods=['POST'])
def analyze_cleaning():
    """Analyze dataset and get AI cleaning recommendations"""
    try:
        df = get_current_dataframe()
        
        # Prepare dataset analysis
        analysis = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
            'sample_data': df.head(5).fillna('').to_dict('records')
        }
        
        # Generate automatic analysis without AI (fallback)
        issues = []
        recommendations = []
        
        # Check for missing values
        missing_cols = {k: v for k, v in analysis['missing_values'].items() if v > 0}
        if missing_cols:
            for col, count in missing_cols.items():
                pct = (count / analysis['shape'][0]) * 100
                priority = "High" if pct > 50 else "Medium" if pct > 10 else "Low"
                issues.append({
                    "column": col,
                    "issue": f"{count} missing values ({pct:.1f}%)",
                    "priority": priority,
                    "action": "Fill with appropriate method or drop if >50%"
                })
            recommendations.append(f"Handle missing values in {len(missing_cols)} columns")
        
        # Check for duplicates
        if analysis['duplicate_rows'] > 0:
            issues.append({
                "column": "All columns",
                "issue": f"{analysis['duplicate_rows']} duplicate rows found",
                "priority": "High",
                "action": "Remove duplicate rows"
            })
            recommendations.append("Remove duplicate rows to ensure data integrity")
        
        # Check data types
        object_cols = [col for col, dtype in analysis['dtypes'].items() if dtype == 'object']
        if object_cols:
            recommendations.append(f"Trim whitespace in {len(object_cols)} text columns")
            issues.append({
                "column": ", ".join(object_cols[:3]) + ("..." if len(object_cols) > 3 else ""),
                "issue": "Text columns may have whitespace or formatting issues",
                "priority": "Medium",
                "action": "Trim whitespace and standardize column names"
            })
        
        # Build summary
        summary = f"Dataset has {analysis['shape'][0]} rows and {analysis['shape'][1]} columns. "
        if missing_cols:
            summary += f"Found missing values in {len(missing_cols)} columns. "
        if analysis['duplicate_rows'] > 0:
            summary += f"Found {analysis['duplicate_rows']} duplicate rows. "
        summary += "Recommend standard cleaning procedures."
        
        cleaning_plan = {
            'summary': summary,
            'issues': issues,
            'recommendations': recommendations if recommendations else ["Dataset appears clean, but standard cleaning is recommended"]
        }
        
        # Try AI enhancement (optional, won't fail if API error)
        try:
            prompt = f"""Analyze this dataset briefly: {analysis['shape'][0]} rows, {analysis['shape'][1]} cols, {len(missing_cols)} cols with missing data, {analysis['duplicate_rows']} duplicates. Provide 2-3 key cleaning recommendations."""
            
            headers = {
                'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'llama-3.1-sonar-small-128k-online',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.2,
                'max_tokens': 500
            }
            
            response = requests.post(PERPLEXITY_API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                ai_response = response.json()
                ai_content = ai_response['choices'][0]['message']['content']
                cleaning_plan['ai_insights'] = ai_content
        except:
            # Silent fail - use automatic analysis
            pass
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'cleaning_plan': cleaning_plan,
            'ai_response': cleaning_plan.get('ai_insights', 'Using automatic analysis')
        })
            
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.route('/api/clean-dataset', methods=['POST'])
def clean_dataset():
    """Clean the dataset based on recommendations"""
    try:
        df = get_current_dataframe()
        original_filename = session.get('original_filename', 'dataset')
        
        # Get cleaning options from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        options = data.get('options', {})
        
        # Store original stats
        original_shape = df.shape
        
        # Apply cleaning operations
        cleaning_log = []
        
        try:
            # 1. Remove duplicate rows
            if options.get('remove_duplicates', True):
                before = len(df)
                df = df.drop_duplicates()
                after = len(df)
                if before != after:
                    cleaning_log.append(f"Removed {before - after} duplicate rows")
                else:
                    cleaning_log.append("No duplicate rows found")
            
            # 2. Handle missing values - Only remove rows with null values
            missing_strategy = options.get('missing_strategy', 'drop_rows')
            
            if missing_strategy == 'drop_rows' or missing_strategy == 'drop':
                # Drop rows where ANY value is null (how='any')
                before = len(df)
                df = df.dropna(how='any')  # Drop if ANY value in the row is null
                after = len(df)
                if before != after:
                    cleaning_log.append(f"Dropped {before - after} rows containing any missing values")
                else:
                    cleaning_log.append("No rows with missing values found")
            
            elif missing_strategy == 'drop_columns':
                threshold = options.get('missing_threshold', 50)  # % threshold
                before_cols = df.shape[1]
                missing_pct = (df.isnull().sum() / len(df)) * 100
                cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
                if cols_to_drop:
                    df = df.drop(columns=cols_to_drop)
                    cleaning_log.append(f"Dropped {len(cols_to_drop)} columns with >{threshold}% missing values: {', '.join(cols_to_drop)}")
                else:
                    cleaning_log.append(f"No columns with >{threshold}% missing values found")
            
            elif missing_strategy == 'fill_mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cleaning_log.append(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
                filled_count = 0
                for col in numeric_cols:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        try:
                            mean_val = df[col].mean()
                            df[col] = df[col].fillna(mean_val)  # Use assignment instead of inplace
                            cleaning_log.append(f"✓ Filled {missing_count} missing values in '{col}' with mean ({mean_val:.2f})")
                            filled_count += 1
                        except Exception as e:
                            cleaning_log.append(f"✗ Warning: Could not fill '{col}' with mean: {str(e)}")
                    else:
                        cleaning_log.append(f"  '{col}' has no missing values")
                if filled_count == 0:
                    cleaning_log.append("No missing values in numeric columns to fill")
                else:
                    cleaning_log.append(f"Successfully filled missing values in {filled_count} columns")
            
            elif missing_strategy == 'fill_median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cleaning_log.append(f"Found {len(numeric_cols)} numeric columns: {', '.join(numeric_cols)}")
                filled_count = 0
                for col in numeric_cols:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        try:
                            median_val = df[col].median()
                            df[col] = df[col].fillna(median_val)  # Use assignment instead of inplace
                            cleaning_log.append(f"✓ Filled {missing_count} missing values in '{col}' with median ({median_val:.2f})")
                            filled_count += 1
                        except Exception as e:
                            cleaning_log.append(f"✗ Warning: Could not fill '{col}' with median: {str(e)}")
                    else:
                        cleaning_log.append(f"  '{col}' has no missing values")
                if filled_count == 0:
                    cleaning_log.append("No missing values in numeric columns to fill")
                else:
                    cleaning_log.append(f"Successfully filled missing values in {filled_count} columns")
            
            elif missing_strategy == 'fill_mode':
                all_cols = df.columns.tolist()
                cleaning_log.append(f"Checking {len(all_cols)} columns for missing values")
                filled_count = 0
                for col in all_cols:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        try:
                            mode_values = df[col].mode()
                            if len(mode_values) > 0:
                                mode_value = mode_values[0]
                                df[col] = df[col].fillna(mode_value)  # Use assignment instead of inplace
                                # Format mode value safely
                                mode_str = str(mode_value)[:50]  # Limit length
                                cleaning_log.append(f"✓ Filled {missing_count} missing values in '{col}' with mode ({mode_str})")
                                filled_count += 1
                        except Exception as e:
                            cleaning_log.append(f"✗ Warning: Could not fill '{col}' with mode: {str(e)}")
                if filled_count == 0:
                    cleaning_log.append("No missing values found to fill")
                else:
                    cleaning_log.append(f"Successfully filled missing values in {filled_count} columns")
            
            elif missing_strategy == 'fill_forward':
                # Track which columns had missing values filled
                cols_filled = []
                for col in df.columns:
                    before_missing = df[col].isnull().sum()
                    if before_missing > 0:
                        df[col].fillna(method='ffill', inplace=True)
                        after_missing = df[col].isnull().sum()
                        if before_missing > after_missing:
                            cols_filled.append(f"{col} ({before_missing - after_missing} values)")
                
                if len(cols_filled) > 0:
                    cleaning_log.append(f"Forward filled missing values in: {', '.join(cols_filled)}")
                else:
                    cleaning_log.append("No missing values to forward fill")
            
            # 3. Remove whitespace from string columns
            if options.get('trim_whitespace', True):
                string_cols = df.select_dtypes(include=['object']).columns
                if len(string_cols) > 0:
                    for col in string_cols:
                        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                    cleaning_log.append(f"Trimmed whitespace from {len(string_cols)} text columns")
                else:
                    cleaning_log.append("No text columns found to trim")
            
            # 4. Standardize column names
            if options.get('standardize_columns', True):
                old_cols = df.columns.tolist()
                df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
                cleaning_log.append("Standardized column names (lowercase, underscores)")
            
            # 5. Remove outliers (optional)
            if options.get('remove_outliers', False):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                before = len(df)
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after = len(df)
                if before != after:
                    cleaning_log.append(f"Removed {before - after} outlier rows using IQR method")
            
            # 6. Convert data types
            if options.get('convert_types', False):
                # Try to convert to appropriate types
                for col in df.columns:
                    # Try numeric conversion
                    if df[col].dtype == 'object':
                        try:
                            df[col] = pd.to_numeric(df[col])
                            cleaning_log.append(f"Converted '{col}' to numeric")
                        except:
                            # Try datetime conversion
                            try:
                                df[col] = pd.to_datetime(df[col])
                                cleaning_log.append(f"Converted '{col}' to datetime")
                            except:
                                pass
        
        except Exception as cleaning_error:
            cleaning_log.append(f"Warning during cleaning: {str(cleaning_error)}")
        
        # Check if dataframe is empty after cleaning
        if len(df) == 0:
            return jsonify({
                'error': 'Cleaning operations resulted in an empty dataset. Try less aggressive cleaning options.',
                'cleaning_log': cleaning_log
            }), 400
        
        # Save cleaned dataset
        base_name = os.path.splitext(original_filename)[0]
        cleaned_filename = f"{base_name}_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        cleaned_filepath = os.path.join(app.config['CLEANED_FOLDER'], cleaned_filename)
        
        # Save with explicit engine
        df.to_excel(cleaned_filepath, index=False, engine='openpyxl')
        
        # Store cleaned file in session
        session['cleaned_file'] = cleaned_filepath
        session['cleaned_filename'] = cleaned_filename
        
        # Update current_file to use cleaned file for visualizations
        session['current_file'] = cleaned_filepath
        session.modified = True  # Ensure session is saved
        
        return jsonify({
            'success': True,
            'message': 'Dataset cleaned successfully!',
            'original_shape': original_shape,
            'cleaned_shape': df.shape,
            'cleaning_log': cleaning_log,
            'rows_removed': original_shape[0] - df.shape[0],
            'columns_removed': original_shape[1] - df.shape[1],
            'download_filename': cleaned_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-cleaned', methods=['GET'])
def download_cleaned():
    """Download the cleaned dataset"""
    try:
        if 'cleaned_file' not in session:
            return jsonify({'error': 'No cleaned file available'}), 400
        
        cleaned_file = session['cleaned_file']
        cleaned_filename = session['cleaned_filename']
        
        if not os.path.exists(cleaned_file):
            return jsonify({'error': 'Cleaned file not found'}), 404
        
        return send_file(
            cleaned_file,
            as_attachment=True,
            download_name=cleaned_filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Data Science Analysis Dashboard Starting...")
    print("="*60)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📊 Allowed file types: {', '.join(ALLOWED_EXTENSIONS)}")
    print(f"🌐 Open your browser to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000)
