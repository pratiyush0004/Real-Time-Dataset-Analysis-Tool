# 🤖 AI-Powered Data Cleaning Feature

## Overview
This dashboard now includes an advanced AI-powered data cleaning system that uses **Perplexity AI** to analyze your datasets and provide intelligent cleaning recommendations.

## Features

### 🧠 AI Analysis
- **Intelligent Dataset Analysis**: Perplexity AI examines your data structure, quality issues, and patterns
- **Priority-Based Recommendations**: Issues are categorized by priority (High/Medium/Low)
- **Actionable Insights**: Get specific recommendations on how to clean each column
- **Missing Value Detection**: Identifies and suggests strategies for handling missing data
- **Data Type Analysis**: Recommends appropriate data type conversions

### 🧹 Automated Cleaning Operations

#### 1. **Remove Duplicate Rows**
- Automatically detects and removes exact duplicate rows
- Preserves the first occurrence of each unique row

#### 2. **Missing Values Handling**
Choose from multiple strategies:
- **Drop Rows**: Remove rows containing any missing values
- **Drop Columns**: Remove columns with high % of missing values (customizable threshold)
- **Fill with Mean**: Replace missing numeric values with column mean
- **Fill with Median**: Replace missing numeric values with column median
- **Fill with Mode**: Replace missing values with most common value
- **Forward Fill**: Propagate last valid observation forward

#### 3. **Text Cleaning**
- **Trim Whitespace**: Remove leading/trailing spaces from text columns
- **Standardize Column Names**: Convert to lowercase, replace spaces with underscores

#### 4. **Data Type Conversion**
- Auto-detect and convert numeric strings to numbers
- Auto-detect and convert date strings to datetime objects
- Optimize data types for memory efficiency

#### 5. **Outlier Removal**
- Uses IQR (Interquartile Range) method
- Removes statistical outliers from numeric columns
- Configurable sensitivity

## How to Use

### Step 1: Upload Your Dataset
1. Click **"Upload Dataset"** in the header
2. Select a CSV or Excel file (up to 50MB)
3. Wait for the file to process

### Step 2: Navigate to Data Cleaning Tab
1. Click on the **"Data Cleaning"** tab in the navigation menu
2. You'll see two panels:
   - **AI-Powered Data Analysis** (left)
   - **Cleaning Options** (right)

### Step 3: Analyze with AI
1. Click **"Analyze Dataset with AI"** button
2. Wait for Perplexity AI to analyze your data (5-15 seconds)
3. Review the AI-generated report:
   - **Summary**: Overall data quality assessment
   - **Issues**: Specific problems found with priority levels
   - **Recommendations**: Actionable steps to improve data quality

### Step 4: Configure Cleaning Options
Select your desired cleaning operations:
- ✅ Remove Duplicate Rows
- Choose missing value strategy from dropdown
- ✅ Trim Whitespace
- ✅ Standardize Column Names
- ☐ Auto-Convert Data Types (optional)
- ☐ Remove Outliers (optional)

### Step 5: Clean Dataset
1. Click **"Clean Dataset"** button
2. Wait for processing to complete
3. Review cleaning results:
   - Original vs cleaned dimensions
   - Number of rows/columns removed
   - Detailed cleaning log

### Step 6: Download Cleaned Data
1. Click **"Download Cleaned Excel File"**
2. The cleaned dataset will be downloaded as an Excel file (.xlsx)
3. Filename includes timestamp: `dataset_cleaned_YYYYMMDD_HHMMSS.xlsx`

## API Endpoints

### `POST /api/analyze-cleaning`
Analyzes the current dataset using Perplexity AI.

**Response:**
```json
{
  "success": true,
  "analysis": {
    "shape": [rows, columns],
    "columns": ["col1", "col2", ...],
    "missing_values": {"col1": 5, "col2": 0},
    "duplicate_rows": 10
  },
  "cleaning_plan": {
    "summary": "Dataset quality overview",
    "issues": [
      {
        "column": "column_name",
        "issue": "description",
        "priority": "High",
        "action": "recommended action"
      }
    ],
    "recommendations": ["recommendation 1", "recommendation 2"]
  }
}
```

### `POST /api/clean-dataset`
Cleans the dataset based on selected options.

**Request Body:**
```json
{
  "options": {
    "remove_duplicates": true,
    "missing_strategy": "fill_mean",
    "trim_whitespace": true,
    "standardize_columns": true,
    "convert_types": false,
    "remove_outliers": false
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset cleaned successfully!",
  "original_shape": [1000, 15],
  "cleaned_shape": [950, 14],
  "rows_removed": 50,
  "columns_removed": 1,
  "cleaning_log": [
    "Removed 20 duplicate rows",
    "Filled missing values in 'age' with mean",
    "Trimmed whitespace from 5 text columns"
  ],
  "download_filename": "dataset_cleaned_20251223_143052.xlsx"
}
```

### `GET /api/download-cleaned`
Downloads the cleaned dataset as an Excel file.

## Perplexity AI Integration

### Model Used
- **Model**: `llama-3.1-sonar-small-128k-online`
- **Features**: 128K context window, online knowledge access
- **Temperature**: 0.2 (more deterministic responses)
- **Max Tokens**: 2000

### API Configuration
The Perplexity API key should be set as an environment variable:
```bash
# Set environment variable (Windows)
set PERPLEXITY_API_KEY=your_api_key_here

# Or (Linux/Mac)
export PERPLEXITY_API_KEY=your_api_key_here
```

### What AI Analyzes
1. **Dataset Structure**: Rows, columns, data types
2. **Data Quality**: Missing values, duplicates, outliers
3. **Column Analysis**: Individual column issues and patterns
4. **Relationships**: Correlations and dependencies
5. **Best Practices**: Industry-standard cleaning approaches

## Examples

### Example 1: Customer Dataset
**Original Issues:**
- 50 duplicate customer records
- Missing email addresses in 200 rows
- Phone numbers formatted inconsistently
- Extra whitespace in name fields

**AI Recommendations:**
1. Remove duplicate customers (High priority)
2. Drop rows with missing emails OR collect emails (High priority)
3. Standardize phone number format (Medium priority)
4. Trim whitespace from name fields (Low priority)

**Cleaning Applied:**
- ✅ Removed 50 duplicates
- ✅ Dropped 200 rows with missing emails
- ✅ Trimmed whitespace
- ✅ Standardized column names
- **Result**: 750 clean rows ready for analysis

### Example 2: Sales Dataset
**Original Issues:**
- Outlier sale amounts (likely errors)
- Date strings in multiple formats
- Missing product categories
- Numeric data stored as text

**AI Recommendations:**
1. Convert date strings to datetime (High priority)
2. Convert numeric strings to numbers (High priority)
3. Fill missing categories with mode (Medium priority)
4. Remove outlier amounts using IQR (Medium priority)

**Cleaning Applied:**
- ✅ Converted dates to datetime
- ✅ Converted amounts to numeric
- ✅ Filled categories with mode
- ✅ Removed 15 outlier rows
- **Result**: Clean, analyzable sales data

## Best Practices

### When to Use Each Missing Value Strategy

1. **Drop Rows**: When missing data is minimal (<5% of rows) and random
2. **Drop Columns**: When column has >50% missing values
3. **Fill with Mean**: For normally distributed numeric data
4. **Fill with Median**: For skewed numeric data or when outliers present
5. **Fill with Mode**: For categorical data
6. **Forward Fill**: For time-series data

### Data Quality Checklist
- [ ] Remove obvious duplicates
- [ ] Handle missing values appropriately
- [ ] Standardize text formatting
- [ ] Convert data types correctly
- [ ] Remove or cap outliers
- [ ] Validate date ranges
- [ ] Check for logical inconsistencies

## Troubleshooting

### AI Analysis Fails
- **Issue**: API timeout or error
- **Solution**: Check internet connection, try again in a moment
- **Note**: Perplexity API has rate limits

### Cleaning Removes Too Many Rows
- **Issue**: Aggressive cleaning settings
- **Solution**: Use less aggressive missing value strategy (fill instead of drop)

### Downloaded File Won't Open
- **Issue**: Excel file corruption
- **Solution**: Ensure openpyxl is installed: `pip install openpyxl`

### Outlier Removal Removes Valid Data
- **Issue**: IQR method too sensitive for your data distribution
- **Solution**: Uncheck "Remove Outliers" or manually filter after download

## File Structure

```
cleaned/
├── dataset_cleaned_20251223_143052.xlsx
├── sales_data_cleaned_20251223_150023.xlsx
└── README.md

uploads/
├── original_dataset.csv
└── README.md
```

## Security Notes

- Uploaded files are stored temporarily in the `uploads/` folder
- Cleaned files are stored in the `cleaned/` folder
- Files are associated with user sessions
- API key is stored server-side (not exposed to clients)
- Maximum file size: 50MB

## Future Enhancements

Potential improvements for future versions:
- [ ] Custom cleaning rules
- [ ] Batch processing for multiple files
- [ ] Cleaning history and undo
- [ ] Advanced imputation methods (KNN, MICE)
- [ ] Data validation rules
- [ ] Export cleaning scripts
- [ ] Schedule automatic cleaning
- [ ] Compare before/after statistics

## Credits

- **AI Provider**: Perplexity AI (llama-3.1-sonar-small-128k-online)
- **Data Processing**: Pandas, NumPy
- **Excel Export**: openpyxl
- **Web Framework**: Flask
- **UI Framework**: Tailwind CSS

---

**Happy Data Cleaning! 🧹✨**
