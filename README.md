# Washington State CPA Firms Analysis Dashboard

A modern, interactive web dashboard for analyzing Washington State CPA (Certified Public Accountant) firm data. Built with Flask, Python data analysis libraries, and a beautiful Tailwind CSS frontend.

## Features

- � **Upload Any Dataset**: Upload your own CSV or Excel files for instant analysis
- 📊 **Interactive Visualizations**: Dynamic charts and graphs powered by Matplotlib and Seaborn
- 🎯 **Real-time Analytics**: Live data analysis with multiple visualization types
- 🔄 **Smart Analysis**: Automatically detects column types (cities, dates, categories, etc.)
- 🏙️ **Top Cities Analysis**: Explore firm distribution across cities
- 💼 **Business Type Insights**: Analyze different business structures and their prevalence
- 📈 **Registration Trends**: Track registrations over time
- 🔍 **Advanced Analytics**: Correlation matrices, lifespan analysis, and location comparisons
- 📱 **Responsive Design**: Beautiful UI that works on all devices
- ⚡ **Fast Performance**: Optimized data loading with intelligent caching
- 🔐 **Secure Uploads**: 50MB file size limit with validation

## Tech Stack

- **Backend**: Python, Flask
- **Data Analysis**: Pandas, NumPy, Matplotlib, Seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Tailwind CSS
- **Icons**: Font Awesome

## Installation

1. **Clone the repository** (or you already have it)

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Make sure your CSV file is in the project root**:
   - `Washington_State_CPA__Certified_Public_Accountant__Firms_20250412.csv`

## Running the Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Open your browser** and navigate to:
```
http://localhost:5000
```

3. **Upload Your Own Dataset** (optional):
   - Click the "Upload Dataset" button in the header
   - Select any CSV or Excel file (.csv, .xlsx, .xls)
   - The dashboard will automatically analyze your data
   - Supports files up to 50MB

## Using Your Own Datasets

The application automatically detects columns in your dataset:
- **Cities/Locations**: Columns with "city" or "location" in the name
- **Business Types**: Columns with "type", "category", "business", or "class"
- **Dates**: Columns with "date", "registration", "created", "issued", or "expires"
- **States/Regions**: Columns with "state" or "region"

### Supported File Formats
- **CSV** (.csv)
- **Excel** (.xlsx, .xls)

### Tips for Best Results
- Include column names in the first row
- Use clear, descriptive column names (e.g., "City", "Registration Date", "Business Type")
- Date columns should be in standard formats (YYYY-MM-DD, MM/DD/YYYY, etc.)
- The system handles missing values automatically

## Dashboard Sections

### 1. Overview
- Key statistics cards showing total firms, cities, business types, and data quality
- Quick view of top cities and business type distribution

### 2. Top Cities
- Detailed visualization of the top 20 cities by number of CPA firms
- Interactive bar chart with firm counts

### 3. Business Types
- Pie chart showing the distribution of different business structures
- Percentage breakdown of each type

### 4. Trends
- Registration trends over time with historical data
- In-state vs out-of-state firm comparison
- Time series analysis with trend lines

### 5. Advanced Analysis
- Firm lifespan analysis by business type
- CPOST /api/upload` - Upload a new dataset file
- `POST /api/reset` - Reset to default dataset
- `GET /api/current-file` - Get current file information
- `orrelation matrix for numeric variables
- Statistical insights and patterns

### 6. Data Table
- Raw data preview with first 50 rows
- Scrollable table with all columns
- Quick data inspection

## Project Structure

```
├── app.py                  # Flask application with API endpoints
├── garvpy.py              # Original data analysis script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Main dashboard HTML template
├── static/                # Static files (empty, CDN used)
└── Washington_State_CPA__Certified_Public_Accountant__Firms_20250412.csv
```

## API Endpoints

- `GET /` - Main dashboard page
- `GET /api/overview` - Get dataset statistics
- `GET /api/top-cities` - Get top 20 cities data
- `GET /api/business-types` - Get business type distribution
- `GET /api/registration-trends` - Get registration trends over time
- `GET /api/lifespan-analysis` - Get firm lifespan analysis
- `GET /api/location-comparison` - Get in-state vs out-of-state comparison
- `GET /api/data-table` - Get sample data table
- `GET /api/correlation-matrix` - Get correlation matrix

## Features Highlights

### Beautiful UI
- Modern gradient design with purple theme
- Smooth animations and transitions
- Card-based layout for easy navigation
- Responsive grid system

##Automatic column detection
- Loading states with skeleton screens
- Error handling with user feedback
- Session-based file management
- Intelligent caching system
- Loading states with skeleton screens
- Error handling with user feedback

### Interactive Navigation
- Tab-based navigation
- Active state indicators
- One-click refresh functionality
- Smooth section transitions

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge
- Opera

## Performance Tips

- Charts are generated on-demand and cached
- Initial page load fetches only overview data
- Additional sections load when accessed
- Refresh button clears cache and reloads all data

## Customization

You can customize the dashboard by:
1. Modifying colors in Tailwind classes
2. Adjusting chart styles in `app.py`
3. Adding new API endpoints for additional analysis
4. Customizing the layout in `index.html`

## Troubleshooting

**Charts not loading?**
- Check that the CSV file path is correct in `app.py`
- Verify your dataset has appropriate columns

**Upload not working?**
- Ensure file is CSV or Excel format
- Check file size (must be under 50MB)
- Verify file contains valid data with headers
- Check console logs for specific error messages

**Analysis failing on custom dataset?**
- Ensure column names are descriptive (e.g., "City", "Date", "Type")
- Check that date columns are in standard formats
- Remove or fix any corrupted data in the file
- The system works best with structured tabular data

**Slow performance?**
- Large datasets may take time to process
- Consider implementing pagination for the data table
- Use caching for frequently accessed data
- Clear browser cache and refresh
- Consider implementing pagination for the data table
- Use caching for frequently accessed data

**Port already in use?**
- Change the port in `app.py`: `app.run(debug=True, port=5001)`

## License

This project is open source and available for educational and commercial use.

## Author

Created with ❤️ for data analysis and visualization

---

Enjoy exploring your CPA firms data with this beautiful dashboard! 🚀
