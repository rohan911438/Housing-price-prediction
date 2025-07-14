# Housing Price Prediction Project

A comprehensive machine learning project that predicts California housing prices using various regression algorithms and advanced feature engineering techniques.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models Used](#models-used)
- [Results](#results)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🎯 Project Overview

This project aims to predict median house values in California using the famous California Housing dataset. The project implements multiple machine learning algorithms including Linear Regression and Random Forest Regressor with hyperparameter tuning to achieve optimal performance.

### Key Objectives:
- Build accurate regression models to predict house prices
- Perform comprehensive exploratory data analysis (EDA)
- Apply feature engineering techniques to improve model performance
- Compare different machine learning algorithms
- Optimize models using hyperparameter tuning

## 📊 Dataset Description

The California Housing dataset contains information about housing districts in California from the 1990 census. Each row represents a housing district.

### Dataset Features:
- **longitude**: Longitude coordinate of the housing district
- **latitude**: Latitude coordinate of the housing district  
- **housing_median_age**: Median age of houses in the district
- **total_rooms**: Total number of rooms in the district
- **total_bedrooms**: Total number of bedrooms in the district
- **population**: Total population in the district
- **households**: Total number of households in the district
- **median_income**: Median income of households in the district
- **ocean_proximity**: Categorical variable indicating proximity to ocean
- **median_house_value**: Target variable - median house value in the district

### Dataset Statistics:
- **Size**: 20,640 housing districts
- **Missing Values**: 207 missing values in total_bedrooms column
- **Data Types**: 9 numerical features, 1 categorical feature

## ✨ Features

- **Data Cleaning**: Handling missing values and data inconsistencies
- **Exploratory Data Analysis**: Comprehensive visualization and statistical analysis
- **Feature Engineering**: Creating new meaningful features from existing ones
- **Data Visualization**: Interactive plots and correlation heatmaps
- **Multiple ML Models**: Linear Regression and Random Forest implementation
- **Model Optimization**: GridSearchCV for hyperparameter tuning
- **Performance Evaluation**: R² score and cross-validation metrics

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd housing-price-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Navigate to and open `main.ipynb`

3. **Run the cells sequentially**
   - Execute cells in order to reproduce the complete analysis

4. **Explore the results**
   - View visualizations, model performance metrics, and predictions

## 🔧 Data Preprocessing

### Data Cleaning Steps:
1. **Missing Value Treatment**: Removed 207 rows with missing `total_bedrooms` values
2. **Data Type Validation**: Ensured appropriate data types for all features
3. **Outlier Detection**: Identified and analyzed extreme values

### Data Transformations:
1. **Log Transformation**: Applied to skewed numerical features
   - `total_rooms`
   - `total_bedrooms` 
   - `population`
   - `households`

2. **Categorical Encoding**: One-hot encoding for `ocean_proximity`
3. **Feature Scaling**: StandardScaler applied for algorithm optimization

## 🛠️ Feature Engineering

### Created Features:
1. **bedroom_ratio**: `total_bedrooms / total_rooms`
   - Represents the proportion of bedrooms to total rooms
   
2. **households_rooms**: `total_rooms / households`
   - Average number of rooms per household

### Benefits:
- Improved model interpretability
- Better capture of underlying patterns
- Enhanced predictive performance

## 🤖 Models Used

### 1. Linear Regression
- **Purpose**: Baseline model for comparison
- **Performance**: R² score evaluation on test set
- **Features**: All engineered features with one-hot encoding

### 2. Random Forest Regressor
- **Purpose**: Advanced ensemble method for improved accuracy
- **Hyperparameter Tuning**: GridSearchCV optimization
- **Parameters Tuned**:
  - `n_estimators`: [30, 100, 300]
  - `max_depth`: [5, 20, 60, 80]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

## 📊 Results

### Model Performance:
- **Linear Regression**: R² score on test set
- **Random Forest (Default)**: Improved performance over linear regression
- **Optimized Random Forest**: Best performance after hyperparameter tuning

### Key Insights:
- `median_income` shows strongest correlation with house prices (0.69)
- Geographic features (`latitude`, `longitude`) important for predictions
- Feature engineering significantly improved model performance
- Random Forest outperformed Linear Regression

## 📁 Project Structure

```
housing-price-prediction/
├── README.md                 # Project documentation
├── main.ipynb               # Main Jupyter notebook with complete analysis
├── housing.csv              # California Housing dataset
├── requirements.txt         # Python dependencies
└── .venv/                   # Virtual environment (created after setup)
```

## 📦 Dependencies

### Core Libraries:
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static plotting
- **seaborn**: Statistical data visualization

### Machine Learning:
- **scikit-learn**: ML algorithms and preprocessing tools
  - LinearRegression
  - RandomForestRegressor
  - train_test_split
  - GridSearchCV
  - StandardScaler

### Jupyter:
- **jupyter**: Interactive notebook environment
- **ipykernel**: Jupyter kernel support

### Installation Command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter ipykernel
```

## 🔍 Key Visualizations

1. **Distribution Plots**: Histograms of all features before and after transformations
2. **Correlation Heatmap**: Feature correlation matrix visualization
3. **Geographic Scatter Plot**: Housing locations colored by price
4. **Feature Relationships**: Scatter plots showing feature interactions

## 📈 Future Enhancements

- **Advanced Models**: XGBoost, Neural Networks
- **Feature Selection**: Recursive feature elimination
- **Cross-Validation**: K-fold cross-validation implementation
- **Deployment**: Web application for real-time predictions
- **Additional Features**: External data integration (crime rates, school districts)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Rohan Kumar**
- GitHub: rohan911438
- LinkedIn: rohan-kumar-1a60b7314
- Email:123131rkorohan@gmail.com

## 🙏 Acknowledgments

- California Housing dataset from the StatLib repository
- Scikit-learn community for excellent ML tools
- Jupyter Project for interactive computing environment

---

**Note**: This project is for educational and demonstration purposes. The model predictions should not be used for actual real estate decisions without further validation and professional consultation.
