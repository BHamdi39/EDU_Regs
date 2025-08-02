import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
import io
import base64
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Econometrics Learning Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with RED color policy for specified sections
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin: 1rem 0;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
    }
    .formula-box {
        background-color: #f0f8ff;
        border: 2px solid #4169e1;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .interpretation-box {
        background-color: #f5f5dc;
        border: 2px solid #daa520;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .results-table {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 1rem;
    }
    .creator-label {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: rgba(0,0,0,0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8rem;
        z-index: 999;
    }
    .metric-box {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    /* RED COLOR POLICY FOR SPECIFIED TEXT */
    .red-header {
        color: #dc143c !important;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .red-text {
        color: #dc143c !important;
        font-weight: bold;
    }
    .red-list {
        color: #dc143c !important;
    }
    .red-hypothesis {
        color: #dc143c !important;
        font-weight: bold;
    }
    .red-interpretation {
        color: #dc143c !important;
    }
    .advanced-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .comparison-table {
        background-color: #f8f9ff;
        border: 1px solid #e1e5fe;
        border-radius: 8px;
        overflow: hidden;
    }
    .export-button {
        background: linear-gradient(45deg, #ff6b6b, #ee5a24);
        color: white !important;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .export-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 107, 107, 0.4);
    }
    .time-series-plot {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 1rem;
        background-color: #f9fff9;
    }
</style>
""", unsafe_allow_html=True)

# Creator label
st.markdown("""
<div class="creator-label">
    Created by HAMDI Boulanouar
</div>
""", unsafe_allow_html=True)

# Language selection
def get_language():
    return st.sidebar.selectbox(
        "🌐 Language / اللغة",
        ["English", "العربية"],
        help="Select your preferred language"
    )

# Translation dictionary
translations = {
    "English": {
        "title": "📊 Econometrics Learning Laboratory",
        "subtitle": "Complete Linear Regression Analysis Tool for Students",
        "upload_data": "📁 Upload Your Dataset",
        "data_preview": "👀 Data Preview",
        "variable_selection": "🎯 Variable Selection",
        "regression_analysis": "📈 Regression Analysis",
        "diagnostics": "🔍 Model Diagnostics",
        "predictions": "🔮 Predictions",
        "dependent_var": "Dependent Variable (Y)",
        "independent_vars": "Independent Variables (X)",
        "run_regression": "Run Regression Analysis",
        "simple_regression": "Simple Linear Regression",
        "multiple_regression": "Multiple Linear Regression"
    },
    "العربية": {
        "title": "📊 مختبر تعلم الاقتصاد القياسي",
        "subtitle": "أداة تحليل الانحدار الخطي الشاملة للطلاب",
        "upload_data": "📁 تحميل البيانات",
        "data_preview": "👀 معاينة البيانات",
        "variable_selection": "🎯 اختيار المتغيرات",
        "regression_analysis": "📈 تحليل الانحدار",
        "diagnostics": "🔍 تشخيص النموذج",
        "predictions": "🔮 التنبؤات",
        "dependent_var": "المتغير التابع (Y)",
        "independent_vars": "المتغيرات المستقلة (X)",
        "run_regression": "تشغيل تحليل الانحدار",
        "simple_regression": "الانحدار الخطي البسيط",
        "multiple_regression": "الانحدار الخطي المتعدد"
    }
}

# Enhanced session state initialization with persistence
def initialize_session_state():
    """Initialize and manage session state with persistence"""
    
    # Core data storage
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'regression_results' not in st.session_state:
        st.session_state.regression_results = None
    
    # Variable selections
    if 'dependent_var' not in st.session_state:
        st.session_state.dependent_var = None
    
    if 'independent_vars' not in st.session_state:
        st.session_state.independent_vars = []
    
    # Analysis settings
    if 'analysis_settings' not in st.session_state:
        st.session_state.analysis_settings = {
            'confidence_level': 95,
            'regression_type': 'OLS',
            'include_diagnostics': True,
            'language': 'English'
        }
    
    # File upload tracking
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    if 'data_hash' not in st.session_state:
        st.session_state.data_hash = None
    
    # Page state tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Data Upload & Preview"
    
    # Analysis history
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []

@st.cache_data(persist=True, show_spinner=False)
def cache_uploaded_data(file_content, file_name):
    """Cache uploaded data with persistence"""
    try:
        if file_name.endswith('.csv'):
            data = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        else:
            data = pd.read_excel(io.BytesIO(file_content))
        
        # Create data hash for change detection
        data_hash = hash(str(data.values.tobytes()) + str(data.columns.tolist()))
        
        return data, data_hash
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def optimize_memory_usage(df):
    """Reduce memory usage of dataframe"""
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def load_sample_data(dataset_type):
    """Load sample data with proper state management"""
    
    np.random.seed(42)
    
    if dataset_type == "wage":
        n = 200
        education = np.random.normal(12, 3, n)
        experience = np.random.exponential(8, n)
        gender = np.random.choice([0, 1], n, p=[0.6, 0.4])
        
        wage = (
            25000 +
            2500 * education +
            800 * experience +
            -100 * experience**2 / 10 +
            -3000 * gender +
            np.random.normal(0, 5000, n)
        )
        
        data = pd.DataFrame({
            'wage': wage,
            'education': education,
            'experience': experience,
            'gender': gender
        })
        dataset_name = "Sample Wage Data"
    
    elif dataset_type == "housing":
        n = 300
        square_feet = np.random.normal(2000, 500, n)
        bedrooms = np.random.choice([2, 3, 4, 5], n, p=[0.2, 0.4, 0.3, 0.1])
        age = np.random.exponential(15, n)
        location_premium = np.random.choice([0, 15000, 30000], n, p=[0.5, 0.3, 0.2])
        
        price = (
            50000 + 
            100 * square_feet + 
            5000 * bedrooms - 
            500 * age + 
            location_premium + 
            np.random.normal(0, 15000, n)
        )
        
        data = pd.DataFrame({
            'price': price,
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'age': age,
            'location_premium': location_premium
        })
        dataset_name = "Sample Housing Data"
    
    else:  # stock data
        n = 250
        market_return = np.random.normal(0.05, 0.15, n)
        company_beta = 1.2
        company_return = 0.02 + company_beta * market_return + np.random.normal(0, 0.1, n)
        volatility = np.abs(np.random.normal(0.2, 0.05, n))
        
        data = pd.DataFrame({
            'company_return': company_return,
            'market_return': market_return,
            'volatility': volatility,
            'trading_volume': np.random.lognormal(10, 1, n)
        })
        dataset_name = "Sample Stock Data"
    
    # Update session state
    st.session_state.data = data
    st.session_state.uploaded_file_name = dataset_name
    st.session_state.data_hash = hash(str(data.values.tobytes()))
    
    # Reset analysis-specific state
    st.session_state.dependent_var = None
    st.session_state.independent_vars = []
    st.session_state.regression_results = None
    
    st.success(f"✅ {dataset_name} loaded successfully!")
    st.experimental_rerun()

def data_upload_page(language):
    """Enhanced data upload page with state persistence"""
    
    t = translations[language]
    
    st.markdown(f'<h2 class="section-header">{t["upload_data"]}</h2>', unsafe_allow_html=True)
    
    # Display current state info
    if st.session_state.data is not None:
        st.success(f"✅ **Data Loaded**: {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
        if st.session_state.uploaded_file_name:
            st.info(f"📁 **File**: {st.session_state.uploaded_file_name}")
    
    # File upload with persistence
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis",
        key="file_uploader"
    )
    
    # Handle file upload and caching
    if uploaded_file is not None:
        # Check if this is a new file
        if (st.session_state.uploaded_file_name != uploaded_file.name or 
            st.session_state.data is None):
            
            try:
                # Cache the uploaded data
                file_content = uploaded_file.getvalue()
                data, data_hash = cache_uploaded_data(file_content, uploaded_file.name)
                
                if data is not None:
                    data = optimize_memory_usage(data)
                    
                    # Update session state
                    st.session_state.data = data
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.data_hash = data_hash
                    
                    # Reset selections if new data
                    st.session_state.dependent_var = None
                    st.session_state.independent_vars = []
                    st.session_state.regression_results = None
                    
                    st.success("✅ Data uploaded successfully!")
                    st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
    
    # Display data if available
    if st.session_state.data is not None:
        display_data_preview(st.session_state.data, language)
    
    # Sample data options with state preservation
    st.markdown("---")
    st.subheader("📊 Or Use Sample Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Wage Dataset", key="wage_data"):
            load_sample_data("wage")
    
    with col2:
        if st.button("🏠 Housing Dataset", key="housing_data"):
            load_sample_data("housing")
    
    with col3:
        if st.button("📈 Stock Dataset", key="stock_data"):
            load_sample_data("stock")

def display_data_preview(data, language):
    """Display data preview with persistent state"""
    
    t = translations[language]
    
    # Data overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📊 Rows", data.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📈 Columns", data.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeric", numeric_cols)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        missing_vals = data.isnull().sum().sum()
        st.metric("❓ Missing", missing_vals)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Quick statistics
    if st.checkbox("📊 Show Descriptive Statistics", key="show_stats"):
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Data visualization
    if st.checkbox("📈 Show Data Visualization", key="show_viz"):
        st.subheader("📈 Data Visualization")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Correlation heatmap
                st.subheader("🔥 Correlation Heatmap")
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(data[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
                st.pyplot(fig)
            
            with col2:
                # Distribution plots
                st.subheader("📊 Distribution Plots")
                selected_col = st.selectbox("Select column for distribution:", numeric_cols)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[selected_col], kde=True, ax=ax)
                plt.title(f'Distribution of {selected_col}')
                st.pyplot(fig)

def regression_analysis_page(language):
    """Enhanced regression analysis page with persistent settings"""
    
    t = translations[language]
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        st.info("👈 Go to 'Data Upload & Preview' to load your dataset")
        return
    
    data = st.session_state.data
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns for regression analysis!")
        return
    
    st.markdown(f'<h2 class="section-header">{t["regression_analysis"]}</h2>', unsafe_allow_html=True)
    
    # Show current data info
    st.info(f"📊 **Current Dataset**: {data.shape[0]} rows × {data.shape[1]} columns | File: {st.session_state.uploaded_file_name}")
    
    # Variable selection with state persistence
    st.markdown(f'<h3 class="section-header">{t["variable_selection"]}</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get current dependent variable or use default
        current_dep = st.session_state.dependent_var if st.session_state.dependent_var in numeric_cols else numeric_cols[0]
        
        dependent_var = st.selectbox(
            f"🎯 {t['dependent_var']}",
            numeric_cols,
            index=numeric_cols.index(current_dep) if current_dep in numeric_cols else 0,
            help="Choose the variable you want to predict",
            key="dep_var_select"
        )
        
        # Update session state
        if st.session_state.dependent_var != dependent_var:
            st.session_state.dependent_var = dependent_var
    
    with col2:
        # Get available independent variables
        available_indep = [col for col in numeric_cols if col != dependent_var]
        
        # Restore previous selections if valid
        current_indep = [var for var in st.session_state.independent_vars if var in available_indep]
        
        independent_vars = st.multiselect(
            f"📊 {t['independent_vars']}",
            available_indep,
            default=current_indep,
            help="Choose variables to predict the dependent variable",
            key="indep_var_select"
        )
        
        # Update session state
        if st.session_state.independent_vars != independent_vars:
            st.session_state.independent_vars = independent_vars
    
    if not independent_vars:
        st.warning("⚠️ Please select at least one independent variable!")
        return
    
    # Show analysis summary
    regression_type = t["simple_regression"] if len(independent_vars) == 1 else t["multiple_regression"]
    st.info(f"🔍 **Analysis Type**: {regression_type}")
    
    if st.button(f"🚀 {t['run_regression']}", type="primary", key="run_regression_btn"):
        with st.spinner("Running regression analysis..."):
            run_regression_analysis(data, dependent_var, independent_vars)

def run_regression_analysis(data, dependent_var, independent_vars):
    """Run regression analysis with results persistence"""
    
    try:
        # Prepare data
        y = data[dependent_var].values
        X = data[independent_vars].values
        X_with_const = sm.add_constant(X)
        
        # Fit OLS model
        model = sm.OLS(y, X_with_const).fit()
        
        # Store results in session state
        st.session_state.regression_results = {
            'model': model,
            'dependent_var': dependent_var,
            'independent_vars': independent_vars,
            'X': X,
            'y': y,
            'X_with_const': X_with_const,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to analysis history
        analysis_record = {
            'dependent_var': dependent_var,
            'independent_vars': independent_vars,
            'r_squared': model.rsquared,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.analysis_history.append(analysis_record)
        
        st.success("✅ Regression analysis completed successfully!")
        
        # Display results
        display_regression_results(model, dependent_var, independent_vars, "English")
        
    except Exception as e:
        st.error(f"❌ Error in regression analysis: {str(e)}")
        st.info("Please check your data and variable selections.")

def display_regression_results(model, dependent_var, independent_vars, language):
    """Display comprehensive regression results with mathematical explanations - FIXED VERSION"""
    
    # Mathematical Foundation - RED TEXT
    st.markdown("""
    <div class="formula-box">
    <h4 class="red-header">📚 Mathematical Foundation</h4>
    """, unsafe_allow_html=True)
    
    if len(independent_vars) == 1:
        st.latex(r"Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i")
        st.markdown("""
        **Simple Linear Regression Model:**
        - Y₍ᵢ₎ = Dependent variable (what we're predicting)
        - β₀ = Intercept (value of Y when X = 0)
        - β₁ = Slope coefficient (change in Y for 1-unit change in X)
        - εᵢ = Error term (unexplained variation)
        """)
    else:
        st.latex(r"Y_i = \beta_0 + \beta_1 X_{1i} + \beta_2 X_{2i} + ... + \beta_k X_{ki} + \varepsilon_i")
        st.markdown("""
        **Multiple Linear Regression Model:**
        - Each β coefficient represents the **partial effect** of that variable
        - "Holding other variables constant" interpretation
        - More complex but more realistic relationships
        """)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # OLS Estimation Explanation - RED TEXT
    st.markdown("""
    <div class="formula-box">
    <h4 class="red-header">🎯 Ordinary Least Squares (OLS) Method</h4>
    """, unsafe_allow_html=True)
    
    st.latex(r"\min_{\beta} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2")
    st.markdown("""
    **The OLS Principle:**
    - Finds the line that **minimizes the sum of squared residuals**
    - Residual = Actual value - Predicted value
    - Why squared? Prevents positive and negative errors from canceling out
    - Results in the "best fitting" line through the data points
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Regression Results Table - FIXED VERSION
    st.subheader("📊 Regression Results Summary")
    
    # FIX: Handle both pandas Series and numpy array cases properly
    try:
        # Method 1: Try to access as pandas Series (if available)
        if hasattr(model.params, 'index') and hasattr(model.params, 'values'):
            # pandas Series case
            param_names = list(model.params.index)
            param_values = list(model.params.values)
            std_errors = list(model.bse.values)
            t_statistics = list(model.tvalues.values)
            p_values = list(model.pvalues.values)
            ci_lower = list(model.conf_int()[0].values)
            ci_upper = list(model.conf_int()[1].values)
        else:
            raise AttributeError("Not a pandas Series")
            
    except (AttributeError, IndexError):
        # Method 2: Handle as numpy arrays or direct values
        try:
            # Create variable names
            param_names = ['Intercept'] + independent_vars
            
            # Convert to numpy arrays and flatten
            param_values = np.atleast_1d(np.array(model.params)).flatten()
            std_errors = np.atleast_1d(np.array(model.bse)).flatten()
            t_statistics = np.atleast_1d(np.array(model.tvalues)).flatten()
            p_values = np.atleast_1d(np.array(model.pvalues)).flatten()
            
            # Handle confidence intervals
            try:
                conf_int = model.conf_int()
                if hasattr(conf_int, 'values'):
                    ci_lower = list(conf_int[0].values)
                    ci_upper = list(conf_int[1].values)
                else:
                    ci_lower = np.atleast_1d(np.array(conf_int[0])).flatten()
                    ci_upper = np.atleast_1d(np.array(conf_int[1])).flatten()
            except:
                # Fallback: calculate approximate confidence intervals
                t_crit = 1.96  # Approximate for large samples
                margin = t_crit * std_errors
                ci_lower = param_values - margin
                ci_upper = param_values + margin
            
            # Convert to lists
            param_values = list(param_values)
            std_errors = list(std_errors)
            t_statistics = list(t_statistics)
            p_values = list(p_values)
            ci_lower = list(ci_lower)
            ci_upper = list(ci_upper)
            
        except Exception as e:
            st.error(f"Error processing regression results: {str(e)}")
            return
    
    # Ensure all lists have the same length
    max_len = max(len(param_names), len(param_values), len(std_errors), 
                  len(t_statistics), len(p_values), len(ci_lower), len(ci_upper))
    
    # Pad shorter lists with NaN or appropriate defaults
    def pad_list(lst, target_len, default_val=np.nan):
        while len(lst) < target_len:
            lst.append(default_val)
        return lst[:target_len]  # Truncate if too long
    
    param_names = pad_list(param_names, max_len, "Unknown")
    param_values = pad_list(param_values, max_len, np.nan)
    std_errors = pad_list(std_errors, max_len, np.nan)
    t_statistics = pad_list(t_statistics, max_len, np.nan)
    p_values = pad_list(p_values, max_len, np.nan)
    ci_lower = pad_list(ci_lower, max_len, np.nan)
    ci_upper = pad_list(ci_upper, max_len, np.nan)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Variable': param_names,
        'Coefficient': param_values,
        'Std Error': std_errors,
        't-statistic': t_statistics,
        'p-value': p_values,
        '95% CI Lower': ci_lower,
        '95% CI Upper': ci_upper
    })
    
    # Format the results table (handle NaN values)
    results_df['Coefficient'] = pd.to_numeric(results_df['Coefficient'], errors='coerce').round(4)
    results_df['Std Error'] = pd.to_numeric(results_df['Std Error'], errors='coerce').round(4)
    results_df['t-statistic'] = pd.to_numeric(results_df['t-statistic'], errors='coerce').round(3)
    results_df['p-value'] = pd.to_numeric(results_df['p-value'], errors='coerce').round(4)
    results_df['95% CI Lower'] = pd.to_numeric(results_df['95% CI Lower'], errors='coerce').round(4)
    results_df['95% CI Upper'] = pd.to_numeric(results_df['95% CI Upper'], errors='coerce').round(4)
    
    st.dataframe(results_df, use_container_width=True)
    
    # Statistical Significance Explanation
    st.markdown("""
    <div class="formula-box">
    <h4 class="red-header">📖 How to Interpret These Results:</h4>
    """, unsafe_allow_html=True)
    
    for i, var in enumerate(param_names):
        if i >= len(results_df):
            break
            
        coef = results_df.iloc[i]['Coefficient']
        pval = results_df.iloc[i]['p-value']
        
        # Handle NaN values
        if pd.isna(coef) or pd.isna(pval):
            continue
            
        if var == 'Intercept' or 'intercept' in var.lower():
            interpretation = f"**Intercept (β₀ = {coef:.4f})**: When all X variables = 0, {dependent_var} = {coef:.4f}"
        else:
            interpretation = f"**{var} (β = {coef:.4f})**: A 1-unit increase in {var} is associated with a {coef:.4f} change in {dependent_var}"
            if len(independent_vars) > 1:
                interpretation += " (holding other variables constant)"
        
        significance = "statistically significant" if pval < 0.05 else "not statistically significant"
        
        st.markdown(f"• {interpretation}")
        st.markdown(f"  - This effect is **{significance}** (p-value = {pval:.4f})")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Fit Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📊 R-squared", f"{model.rsquared:.4f}")
        st.markdown("% of variation explained")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📈 Adjusted R²", f"{model.rsquared_adj:.4f}")
        st.markdown("Penalizes extra variables")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("🎯 F-statistic", f"{model.fvalue:.2f}")
        st.markdown(f"p-value: {model.f_pvalue:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📏 Observations", f"{int(model.nobs)}")
        st.metric("📊 DoF", f"{int(model.df_resid)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # F-test Explanation
    st.markdown("""
    <div class="formula-box">
    <h4 class="red-header">🧪 F-Test for Overall Model Significance</h4>
    """, unsafe_allow_html=True)
    
    st.latex(r"H_0: \beta_1 = \beta_2 = ... = \beta_k = 0")
    st.latex(r"H_A: \text{At least one } \beta_j \neq 0")
    
    st.markdown(f"""
    **F-statistic = {model.fvalue:.2f}** with p-value = **{model.f_pvalue:.4f}**
    
    **Interpretation:**
    - Tests whether ALL independent variables together have no effect on Y
    - If p-value < 0.05: Reject H₀ - the model is statistically significant
    - **Conclusion**: {"The model is statistically significant" if model.f_pvalue < 0.05 else "The model is not statistically significant"}
    """)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization
    st.subheader("📈 Regression Visualization")
    
    if len(independent_vars) == 1:
        # Simple regression plot
        X_data = st.session_state.regression_results['X'].flatten()
        y_data = st.session_state.regression_results['y']
        
        fig = px.scatter(
            x=X_data,
            y=y_data,
            title=f"Simple Linear Regression: {dependent_var} vs {independent_vars[0]}"
        )
        
        # Add regression line
        x_range = np.linspace(X_data.min(), X_data.max(), 100)
        y_pred_line = param_values[0] + param_values[1] * x_range
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_pred_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            xaxis_title=independent_vars[0],
            yaxis_title=dependent_var,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Multiple regression - show actual vs predicted
        y_pred = model.predict()
        y_actual = st.session_state.regression_results['y']
        
        fig = px.scatter(
            x=y_pred,
            y=y_actual,
            title="Multiple Regression: Actual vs Predicted Values"
        )
        
        # Add perfect prediction line
        min_val = min(min(y_pred), min(y_actual))
        max_val = max(max(y_pred), max(y_actual))
        
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Predicted Values",
            yaxis_title="Actual Values"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual Analysis Preview
    st.subheader("🔍 Quick Residual Check")
    residuals = model.resid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Fitted
    ax1.scatter(model.fittedvalues, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted Values')
    
    # QQ plot for normality
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)')
    
    st.pyplot(fig)
    
    st.info("💡 **Next Steps**: Check the 'Model Diagnostics' tab for comprehensive assumption testing!")

def diagnostics_page(language):
    """Comprehensive diagnostic tests for regression assumptions"""
    
    if st.session_state.regression_results is None:
        st.warning("⚠️ Please run regression analysis first!")
        return
    
    st.markdown('<h2 class="section-header">🔍 Model Diagnostics & Assumption Testing</h2>', 
                unsafe_allow_html=True)
    
    model = st.session_state.regression_results['model']
    X = st.session_state.regression_results['X']
    y = st.session_state.regression_results['y']
    X_with_const = st.session_state.regression_results['X_with_const']
    independent_vars = st.session_state.regression_results['independent_vars']
    
    # Overview of assumptions - RED TEXT
    st.markdown("""
    <div class="formula-box">
    <h4 class="red-header">📋 The Five Key Assumptions We're Testing:</h4>
    <div class="red-list">
    1. <strong>Linearity</strong>: The relationship is actually linear<br>
    2. <strong>Homoskedasticity</strong>: Constant error variance<br>
    3. <strong>No Autocorrelation</strong>: Errors are independent<br>
    4. <strong>Normality</strong>: Errors are normally distributed<br>
    5. <strong>No Multicollinearity</strong>: Independent variables aren't perfectly correlated
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Test 1: Heteroskedasticity Tests
    st.subheader("🔥 Test 1: Heteroskedasticity (Constant Variance)")
    
    # Breusch-Pagan Test
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X_with_const)
    
    st.markdown("""
    <div class="formula-box">
    <h5 class="red-header">📚 Breusch-Pagan Test</h5>
    <div class="red-hypothesis">
    <strong>Null Hypothesis (H₀): Homoskedasticity (constant variance)</strong><br>
    <strong>Alternative Hypothesis (H₁): Heteroskedasticity (non-constant variance)</strong>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Breusch-Pagan Statistic", f"{bp_stat:.4f}")
        st.metric("📊 p-value", f"{bp_pvalue:.4f}")
    
    with col2:
        if bp_pvalue < 0.05:
            st.error("❌ **Reject H₀**: Heteroskedasticity detected!")
            st.markdown("**Solution**: Use robust standard errors or transform variables")
        else:
            st.success("✅ **Fail to reject H₀**: No evidence of heteroskedasticity")
    
    # Visual test for heteroskedasticity
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model.fittedvalues, model.resid, alpha=0.6)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('Fitted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Fitted Values (Heteroskedasticity Check)')
    st.pyplot(fig)
    
    st.markdown("""
    <div class="interpretation-box">
    <div class="red-interpretation">
    <strong>How to interpret this plot:</strong><br>
    • <strong>Good: Random scatter around zero line</strong><br>
    • <strong>Bad: Funnel shape (variance increases) or systematic patterns</strong>
    </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Test 2: Autocorrelation (Durbin-Watson Test)
    st.subheader("🔄 Test 2: Autocorrelation (Independence of Errors)")
    
    dw_stat = durbin_watson(model.resid)
    
    st.markdown("""
    <div class="formula-box">
    <h5 class="red-header">📚 Durbin-Watson Test</h5>
    """, unsafe_allow_html=True)
    
    st.latex(r"DW = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n}e_t^2}")
    
    st.markdown("""
    <strong>Interpretation Rules:</strong><br>
    • DW ≈ 2.0: No autocorrelation<br>
    • DW < 1.5: Positive autocorrelation<br>
    • DW > 2.5: Negative autocorrelation
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Durbin-Watson Statistic", f"{dw_stat:.4f}")
    
    with col2:
        if 1.5 <= dw_stat <= 2.5:
            st.success("✅ **No significant autocorrelation**")
        elif dw_stat < 1.5:
            st.error("❌ **Positive autocorrelation detected**")
            st.markdown("**Solution**: Add lagged variables or use robust standard errors")
        else:
            st.error("❌ **Negative autocorrelation detected**")
            st.markdown("**Solution**: Check for over-differencing")
    
    # Test 3: Normality (Jarque-Bera Test) - FIXED VERSION
    st.subheader("📊 Test 3: Normality of Residuals")
    
    # FIX: jarque_bera returns 4 values, not 2
    try:
        jb_result = jarque_bera(model.resid)
        jb_stat = jb_result[0]
        jb_pvalue = jb_result[1]
    except:
        # Alternative calculation if jarque_bera fails
        from scipy.stats import jarque_bera as scipy_jb
        jb_stat, jb_pvalue = scipy_jb(model.resid)
    
    st.markdown("""
    <div class="formula-box">
    <h5 class="red-header">📚 Jarque-Bera Test</h5>
    """, unsafe_allow_html=True)
    
    st.latex(r"JB = \frac{n}{6}\left[S^2 + \frac{1}{4}(K-3)^2\right]")
    
    st.markdown("""
    Where:<br>
    • S = Skewness of residuals<br>
    • K = Kurtosis of residuals<br>
    • Normal distribution: S=0, K=3<br><br>
    <strong>H₀:</strong> Residuals are normally distributed<br>
    <strong>H₁:</strong> Residuals are not normally distributed
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("🎯 Jarque-Bera Statistic", f"{jb_stat:.4f}")
        st.metric("📊 p-value", f"{jb_pvalue:.4f}")
        
        # Calculate skewness and kurtosis
        from scipy.stats import skew, kurtosis
        resid_skew = skew(model.resid)
        resid_kurt = kurtosis(model.resid) + 3  # Adding 3 for normal kurtosis
        
        st.metric("📈 Skewness", f"{resid_skew:.4f}")
        st.metric("📊 Kurtosis", f"{resid_kurt:.4f}")
    
    with col2:
        if jb_pvalue < 0.05:
            st.error("❌ **Reject H₀**: Residuals are not normally distributed!")
            st.markdown("**Solutions**: Transform variables, use robust regression, or larger sample")
        else:
            st.success("✅ **Fail to reject H₀**: Residuals appear normally distributed")
    
    # Visual normality tests
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of residuals
    ax1.hist(model.resid, bins=20, density=True, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Density')
    ax1.set_title('Histogram of Residuals')
    
    # Add normal curve
    x_norm = np.linspace(model.resid.min(), model.resid.max(), 100)
    y_norm = stats.norm.pdf(x_norm, loc=np.mean(model.resid), scale=np.std(model.resid))
    ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
    ax1.legend()
    
    # Q-Q plot
    stats.probplot(model.resid, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot: Normal Distribution Check')
    
    st.pyplot(fig)
    
    # Test 4: Multicollinearity (VIF)
    if len(independent_vars) > 1:
        st.subheader("🔗 Test 4: Multicollinearity (VIF Analysis)")
        
        st.markdown("""
        <div class="formula-box">
        <h5 class="red-text">📚 Variance Inflation Factor (VIF)</h5>
        """, unsafe_allow_html=True)
        
        st.latex(r"VIF_j = \frac{1}{1-R_j^2}")
        
        st.markdown("""
        Where R²ⱼ comes from regressing Xⱼ on all other X variables<br><br>
        <strong>Interpretation:</strong><br>
        • VIF = 1: No correlation with other variables<br>
        • VIF < 5: Acceptable multicollinearity<br>
        • VIF > 10: Problematic multicollinearity
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate VIF for each variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = independent_vars
        
        vif_values = []
        for i in range(len(independent_vars)):
            vif = variance_inflation_factor(X, i)
            vif_values.append(vif)
        
        vif_data["VIF"] = vif_values
        vif_data["VIF"] = vif_data["VIF"].round(3)
        
        # Add interpretation
        vif_data["Interpretation"] = vif_data["VIF"].apply(
            lambda x: "✅ Low" if x < 5 else ("⚠️ Moderate" if x < 10 else "❌ High")
        )
        
        st.dataframe(vif_data, use_container_width=True)
        
        # Check for problematic multicollinearity
        max_vif = vif_data["VIF"].max()
        if max_vif > 10:
            st.error(f"❌ **High multicollinearity detected!** Maximum VIF = {max_vif:.3f}")
            st.markdown("""
            **Solutions:**
            - Remove highly correlated variables
            - Combine correlated variables into indices
            - Use ridge regression
            - Collect more data
            """)
        elif max_vif > 5:
            st.warning(f"⚠️ **Moderate multicollinearity present.** Maximum VIF = {max_vif:.3f}")
        else:
            st.success(f"✅ **Low multicollinearity.** Maximum VIF = {max_vif:.3f}")
    
    # Overall Diagnostic Summary
    st.subheader("📋 Diagnostic Summary")
    
    # Count passed tests
    tests_passed = 0
    total_tests = 4
    
    summary_data = []
    
    # Heteroskedasticity
    hetero_pass = bp_pvalue >= 0.05
    tests_passed += hetero_pass
    summary_data.append({
        "Test": "Heteroskedasticity (Breusch-Pagan)",
        "Status": "✅ Pass" if hetero_pass else "❌ Fail",
        "p-value": f"{bp_pvalue:.4f}",
        "Issue": "None" if hetero_pass else "Non-constant variance"
    })
    
    # Autocorrelation
    auto_pass = 1.5 <= dw_stat <= 2.5
    tests_passed += auto_pass
    summary_data.append({
        "Test": "Autocorrelation (Durbin-Watson)",
        "Status": "✅ Pass" if auto_pass else "❌ Fail",
        "p-value": f"{dw_stat:.4f}",
        "Issue": "None" if auto_pass else "Correlated errors"
    })
    
    # Normality
    norm_pass = jb_pvalue >= 0.05
    tests_passed += norm_pass
    summary_data.append({
        "Test": "Normality (Jarque-Bera)",
        "Status": "✅ Pass" if norm_pass else "❌ Fail",
        "p-value": f"{jb_pvalue:.4f}",
        "Issue": "None" if norm_pass else "Non-normal residuals"
    })
    
    # Multicollinearity
    if len(independent_vars) > 1:
        max_vif = vif_data["VIF"].max()
        multi_pass = max_vif < 5
        tests_passed += multi_pass
        summary_data.append({
            "Test": "Multicollinearity (VIF)",
            "Status": "✅ Pass" if multi_pass else "❌ Fail",
            "p-value": f"{max_vif:.3f}",
            "Issue": "None" if multi_pass else "High correlation between predictors"
        })
    else:
        total_tests = 3  # No multicollinearity test for simple regression
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Overall assessment
    pass_rate = tests_passed / total_tests
    
    if pass_rate >= 0.75:
        st.success(f"✅ **Model Quality: GOOD** ({tests_passed}/{total_tests} tests passed)")
        st.markdown("Your regression model meets most key assumptions!")
    elif pass_rate >= 0.5:
        st.warning(f"⚠️ **Model Quality: MODERATE** ({tests_passed}/{total_tests} tests passed)")
        st.markdown("Some assumptions are violated - consider remedial measures.")
    else:
        st.error(f"❌ **Model Quality: POOR** ({tests_passed}/{total_tests} tests passed)")
        st.markdown("Multiple assumptions violated - model results may be unreliable.")

def advanced_diagnostics_page():
    """Advanced diagnostic features"""
    
    if st.session_state.regression_results is None:
        st.warning("⚠️ Please run regression analysis first!")
        return
    
    st.markdown('<h2 class="section-header">🔬 Advanced Diagnostics</h2>', unsafe_allow_html=True)
    
    model = st.session_state.regression_results['model']
    X = st.session_state.regression_results['X']
    y = st.session_state.regression_results['y']
    
    # Leverage and Influence Analysis
    st.subheader("📊 Leverage and Influence Analysis")
    
    # Calculate diagnostic measures
    leverage = model.get_influence().hat_matrix_diag
    cooks_distance = model.get_influence().cooks_distance[0]
    standardized_residuals = model.get_influence().resid_studentized_internal
    
    # Create influence plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=leverage,
        y=standardized_residuals,
        mode='markers',
        marker=dict(
            size=cooks_distance * 100,
            color=cooks_distance,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Cook's Distance")
        ),
        text=[f"Obs {i+1}<br>Cook's D: {d:.3f}" for i, d in enumerate(cooks_distance)],
        hovertemplate='Leverage: %{x:.3f}<br>Std. Residual: %{y:.3f}<br>%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Influence Plot: Leverage vs Standardized Residuals",
        xaxis_title="Leverage",
        yaxis_title="Standardized Residuals",
        showlegend=False
    )
    
    # Add reference lines
    fig.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="High Residual (+2)")
    fig.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="High Residual (-2)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Identify influential observations
    high_leverage = leverage > 2 * len(model.params) / len(y)
    high_cooks = cooks_distance > 4 / len(y)
    
    if np.any(high_leverage | high_cooks):
        st.warning("⚠️ **Potentially Influential Observations Detected:**")
        
        influence_df = pd.DataFrame({
            'Observation': range(1, len(y) + 1),
            'Leverage': leverage,
            'Cooks_Distance': cooks_distance,
            'High_Leverage': high_leverage,
            'High_Cooks': high_cooks
        })
        
        problematic = influence_df[influence_df['High_Leverage'] | influence_df['High_Cooks']]
        st.dataframe(problematic, use_container_width=True)
        
        st.markdown("""
        <div class="interpretation-box">
        <h5 class="red-text">📖 What This Means:</h5>
        <div class="red-list">
        • <strong>High Leverage:</strong> Unusual X values that can strongly influence the regression line</strong><br>
        • <strong>High Cook's Distance:</strong> Observations that significantly change coefficients when removed</strong><br>
        • <strong>Recommendation:</strong> Investigate these observations for data errors or consider robust regression</strong><br>
        </div>
        """, unsafe_allow_html=True)

def model_comparison_page():
    """Compare different regression models"""
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    st.markdown('<h2 class="section-header">⚖️ Model Comparison</h2>', unsafe_allow_html=True)
    
    data = st.session_state.data
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Variable selection
    col1, col2 = st.columns(2)
    
    with col1:
        y_var = st.selectbox("Dependent Variable", numeric_cols)
    
    with col2:
        x_vars = st.multiselect("Independent Variables", 
                               [col for col in numeric_cols if col != y_var])
    
    if len(x_vars) < 1:
        st.warning("Please select at least one independent variable!")
        return
    
    if st.button("🔄 Compare Models", type="primary"):
        y = data[y_var].values
        X = data[x_vars].values
        
        results_comparison = []
        
        # OLS Results
        X_with_const = sm.add_constant(X)
        ols_model = sm.OLS(y, X_with_const).fit()
        results_comparison.append({
            'Model': 'OLS',
            'R²': ols_model.rsquared,
            'Adj R²': ols_model.rsquared_adj,
            'AIC': ols_model.aic,
            'BIC': ols_model.bic,
            'RMSE': np.sqrt(np.mean(ols_model.resid**2))
        })
        
        # Ridge Regression
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        
        for alpha in [0.1, 1.0]:
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(X_with_const, y)
            y_pred = ridge.predict(X_with_const)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(np.mean((y - y_pred)**2))
            
            results_comparison.append({
                'Model': f'Ridge (α={alpha})',
                'R²': r2,
                'Adj R²': 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(x_vars) - 1),
                'AIC': 'N/A',
                'BIC': 'N/A',
                'RMSE': rmse
            })
        
        # Lasso Regression
        from sklearn.linear_model import Lasso
        
        lasso = Lasso(alpha=0.1, fit_intercept=False, max_iter=2000)
        lasso.fit(X_with_const, y)
        y_pred = lasso.predict(X_with_const)
        
        r2 = r2_score(y, y_pred)
        results_comparison.append({
            'Model': 'Lasso (α=0.1)',
            'R²': r2,
            'Adj R²': 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(x_vars) - 1),
            'AIC': 'N/A',
            'BIC': 'N/A',
            'RMSE': np.sqrt(np.mean((y - y_pred)**2))
        })
        
        # Polynomial Features
        if len(x_vars) <= 3:  # Only for small number of variables
            poly = PolynomialFeatures(degree=2, include_bias=True)
            X_poly = poly.fit_transform(X)
            
            poly_model = sm.OLS(y, X_poly).fit()
            
            results_comparison.append({
                'Model': 'Polynomial (degree=2)',
                'R²': poly_model.rsquared,
                'Adj R²': poly_model.rsquared_adj,
                'AIC': poly_model.aic,
                'BIC': poly_model.bic,
                'RMSE': np.sqrt(np.mean(poly_model.resid**2))
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(results_comparison)
        comparison_df = comparison_df.round(4)
        
        st.subheader("📊 Model Comparison Results")
        st.dataframe(comparison_df, use_container_width=True)
        
        # Highlight best models
        st.markdown("""
        <div class="interpretation-box">
        <h5 class="red-text">🏆 Model Selection Criteria:</h5>
        <div class="red-list">
        • <strong>Highest R²:</strong> Best fit to training data<br>
        • <strong>Highest Adjusted R²:</strong> Best fit considering model complexity<br>
        • <strong>Lowest AIC/BIC:</strong> Best balance of fit and simplicity<br>
        • <strong>Lowest RMSE:</strong> Best prediction accuracy
        </div>
        """, unsafe_allow_html=True)

def time_series_features():
    """Add time series specific features"""
    
    st.markdown('<h2 class="section-header">📈 Time Series Regression</h2>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    data = st.session_state.data
    
    # Check if there's a date column
    date_cols = data.select_dtypes(include=['datetime64', 'object']).columns.tolist()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    st.subheader("📅 Time Series Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if date_cols:
            date_col = st.selectbox("Date Column (optional)", ['None'] + date_cols)
        else:
            st.info("No date column detected - using row index as time")
            date_col = 'None'
    
    with col2:
        y_var = st.selectbox("Dependent Variable", numeric_cols)
    
    # Time series specific options
    st.subheader("⚙️ Time Series Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_trend = st.checkbox("Include Linear Trend", value=True)
        include_seasonal = st.checkbox("Include Seasonal Dummies")
    
    with col2:
        max_lags = st.slider("Maximum Lags to Include", 0, 10, 2)
        include_ar_terms = st.checkbox("Include Autoregressive Terms", value=True)
    
    with col3:
        x_vars = st.multiselect("Additional Variables", 
                               [col for col in numeric_cols if col != y_var])
    
    if st.button("🚀 Run Time Series Regression", type="primary"):
        # Prepare time series data
        ts_data = data.copy()
        
        if date_col != 'None':
            ts_data[date_col] = pd.to_datetime(ts_data[date_col])
            ts_data = ts_data.sort_values(date_col)
        
        y = ts_data[y_var].values
        X_list = []
        var_names = []
        
        # Add intercept
        X_list.append(np.ones(len(y)))
        var_names.append('Intercept')
        
        # Add trend
        if include_trend:
            trend = np.arange(1, len(y) + 1)
            X_list.append(trend)
            var_names.append('Trend')
        
        # Add seasonal dummies (quarterly)
        if include_seasonal:
            for quarter in range(1, 4):  # Q1, Q2, Q3 (Q4 is reference)
                seasonal = np.array([(i % 4) == (quarter - 1) for i in range(len(y))]).astype(int)
                X_list.append(seasonal)
                var_names.append(f'Q{quarter}')
        
        # Add lagged dependent variables (AR terms)
        if include_ar_terms and max_lags > 0:
            for lag in range(1, max_lags + 1):
                y_lag = np.concatenate([np.full(lag, np.nan), y[:-lag]])
                X_list.append(y_lag)
                var_names.append(f'{y_var}_lag{lag}')
        
        # Add other variables
        for var in x_vars:
            X_list.append(ts_data[var].values)
            var_names.append(var)
        
        # Combine and handle missing values
        X = np.column_stack(X_list)
        
        # Remove rows with missing values
        valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(y) < 10:
            st.error("❌ Not enough valid observations after handling lags and missing values!")
            return
        
        # Fit model
        ts_model = sm.OLS(y, X).fit()
        
        # Display results
        st.subheader("📊 Time Series Regression Results")
        
        # Create results table
        results_df = pd.DataFrame({
            'Variable': var_names,
            'Coefficient': ts_model.params,
            'Std Error': ts_model.bse,
            't-statistic': ts_model.tvalues,
            'p-value': ts_model.pvalues
        }).round(4)
        
        st.dataframe(results_df, use_container_width=True)
        
        # Model statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R²", f"{ts_model.rsquared:.4f}")
        with col2:
            st.metric("Adj R²", f"{ts_model.rsquared_adj:.4f}")
        with col3:
            st.metric("DW Stat", f"{durbin_watson(ts_model.resid):.4f}")
        with col4:
            st.metric("Observations", len(y))

def predictions_page(language):
    """Prediction interface with confidence intervals"""
    
    if st.session_state.regression_results is None:
        st.warning("⚠️ Please run regression analysis first!")
        return
    
    st.markdown('<h2 class="section-header">🔮 Predictions & Confidence Intervals</h2>', 
                unsafe_allow_html=True)
    
    model = st.session_state.regression_results['model']
    independent_vars = st.session_state.regression_results['independent_vars']
    dependent_var = st.session_state.regression_results['dependent_var']
    
    # Explanation of prediction types
    st.markdown("""
    <div class="formula-box"> 
    <h4 class="red-text">📚 Types of Predictions</h4>
    <div class="red-list">
    <strong>1. Point Prediction:</strong> Single best guess for Y<br>
    <strong>2. Confidence Interval:</strong> Range for the MEAN of Y at given X values<br>
    <strong>3. Prediction Interval:</strong> Range for an INDIVIDUAL Y value (wider)
    </div>
    """, unsafe_allow_html=True)
    
    # Mathematical formulas
    st.markdown("""
    <div class="formula-box">
    <h5 class="red-text">🧮 Mathematical Formulas5></h5>
    """, unsafe_allow_html=True)
    
    st.latex(r"\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X_1 + ... + \hat{\beta}_k X_k")
    
    st.markdown("**Confidence Interval for Mean Response:**")
    st.latex(r"\hat{Y} \pm t_{\alpha/2} \cdot SE(\hat{Y})")
    
    st.markdown("**Prediction Interval for Individual Response:**")
    st.latex(r"\hat{Y} \pm t_{\alpha/2} \cdot SE(prediction)")
    
    st.markdown("where SE(prediction) > SE(Ŷ) because it includes both estimation uncertainty AND individual variation")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input section for new predictions
    st.subheader("📊 Make New Predictions")
    
    col1, col2 = st.columns(2)
    
    # Get input values for prediction
    new_values = {}
    
    with col1:
        st.markdown("**Enter values for prediction:**")
        for var in independent_vars:
            # Get some reasonable bounds from the data
            data = st.session_state.data
            if var in data.columns:
                min_val = float(data[var].min())
                max_val = float(data[var].max())
                mean_val = float(data[var].mean())
                
                new_values[var] = st.number_input(
                    f"{var}",
                    min_value=min_val * 0.1,  # Allow some extrapolation
                    max_value=max_val * 1.5,
                    value=mean_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range in data: {min_val:.2f} to {max_val:.2f}"
                )
            else:
                new_values[var] = st.number_input(f"{var}", value=0.0)
    
    with col2:
        confidence_level = st.slider(
            "Confidence Level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help="Higher confidence = wider intervals"
        )
        
        alpha = (100 - confidence_level) / 100
        
        if st.button("🚀 Calculate Prediction", type="primary"):
            # Prepare prediction data
            X_new = [1] + [new_values.get(var, 0) for var in independent_vars]  # Add intercept
            X_new_array = np.array(X_new).reshape(1, -1)
            
            # Point prediction
            point_pred = model.predict(X_new_array)[0]
            
            # Calculate standard errors for intervals
            # This is a simplified version - in practice, you'd use the full covariance matrix
            prediction_se = np.sqrt(model.mse_resid * (1 + np.sum(X_new_array[0] ** 2) / len(model.resid)))
            confidence_se = np.sqrt(model.mse_resid * np.sum(X_new_array[0] ** 2) / len(model.resid))
            
            # Critical t-value
            t_crit = stats.t.ppf(1 - alpha/2, df=model.df_resid)
            
            # Intervals
            conf_lower = point_pred - t_crit * confidence_se
            conf_upper = point_pred + t_crit * confidence_se
            
            pred_lower = point_pred - t_crit * prediction_se
            pred_upper = point_pred + t_crit * prediction_se
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("📍 Point Prediction", f"{point_pred:.4f}")
                st.markdown("Best single guess")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("📊 Confidence Interval", f"[{conf_lower:.4f}, {conf_upper:.4f}]")
                st.markdown("For the AVERAGE response")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                st.metric("🎯 Prediction Interval", f"[{pred_lower:.4f}, {pred_upper:.4f}]")
                st.markdown("For an INDIVIDUAL response")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Interpretation
            st.markdown(f"""
            <div class="interpretation-box">
            <h5 class="red-text">📖 Interpretation:</h5>
            <div class="red-list">
            <strong>Point Prediction:</strong> Our best estimate is that {dependent_var} = {point_pred:.4f}<br><br>
            <strong>Confidence Interval:</strong> We are {confidence_level}% confident that the AVERAGE {dependent_var} 
            for all individuals with these characteristics is between {conf_lower:.4f} and {conf_upper:.4f}<br><br>
            <strong>Prediction Interval:</strong> We are {confidence_level}% confident that a SPECIFIC individual 
            with these characteristics will have {dependent_var} between {pred_lower:.4f} and {pred_upper:.4f}
            </div>
            """, unsafe_allow_html=True)

def export_results_feature():
    """Export regression results to various formats - FIXED VERSION"""
    
    if st.session_state.regression_results is None:
        st.warning("⚠️ Please run regression analysis first!")
        return
    
    st.markdown('<h2 class="section-header">📤 Export Results</h2>', unsafe_allow_html=True)
    
    model = st.session_state.regression_results['model']
    independent_vars = st.session_state.regression_results['independent_vars']
    
    # Export options
    export_format = st.selectbox(
        "Choose Export Format:",
        ["Summary Report (TXT)", "Excel Workbook", "LaTeX Table", "CSV Data", "Python Code"]
    )
    
    if export_format == "Summary Report (TXT)":
        st.markdown("""
        ### 📄 Summary Report
        Generate a comprehensive text report with all regression results, diagnostics, and explanations.
        """)
        
        if st.button("📄 Generate Report"):
            # Create a comprehensive text summary
            report_text = f"""
ECONOMETRICS REGRESSION ANALYSIS REPORT
=====================================

Generated by: Econometrics Learning Lab - Created by HAMDI Boulanouar
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

MODEL SUMMARY:
{model.summary()}

MODEL DIAGNOSTICS:
- R-squared: {model.rsquared:.4f}
- Adjusted R-squared: {model.rsquared_adj:.4f}
- F-statistic: {model.fvalue:.4f}
- Prob (F-statistic): {model.f_pvalue:.4f}
- Number of observations: {int(model.nobs)}
- Durbin-Watson: {durbin_watson(model.resid):.4f}

COEFFICIENTS INTERPRETATION:
"""
            
            # FIX: Handle both pandas Series and numpy array cases
            try:
                # Try pandas Series approach first
                if hasattr(model.params, 'index'):
                    param_names = model.params.index
                    param_values = model.params.values
                    pvalue_values = model.pvalues.values
                else:
                    raise AttributeError("Not pandas Series")
            except AttributeError:
                # Handle numpy array case
                param_names = ['Intercept'] + independent_vars
                param_values = np.array(model.params) if hasattr(model.params, '__iter__') else [model.params]
                pvalue_values = np.array(model.pvalues) if hasattr(model.pvalues, '__iter__') else [model.pvalues]
            
            for i, param_name in enumerate(param_names):
                coef = param_values[i] if i < len(param_values) else 0
                pval = pvalue_values[i] if i < len(pvalue_values) else 1
                significance = "significant" if pval < 0.05 else "not significant"
                report_text += f"- {param_name}: {coef:.4f} (p-value: {pval:.4f}) - {significance}\n"
            
            report_text += f"""

DIAGNOSTIC TESTS SUMMARY:
- Heteroskedasticity test needed
- Normality test needed
- Multicollinearity check needed

Created with Econometrics Learning Lab by HAMDI Boulanouar
            """
            
            # Create download button
            st.download_button(
                label="📥 Download Report",
                data=report_text.encode(),
                file_name="regression_analysis_report.txt",
                mime="text/plain"
            )
    
    elif export_format == "LaTeX Table":
        st.markdown("""
        ### 📝 LaTeX Table
        Generate LaTeX code for academic papers.
        """)
        
        try:
            latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Regression Results}}
\\begin{{tabular}}{{lcccc}}
\\hline\\hline
Variable & Coefficient & Std. Error & t-statistic & p-value \\\\
\\hline
"""
            
            # FIX: Handle parameter access properly
            try:
                # Try pandas Series approach
                if hasattr(model.params, 'index'):
                    param_items = list(zip(model.params.index, model.params.values, 
                                         model.bse.values, model.tvalues.values, model.pvalues.values))
                else:
                    raise AttributeError("Not pandas Series")
            except AttributeError:
                # Handle numpy array case
                param_names = ['Intercept'] + independent_vars
                param_values = np.array(model.params).flatten()
                std_errors = np.array(model.bse).flatten()
                t_values = np.array(model.tvalues).flatten()
                p_values = np.array(model.pvalues).flatten()
                
                param_items = list(zip(param_names, param_values, std_errors, t_values, p_values))
            
            for name, coef, se, t_val, p_val in param_items:
                latex_code += f"{name} & {coef:.4f} & {se:.4f} & {t_val:.3f} & {p_val:.4f} \\\\\n"
            
            latex_code += f"""\\hline
\\multicolumn{{5}}{{l}}{{R² = {model.rsquared:.4f}, Adj. R² = {model.rsquared_adj:.4f}, N = {int(model.nobs)}}} \\\\
\\hline\\hline
\\end{{tabular}}
\\label{{tab:regression}}
\\end{{table}}
"""
            
            st.code(latex_code, language='latex')
            
            st.download_button(
                label="📥 Download LaTeX Code",
                data=latex_code,
                file_name="regression_table.tex",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error generating LaTeX table: {str(e)}")

def educational_materials_page(language):
    """Educational materials and explanations"""
    
    st.markdown('<h2 class="section-header">📚 Educational Materials</h2>', 
                unsafe_allow_html=True)
    
    # Create tabs for different topics
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Basic Concepts",
        "🧮 Mathematical Foundations", 
        "🔍 Diagnostic Tests",
        "📈 Interpretation Guide",
        "🎯 Common Mistakes"
    ])
    
    with tab1:
        st.markdown("""
        ## 📊 Basic Concepts in Linear Regression
        
        ### What is Linear Regression?
        Linear regression is a statistical method that models the relationship between:
        - **Dependent Variable (Y)**: What you want to predict
        - **Independent Variable(s) (X)**: What you use to make predictions
        
        ### Why Use Regression?
        1. **Prediction**: Forecast future values
        2. **Explanation**: Understand relationships between variables
        3. **Control**: Isolate effects of individual variables
        
        ### Simple vs. Multiple Regression
        
        **Simple Linear Regression:**
        - One dependent variable (Y)
        - One independent variable (X)
        - Equation: Y = β₀ + β₁X + ε
        
        **Multiple Linear Regression:**
        - One dependent variable (Y)  
        - Multiple independent variables (X₁, X₂, ..., Xₖ)
        - Equation: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₖXₖ + ε
        
        ### Key Terms Explained
        
        **β₀ (Beta-zero, Intercept):**
        - The value of Y when all X variables equal zero
        - The "baseline" or "starting point"
        
        **β₁, β₂, ... (Slope Coefficients):**
        - How much Y changes when that X variable increases by 1 unit
        - In multiple regression: "holding all other variables constant"
        
        **ε (Epsilon, Error Term):**
        - Everything that affects Y but isn't in our model
        - Random variations, measurement errors, omitted variables
        """)
        
        # Interactive example
        st.markdown("### 🎮 Interactive Example")
        
        # Create sample data for demonstration
        np.random.seed(42)
        x_demo = np.linspace(0, 10, 50)
        
        col1, col2 = st.columns(2)
        
        with col1:
            intercept = st.slider("Intercept (β₀)", -10, 10, 2)
            slope = st.slider("Slope (β₁)", -2, 2, 1)
            noise = st.slider("Error Term (noise)", 0, 5, 1)
        
        with col2:
            # Generate data with user parameters
            y_demo = intercept + slope * x_demo + np.random.normal(0, noise, 50)
            
            fig = px.scatter(x=x_demo, y=y_demo, title="Interactive Regression Demo")
            
            # Add true line
            y_true = intercept + slope * x_demo
            fig.add_trace(go.Scatter(
                x=x_demo, y=y_true, mode='lines', 
                name=f'True Line: Y = {intercept} + {slope}X',
                line=dict(color='red', width=3)
            ))
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("""
        ## 🧮 Mathematical Foundations
        
        ### The Ordinary Least Squares (OLS) Method
        
        OLS finds the "best fitting" line by minimizing the sum of squared errors.
        """)
        
        st.latex(r"\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (Y_i - \beta_0 - \beta_1 X_i)^2")
        
        st.markdown("""
        ### Why Minimize Squared Errors?
        1. **Prevents cancellation**: Positive and negative errors don't cancel out
        2. **Penalizes large errors**: Squared term gives more weight to big mistakes
        3. **Mathematical convenience**: Has a unique solution
        
        ### OLS Formulas (Simple Regression)
        """)
        
        st.latex(r"\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^{n}(X_i - \bar{X})^2}")
        
        st.latex(r"\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1\bar{X}")
        
        st.markdown("""
                ### Matrix Form (Multiple Regression)
        """)
        
        st.latex(r"\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}")
        
        st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}")
        
        st.markdown("""
        ### Key Properties of OLS Estimators
        
        Under certain assumptions, OLS estimators are **BLUE**:
        - **B**est: Minimum variance among all linear unbiased estimators
        - **L**inear: Linear combinations of the dependent variable
        - **U**nbiased: E[β̂] = β (correct on average)
        - **E**stimators: Sample-based estimates of population parameters
        
        ### R-squared: Measuring Model Fit
        """)
        
        st.latex(r"R^2 = 1 - \frac{SSR}{TSS} = 1 - \frac{\sum(Y_i - \hat{Y}_i)^2}{\sum(Y_i - \bar{Y})^2}")
        
        st.markdown("""
        **Interpretation:**
        - R² = 0: Model explains 0% of variation (no better than just using the mean)
        - R² = 1: Model explains 100% of variation (perfect fit)
        - R² = 0.7: Model explains 70% of variation in Y
        """)
    
    with tab3:
        st.markdown("""
        ## 🔍 Diagnostic Tests Explained
        
        ### The Five Key Assumptions
        
        #### 1. Linearity
        **What it means:** The relationship between X and Y is actually linear
        
        **How to check:** Plot Y vs X, look for straight-line relationship
        
        **What if violated:** 
        - Coefficients will be biased
        - Predictions will be systematically wrong
        
        **Solutions:**
        - Transform variables (log, square root, etc.)
        - Add polynomial terms (X²)
        - Use non-linear regression methods
        
        #### 2. Homoskedasticity (Constant Variance)
        **What it means:** The spread of errors is the same for all values of X
        
        **Test:** Breusch-Pagan test
        - H₀: Homoskedasticity (good)
        - H₁: Heteroskedasticity (bad)
        
        **Visual check:** Plot residuals vs fitted values
        - Good: Random scatter
        - Bad: Funnel or cone shape
        
        **Solutions:**
        - Use robust standard errors
        - Transform variables
        - Weighted least squares
        
        #### 3. No Autocorrelation (Independence)
        **What it means:** Errors for different observations are unrelated
        
        **Test:** Durbin-Watson test
        """)
        
        st.latex(r"DW = \frac{\sum_{t=2}^{n}(e_t - e_{t-1})^2}{\sum_{t=1}^{n}e_t^2}")
        
        st.markdown("""
        **Interpretation:**
        - DW ≈ 2: No autocorrelation ✅
        - DW < 1.5: Positive autocorrelation ❌
        - DW > 2.5: Negative autocorrelation ❌
        
        **Solutions:**
        - Add lagged variables
        - Use robust standard errors
        - Check for omitted variables
        
        #### 4. Normality of Errors
        **What it means:** Errors follow a normal (bell-curve) distribution
        
        **Test:** Jarque-Bera test
        """)
        
        st.latex(r"JB = \frac{n}{6}\left[S^2 + \frac{(K-3)^2}{4}\right]")
        
        st.markdown("""
        Where:
        - S = Skewness (asymmetry)
        - K = Kurtosis (tail thickness)
        - For normal distribution: S=0, K=3
        
        **Visual checks:**
        - Histogram of residuals should be bell-shaped
        - Q-Q plot points should lie on straight line
        
        #### 5. No Perfect Multicollinearity
        **What it means:** Independent variables aren't perfectly correlated
        
        **Test:** Variance Inflation Factor (VIF)
        """)
        
        st.latex(r"VIF_j = \frac{1}{1-R_j^2}")
        
        st.markdown("""
        **Interpretation:**
        - VIF < 5: Low multicollinearity ✅
        - VIF 5-10: Moderate multicollinearity ⚠️
        - VIF > 10: High multicollinearity ❌
        
        **Solutions:**
        - Remove highly correlated variables
        - Combine variables into indices
        - Use ridge regression
        - Collect more data
        """)
    
    with tab4:
        st.markdown("""
        ## 📈 Interpretation Guide
        
        ### Reading Regression Output
        
        #### Coefficient Interpretation
        
        **Simple Regression: Y = β₀ + β₁X**
        - β₁ = 2.5: "A 1-unit increase in X is associated with a 2.5-unit increase in Y"
        
        **Multiple Regression: Y = β₀ + β₁X₁ + β₂X₂**
        - β₁ = 2.5: "A 1-unit increase in X₁ is associated with a 2.5-unit increase in Y, **holding X₂ constant**"
        
        #### Statistical Significance
        
        **t-statistic:** Measures how many standard errors the coefficient is away from zero
        - Large |t| = Strong evidence that coefficient ≠ 0
        - Small |t| = Weak evidence
        
        **p-value:** Probability of getting this result if the true coefficient were zero
        - p < 0.05: Statistically significant at 5% level
        - p < 0.01: Statistically significant at 1% level
        
        **Confidence Intervals:** Range of plausible values for the true coefficient
        - 95% CI [1.2, 3.8]: We're 95% confident the true coefficient is between 1.2 and 3.8
        
        #### Economic vs. Statistical Significance
        
        **Statistical Significance:** Is the effect different from zero?
        **Economic Significance:** Is the effect large enough to matter?
        
        Example: Education coefficient = 100, p-value = 0.001
        - Statistically significant: Yes (p < 0.05)
        - Economically significant: One extra year of education increases wage by $100 - is this meaningful?
        
        #### F-test for Overall Significance
        Tests: H₀: All slope coefficients = 0 (model is useless)
        
        - High F-statistic, low p-value: Model is useful
        - Low F-statistic, high p-value: Model adds no value
        
        ### Common Interpretation Mistakes
        
        ❌ **"X causes Y"** → ✅ **"X is associated with Y"**
        - Regression shows correlation, not necessarily causation
        
        ❌ **"R² = 0.3 means the model is bad"** → ✅ **"Depends on context"**
        - In finance, R² = 0.1 might be impressive
        - In physics, R² = 0.95 might be expected
        
        ❌ **"Insignificant coefficient means no relationship"** → ✅ **"No evidence of relationship"**
        - Lack of evidence ≠ evidence of lack
        """)
    
    with tab5:
        st.markdown("""
        ## 🎯 Common Mistakes to Avoid
        
        ### 1. Data Issues
        
        ❌ **Using the wrong variable types**
        - Don't use categorical variables as numeric without proper coding
        - Example: Coding Male=1, Female=2 implies ordering
        
        ✅ **Proper solution:** Use dummy variables (Male=0, Female=1)
        
        ❌ **Ignoring missing data**
        - Simply dropping all rows with any missing data can cause bias
        
        ✅ **Better approaches:** 
        - Analyze missing data patterns
        - Consider imputation methods
        - Use robust methods
        
        ### 2. Model Specification Errors
        
        ❌ **Assuming linearity without checking**
        - Just because you fit a linear model doesn't mean the relationship is linear
        
        ✅ **Always plot your data first**
        - Scatter plots for continuous variables
        - Box plots for categorical variables
        
        ❌ **Omitting important variables**
        - Leads to biased coefficients
        - Example: Studying education's effect on wages without controlling for experience
        
        ✅ **Think theoretically about what should be included**
        
        ### 3. Statistical Testing Mistakes
        
        ❌ **Multiple testing without adjustment**
        - Testing 20 variables at 5% significance level → expect 1 false positive
        
        ✅ **Use appropriate corrections** (Bonferroni, etc.)
        
        ❌ **Confusing statistical and practical significance**
        - With large samples, tiny effects can be statistically significant
        - Always ask: "Is this effect meaningful in practice?"
        
        ### 4. Interpretation Errors
        
        ❌ **Causal language for observational data**
        - "Education causes higher wages" vs. "Education is associated with higher wages"
        
        ✅ **Be careful about causal claims**
        - Consider reverse causation
        - Think about omitted variables
        - Use instrumental variables if available
        
        ❌ **Extrapolation beyond data range**
        - Don't predict for X values far outside your sample range
        
        ✅ **Stay within reasonable bounds of your data**
        
        ### 5. Diagnostic Mistakes
        
        ❌ **Ignoring assumption violations**
        - "My R² is high, so the model must be good"
        
        ✅ **Always check assumptions**
        - Visual diagnostics
        - Formal tests
        - Consider remedial measures
        
        ❌ **Over-relying on automated model selection**
        - Stepwise regression can be misleading
        - P-hacking through trying many specifications
        
        ✅ **Use theory to guide model building**
        - Start with theoretical model
        - Make changes based on diagnostics
        - Report sensitivity analyses
        
        ### 6. Reporting Mistakes
        
        ❌ **Only reporting coefficients and p-values**
        - Readers need context and interpretation
        
        ✅ **Provide complete information:**
        - Economic interpretation of coefficients
        - Confidence intervals
        - Model fit statistics
        - Diagnostic test results
        - Sample description
        
        ❌ **Not discussing limitations**
        - Every model has limitations
        
        ✅ **Be transparent about:**
        - Data limitations
        - Assumption violations
        - Alternative explanations
        - Generalizability concerns
        """)

# Enhanced sidebar with state persistence and history
def enhanced_sidebar():
    """Enhanced sidebar with state persistence and history"""
    
    st.sidebar.markdown("---")
    
    # Current session info
    with st.sidebar.expander("📊 Current Session"):
        if st.session_state.data is not None:
            st.write(f"**Data**: {st.session_state.uploaded_file_name}")
            st.write(f"**Rows**: {st.session_state.data.shape[0]}")
            st.write(f"**Columns**: {st.session_state.data.shape[1]}")
            
            if st.session_state.dependent_var:
                st.write(f"**Dependent Var**: {st.session_state.dependent_var}")
            
            if st.session_state.independent_vars:
                st.write(f"**Independent Vars**: {len(st.session_state.independent_vars)}")
        else:
            st.write("No data loaded")
    
    # Analysis history
    if st.session_state.analysis_history:
        with st.sidebar.expander(f"📈 Analysis History ({len(st.session_state.analysis_history)})"):
            for i, analysis in enumerate(reversed(st.session_state.analysis_history[-5:])):
                st.write(f"**{i+1}.** {analysis['dependent_var']} ~ {'+'.join(analysis['independent_vars'][:2])}{'...' if len(analysis['independent_vars']) > 2 else ''}")
                st.write(f"R² = {analysis['r_squared']:.3f}")
                st.write("---")
    
    # Quick help section
    with st.sidebar.expander("❓ Quick Help"):
        st.markdown("""
        **Getting Started:**
        1. Upload your data (CSV/Excel)
        2. Select variables for analysis
        3. Run regression analysis
        4. Check diagnostics
        5. Make predictions
        
        **Need Help?**
        - Check Educational Materials tab
        - All formulas are explained
        - Red text shows key concepts
        """)
    
    # Session management
    with st.sidebar.expander("💾 Session Management"):
        if st.button("🔄 Reset Session", key="reset_session"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("Session reset!")
            st.experimental_rerun()
    
    # App information
    with st.sidebar.expander("ℹ️ About This App"):
        st.markdown("""
        **Econometrics Learning Lab**
        
        A comprehensive educational tool for learning linear regression and econometrics.
        
        **Features:**
        - Complete OLS analysis
        - Diagnostic testing
        - Model comparison
        - Time series analysis
        - Export capabilities
        - Bilingual support
        
        **Created by:** HAMDI Boulanouar
        """)

# Updated main function with proper initialization
def main():
    # Initialize session state first
    initialize_session_state()
    
    # Get language preference (persistent)
    language = get_language()
    st.session_state.analysis_settings['language'] = language
    
    t = translations[language]
    
    # Main title
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2rem; color: #666;">{t["subtitle"]}</p>', 
                unsafe_allow_html=True)
    
    # Enhanced sidebar
    enhanced_sidebar()
    
    # Navigation with state persistence
    st.sidebar.title("📋 Navigation")
    pages = [
        "Data Upload & Preview",
        "Regression Analysis", 
        "Model Diagnostics",
        "Advanced Diagnostics",
        "Model Comparison",
        "Time Series Analysis",
        "Predictions",
        "Export Results",
        "Educational Materials"
    ]
    
    # Get current page from session state or default
    current_page_index = 0
    if st.session_state.current_page in pages:
        current_page_index = pages.index(st.session_state.current_page)
    
    selected_page = st.sidebar.selectbox(
        "Choose Analysis Step:", 
        pages, 
        index=current_page_index,
        key="page_selector"
    )
    
    # Update current page in session state
    st.session_state.current_page = selected_page
    
    # Route to appropriate page
    if selected_page == "Data Upload & Preview":
        data_upload_page(language)
    elif selected_page == "Regression Analysis":
        regression_analysis_page(language)
    elif selected_page == "Model Diagnostics":
        diagnostics_page(language)
    elif selected_page == "Advanced Diagnostics":
        advanced_diagnostics_page()
    elif selected_page == "Model Comparison":
        model_comparison_page()
    elif selected_page == "Time Series Analysis":
        time_series_features()
    elif selected_page == "Predictions":
        predictions_page(language)
    elif selected_page == "Export Results":
        export_results_feature()
    elif selected_page == "Educational Materials":
        educational_materials_page(language)

# Initialize the app
if __name__ == "__main__":
    try:
        # Run main application
        main()
        
    except Exception as e:
        st.error("❌ Application Error")
        st.exception(e)
        
        # Offer recovery options
        if st.button("🔄 Try to Recover Session"):
            initialize_session_state()
            st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>📊 <strong>Econometrics Learning Laboratory</strong></p>
            <p>Created with ❤️ by <strong>HAMDI Boulanouar</strong></p>
            <p><em>Making econometrics accessible to everyone</em></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
# Enhanced data display functions
def display_full_data_table(data, max_rows=None):
    """Display complete data table with pagination options"""
    
    st.subheader("📋 Complete Dataset")
    
    # Display options
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_all = st.checkbox("📊 Show All Rows", value=False, key="show_all_data")
    
    with col2:
        if not show_all:
            rows_to_show = st.selectbox(
                "Rows to Display", 
                [10, 25, 50, 100, 200], 
                index=0,
                key="rows_display"
            )
        else:
            rows_to_show = len(data)
    
    with col3:
        start_row = st.number_input(
            "Start from Row", 
            min_value=1, 
            max_value=len(data), 
            value=1,
            key="start_row"
        ) - 1
    
    with col4:
        st.metric("Total Rows", len(data))
    
    # Display the data
    if show_all:
        st.info(f"📊 Showing all {len(data)} rows")
        st.dataframe(data, use_container_width=True, height=600)
    else:
        end_row = min(start_row + rows_to_show, len(data))
        st.info(f"📊 Showing rows {start_row + 1} to {end_row} of {len(data)}")
        st.dataframe(data.iloc[start_row:end_row], use_container_width=True, height=400)
    
    return show_all, rows_to_show

def save_analysis_results():
    """Save complete analysis results to session state with timestamp"""
    
    if st.session_state.regression_results:
        # Create comprehensive results package
        results_package = {
            'model_summary': str(st.session_state.regression_results['model'].summary()),
            'coefficients': get_coefficients_data(),
            'model_stats': get_model_statistics(),
            'diagnostic_results': get_diagnostic_results() if st.session_state.get('last_diagnostics') else None,
            'data_info': {
                'shape': st.session_state.data.shape,
                'columns': list(st.session_state.data.columns),
                'dependent_var': st.session_state.dependent_var,
                'independent_vars': st.session_state.independent_vars
            },
            'timestamp': datetime.now().isoformat(),
            'session_id': generate_session_id()
        }
        
        # Save to session state with history
        if 'saved_analyses' not in st.session_state:
            st.session_state.saved_analyses = []
        
        st.session_state.saved_analyses.append(results_package)
        
        # Keep only last 10 analyses to manage memory
        if len(st.session_state.saved_analyses) > 10:
            st.session_state.saved_analyses = st.session_state.saved_analyses[-10:]
        
        return True
    return False

def generate_session_id():
    """Generate unique session ID"""
    import hashlib
    timestamp = str(datetime.now().timestamp())
    return hashlib.md5(timestamp.encode()).hexdigest()[:8]

def get_coefficients_data():
    """Extract coefficients data in a structured format"""
    model = st.session_state.regression_results['model']
    independent_vars = st.session_state.regression_results['independent_vars']
    
    try:
        if hasattr(model.params, 'index'):
            return {
                'variables': list(model.params.index),
                'coefficients': list(model.params.values),
                'std_errors': list(model.bse.values),
                't_statistics': list(model.tvalues.values),
                'p_values': list(model.pvalues.values),
                'conf_int_lower': list(model.conf_int()[0].values),
                'conf_int_upper': list(model.conf_int()[1].values)
            }
    except:
        return {
            'variables': ['Intercept'] + independent_vars,
            'coefficients': list(np.array(model.params).flatten()),
            'std_errors': list(np.array(model.bse).flatten()),
            't_statistics': list(np.array(model.tvalues).flatten()),
            'p_values': list(np.array(model.pvalues).flatten()),
            'conf_int_lower': list(np.array(model.conf_int()[0]).flatten()),
            'conf_int_upper': list(np.array(model.conf_int()[1]).flatten())
        }

def get_model_statistics():
    """Extract model statistics"""
    model = st.session_state.regression_results['model']
    
    return {
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'f_statistic': model.fvalue,
        'f_pvalue': model.f_pvalue,
        'aic': model.aic,
        'bic': model.bic,
        'observations': int(model.nobs),
        'df_residuals': int(model.df_resid)
    }

def get_diagnostic_results():
    """Get diagnostic test results if available"""
    try:
        model = st.session_state.regression_results['model']
        X_with_const = st.session_state.regression_results['X_with_const']
        
        # Breusch-Pagan test
        bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, X_with_const)
        
        # Durbin-Watson test
        dw_stat = durbin_watson(model.resid)
        
        # Jarque-Bera test
        jb_result = jarque_bera(model.resid)
        jb_stat, jb_pvalue = jb_result[0], jb_result[1]
        
        return {
            'breusch_pagan': {'statistic': bp_stat, 'pvalue': bp_pvalue},
            'durbin_watson': {'statistic': dw_stat},
            'jarque_bera': {'statistic': jb_stat, 'pvalue': jb_pvalue}
        }
    except:
        return None

# Enhanced data upload page with reset functionality
def enhanced_data_upload_page(language):
    """Enhanced data upload page with reset and full display options"""
    
    t = translations[language]
    
    st.markdown(f'<h2 class="section-header">{t["upload_data"]}</h2>', unsafe_allow_html=True)
    
    # Current data status and reset options
    if st.session_state.data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"✅ **Data Loaded**: {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")
            if st.session_state.uploaded_file_name:
                st.info(f"📁 **File**: {st.session_state.uploaded_file_name}")
        
        with col2:
            # Reset data button
            if st.button("🔄 **Reset Current Data**", type="secondary", key="reset_data_btn"):
                reset_current_data()
                st.success("✅ Data reset successfully!")
                st.experimental_rerun()
        
        with col3:
            # Save current analysis button
            if st.session_state.regression_results:
                if st.button("💾 **Save Current Analysis**", type="primary", key="save_analysis_btn"):
                    if save_analysis_results():
                        st.success("✅ Analysis saved successfully!")
                    else:
                        st.error("❌ Failed to save analysis")
    
    # File upload section
    st.markdown("### 📁 Upload New Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for analysis",
        key="file_uploader_enhanced"
    )
    
    # Upload options
    col1, col2 = st.columns(2)
    with col1:
        replace_current = st.checkbox(
            "🔄 Replace current data", 
            value=True, 
            help="Check to replace existing data, uncheck to keep both"
        )
    
    with col2:
        auto_preview = st.checkbox(
            "👀 Auto-preview data", 
            value=True, 
            help="Automatically show data preview after upload"
        )
    
    # Handle file upload
    if uploaded_file is not None:
        if replace_current or st.session_state.data is None:
            try:
                # Load new data
                file_content = uploaded_file.getvalue()
                data, data_hash = cache_uploaded_data(file_content, uploaded_file.name)
                
                if data is not None:
                    data = optimize_memory_usage(data)
                    
                    # Update session state
                    st.session_state.data = data
                    st.session_state.uploaded_file_name = uploaded_file.name
                    st.session_state.data_hash = data_hash
                    
                    # Reset analysis-specific state only if replacing
                    if replace_current:
                        st.session_state.dependent_var = None
                        st.session_state.independent_vars = []
                        st.session_state.regression_results = None
                    
                    st.success("✅ Data uploaded successfully!")
                    
                    if auto_preview:
                        st.experimental_rerun()
                
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
    
    # Display data if available
    if st.session_state.data is not None:
        enhanced_data_preview(st.session_state.data, language)
    
    # Sample data section
    st.markdown("---")
    st.subheader("📊 Or Use Sample Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💰 Wage Dataset", key="wage_data_enhanced"):
            load_sample_data_enhanced("wage", replace_current)
    
    with col2:
        if st.button("🏠 Housing Dataset", key="housing_data_enhanced"):
            load_sample_data_enhanced("housing", replace_current)
    
    with col3:
        if st.button("📈 Stock Dataset", key="stock_data_enhanced"):
            load_sample_data_enhanced("stock", replace_current)
    
    with col4:
        if st.button("🏭 Production Dataset", key="production_data_enhanced"):
            load_sample_data_enhanced("production", replace_current)

def reset_current_data():
    """Reset current data and related analysis"""
    # Clear data-related session state
    st.session_state.data = None
    st.session_state.uploaded_file_name = None
    st.session_state.data_hash = None
    
    # Clear analysis results
    st.session_state.dependent_var = None
    st.session_state.independent_vars = []
    st.session_state.regression_results = None
    
    # Mark last diagnostics as cleared
    if 'last_diagnostics' in st.session_state:
        del st.session_state.last_diagnostics

def load_sample_data_enhanced(dataset_type, replace_current=True):
    """Enhanced sample data loading with replace option"""
    
    if not replace_current and st.session_state.data is not None:
        st.warning("⚠️ Cannot load sample data: Current data would be replaced. Check 'Replace current data' option.")
        return
    
    np.random.seed(42)
    
    if dataset_type == "wage":
        n = 500  # Larger sample
        education = np.random.normal(12, 3, n)
        experience = np.random.exponential(8, n)
        gender = np.random.choice([0, 1], n, p=[0.6, 0.4])
        region = np.random.choice([1, 2, 3, 4], n)  # Different regions
        
        wage = (
            25000 +
            2500 * education +
            800 * experience +
            -100 * experience**2 / 10 +
            -3000 * gender +
            1000 * region +
            np.random.normal(0, 5000, n)
        )
        
        data = pd.DataFrame({
            'wage': wage,
            'education': education,
            'experience': experience,
            'gender': gender,
            'region': region
        })
        dataset_name = "Sample Wage Data (Enhanced)"
    
    elif dataset_type == "housing":
        n = 400
        square_feet = np.random.normal(2000, 500, n)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        bathrooms = np.random.normal(2, 0.5, n)
        age = np.random.exponential(15, n)
        location_premium = np.random.choice([0, 15000, 30000, 50000], n, p=[0.4, 0.3, 0.2, 0.1])
        garage = np.random.choice([0, 1], n, p=[0.3, 0.7])
        
        price = (
            50000 + 
            100 * square_feet + 
            5000 * bedrooms +
            3000 * bathrooms -
            500 * age + 
            location_premium +
            8000 * garage +
            np.random.normal(0, 15000, n)
        )
        
        data = pd.DataFrame({
            'price': price,
            'square_feet': square_feet,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'age': age,
            'location_premium': location_premium,
            'garage': garage
        })
        dataset_name = "Sample Housing Data (Enhanced)"
    
    elif dataset_type == "stock":
        n = 300
        market_return = np.random.normal(0.05, 0.15, n)
        company_beta = np.random.normal(1.2, 0.3, n)
        company_return = 0.02 + company_beta * market_return + np.random.normal(0, 0.1, n)
        volatility = np.abs(np.random.normal(0.2, 0.05, n))
        market_cap = np.random.lognormal(15, 2, n)
        
        data = pd.DataFrame({
            'company_return': company_return,
            'market_return': market_return,
            'volatility': volatility,
            'trading_volume': np.random.lognormal(10, 1, n),
            'market_cap': market_cap,
            'beta': company_beta
        })
        dataset_name = "Sample Stock Data (Enhanced)"
    
    else:  # production data
        n = 250
        labor = np.random.exponential(50, n)
        capital = np.random.exponential(100, n)
        technology = np.random.normal(1, 0.1, n)
        energy = np.random.exponential(20, n)
        
        output = technology * (labor ** 0.6) * (capital ** 0.3) * (energy ** 0.1) + np.random.normal(0, 5, n)
        
        data = pd.DataFrame({
            'output': output,
            'labor': labor,
            'capital': capital,
            'technology': technology,
            'energy': energy
        })
        dataset_name = "Sample Production Data"
    
    # Update session state
    st.session_state.data = data
    st.session_state.uploaded_file_name = dataset_name
    st.session_state.data_hash = hash(str(data.values.tobytes()))
    
    # Reset analysis-specific state
    if replace_current:
        st.session_state.dependent_var = None
        st.session_state.independent_vars = []
        st.session_state.regression_results = None
    
    st.success(f"✅ {dataset_name} loaded successfully!")
    st.experimental_rerun()

def enhanced_data_preview(data, language):
    """Enhanced data preview with full display options"""
    
    st.markdown("### 📊 Data Overview")
    
    # Enhanced metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📊 Rows", data.shape[0])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("📈 Columns", data.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        numeric_cols = len(data.select_dtypes(include=[np.number]).columns)
        st.metric("🔢 Numeric", numeric_cols)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        missing_vals = data.isnull().sum().sum()
        st.metric("❓ Missing", missing_vals)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        memory_usage = data.memory_usage(deep=True).sum() / (1024**2)
        st.metric("💾 Memory (MB)", f"{memory_usage:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Full data display with options
    show_all, rows_shown = display_full_data_table(data)
    
    # Data info section
    if st.expander("📋 Column Information", expanded=False):
        column_info = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null': data.count(),
            'Null Count': data.isnull().sum(),
            'Unique Values': data.nunique()
        })
        st.dataframe(column_info, use_container_width=True)
    
    # Enhanced statistics
    if st.checkbox("📊 Show Descriptive Statistics", key="show_stats_enhanced"):
        st.subheader("📊 Descriptive Statistics")
        
        # Options for statistics
        col1, col2 = st.columns(2)
        with col1:
            include_all = st.checkbox("Include all columns", value=False)
        with col2:
            percentiles = st.multiselect(
                "Additional percentiles",
                [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99],
                default=[0.25, 0.75]
            )
        
        if include_all:
            stats_df = data.describe(include='all', percentiles=percentiles)
        else:
            stats_df = data.describe(percentiles=percentiles)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Download statistics
        csv_stats = stats_df.to_csv()
        st.download_button(
            label="📥 Download Statistics CSV",
            data=csv_stats,
            file_name="descriptive_statistics.csv",
            mime="text/csv"
        )

# Enhanced regression results with full tables
def enhanced_display_regression_results(model, dependent_var, independent_vars, language):
    """Enhanced regression results display with full tables and save options"""
    
    # All previous regression results code here...
    # [Include the complete display_regression_results function from before]
    
    # Add save functionality
    st.markdown("### 💾 Save Analysis Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save to Session", key="save_to_session"):
            if save_analysis_results():
                st.success("✅ Analysis saved to session!")
            else:
                st.error("❌ Failed to save analysis")
    
    with col2:
        # Export coefficients table
        coeffs_data = get_coefficients_data()
        coeffs_df = pd.DataFrame(coeffs_data)
        csv_coeffs = coeffs_df.to_csv(index=False)
        
        st.download_button(
            label="📊 Download Coefficients",
            data=csv_coeffs,
            file_name="regression_coefficients.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export model statistics
        model_stats = get_model_statistics()
        stats_df = pd.DataFrame([model_stats])
        csv_stats = stats_df.to_csv(index=False)
        
        st.download_button(
            label="📈 Download Model Stats",
            data=csv_stats,
            file_name="model_statistics.csv",
            mime="text/csv"
        )

# Enhanced sidebar with saved analyses
def enhanced_sidebar_with_history():
    """Enhanced sidebar with complete analysis history"""
    
    st.sidebar.markdown("---")
    
    # Current session info
    with st.sidebar.expander("📊 Current Session"):
        if st.session_state.data is not None:
            st.write(f"**Data**: {st.session_state.uploaded_file_name}")
            st.write(f"**Shape**: {st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
            
            if st.session_state.dependent_var:
                st.write(f"**Dependent**: {st.session_state.dependent_var}")
            
            if st.session_state.independent_vars:
                st.write(f"**Independent**: {', '.join(st.session_state.independent_vars[:3])}{'...' if len(st.session_state.independent_vars) > 3 else ''}")
        else:
            st.write("❌ No data loaded")
    
    # Saved analyses history
    if st.session_state.get('saved_analyses'):
        with st.sidebar.expander(f"💾 Saved Analyses ({len(st.session_state.saved_analyses)})"):
            for i, analysis in enumerate(reversed(st.session_state.saved_analyses)):
                with st.container():
                    st.write(f"**Analysis {i+1}** (ID: {analysis['session_id']})")
                    st.write(f"📊 R² = {analysis['model_stats']['r_squared']:.3f}")
                    st.write(f"📅 {datetime.fromisoformat(analysis['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                    
                    if st.button(f"📥 Export Analysis {i+1}", key=f"export_analysis_{i}"):
                        # Export complete analysis
                        analysis_json = json.dumps(analysis, indent=2, default=str)
                        st.download_button(
                            label="Download Complete Analysis",
                            data=analysis_json,
                            file_name=f"analysis_{analysis['session_id']}.json",
                            mime="application/json",
                            key=f"download_analysis_{i}"
                        )
                    st.write("---")
    
    # Session management with enhanced options
    with st.sidebar.expander("🔧 Session Management"):
        if st.button("💾 Save Current Session", key="save_current_session"):
            if save_analysis_results():
                st.success("✅ Session saved!")
            else:
                st.warning("⚠️ No analysis to save")
        
        if st.button("🔄 Reset All Data", key="reset_all_data"):
            reset_current_data()
            st.success("✅ All data reset!")
            st.experimental_rerun()
        
        if st.button("🧹 Clear Analysis History", key="clear_history"):
            st.session_state.saved_analyses = []
            st.success("✅ History cleared!")
            st.experimental_rerun()
        
        # Memory usage info
        if st.session_state.data is not None:
            memory_mb = st.session_state.data.memory_usage(deep=True).sum() / (1024**2)
            st.info(f"💾 Memory: {memory_mb:.1f} MB")

# Update the main function to use enhanced features
def main():
    # Initialize session state first
    initialize_session_state()
    
    # Get language preference
    language = get_language()
    t = translations[language]
    
    # Main title
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; font-size: 1.2rem; color: #666;">{t["subtitle"]}</p>', 
                unsafe_allow_html=True)
    
    # Enhanced sidebar with history
    enhanced_sidebar_with_history()
    
    # Navigation
    st.sidebar.title("📋 Navigation")
    pages = [
        "Data Upload & Preview",
        "Regression Analysis", 
        "Model Diagnostics",
        "Advanced Diagnostics",
        "Model Comparison",
        "Time Series Analysis",
        "Predictions",
        "Export Results",
        "Educational Materials"
    ]
    
    selected_page = st.sidebar.selectbox("Choose Analysis Step:", pages)
    
    # Route to appropriate page (use enhanced versions)
    if selected_page == "Data Upload & Preview":
        enhanced_data_upload_page(language)  # Use enhanced version
    elif selected_page == "Regression Analysis":
        regression_analysis_page(language)
    elif selected_page == "Model Diagnostics":
        diagnostics_page(language)
    # ... other pages remain the same


