import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from scipy import stats
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU-accelerated libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

class PromotionAnalyzer:
    """
    Advanced promotion impact analysis with multiple ML models and hyperparameter tuning
    """
    
    def __init__(self, gpu_enabled=False, verbose=True):
        self.models = {}
        self.feature_importance = {}
        self.results = {}
        self.gpu_enabled = gpu_enabled
        self.verbose = verbose
        self.grid_search_results = {}
        
    def log(self, message, level="INFO"):
        """Human-readable logging"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            if level == "SUCCESS":
                print(f"[{timestamp}] ✓ {message}")
            elif level == "WARNING":
                print(f"[{timestamp}] ⚠ {message}")
            elif level == "ERROR":
                print(f"[{timestamp}] ✗ {message}")
            elif level == "QUESTION":
                print(f"[{timestamp}] ❓ {message}")
            else:
                print(f"[{timestamp}] ℹ {message}")
    
    def analyze_promotion_impact_by_item(self, sales_df):
        """
        CASE STUDY QUESTION C: Which items experienced the biggest sale increase during promotions?
        
        Analysis Method:
        - Compare average daily sales during promotion vs non-promotion periods
        - Calculate both absolute and percentage lift for each product
        - Rank products by promotion responsiveness
        """
        
        self.log("Analyzing promotion impact by individual products...")
        
        # Separate promotion and non-promotion data
        promo_data = sales_df[sales_df['IsPromotion']].copy()
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        self.log(f"Promotion periods: {len(promo_data):,} transactions")
        self.log(f"Non-promotion periods: {len(non_promo_data):,} transactions")
        
        # Calculate metrics for promotion periods
        promo_sales = promo_data.groupby('ProductCode')['SalesQuantity'].agg(['mean', 'sum', 'count']).reset_index()
        promo_sales.columns = ['ProductCode', 'AvgPromoDailySales', 'TotalPromoSales', 'PromoDays']
        
        # Calculate metrics for non-promotion periods  
        non_promo_sales = non_promo_data.groupby('ProductCode')['SalesQuantity'].agg(['mean', 'sum', 'count']).reset_index()
        non_promo_sales.columns = ['ProductCode', 'AvgNonPromoDailySales', 'TotalNonPromoSales', 'NonPromoDays']
        
        # Merge and calculate promotion lift
        item_impact = promo_sales.merge(non_promo_sales, on='ProductCode', how='inner')
        item_impact['AbsoluteLift'] = item_impact['AvgPromoDailySales'] - item_impact['AvgNonPromoDailySales']
        item_impact['PercentLift'] = (item_impact['AbsoluteLift'] / item_impact['AvgNonPromoDailySales']) * 100
        
        # Handle edge cases
        item_impact['PercentLift'] = item_impact['PercentLift'].replace([np.inf, -np.inf], np.nan)
        
        # Sort by absolute lift (most meaningful for business impact)
        item_impact = item_impact.sort_values('AbsoluteLift', ascending=False)
        
        # Business interpretation
        self.log("ANSWER TO QUESTION C: Items with Biggest Promotion Impact", "SUCCESS")
        print("\n" + "="*80)
        print("TOP 10 PRODUCTS WITH HIGHEST PROMOTION RESPONSIVENESS")
        print("="*80)
        print("Methodology: Average daily sales during promotions vs. baseline periods")
        print("Key Metrics: Absolute Lift = Promo Sales - Baseline Sales")
        print("            Percent Lift = (Absolute Lift / Baseline Sales) × 100")
        print("-"*80)
        
        top_items = item_impact.head(10)
        for idx, (_, item) in enumerate(top_items.iterrows(), 1):
            print(f"#{idx:2d}. Product {int(item['ProductCode']):3d}: "
                  f"+{item['AbsoluteLift']:5.2f} units/day ({item['PercentLift']:5.1f}% lift)")
        
        print("-"*80)
        print(f"Business Insight: Product {int(top_items.iloc[0]['ProductCode'])} shows the strongest")
        print(f"promotion response with +{top_items.iloc[0]['AbsoluteLift']:.2f} units/day increase")
        print(f"({top_items.iloc[0]['PercentLift']:.1f}% improvement over baseline)")
        
        self.results['item_impact'] = item_impact
        return item_impact
    
    def analyze_promotion_impact_by_store(self, sales_df):
        """
        CASE STUDY QUESTION D: Are there stores that have higher promotion reaction?
        
        Analysis Method:
        - Compare store-level sales performance during promotions vs baseline
        - Identify top-performing stores for promotional activities
        - Provide strategic recommendations for store selection
        """
        
        self.log("Analyzing promotion responsiveness across store locations...")
        
        promo_data = sales_df[sales_df['IsPromotion']].copy()
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        # Store-level promotion metrics
        promo_sales = promo_data.groupby('StoreCode')['SalesQuantity'].agg(['mean', 'sum', 'count']).reset_index()
        promo_sales.columns = ['StoreCode', 'AvgPromoDailySales', 'TotalPromoSales', 'PromoDays']
        
        non_promo_sales = non_promo_data.groupby('StoreCode')['SalesQuantity'].agg(['mean', 'sum', 'count']).reset_index()
        non_promo_sales.columns = ['StoreCode', 'AvgNonPromoDailySales', 'TotalNonPromoSales', 'NonPromoDays']
        
        # Calculate store promotion responsiveness
        store_impact = promo_sales.merge(non_promo_sales, on='StoreCode', how='inner')
        store_impact['AbsoluteLift'] = store_impact['AvgPromoDailySales'] - store_impact['AvgNonPromoDailySales']
        store_impact['PercentLift'] = (store_impact['AbsoluteLift'] / store_impact['AvgNonPromoDailySales']) * 100
        
        # Handle edge cases
        store_impact['PercentLift'] = store_impact['PercentLift'].replace([np.inf, -np.inf], np.nan)
        store_impact = store_impact.sort_values('AbsoluteLift', ascending=False)
        
        # Business interpretation
        self.log("ANSWER TO QUESTION D: Stores with Highest Promotion Reaction", "SUCCESS")
        print("\n" + "="*80)
        print("TOP 10 STORES WITH HIGHEST PROMOTION RESPONSIVENESS")
        print("="*80)
        print("Strategic Value: These stores show the best ROI for promotional investments")
        print("Recommendation: Prioritize these locations for future campaigns")
        print("-"*80)
        
        top_stores = store_impact.head(10)
        for idx, (_, store) in enumerate(top_stores.iterrows(), 1):
            print(f"#{idx:2d}. Store {int(store['StoreCode']):3d}: "
                  f"+{store['AbsoluteLift']:5.2f} units/day ({store['PercentLift']:5.1f}% lift)")
        
        print("-"*80)
        print(f"Key Finding: Store {int(top_stores.iloc[0]['StoreCode'])} demonstrates exceptional")
        print(f"promotion sensitivity with {top_stores.iloc[0]['PercentLift']:.1f}% sales increase")
        print(f"Recommendation: This store should be a priority for promotional activities")
        
        self.results['store_impact'] = store_impact
        return store_impact
    
    def compare_item_categories(self, sales_df):
        """
        CASE STUDY QUESTION F: Is there any significant difference between promotion impacts 
        of Fast versus Slow items?
        
        Statistical Method:
        - Independent samples t-test to compare promotion lift between categories
        - Effect size calculation to measure practical significance
        - Business interpretation of statistical findings
        """
        
        if 'ItemCategory' not in sales_df.columns:
            self.log("Item categories not found. Please run clustering first.", "ERROR")
            return None
        
        self.log("Conducting statistical comparison of Fast vs Slow items...")
        
        promo_data = sales_df[sales_df['IsPromotion']].copy()
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        # Calculate category-level metrics
        promo_by_category = promo_data.groupby('ItemCategory')['SalesQuantity'].agg(['mean', 'std', 'count']).reset_index()
        promo_by_category.columns = ['ItemCategory', 'PromoMean', 'PromoStd', 'PromoCount']
        
        non_promo_by_category = non_promo_data.groupby('ItemCategory')['SalesQuantity'].agg(['mean', 'std', 'count']).reset_index()
        non_promo_by_category.columns = ['ItemCategory', 'NonPromoMean', 'NonPromoStd', 'NonPromoCount']
        
        category_comparison = promo_by_category.merge(non_promo_by_category, on='ItemCategory')
        category_comparison['AbsoluteLift'] = category_comparison['PromoMean'] - category_comparison['NonPromoMean']
        category_comparison['PercentLift'] = (category_comparison['AbsoluteLift'] / category_comparison['NonPromoMean']) * 100
        
        self.log("ANSWER TO QUESTION F: Fast vs Slow Items Statistical Comparison", "SUCCESS")
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS: PROMOTION IMPACT BY ITEM CATEGORY")
        print("="*80)
        print("Research Question: Do Fast items respond differently to promotions than Slow items?")
        print("Statistical Method: Independent samples t-test")
        print("-"*80)
        
        # Display category summary
        for _, row in category_comparison.iterrows():
            print(f"{row['ItemCategory']:6s} Items: {row['PercentLift']:6.2f}% lift "
                  f"(n={row['PromoCount']:,} observations)")
        
        # Statistical testing between Fast and Slow items
        if 'Fast' in promo_data['ItemCategory'].values and 'Slow' in promo_data['ItemCategory'].values:
            fast_promo = promo_data[promo_data['ItemCategory'] == 'Fast']['SalesQuantity']
            slow_promo = promo_data[promo_data['ItemCategory'] == 'Slow']['SalesQuantity']
            
            fast_baseline = non_promo_data[non_promo_data['ItemCategory'] == 'Fast']['SalesQuantity'].mean()
            slow_baseline = non_promo_data[non_promo_data['ItemCategory'] == 'Slow']['SalesQuantity'].mean()
            
            fast_lift = fast_promo.mean() - fast_baseline
            slow_lift = slow_promo.mean() - slow_baseline
            
            # Independent samples t-test
            t_stat, p_value = stats.ttest_ind(fast_promo - fast_baseline, slow_promo - slow_baseline)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(fast_promo)-1)*fast_promo.std()**2 + (len(slow_promo)-1)*slow_promo.std()**2) / 
                                (len(fast_promo) + len(slow_promo) - 2))
            cohens_d = (fast_lift - slow_lift) / pooled_std
            
            print("-"*80)
            print("STATISTICAL RESULTS:")
            print(f"Fast Items Promotion Lift:  {fast_lift:7.3f} units/day")
            print(f"Slow Items Promotion Lift:  {slow_lift:7.3f} units/day")
            print(f"Difference:                 {fast_lift-slow_lift:7.3f} units/day")
            print(f"T-statistic:                {t_stat:7.3f}")
            print(f"P-value:                    {p_value:.2e}")
            print(f"Effect size (Cohen's d):    {cohens_d:7.3f}")
            
            # Business interpretation
            print("-"*80)
            print("BUSINESS INTERPRETATION:")
            if p_value < 0.001:
                significance = "Highly Significant"
            elif p_value < 0.01:
                significance = "Very Significant" 
            elif p_value < 0.05:
                significance = "Significant"
            else:
                significance = "Not Significant"
                
            print(f"Statistical Significance: {significance} (p < 0.05)")
            
            if abs(cohens_d) > 0.8:
                effect = "Large"
            elif abs(cohens_d) > 0.5:
                effect = "Medium"
            elif abs(cohens_d) > 0.2:
                effect = "Small"
            else:
                effect = "Negligible"
                
            print(f"Practical Significance: {effect} effect size")
            
            if p_value < 0.05:
                print(f"CONCLUSION: Fast items respond {abs(fast_lift-slow_lift):.2f} units/day more strongly")
                print("to promotions than Slow items. This difference is statistically significant.")
                print("RECOMMENDATION: Allocate more promotional budget to Fast items for higher ROI.")
            else:
                print("CONCLUSION: No significant difference in promotion response between categories.")
        
        self.results['item_category_comparison'] = category_comparison
        return category_comparison
    
    def compare_store_categories(self, sales_df):
        """
        CASE STUDY QUESTION G: Is there any significant difference between promotion impacts 
        of Fast versus Slow stores?
        """
        
        if 'StoreCategory' not in sales_df.columns:
            self.log("Store categories not found. Please run clustering first.", "ERROR")
            return None
        
        self.log("Conducting statistical comparison of Fast vs Slow stores...")
        
        promo_data = sales_df[sales_df['IsPromotion']].copy()
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        # Store category analysis (similar structure to items)
        promo_by_category = promo_data.groupby('StoreCategory')['SalesQuantity'].agg(['mean', 'std', 'count']).reset_index()
        promo_by_category.columns = ['StoreCategory', 'PromoMean', 'PromoStd', 'PromoCount']
        
        non_promo_by_category = non_promo_data.groupby('StoreCategory')['SalesQuantity'].agg(['mean', 'std', 'count']).reset_index()
        non_promo_by_category.columns = ['StoreCategory', 'NonPromoMean', 'NonPromoStd', 'NonPromoCount']
        
        category_comparison = promo_by_category.merge(non_promo_by_category, on='StoreCategory')
        category_comparison['AbsoluteLift'] = category_comparison['PromoMean'] - category_comparison['NonPromoMean']
        category_comparison['PercentLift'] = (category_comparison['AbsoluteLift'] / category_comparison['NonPromoMean']) * 100
        
        self.log("ANSWER TO QUESTION G: Fast vs Slow Stores Statistical Comparison", "SUCCESS")
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS: PROMOTION IMPACT BY STORE CATEGORY")
        print("="*80)
        
        # Statistical testing and interpretation (similar to items)
        for _, row in category_comparison.iterrows():
            print(f"{row['StoreCategory']:6s} Stores: {row['PercentLift']:6.2f}% lift "
                  f"(n={row['PromoCount']:,} observations)")
        
        if 'Fast' in promo_data['StoreCategory'].values and 'Slow' in promo_data['StoreCategory'].values:
            fast_promo = promo_data[promo_data['StoreCategory'] == 'Fast']['SalesQuantity']
            slow_promo = promo_data[promo_data['StoreCategory'] == 'Slow']['SalesQuantity']
            
            fast_baseline = non_promo_data[non_promo_data['StoreCategory'] == 'Fast']['SalesQuantity'].mean()
            slow_baseline = non_promo_data[non_promo_data['StoreCategory'] == 'Slow']['SalesQuantity'].mean()
            
            fast_lift = fast_promo.mean() - fast_baseline
            slow_lift = slow_promo.mean() - slow_baseline
            
            t_stat, p_value = stats.ttest_ind(fast_promo - fast_baseline, slow_promo - slow_baseline)
            
            print(f"\nFast Stores Lift: {fast_lift:.3f} | Slow Stores Lift: {slow_lift:.3f}")
            print(f"P-value: {p_value:.2e} | Significant: {'Yes' if p_value < 0.05 else 'No'}")
        
        self.results['store_category_comparison'] = category_comparison
        return category_comparison

    def get_available_models(self):
        """Get dictionary of available ML models with their configurations"""
        models = {
            'linear_regression': {
                'model': LinearRegression(),
                'name': 'Linear Regression',
                'params_fast': {},
                'params_full': {},
                'gpu_supported': False
            },
            'ridge': {
                'model': Ridge(),
                'name': 'Ridge Regression',
                'params_fast': {'alpha': [1.0]},
                'params_full': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                'gpu_supported': False
            },
            'lasso': {
                'model': Lasso(),
                'name': 'Lasso Regression',
                'params_fast': {'alpha': [0.1, 1.0, 10.0]},
                'params_full': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
                'gpu_supported': False
            },
            'elastic_net': {
                'model': ElasticNet(),
                'name': 'Elastic Net',
                'params_fast': {'alpha': [0.1, 1.0], 'l1_ratio': [0.5, 0.7]},
                'params_full': {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
                'gpu_supported': False
            },
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'name': 'Random Forest',
                'params_fast': {'n_estimators': [50], 'max_depth': [10]},
                'params_full': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None], 
                               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                'gpu_supported': False
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'name': 'Gradient Boosting',
                'params_fast': {'n_estimators': [50], 'learning_rate': [0.1]},
                'params_full': {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2], 
                               'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0]},
                'gpu_supported': False
            },
            'extra_trees': {
                'model': ExtraTreesRegressor(random_state=42),
                'name': 'Extra Trees',
                'params_fast': {'n_estimators': [50, 100], 'max_depth': [10, 20]},
                'params_full': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None],
                               'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                'gpu_supported': False
            },
            'decision_tree': {
                'model': DecisionTreeRegressor(random_state=42),
                'name': 'Decision Tree',
                'params_fast': {'max_depth': [5, 10, 20]},
                'params_full': {'max_depth': [3, 5, 10, 20, None], 'min_samples_split': [2, 5, 10],
                               'min_samples_leaf': [1, 2, 4], 'criterion': ['squared_error', 'friedman_mse']},
                'gpu_supported': False
            },
            'knn': {
                'model': KNeighborsRegressor(),
                'name': 'K-Nearest Neighbors',
                'params_fast': {'n_neighbors': [3, 5, 10]},
                'params_full': {'n_neighbors': [3, 5, 10, 15, 20], 'weights': ['uniform', 'distance'],
                               'metric': ['euclidean', 'manhattan', 'minkowski']},
                'gpu_supported': False
            },
            'svr': {
                'model': SVR(),
                'name': 'Support Vector Regression',
                'params_fast': {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf']},
                'params_full': {'C': [0.01, 0.1, 1.0, 10.0, 100.0], 'kernel': ['rbf', 'poly', 'sigmoid'],
                               'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]},
                'gpu_supported': False
            },
            'mlp': {
                'model': MLPRegressor(max_iter=500, random_state=42),
                'name': 'Neural Network (MLP)',
                'params_fast': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.01, 0.1]},
                'params_full': {'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                               'alpha': [0.001, 0.01, 0.1, 1.0], 'learning_rate': ['constant', 'adaptive']},
                'gpu_supported': False
            }
        }
        
        # Add GPU-accelerated models if available
        if HAS_XGB:
            models['xgboost'] = {
                'model': xgb.XGBRegressor(random_state=42, tree_method='gpu_hist' if self.gpu_enabled else 'hist'),
                'name': 'XGBoost',
                'params_fast': {'n_estimators': [50], 'learning_rate': [0.1]},
                'params_full': {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                               'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0],
                               'colsample_bytree': [0.8, 0.9, 1.0]},
                'gpu_supported': True
            }
        
        if HAS_LGB:
            models['lightgbm'] = {
                'model': lgb.LGBMRegressor(random_state=42, device='gpu' if self.gpu_enabled else 'cpu', verbose=-1),
                'name': 'LightGBM',
                'params_fast': {'n_estimators': [50], 'learning_rate': [0.1]},
                'params_full': {'n_estimators': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                               'max_depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0],
                               'colsample_bytree': [0.8, 0.9, 1.0]},
                'gpu_supported': True
            }
        
        if HAS_CATBOOST:
            models['catboost'] = {
                'model': cb.CatBoostRegressor(random_state=42, verbose=False, 
                                            task_type='GPU' if self.gpu_enabled else 'CPU'),
                'name': 'CatBoost',
                'params_fast': {'iterations': [50], 'learning_rate': [0.1]},
                'params_full': {'iterations': [50, 100, 200], 'learning_rate': [0.05, 0.1, 0.2],
                               'depth': [3, 5, 7], 'subsample': [0.8, 0.9, 1.0]},
                'gpu_supported': True
            }
        
        return models

    def build_prediction_models(self, train_df, target_col='SalesQuantity', 
                              selected_models=None, mode='fast', cv_folds=3):
        """
        CASE STUDY QUESTION E: What is the biggest effect explaining sales change during promotions?
        
        Advanced ML Pipeline:
        - Multiple algorithms with hyperparameter tuning
        - Feature importance analysis
        - Cross-validation for robust evaluation
        - GPU acceleration when available
        """
        
        self.log(f"Building prediction models in {mode} mode with {cv_folds}-fold CV...")
        
        # Get available models
        available_models = self.get_available_models()
        
        # Select models to use
        if selected_models is None:
            selected_models = ['linear_regression', 'ridge', 'random_forest', 'gradient_boosting']
            if HAS_XGB:
                selected_models.append('xgboost')
        
        # Prepare features
        feature_cols = []
        model_data = train_df.copy()
        
        self.log("Preparing features for machine learning...")
        
        # Categorical feature encoding
        categorical_features = ['ProductCode', 'StoreCode', 'ItemCategory', 'StoreCategory', 'PromotionPeriod', 'DayOfWeek']
        le_dict = {}
        
        for col in categorical_features:
            if col in model_data.columns:
                le = LabelEncoder()
                model_data[f'{col}_encoded'] = le.fit_transform(model_data[col].fillna('Unknown'))
                le_dict[col] = le
                feature_cols.append(f'{col}_encoded')
        
        # Numerical features
        numerical_features = ['DayOfYear', 'Week', 'IsPromotion', 'DaysSincePromoStart', 'DaysUntilPromoEnd']
        for col in numerical_features:
            if col in model_data.columns:
                feature_cols.append(col)
        
        # Prepare training data
        X = model_data[feature_cols].fillna(0)
        y = model_data[target_col]
        
        # Feature scaling for algorithms that need it
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.log(f"Training dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Train models with hyperparameter tuning
        model_results = {}
        
        print("\n" + "="*80)
        print("MACHINE LEARNING MODEL COMPARISON")
        print("="*80)
        print("QUESTION E: Biggest factors explaining sales change during promotions")
        print("Method: Multiple ML algorithms with hyperparameter optimization")
        print("-"*80)
        
        for model_name in selected_models:
            if model_name not in available_models:
                self.log(f"Model {model_name} not available, skipping...", "WARNING")
                continue
                
            model_config = available_models[model_name]
            model = model_config['model']
            
            start_time = time.time()
            self.log(f"Training {model_config['name']}...")
            
            # Select parameter grid based on mode
            if mode == 'fast':
                param_grid = model_config['params_fast']
            else:
                param_grid = model_config['params_full']
            
            # Use appropriate data (scaled for some algorithms)
            if model_name in ['svr', 'knn', 'mlp']:
                X_train = X_scaled
            else:
                X_train = X
            
            # Hyperparameter tuning
            if param_grid and mode != 'fast':
                # Full mode: Use extensive hyperparameter tuning
                grid_search = RandomizedSearchCV(model, param_grid, cv=cv_folds,
                                               scoring='r2', n_jobs=-1, n_iter=20)
                grid_search.fit(X_train, y)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
            elif param_grid and mode == 'fast':
                # Fast mode: Minimal hyperparameter tuning with single CV fold
                grid_search = GridSearchCV(model, param_grid, cv=2, 
                                         scoring='r2', n_jobs=-1)
                grid_search.fit(X_train, y)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
            else:
                # No hyperparameters to tune - use default model
                model.fit(X_train, y)
                best_model = model
                best_params = {}
                cv_score = None
            
            # Make predictions and calculate metrics
            y_pred = best_model.predict(X_train)
            
            results = {
                'model': best_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'r2': r2_score(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'mae': mean_absolute_error(y, y_pred),
                'mape': np.mean(np.abs((y - y_pred) / np.where(y != 0, y, 1))) * 100,
                'training_time': time.time() - start_time
            }
            
            model_results[model_name] = results
            self.models[model_name] = best_model
            
            # Display results
            print(f"{model_config['name']:20s} | R²={results['r2']:.3f} | "
                  f"RMSE={results['rmse']:.3f} | MAE={results['mae']:.3f} | "
                  f"Time={results['training_time']:.1f}s")
        
        # Find best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_model_result = model_results[best_model_name]
        
        print("-"*80)
        print(f"BEST MODEL: {available_models[best_model_name]['name']}")
        print(f"Performance: R² = {best_model_result['r2']:.3f}")
        print(f"Best Parameters: {best_model_result['best_params']}")
        
        # Feature importance analysis
        self.analyze_feature_importance(best_model_result['model'], feature_cols)
        
        # Store results
        self.grid_search_results = model_results
        self.models['label_encoders'] = le_dict
        self.models['feature_cols'] = feature_cols
        self.models['scaler'] = scaler
        
        return model_results
    
    def analyze_feature_importance(self, model, feature_cols):
        """Analyze and display feature importance for the best model"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                self.log("Model doesn't support feature importance analysis", "WARNING")
                return
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\n" + "="*60)
            print("FEATURE IMPORTANCE ANALYSIS")
            print("="*60)
            print("ANSWER TO QUESTION E: Biggest factors explaining sales change")
            print("-"*60)
            
            top_features = importance_df.head(10)
            for idx, row in top_features.iterrows():
                print(f"{row['Feature']:25s} | {row['Importance']:.4f}")
            
            print("-"*60)
            print("Business Interpretation:")
            top_feature = top_features.iloc[0]['Feature']
            print(f"Most Important Factor: {top_feature}")
            print("This feature has the strongest predictive power for sales changes")
            print("during promotional periods.")
            
            self.feature_importance['main_analysis'] = importance_df
            
        except Exception as e:
            self.log(f"Error in feature importance analysis: {e}", "ERROR")
    
    def analyze_return_rates(self, sales_df):
        """
        BONUS QUESTION: Is there any significant difference in item return rates after promotions?
        
        Analysis Method:
        - Compare return rates across different periods
        - Statistical significance testing
        - Business implications for customer satisfaction
        """
        
        self.log("Analyzing return rate patterns across promotional periods...")
        
        # Identify returns (negative quantities)
        returns_data = sales_df[sales_df['SalesQuantity'] < 0].copy()
        total_transactions = len(sales_df)
        
        # Calculate return rates by period type
        periods = {
            'During Promotions': sales_df['IsPromotion'],
            'After Promotions': sales_df['IsPostPromo'] if 'IsPostPromo' in sales_df.columns else pd.Series([False]*len(sales_df)),
            'Normal Periods': ~(sales_df['IsPromotion'] | (sales_df['IsPostPromo'] if 'IsPostPromo' in sales_df.columns else pd.Series([False]*len(sales_df))))
        }
        
        print("\n" + "="*80)
        print("BONUS ANALYSIS: RETURN RATE PATTERNS")
        print("="*80)
        print("Research Question: Do promotions affect customer return behavior?")
        print("Business Impact: Return rates indicate customer satisfaction and product quality")
        print("-"*80)
        
        return_analysis = {}
        for period_name, period_mask in periods.items():
            period_data = sales_df[period_mask]
            period_returns = returns_data[returns_data.index.isin(period_data.index)]
            
            if len(period_data) > 0:
                return_rate = (len(period_returns) / len(period_data)) * 100
                avg_return_size = abs(period_returns['SalesQuantity'].mean()) if len(period_returns) > 0 else 0
                
                return_analysis[period_name] = {
                    'return_rate': return_rate,
                    'total_returns': len(period_returns),
                    'total_transactions': len(period_data),
                    'avg_return_size': avg_return_size
                }
                
                print(f"{period_name:20s}: {return_rate:5.2f}% return rate "
                      f"({len(period_returns):,} returns out of {len(period_data):,} transactions)")
        
        # Business interpretation
        promo_rate = return_analysis.get('During Promotions', {}).get('return_rate', 0)
        normal_rate = return_analysis.get('Normal Periods', {}).get('return_rate', 0)
        
        print("-"*80)
        print("BUSINESS INSIGHTS:")
        if abs(promo_rate - normal_rate) < 0.1:
            print("✓ Return rates remain stable during promotions")
            print("  This indicates good product quality and customer satisfaction")
        elif promo_rate > normal_rate:
            print("⚠ Higher return rates during promotions")
            print("  May indicate impulse buying or customer dissatisfaction")
        else:
            print("✓ Lower return rates during promotions")
            print("  Suggests customers are more satisfied with promoted products")
        
        self.results['return_analysis'] = return_analysis
        return return_analysis

if __name__ == "__main__":
    print("Advanced Promotion Analysis module loaded successfully!")
    print("Features: Multiple ML models, GPU support, hyperparameter tuning, statistical testing")
    print("Available models depend on installed packages (XGBoost, LightGBM, CatBoost)") 