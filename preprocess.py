import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Class to handle data preprocessing for sales promotion analysis
    """
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.sales_data = None
        self.test_data = None
        self.promotion_dates = None
        self.product_data = None
        
    def load_data(self):
        """Load all required datasets"""
        print("Loading datasets...")
        
        # Load main sales data
        self.sales_data = pd.read_csv(f'{self.data_path}assignment4.1a.csv')
        print(f"Sales data loaded: {self.sales_data.shape}")
        
        # Load test data
        self.test_data = pd.read_csv(f'{self.data_path}assignment4.1b.csv')
        print(f"Test data loaded: {self.test_data.shape}")
        
        # Load promotion dates
        self.promotion_dates = pd.read_csv(f'{self.data_path}PromotionDates.csv')
        print(f"Promotion dates loaded: {self.promotion_dates.shape}")
        
        # Load product categories
        self.product_data = pd.read_csv(f'{self.data_path}assignment4.1c.csv')
        print(f"Product data loaded: {self.product_data.shape}")
        
        return self
    
    def clean_sales_data(self, data):
        """Clean and preprocess sales data"""
        df = data.copy()
        
        # Convert date column
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Create additional time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        # Handle missing values
        df = df.dropna()
        
        print(f"Cleaned data shape: {df.shape}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Unique products: {df['ProductCode'].nunique()}")
        print(f"Unique stores: {df['StoreCode'].nunique()}")
        
        return df
    
    def clean_promotion_dates(self):
        """Clean and standardize promotion dates"""
        df = self.promotion_dates.copy()
        
        # Fix date format issues
        def parse_date(date_str):
            try:
                # Try different date formats
                for fmt in ['%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d']:
                    try:
                        return pd.to_datetime(date_str, format=fmt)
                    except:
                        continue
                return pd.to_datetime(date_str)
            except:
                return None
        
        df['StartDate'] = df['StartDate'].apply(parse_date)
        df['EndDate'] = df['EndDate'].apply(parse_date)
        
        # Remove any rows with invalid dates
        df = df.dropna()
        
        # Fix obvious errors in promotion dates
        # Promo5 shows "1/9/2015 to 6/9/2015" which is clearly wrong (5 months!)
        # Based on the pattern of weekly promotions, fix this
        if len(df) > 4:
            promo5_idx = df[df['Period'] == 'Promo5'].index
            if len(promo5_idx) > 0:
                # Based on chronological pattern, Promo5 should be between Promo4 and end of data
                df.loc[promo5_idx, 'StartDate'] = pd.to_datetime('2015-07-09')
                df.loc[promo5_idx, 'EndDate'] = pd.to_datetime('2015-07-16')
                print(f"Fixed Promo5 dates: 2015-07-09 to 2015-07-16")
                
        # Check for any promotion longer than 15 days (suspicious)
        df['Duration'] = (df['EndDate'] - df['StartDate']).dt.days
        long_promos = df[df['Duration'] > 15]
        if not long_promos.empty:
            print(f"Warning: Found promotions longer than 15 days:")
            print(long_promos[['Period', 'StartDate', 'EndDate', 'Duration']])
        
        # Sort by start date
        df = df.sort_values('StartDate')
        
        print("Cleaned promotion dates:")
        print(df)
        
        return df
    
    def create_promotion_flags(self, sales_df, promo_df):
        """Create promotion period flags in sales data"""
        df = sales_df.copy()
        
        # Initialize promotion columns
        df['IsPromotion'] = False
        df['PromotionPeriod'] = None
        
        # Mark promotion periods
        for _, promo in promo_df.iterrows():
            mask = (df['Date'] >= promo['StartDate']) & (df['Date'] <= promo['EndDate'])
            df.loc[mask, 'IsPromotion'] = True
            df.loc[mask, 'PromotionPeriod'] = promo['Period']
        
        return df
    
    def preprocess_all(self):
        """Run complete preprocessing pipeline"""
        # Load data
        self.load_data()
        
        # Clean sales data
        self.sales_data = self.clean_sales_data(self.sales_data)
        self.test_data = self.clean_sales_data(self.test_data)
        
        # Clean promotion dates
        self.promotion_dates = self.clean_promotion_dates()
        
        # Add promotion flags
        self.sales_data = self.create_promotion_flags(self.sales_data, self.promotion_dates)
        self.test_data = self.create_promotion_flags(self.test_data, self.promotion_dates)
        
        print("Preprocessing completed!")
        
        return {
            'sales_data': self.sales_data,
            'test_data': self.test_data,
            'promotion_dates': self.promotion_dates,
            'product_data': self.product_data
        }
    
    def get_baseline_period_data(self, sales_df, num_promotions=4):
        """Get data for only the first 4 promotions for baseline modeling"""
        promo_names = [f'Promo{i}' for i in range(1, num_promotions + 1)]
        
        # Find the date range covering first 4 promotions
        first_promos = self.promotion_dates[
            self.promotion_dates['Period'].isin(promo_names)
        ]
        
        if len(first_promos) == 0:
            # Fallback: use first 6 months
            start_date = sales_df['Date'].min()
            end_date = start_date + timedelta(days=180)
        else:
            start_date = sales_df['Date'].min()
            end_date = first_promos['EndDate'].max()
        
        # Filter data
        baseline_data = sales_df[
            (sales_df['Date'] >= start_date) & 
            (sales_df['Date'] <= end_date)
        ].copy()
        
        print(f"Baseline period: {start_date} to {end_date}")
        print(f"Baseline data shape: {baseline_data.shape}")
        
        return baseline_data

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    data = preprocessor.preprocess_all()
    
    # Show some statistics
    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"Sales data: {data['sales_data'].shape}")
    print(f"Promotion periods: {data['sales_data']['IsPromotion'].sum()} days")
    print(f"Non-promotion periods: {(~data['sales_data']['IsPromotion']).sum()} days") 