import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngineer:
    """
    Class to handle feature engineering for sales promotion analysis
    """
    
    def __init__(self):
        self.item_clusters = None
        self.store_clusters = None
        self.item_metrics = None
        self.store_metrics = None
        
    def calculate_item_metrics(self, sales_df):
        """Calculate metrics for item clustering (Fast/Medium/Slow items)"""
        # Filter non-promotion periods for baseline metrics
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        # Calculate weekly sales per store for each item
        item_metrics = []
        
        for item in sales_df['ProductCode'].unique():
            item_data = non_promo_data[non_promo_data['ProductCode'] == item]
            
            if len(item_data) == 0:
                continue
                
            # Calculate metrics per store
            store_weekly_sales = []
            for store in item_data['StoreCode'].unique():
                store_item_data = item_data[item_data['StoreCode'] == store]
                
                # Group by week and sum quantities
                weekly_sales = store_item_data.groupby('Week')['SalesQuantity'].sum()
                
                if len(weekly_sales) > 0:
                    store_weekly_sales.extend(weekly_sales.tolist())
            
            if len(store_weekly_sales) > 0:
                avg_weekly_sale_per_store = np.mean(store_weekly_sales)
                total_sales = item_data['SalesQuantity'].sum()
                num_stores = item_data['StoreCode'].nunique()
                num_weeks = len(item_data['Week'].unique())
                sales_volatility = np.std(store_weekly_sales) if len(store_weekly_sales) > 1 else 0
                
                item_metrics.append({
                    'ProductCode': item,
                    'AvgWeeklySalePerStore': avg_weekly_sale_per_store,
                    'TotalSales': total_sales,
                    'NumStores': num_stores,
                    'NumWeeks': num_weeks,
                    'SalesVolatility': sales_volatility,
                    'SalesPerStorePerWeek': total_sales / (num_stores * num_weeks) if (num_stores * num_weeks) > 0 else 0
                })
        
        self.item_metrics = pd.DataFrame(item_metrics)
        return self.item_metrics
    
    def calculate_store_metrics(self, sales_df):
        """Calculate metrics for store clustering (Fast/Medium/Slow stores)"""
        # Filter non-promotion periods for baseline metrics
        non_promo_data = sales_df[~sales_df['IsPromotion']].copy()
        
        store_metrics = []
        
        for store in sales_df['StoreCode'].unique():
            store_data = non_promo_data[non_promo_data['StoreCode'] == store]
            
            if len(store_data) == 0:
                continue
            
            # Calculate metrics per item
            item_weekly_sales = []
            for item in store_data['ProductCode'].unique():
                item_store_data = store_data[store_data['ProductCode'] == item]
                
                # Group by week and sum quantities
                weekly_sales = item_store_data.groupby('Week')['SalesQuantity'].sum()
                
                if len(weekly_sales) > 0:
                    item_weekly_sales.extend(weekly_sales.tolist())
            
            if len(item_weekly_sales) > 0:
                avg_weekly_sale_per_item = np.mean(item_weekly_sales)
                total_sales = store_data['SalesQuantity'].sum()
                num_items = store_data['ProductCode'].nunique()
                num_weeks = len(store_data['Week'].unique())
                sales_volatility = np.std(item_weekly_sales) if len(item_weekly_sales) > 1 else 0
                
                store_metrics.append({
                    'StoreCode': store,
                    'AvgWeeklySalePerItem': avg_weekly_sale_per_item,
                    'TotalSales': total_sales,
                    'NumItems': num_items,
                    'NumWeeks': num_weeks,
                    'SalesVolatility': sales_volatility,
                    'SalesPerItemPerWeek': total_sales / (num_items * num_weeks) if (num_items * num_weeks) > 0 else 0
                })
        
        self.store_metrics = pd.DataFrame(store_metrics)
        return self.store_metrics
    
    def cluster_items(self, method='kmeans', manual_thresholds=None):
        """Cluster items into Fast/Medium/Slow categories"""
        if self.item_metrics is None:
            raise ValueError("Item metrics not calculated. Run calculate_item_metrics first.")
        
        df = self.item_metrics.copy()
        
        if method == 'manual' and manual_thresholds:
            # Manual clustering based on thresholds
            low_threshold, high_threshold = manual_thresholds
            
            df['ItemCategory'] = 'Medium'
            df.loc[df['AvgWeeklySalePerStore'] <= low_threshold, 'ItemCategory'] = 'Slow'
            df.loc[df['AvgWeeklySalePerStore'] >= high_threshold, 'ItemCategory'] = 'Fast'
            
        else:
            # KMeans clustering
            features = ['AvgWeeklySalePerStore', 'SalesPerStorePerWeek']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Map clusters to Fast/Medium/Slow based on average sales
            df['Cluster'] = clusters
            cluster_means = df.groupby('Cluster')['AvgWeeklySalePerStore'].mean()
            
            # Sort clusters by sales performance
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {
                sorted_clusters[0]: 'Slow',
                sorted_clusters[1]: 'Medium', 
                sorted_clusters[2]: 'Fast'
            }
            
            df['ItemCategory'] = df['Cluster'].map(cluster_mapping)
        
        self.item_clusters = df
        
        # Print clustering results
        print("Item Clustering Results:")
        print(df['ItemCategory'].value_counts())
        print("\nItem Category Statistics:")
        print(df.groupby('ItemCategory')['AvgWeeklySalePerStore'].agg(['count', 'mean', 'std']).round(2))
        
        return df
    
    def cluster_stores(self, method='kmeans', manual_thresholds=None):
        """Cluster stores into Fast/Medium/Slow categories"""
        if self.store_metrics is None:
            raise ValueError("Store metrics not calculated. Run calculate_store_metrics first.")
        
        df = self.store_metrics.copy()
        
        if method == 'manual' and manual_thresholds:
            # Manual clustering based on thresholds
            low_threshold, high_threshold = manual_thresholds
            
            df['StoreCategory'] = 'Medium'
            df.loc[df['AvgWeeklySalePerItem'] <= low_threshold, 'StoreCategory'] = 'Slow'
            df.loc[df['AvgWeeklySalePerItem'] >= high_threshold, 'StoreCategory'] = 'Fast'
            
        else:
            # KMeans clustering
            features = ['AvgWeeklySalePerItem', 'SalesPerItemPerWeek']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[features])
            
            kmeans = KMeans(n_clusters=3, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Map clusters to Fast/Medium/Slow based on average sales
            df['Cluster'] = clusters
            cluster_means = df.groupby('Cluster')['AvgWeeklySalePerItem'].mean()
            
            # Sort clusters by sales performance
            sorted_clusters = cluster_means.sort_values().index.tolist()
            cluster_mapping = {
                sorted_clusters[0]: 'Slow',
                sorted_clusters[1]: 'Medium',
                sorted_clusters[2]: 'Fast'
            }
            
            df['StoreCategory'] = df['Cluster'].map(cluster_mapping)
        
        self.store_clusters = df
        
        # Print clustering results
        print("Store Clustering Results:")
        print(df['StoreCategory'].value_counts())
        print("\nStore Category Statistics:")
        print(df.groupby('StoreCategory')['AvgWeeklySalePerItem'].agg(['count', 'mean', 'std']).round(2))
        
        return df
    
    def add_cluster_features(self, sales_df):
        """Add cluster information to sales data"""
        df = sales_df.copy()
        
        # Add item categories
        if self.item_clusters is not None:
            item_category_map = dict(zip(self.item_clusters['ProductCode'], self.item_clusters['ItemCategory']))
            df['ItemCategory'] = df['ProductCode'].map(item_category_map)
        
        # Add store categories
        if self.store_clusters is not None:
            store_category_map = dict(zip(self.store_clusters['StoreCode'], self.store_clusters['StoreCategory']))
            df['StoreCategory'] = df['StoreCode'].map(store_category_map)
        
        return df
    
    def create_promotion_features(self, sales_df):
        """Create promotion-related features"""
        df = sales_df.copy()
        
        # Days since promotion start
        df['DaysSincePromoStart'] = 0
        df['DaysUntilPromoEnd'] = 0
        
        for promo_period in df['PromotionPeriod'].dropna().unique():
            promo_mask = df['PromotionPeriod'] == promo_period
            promo_dates = df[promo_mask]['Date']
            
            if len(promo_dates) > 0:
                start_date = promo_dates.min()
                end_date = promo_dates.max()
                
                df.loc[promo_mask, 'DaysSincePromoStart'] = (df.loc[promo_mask, 'Date'] - start_date).dt.days
                df.loc[promo_mask, 'DaysUntilPromoEnd'] = (end_date - df.loc[promo_mask, 'Date']).dt.days
        
        # Pre and post promotion periods
        df['IsPrePromo'] = False
        df['IsPostPromo'] = False
        
        # Mark 7 days before and after promotions
        for _, row in df[df['IsPromotion']].groupby('PromotionPeriod')['Date'].agg(['min', 'max']).iterrows():
            promo_start, promo_end = row['min'], row['max']
            
            # Pre-promotion period
            pre_mask = (df['Date'] >= promo_start - pd.Timedelta(days=7)) & (df['Date'] < promo_start)
            df.loc[pre_mask, 'IsPrePromo'] = True
            
            # Post-promotion period
            post_mask = (df['Date'] > promo_end) & (df['Date'] <= promo_end + pd.Timedelta(days=7))
            df.loc[post_mask, 'IsPostPromo'] = True
        
        return df
    
    def calculate_baseline_sales(self, sales_df):
        """Calculate baseline sales for each item-store combination"""
        # Use non-promotion periods to calculate baseline
        baseline_data = sales_df[~sales_df['IsPromotion']].copy()
        
        # Calculate weekly baseline for each item-store combination
        baseline_weekly = baseline_data.groupby(['ProductCode', 'StoreCode', 'Week'])['SalesQuantity'].sum().reset_index()
        baseline_avg = baseline_weekly.groupby(['ProductCode', 'StoreCode'])['SalesQuantity'].mean().reset_index()
        baseline_avg.rename(columns={'SalesQuantity': 'BaselineSales'}, inplace=True)
        
        # Merge back to main data
        df = sales_df.merge(baseline_avg, on=['ProductCode', 'StoreCode'], how='left')
        
        # Calculate promotion lift
        df['SalesLift'] = df['SalesQuantity'] - df['BaselineSales'].fillna(0)
        df['SalesLiftPercent'] = (df['SalesLift'] / df['BaselineSales'].fillna(1)) * 100
        
        return df
    
    def visualize_clusters(self):
        """Create visualizations for item and store clusters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Item clustering visualization
        if self.item_clusters is not None:
            # Scatter plot of item metrics
            axes[0, 0].scatter(self.item_clusters['AvgWeeklySalePerStore'], 
                             self.item_clusters['SalesPerStorePerWeek'],
                             c=pd.Categorical(self.item_clusters['ItemCategory']).codes,
                             alpha=0.6)
            axes[0, 0].set_xlabel('Avg Weekly Sale Per Store')
            axes[0, 0].set_ylabel('Sales Per Store Per Week')
            axes[0, 0].set_title('Item Clustering')
            axes[0, 0].legend(self.item_clusters['ItemCategory'].unique())
            
            # Box plot of item categories
            sns.boxplot(data=self.item_clusters, x='ItemCategory', y='AvgWeeklySalePerStore', ax=axes[0, 1])
            axes[0, 1].set_title('Item Categories Distribution')
        
        # Store clustering visualization
        if self.store_clusters is not None:
            # Scatter plot of store metrics
            axes[1, 0].scatter(self.store_clusters['AvgWeeklySalePerItem'],
                             self.store_clusters['SalesPerItemPerWeek'],
                             c=pd.Categorical(self.store_clusters['StoreCategory']).codes,
                             alpha=0.6)
            axes[1, 0].set_xlabel('Avg Weekly Sale Per Item')
            axes[1, 0].set_ylabel('Sales Per Item Per Week')
            axes[1, 0].set_title('Store Clustering')
            axes[1, 0].legend(self.store_clusters['StoreCategory'].unique())
            
            # Box plot of store categories
            sns.boxplot(data=self.store_clusters, x='StoreCategory', y='AvgWeeklySalePerItem', ax=axes[1, 1])
            axes[1, 1].set_title('Store Categories Distribution')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Test feature engineering (placeholder)
    print("Feature engineering module loaded successfully!")
    print("Use this module with preprocessed data to create clusters and features.") 