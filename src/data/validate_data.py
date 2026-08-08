"""
Comprehensive data quality checks for manga recommender dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """Validate all collected datasets"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.issues = []
        self.warnings = []
        self.stats = {}
        
    def log_issue(self, severity, message):
        """Log data quality issues"""
        if severity == 'ERROR':
            self.issues.append(f"ERROR: {message}")
        elif severity == 'WARNING':
            self.warnings.append(f"WARNING: {message}")
        elif severity == 'INFO':
            print(f"ℹINFO: {message}")
    
    def validate_manga_metadata(self):
        """Validate manga_metadata.csv"""
        print("\n" + "="*70)
        print("VALIDATING MANGA METADATA")
        print("="*70)
        
        filepath = self.data_dir / 'manga_metadata.csv'
        
        if not filepath.exists():
            self.log_issue('ERROR', f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        self.log_issue('INFO', f"Loaded {len(df)} manga records")
        
        # Check required columns
        required_cols = ['manga_id', 'title', 'genres', 'score', 'members']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log_issue('ERROR', f"Missing required columns: {missing_cols}")
            return None
        
        # Check for duplicates
        duplicates = df['manga_id'].duplicated().sum()
        if duplicates > 0:
            self.log_issue('ERROR', f"Found {duplicates} duplicate manga_id values")
        else:
            self.log_issue('INFO', "No duplicate manga_id values")
        
        # Check for missing values
        print("\nMissing Values:")
        missing_summary = df.isnull().sum()
        print(missing_summary[missing_summary > 0])
        
        critical_nulls = df[['manga_id', 'title']].isnull().sum()
        if critical_nulls.any():
            self.log_issue('ERROR', f"Critical fields have nulls: {critical_nulls[critical_nulls > 0]}")
        
        # Genre validation
        genres_missing = df['genres'].isnull().sum()
        if genres_missing > len(df) * 0.1:  # More than 10% missing
            self.log_issue('WARNING', f"{genres_missing} manga missing genres ({genres_missing/len(df)*100:.1f}%)")
        
        # Score validation
        scores = df['score'].dropna()
        if (scores < 0).any() or (scores > 10).any():
            self.log_issue('ERROR', "Score values outside valid range [0, 10]")
        
        avg_score = scores.mean()
        if avg_score < 5 or avg_score > 9:
            self.log_issue('WARNING', f"Average score ({avg_score:.2f}) seems unusual")
        else:
            self.log_issue('INFO', f"Average score: {avg_score:.2f} (reasonable)")
        
        # Date validation
        df['published_from'] = pd.to_datetime(df['published_from'], errors='coerce')
        date_nulls = df['published_from'].isnull().sum()
        if date_nulls > 0:
            self.log_issue('WARNING', f"{date_nulls} manga missing publication dates")
        
        valid_dates = df['published_from'].dropna()
        if len(valid_dates) > 0:
            date_range = f"{valid_dates.min().date()} to {valid_dates.max().date()}"
            self.log_issue('INFO', f"Date range: {date_range}")
        
        # Check for decade coverage (2014-2024)
        df['pub_year'] = df['published_from'].dt.year
        year_counts = df['pub_year'].value_counts().sort_index()
        print("\nManga by Year:")
        print(year_counts)
        
        if year_counts.empty or year_counts.index.min() > 2014:
            self.log_issue('WARNING', "Limited coverage of early years (2014-2016)")
        
        # Genre distribution
        all_genres = []
        for genres_str in df['genres'].dropna():
            all_genres.extend(genres_str.split(','))
        
        unique_genres = len(set(all_genres))
        self.log_issue('INFO', f"Found {unique_genres} unique genres")
        
        if unique_genres < 10:
            self.log_issue('WARNING', "Very few genres - data might be limited")
        
        # Stats summary
        self.stats['manga'] = {
            'total_count': len(df),
            'avg_score': avg_score,
            'unique_genres': unique_genres,
            'date_coverage': f"{valid_dates.min().year}-{valid_dates.max().year}" if len(valid_dates) > 0 else "N/A"
        }
        
        return df
    
    def validate_sales_data(self, manga_df):
        """Validate manga_sales.csv"""
        print("\n" + "="*70)
        print("VALIDATING SALES DATA")
        print("="*70)
        
        filepath = self.data_dir / 'manga_sales.csv'
        
        if not filepath.exists():
            self.log_issue('ERROR', f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        self.log_issue('INFO', f"Loaded {len(df)} sales records")
        
        # Check required columns
        required_cols = ['manga_id', 'title', 'volume', 'week_number', 'date', 'sales_count', 'rank']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log_issue('ERROR', f"Missing required columns: {missing_cols}")
            return None
        
        # Check manga_id alignment with metadata
        if manga_df is not None:
            sales_manga_ids = set(df['manga_id'].unique())
            metadata_manga_ids = set(manga_df['manga_id'].unique())
            
            orphan_sales = sales_manga_ids - metadata_manga_ids
            if orphan_sales:
                self.log_issue('WARNING', f"{len(orphan_sales)} manga_ids in sales but not in metadata")
            
            manga_without_sales = metadata_manga_ids - sales_manga_ids
            if manga_without_sales:
                self.log_issue('WARNING', f"{len(manga_without_sales)} manga in metadata have no sales data")
            else:
                self.log_issue('INFO', "All metadata manga have sales records")
        
        # Sales value validation
        if (df['sales_count'] < 0).any():
            self.log_issue('ERROR', "Negative sales_count values found")
        
        if (df['sales_count'] == 0).sum() > len(df) * 0.01:
            self.log_issue('WARNING', "More than 1% of sales records are zero")
        
        sales_stats = df['sales_count'].describe()
        print("\nSales Statistics:")
        print(sales_stats)
        
        # Check for unrealistic values
        if sales_stats['max'] > 1_000_000:
            self.log_issue('WARNING', f"Very high max sales: {sales_stats['max']:,.0f}")
        
        if sales_stats['mean'] < 100:
            self.log_issue('WARNING', f"Very low average sales: {sales_stats['mean']:.0f}")
        
        # Date validation
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        date_nulls = df['date'].isnull().sum()
        if date_nulls > 0:
            self.log_issue('ERROR', f"{date_nulls} sales records have invalid dates")
        
        date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
        self.log_issue('INFO', f"Sales date range: {date_range}")
        
        # Check for gaps in time series
        df_sorted = df.sort_values('date')
        date_diffs = df_sorted['date'].diff()
        max_gap = date_diffs.max()
        if max_gap > pd.Timedelta(days=90):
            self.log_issue('WARNING', f"Large gap in sales data: {max_gap.days} days")
        
        # Volume validation
        volume_stats = df.groupby('manga_id')['volume'].max().describe()
        print("\nVolumes per Manga:")
        print(volume_stats)
        
        if volume_stats['max'] > 100:
            self.log_issue('WARNING', f"Some manga have {volume_stats['max']:.0f} volumes (very long series)")
        
        # Week number validation
        if (df['week_number'] < 1).any() or (df['week_number'] > 52).any():
            self.log_issue('WARNING', "week_number values outside expected range [1, 52]")
        
        # Check for complete time series (each volume should have multiple weeks)
        weeks_per_volume = df.groupby(['manga_id', 'volume']).size()
        if (weeks_per_volume < 4).sum() > len(weeks_per_volume) * 0.1:
            self.log_issue('WARNING', "Many volumes have fewer than 4 weeks of data")
        
        # Rank validation
        if (df['rank'] < 1).any():
            self.log_issue('ERROR', "Rank values less than 1 found")
        
        # Stats summary
        self.stats['sales'] = {
            'total_records': len(df),
            'unique_manga': df['manga_id'].nunique(),
            'avg_sales': sales_stats['mean'],
            'date_range': date_range
        }
        
        return df
    
    def validate_users(self):
        """Validate users.csv"""
        print("\n" + "="*70)
        print("VALIDATING USER DATA")
        print("="*70)
        
        filepath = self.data_dir / 'users.csv'
        
        if not filepath.exists():
            self.log_issue('ERROR', f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        self.log_issue('INFO', f"Loaded {len(df)} users")
        
        # Check required columns
        required_cols = ['user_id', 'preferred_genres', 'avg_rating', 'activity_level']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log_issue('ERROR', f"Missing required columns: {missing_cols}")
            return None
        
        # Check for duplicates
        duplicates = df['user_id'].duplicated().sum()
        if duplicates > 0:
            self.log_issue('ERROR', f"Found {duplicates} duplicate user_id values")
        else:
            self.log_issue('INFO', "No duplicate user_id values")
        
        # Rating validation
        ratings = df['avg_rating'].dropna()
        if (ratings < 1).any() or (ratings > 10).any():
            self.log_issue('ERROR', "avg_rating values outside valid range [1, 10]")
        
        rating_mean = ratings.mean()
        self.log_issue('INFO', f"Average user rating: {rating_mean:.2f}")
        
        # Activity level validation
        activity_levels = df['activity_level'].value_counts()
        print("\nUser Activity Levels:")
        print(activity_levels)
        
        valid_levels = {'low', 'medium', 'high'}
        invalid_levels = set(df['activity_level'].unique()) - valid_levels
        if invalid_levels:
            self.log_issue('ERROR', f"Invalid activity levels: {invalid_levels}")
        
        # Genre preference validation
        users_without_genres = df['preferred_genres'].isnull().sum()
        if users_without_genres > 0:
            self.log_issue('WARNING', f"{users_without_genres} users have no genre preferences")
        
        # Stats summary
        self.stats['users'] = {
            'total_count': len(df),
            'avg_rating': rating_mean,
            'activity_distribution': activity_levels.to_dict()
        }
        
        return df
    
    def validate_interactions(self, users_df, manga_df):
        """Validate user_interactions.csv"""
        print("\n" + "="*70)
        print("VALIDATING USER INTERACTIONS")
        print("="*70)
        
        filepath = self.data_dir / 'user_interactions.csv'
        
        if not filepath.exists():
            self.log_issue('ERROR', f"File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        self.log_issue('INFO', f"Loaded {len(df)} interactions")
        
        # Check required columns
        required_cols = ['user_id', 'manga_id', 'interaction_type', 'rating', 'timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.log_issue('ERROR', f"Missing required columns: {missing_cols}")
            return None
        
        # Check user_id alignment
        if users_df is not None:
            interaction_users = set(df['user_id'].unique())
            metadata_users = set(users_df['user_id'].unique())
            
            orphan_users = interaction_users - metadata_users
            if orphan_users:
                self.log_issue('ERROR', f"{len(orphan_users)} user_ids in interactions but not in users.csv")
            else:
                self.log_issue('INFO', "All interaction user_ids exist in users.csv")
        
        # Check manga_id alignment
        if manga_df is not None:
            interaction_manga = set(df['manga_id'].unique())
            metadata_manga = set(manga_df['manga_id'].unique())
            
            orphan_manga = interaction_manga - metadata_manga
            if orphan_manga:
                self.log_issue('WARNING', f"{len(orphan_manga)} manga_ids in interactions but not in metadata")
        
        # Interaction type validation
        interaction_counts = df['interaction_type'].value_counts()
        print("\nInteraction Types:")
        print(interaction_counts)
        
        valid_types = {'purchased', 'rated', 'bookmarked', 'viewed'}
        invalid_types = set(df['interaction_type'].unique()) - valid_types
        if invalid_types:
            self.log_issue('ERROR', f"Invalid interaction types: {invalid_types}")
        
        # Rating validation
        ratings = df['rating'].dropna()
        if (ratings < 1).any() or (ratings > 10).any():
            self.log_issue('ERROR', "Rating values outside valid range [1, 10]")
        
        rating_stats = ratings.describe()
        print("\nRating Statistics:")
        print(rating_stats)
        
        self.log_issue('INFO', f"Average interaction rating: {rating_stats['mean']:.2f}")
        
        # Timestamp validation
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        timestamp_nulls = df['timestamp'].isnull().sum()
        if timestamp_nulls > 0:
            self.log_issue('ERROR', f"{timestamp_nulls} interactions have invalid timestamps")
        
        # Check for future timestamps
        future_timestamps = (df['timestamp'] > pd.Timestamp.now()).sum()
        if future_timestamps > 0:
            self.log_issue('ERROR', f"{future_timestamps} interactions have future timestamps")
        
        # Check interaction distribution per user
        interactions_per_user = df.groupby('user_id').size()
        print("\nInteractions per User:")
        print(interactions_per_user.describe())
        
        users_with_few_interactions = (interactions_per_user < 3).sum()
        if users_with_few_interactions > len(interactions_per_user) * 0.2:
            self.log_issue('WARNING', f"{users_with_few_interactions} users have fewer than 3 interactions")
        
        # Check for duplicate interactions
        duplicates = df.duplicated(subset=['user_id', 'manga_id', 'interaction_type']).sum()
        if duplicates > 0:
            self.log_issue('WARNING', f"{duplicates} duplicate user-manga-type combinations")
        
        # Stats summary
        self.stats['interactions'] = {
            'total_count': len(df),
            'unique_users': df['user_id'].nunique(),
            'unique_manga': df['manga_id'].nunique(),
            'avg_rating': rating_stats['mean'],
            'type_distribution': interaction_counts.to_dict()
        }
        
        return df
    
    def cross_validate(self, manga_df, sales_df, users_df, interactions_df):
        """Cross-dataset validation"""
        print("\n" + "="*70)
        print("CROSS-DATASET VALIDATION")
        print("="*70)
        
        if manga_df is None or sales_df is None:
            self.log_issue('ERROR', "Cannot perform cross-validation without manga and sales data")
            return
        
        # Check if popular manga (high members) have high sales
        if 'members' in manga_df.columns:
            top_popular = manga_df.nlargest(20, 'members')['manga_id']
            top_sales = sales_df.groupby('manga_id')['sales_count'].sum().nlargest(20).index
            
            overlap = len(set(top_popular) & set(top_sales))
            overlap_pct = overlap / 20 * 100
            
            if overlap_pct < 30:
                self.log_issue('WARNING', f"Low overlap ({overlap_pct:.0f}%) between popular manga and high sales")
            else:
                self.log_issue('INFO', f"✓ Good overlap ({overlap_pct:.0f}%) between popularity and sales")
        
        # Check if high-rated manga have high interaction counts
        if interactions_df is not None:
            high_rated = manga_df[manga_df['score'] > 8.5]['manga_id']
            interaction_counts = interactions_df.groupby('manga_id').size()
            
            avg_interactions_high_rated = interaction_counts[interaction_counts.index.isin(high_rated)].mean()
            avg_interactions_overall = interaction_counts.mean()
            
            if avg_interactions_high_rated > avg_interactions_overall * 1.5:
                self.log_issue('INFO', f"High-rated manga have {avg_interactions_high_rated/avg_interactions_overall:.1f}x more interactions")
            else:
                self.log_issue('WARNING', "High-rated manga don't show expected interaction boost")
    
    def generate_report(self):
        """Generate final validation report"""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        print(f"\nDataset Statistics:")
        for dataset, stats in self.stats.items():
            print(f"\n  {dataset.upper()}:")
            for key, value in stats.items():
                print(f"    {key}: {value}")
        
        if self.issues:
            print(f"\nERRORS FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
        else:
            print("\nNo critical errors found!")
        
        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        # Overall verdict
        print("\n" + "="*70)
        if not self.issues:
            if len(self.warnings) <= 3:
                print("DATA QUALITY: EXCELLENT - Ready to proceed!")
            else:
                print("DATA QUALITY: GOOD - Minor issues, but usable")
        else:
            print("DATA QUALITY: ISSUES FOUND - Review errors before proceeding")
        print("="*70)
    
    def run_all_validations(self):
        """Run complete validation suite"""
        print("\nStarting Data Quality Validation...")
        
        manga_df = self.validate_manga_metadata()
        sales_df = self.validate_sales_data(manga_df)
        users_df = self.validate_users()
        interactions_df = self.validate_interactions(users_df, manga_df)
        
        self.cross_validate(manga_df, sales_df, users_df, interactions_df)
        self.generate_report()
        
        return len(self.issues) == 0


if __name__ == "__main__":
    validator = DataValidator()
    success = validator.run_all_validations()
    
    exit(0 if success else 1)