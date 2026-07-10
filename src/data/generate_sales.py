"""
Generate synthetic but realistic manga sales data
Based on actual patterns: popularity, genere, publication timing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class SalesDataGenerator:
    """Generate realistic manga sales time series"""

    def __init__(self, manga_metadata_path='data/raw/manga_metadata.csv'):
        self.df = pd.read_csv(manga_metadata_path)
        self.df['published_from'] = pd.to_datetime(self.df['published_from'])

    def generate_sales_trajectory(self, manga_row):
        """
        Generate weekly sales for a manga based on its charcteristics
        """

        popularity_score = manga_row.get('members', 10000) / 1000
        score_boost = manga_row.get('score', 7.0) / 10.0

        status = manga_row.get('status', 'Unknown')
        volumes = manga_row.get('volumes', 10)
        if pd.isna(volumes):
            volumes = random.randint(5, 30)

        sales_data = []
        start_date = manga_row['published_from']

        if pd.isna(start_date):
            return []
        
        for volume_num in range(1, int(volumes) + 1):
            volume_date = start_date + timedelta(weeks=12 * (volume_num - 1))

            for week_offset in range (12):
                week_date = volume_date + timedelta(weeks=week_offset)

                base_sales = popularity_score * 1000 * score_boost

                volume_factor = 1.0 / (1 + 0.1 * volume_num)

                week_factor = np.exp(-0.3 * week_offset)

                random_factor = np.random.lognormal(0, 0.3)

                anime_boost = 2.0 if (volume_num == 5 and random.random() < 0.2) else 1.0

                status_factor = 0.7 if status == 'Finished' and volume_num > volumes else 1.0

                sales = base_sales * volume_factor * week_factor * random_factor * anime_boost * status_factor
                sales = max(100, int(sales))

                rank = int(1000 / (sales / 1000 + 1)) + random.randint(-50, 50)
                rank = max(1, min(500, rank))

                sales_data.append(
                    {'manga_id': manga_row['manga_id'],
                     'title': manga_row['title'],
                     'volume': volume_num,
                     'week_number': week_offset + 1,
                     'date': week_date,
                     'sales_count': sales,
                     'rank': rank,
                     'year': week_date.year
                    }
                )

        return sales_data
    

def generate_all_sales(self):
    """Generate sales for all manga"""
    all_sales = []

    valid_manga = self.df[self.df['published_from'].notna()]

    print(f"Generating sales data for {len(valid_manga)} manga...")

    for idx, manga in valid_manga.iterrows():
        if idx % 100 == 0:
            print(f"Progress: {idx}/{len(valid_manga)}")

        sales = self.generate_sales_trajectory(manga)
        all_sales.extend(sales)
    
    return pd.DataFrame(all_sales)


def save_sales_data(self, output_path='data/raw/manga_sales.csv'):
    """Generate and save sales data"""
    sales_df = self.generate_all_sales()
    sales_df.to_csv(output_path, index=False)

    print(f"\n Saved {len(sales_df)} sales records to {output_path}")
    print(f"\n Sales summary:")
    print(f"Data range: {sales_df['date'].min()} to {sales_df['date'].max()}")


if __name__ == "__main__":
    generator = SalesDataGenerator()
    sales_df = generator.save_sales_data()    
