from datetime import datetime, timedelta
from datasets import load_dataset
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_new_trending_items(dataset_repo="reach-vb/trending-repos", days=7, max_age_months=1, top_n_per_day=100):
    """
    Find new trending models/datasets that first appeared in the specified time window.
    
    Args:
        dataset_repo (str): Hugging Face dataset repository
        days (int): Number of days to look back for new trending items
        max_age_months (int): Exclude models/datasets last modified more than this many months ago
        top_n_per_day (int): Only consider top N trending items per day (default 100)
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_repo)
        
        # Convert to pandas DataFrames
        models_df = dataset["models"].to_pandas()
        datasets_df = dataset["datasets"].to_pandas()
        
        # Convert dates to datetime and ensure consistent timezone (UTC)
        models_df['collected_at'] = pd.to_datetime(models_df['collected_at'], utc=True)
        models_df['last_modified'] = pd.to_datetime(models_df['last_modified'], utc=True)
        datasets_df['collected_at'] = pd.to_datetime(datasets_df['collected_at'], utc=True)
        datasets_df['last_modified'] = pd.to_datetime(datasets_df['last_modified'], utc=True)
        
        # Filter to only keep top N items per day
        models_df = models_df.sort_values(['collected_at', 'downloads'], ascending=[True, False])
        models_df = models_df.groupby('collected_at').head(top_n_per_day).reset_index(drop=True)
        
        datasets_df = datasets_df.sort_values(['collected_at', 'downloads'], ascending=[True, False])
        datasets_df = datasets_df.groupby('collected_at').head(top_n_per_day).reset_index(drop=True)
        
        # Get the most recent collection date (timezone-aware)
        latest_date = max(models_df['collected_at'].max(), datasets_df['collected_at'].max())
        cutoff_date = latest_date - timedelta(days=days)
        age_cutoff_date = latest_date - timedelta(days=30*max_age_months)
        
        logger.info(f"Latest data collected at: {latest_date}")
        logger.info(f"Looking for items that weren't trending before: {cutoff_date}")
        logger.info(f"Excluding items last modified before: {age_cutoff_date}")

        # Get only IDs for comparison
        models_before_ids = set(models_df[models_df['collected_at'] < cutoff_date]['id'].unique())
        models_after_ids = set(models_df[models_df['collected_at'] >= cutoff_date]['id'].unique())
        
        datasets_before_ids = set(datasets_df[datasets_df['collected_at'] < cutoff_date]['id'].unique())
        datasets_after_ids = set(datasets_df[datasets_df['collected_at'] >= cutoff_date]['id'].unique())

        # Find truly new items (in after but not in before)
        new_models_ids = models_after_ids - models_before_ids
        new_datasets_ids = datasets_after_ids - datasets_before_ids

        # Get complete information only for the new items
        new_models = (models_df[models_df['id'].isin(new_models_ids)]
                     .sort_values('collected_at', ascending=False)
                     .drop_duplicates('id', keep='first'))
        
        new_datasets = (datasets_df[datasets_df['id'].isin(new_datasets_ids)]
                       .sort_values('collected_at', ascending=False)
                       .drop_duplicates('id', keep='first'))

        # Apply age filter (now comparing timezone-aware datetimes)
        new_models = new_models[new_models['last_modified'] >= age_cutoff_date]
        new_datasets = new_datasets[new_datasets['last_modified'] >= age_cutoff_date]

        # Print results
        print(f"\n=== New Trending Models (first appeared in last {days} days, modified within {max_age_months} month(s)) ===")
        if not new_models.empty:
            for _, row in new_models.iterrows():
                modified_days = (latest_date - row['last_modified']).days
                print(f"{row['collected_at']}: {row['id']} (Modified {modified_days}d ago, Downloads: {row['downloads']}, Likes: {row['likes']})")
            print(f"\nTotal new models: {len(new_models)}")
        else:
            print("No new trending models found.")
        
        print(f"\n=== New Trending Datasets (first appeared in last {days} days, modified within {max_age_months} month(s)) ===")
        if not new_datasets.empty:
            for _, row in new_datasets.iterrows():
                modified_days = (latest_date - row['last_modified']).days
                print(f"{row['collected_at']}: {row['id']} (Modified {modified_days}d ago, Downloads: {row['downloads']}, Likes: {row['likes']})")
            print(f"\nTotal new datasets: {len(new_datasets)}")
        else:
            print("No new trending datasets found.")
        
        return {
            'new_models': new_models,
            'new_datasets': new_datasets,
            'new_models_count': len(new_models),
            'new_datasets_count': len(new_datasets)
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    # Example usage with all parameters
    results = find_new_trending_items(
        days=7,
        max_age_months=1,
        top_n_per_day=100
    )