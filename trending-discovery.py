from datetime import datetime, timedelta
from datasets import load_dataset
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_new_trending_items(dataset_repo="reach-vb/trending-repos", days=7):
    """
    Find models/datasets that appeared in trending in the last specified days
    and weren't in the trending list at all before that period.
    
    Args:
        dataset_repo (str): Hugging Face dataset repository
        days (int): Number of days to look back for new trending items
    """
    try:
        # Load the dataset
        dataset = load_dataset(dataset_repo)
        
        # Convert to pandas DataFrames
        models_df = dataset["models"].to_pandas()
        datasets_df = dataset["datasets"].to_pandas()
        
        # Convert collected_at to datetime
        models_df['collected_at'] = pd.to_datetime(models_df['collected_at'])
        datasets_df['collected_at'] = pd.to_datetime(datasets_df['collected_at'])
        
        # Get the most recent collection date
        latest_date = max(models_df['collected_at'].max(), datasets_df['collected_at'].max())
        cutoff_date = latest_date - timedelta(days=days)
        
        logger.info(f"Latest data collected at: {latest_date}")
        logger.info(f"Looking for items that weren't trending before: {cutoff_date}")
        
        # Find all items that existed before the cutoff date
        existing_models_before = set(models_df[models_df['collected_at'] < cutoff_date]['id'].unique())
        existing_datasets_before = set(datasets_df[datasets_df['collected_at'] < cutoff_date]['id'].unique())
        
        # Find items that appeared after cutoff date
        recent_models = models_df[models_df['collected_at'] >= cutoff_date]
        recent_datasets = datasets_df[datasets_df['collected_at'] >= cutoff_date]
        
        # Filter to only keep items that weren't in the earlier period at all
        truly_new_models = recent_models[~recent_models['id'].isin(existing_models_before)]
        truly_new_datasets = recent_datasets[~recent_datasets['id'].isin(existing_datasets_before)]
        
        # Get the most recent entry for each new item (de-duplicate)
        new_models = (truly_new_models.sort_values('collected_at', ascending=False)
                                     .drop_duplicates('id', keep='first'))
        new_datasets = (truly_new_datasets.sort_values('collected_at', ascending=False)
                                       .drop_duplicates('id', keep='first'))
        
        # Sort by collection date (newest first)
        new_models = new_models.sort_values('collected_at', ascending=False)
        new_datasets = new_datasets.sort_values('collected_at', ascending=False)
        
        # Print results
        print("\n=== New Trending Models (not trending before last {} days) ===".format(days))
        if not new_models.empty:
            for _, row in new_models.iterrows():
                print(f"{row['collected_at']}: {row['id']} (Downloads: {row['downloads']}, Likes: {row['likes']})")
        else:
            print("No new trending models found.")
        
        print("\n=== New Trending Datasets (not trending before last {} days) ===".format(days))
        if not new_datasets.empty:
            for _, row in new_datasets.iterrows():
                print(f"{row['collected_at']}: {row['id']} (Downloads: {row['downloads']}, Likes: {row['likes']})")
        else:
            print("No new trending datasets found.")
        
        return {
            'new_models': new_models,
            'new_datasets': new_datasets
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    # Example usage - finds items that weren't trending at all before the last 7 days
    results = find_new_trending_items(days=7)