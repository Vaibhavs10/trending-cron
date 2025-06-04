from datetime import datetime, timedelta
from datasets import load_dataset
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_new_trending_items(dataset_repo="reach-vb/trending-repos", days=7):
    """
    Find models/datasets that appeared in trending in the last specified days.
    
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
        logger.info(f"Looking for items that first appeared after: {cutoff_date}")
        
        # Find models that first appeared in the last 'days' days
        recent_models = models_df[models_df['collected_at'] >= cutoff_date]
        first_appearance_models = recent_models.groupby('id').agg({'collected_at': 'min'}).reset_index()
        new_models = first_appearance_models[first_appearance_models['collected_at'] >= cutoff_date]
        
        # Find datasets that first appeared in the last 'days' days
        recent_datasets = datasets_df[datasets_df['collected_at'] >= cutoff_date]
        first_appearance_datasets = recent_datasets.groupby('id').agg({'collected_at': 'min'}).reset_index()
        new_datasets = first_appearance_datasets[first_appearance_datasets['collected_at'] >= cutoff_date]
        
        # Get full information for the new items
        new_models_full = models_df.merge(new_models, on=['id', 'collected_at'])
        new_datasets_full = datasets_df.merge(new_datasets, on=['id', 'collected_at'])
        
        # Sort by collection date (newest first)
        new_models_full = new_models_full.sort_values('collected_at', ascending=False)
        new_datasets_full = new_datasets_full.sort_values('collected_at', ascending=False)
        
        # Print results
        print("\n=== New Trending Models (last {} days) ===".format(days))
        if not new_models_full.empty:
            for _, row in new_models_full.iterrows():
                print(f"{row['collected_at']}: {row['id']} (Downloads: {row['downloads']}, Likes: {row['likes']})")
        else:
            print("No new trending models found.")
        
        print("\n=== New Trending Datasets (last {} days) ===".format(days))
        if not new_datasets_full.empty:
            for _, row in new_datasets_full.iterrows():
                print(f"{row['collected_at']}: {row['id']} (Downloads: {row['downloads']}, Likes: {row['likes']})")
        else:
            print("No new trending datasets found.")
        
        return {
            'new_models': new_models_full,
            'new_datasets': new_datasets_full
        }
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    # Example usage - finds items that first appeared in trending in the last 7 days
    results = find_new_trending_items(days=7)