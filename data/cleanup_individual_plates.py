#!/usr/bin/env python3

import os
import argparse
import glob
from pathlib import Path

def cleanup_individual_plates(data_dir):
    """
    Clean up individual JUMP plate files after aggregation.
    Keeps only the aggregated files: filtered.parquet and centered.filtered.parquet
    """
    print(f"🧹 Cleaning up individual plate files in: {data_dir}")
    
    # Check if aggregated files exist
    filtered_file = os.path.join(data_dir, "filtered.parquet")
    centered_filtered_file = os.path.join(data_dir, "centered.filtered.parquet")
    
    if not os.path.exists(filtered_file):
        print(f"❌ ERROR: Aggregated file not found: {filtered_file}")
        print("Please run aggregation first!")
        return False
    
    print(f"✅ Found aggregated files:")
    print(f"  - Original: {filtered_file}")
    if os.path.exists(centered_filtered_file):
        print(f"  - Centered: {centered_filtered_file}")
    
    # Find individual plate files to delete
    # Pattern: source.batch.plate.plate.parquet (with or without .centered)
    pattern = os.path.join(data_dir, "*.*.*.*.parquet")
    all_files = glob.glob(pattern)
    
    # Exclude the aggregated files
    files_to_delete = [
        f for f in all_files 
        if not f.endswith("filtered.parquet")
    ]
    
    if not files_to_delete:
        print("ℹ️  No individual plate files found to delete.")
        return True
    
    print(f"Found {len(files_to_delete)} individual plate files to delete")
    
    # Calculate total size
    total_size = sum(os.path.getsize(f) for f in files_to_delete if os.path.exists(f))
    size_mb = total_size / (1024 * 1024)
    print(f"Total size to be freed: {size_mb:.1f} MB")
    
    # Delete files
    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
            if deleted_count % 100 == 0:
                print(f"  Deleted {deleted_count} files...")
        except OSError as e:
            print(f"  Failed to delete {file_path}: {e}")
    
    print(f"✅ Cleanup completed! Deleted {deleted_count} individual plate files")
    print(f"💾 Freed up {size_mb:.1f} MB of disk space")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Clean up individual JUMP plate files after aggregation")
    parser.add_argument("data_dir", help="Directory containing the plate files")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ ERROR: Directory not found: {args.data_dir}")
        return 1
    
    success = cleanup_individual_plates(args.data_dir)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 