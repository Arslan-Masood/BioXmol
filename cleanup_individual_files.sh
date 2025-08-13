#!/bin/bash

# Script to clean up individual JUMP plate files after aggregation
# This will delete files like: source_*.*.*.*.parquet (original only)
# But preserve: *.centered.parquet, filtered.parquet and centered.filtered.parquet

DATA_DIR="/scratch/work/masooda1/datasets/jump_data"

echo "JUMP Data Cleanup Script"
echo "========================"
echo "Data directory: $DATA_DIR"
echo ""

# Check if aggregated files exist first
ORIGINAL_AGG="${DATA_DIR}/filtered.parquet"
CENTERED_AGG="${DATA_DIR}/centered.filtered.parquet"

if [ ! -f "$ORIGINAL_AGG" ]; then
    echo "WARNING: Original aggregated file not found: $ORIGINAL_AGG"
    echo "Please run aggregation first before cleaning up individual files!"
    exit 1
fi

echo "✓ Found original aggregated file:"
echo "  - Original: $ORIGINAL_AGG ($(du -h "$ORIGINAL_AGG" | cut -f1))"

if [ -f "$CENTERED_AGG" ]; then
    echo "  - Centered: $CENTERED_AGG ($(du -h "$CENTERED_AGG" | cut -f1))"
fi
echo ""

# Count files to be deleted
cd "$DATA_DIR"

# Count original individual files (source_*.*.*.*.parquet but not *.centered.parquet or *filtered*)
ORIGINAL_COUNT=$(find . -name "source_*.*.*.*.parquet" ! -name "*.centered.parquet" ! -name "*filtered*" | wc -l)

# Count centered individual files that will be PRESERVED
CENTERED_COUNT=$(find . -name "source_*.*.*.*.centered.parquet" | wc -l)

echo "Files to be processed:"
echo "  - Original individual files to DELETE: $ORIGINAL_COUNT"
echo "  - Centered individual files to PRESERVE: $CENTERED_COUNT"
echo ""

if [ $ORIGINAL_COUNT -eq 0 ]; then
    echo "No original individual files found to delete."
    exit 0
fi

# Calculate space to be freed
echo "Calculating space that will be freed..."
SPACE_TO_FREE=$(find . -name "source_*.*.*.*.parquet" ! -name "*.centered.parquet" ! -name "*filtered*" -exec du -ch {} + | tail -1 | cut -f1)
echo "Estimated space to be freed: $SPACE_TO_FREE"
echo ""

# Show some examples of files to be deleted
echo "Examples of files to be DELETED:"
find . -name "source_*.*.*.*.parquet" ! -name "*.centered.parquet" ! -name "*filtered*" | head -5
if [ $ORIGINAL_COUNT -gt 5 ]; then
    echo "... and $((ORIGINAL_COUNT - 5)) more original files"
fi
echo ""

echo "Examples of files to be PRESERVED:"
find . -name "source_*.*.*.*.centered.parquet" | head -3
if [ $CENTERED_COUNT -gt 3 ]; then
    echo "... and $((CENTERED_COUNT - 3)) more centered files"
fi
echo ""

# Confirmation prompt
read -p "Are you sure you want to delete $ORIGINAL_COUNT ORIGINAL files (keeping centered files)? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Operation cancelled."
    exit 0
fi

echo ""
echo "Starting cleanup of ORIGINAL files only..."

# Delete original individual files only
echo "Deleting original individual files..."
DELETED_ORIG=0
for file in $(find . -name "source_*.*.*.*.parquet" ! -name "*.centered.parquet" ! -name "*filtered*"); do
    if rm "$file" 2>/dev/null; then
        ((DELETED_ORIG++))
        if [ $((DELETED_ORIG % 100)) -eq 0 ]; then
            echo "  Deleted $DELETED_ORIG original files..."
        fi
    else
        echo "  Failed to delete: $file"
    fi
done

echo ""
echo "Cleanup completed!"
echo "  - Original files deleted: $DELETED_ORIG"
echo "  - Centered files preserved: $CENTERED_COUNT"
echo ""

# Show remaining files
echo "Remaining files in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.parquet 2>/dev/null || echo "No .parquet files found"

echo ""
echo "Cleanup successful! Original individual plate files have been removed."
echo "Your data is preserved in:"
echo "  - $ORIGINAL_AGG (aggregated original data)"
if [ -f "$CENTERED_AGG" ]; then
    echo "  - $CENTERED_AGG (aggregated centered data)"
fi
echo "  - Individual centered files (*.centered.parquet) - PRESERVED" 