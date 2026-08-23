#!/bin/bash

# CONFIGURATION
FILE_NAME="history_log.txt"
START_DATE="2026-08-22"   # Format: YYYY-MM-DD
END_DATE="2026-08-17"     # Format: YYYY-MM-DD
COMMIT_INTERVAL="1 day"   # Change frequency (e.g., "1 day", "2 days", "12 hours")

# Initialize the file
echo "# Project History Log" > "$FILE_NAME"
git add "$FILE_NAME"

# Convert dates to seconds for calculation
start_sec=$(date -d "$START_DATE" +%s)
end_sec=$(date -d "$END_DATE" +%s)

current_sec=$start_sec

echo "Generating commits from $START_DATE to $END_DATE..."

while [ $current_sec -le $end_sec ]; do
    # Format the current date for the commit
    commit_date=$(date -d "@$current_sec" "+%Y-%m-%d %H:%M:%S")
    readable_date=$(date -d "@$current_sec" "+%Y-%m-%d")
    
    # Append new content to the file (required to create a new commit state)
    echo "Entry created on $readable_date" >> "$FILE_NAME"
    
    # Stage the change
    git add "$FILE_NAME"
    
    # Commit with backdated environment variables
    # Both AUTHOR and COMMITTER dates must be set to match
    GIT_AUTHOR_DATE="$commit_date" \
    GIT_COMMITTER_DATE="$commit_date" \
    git commit -m "Update log for $readable_date"
    
    if [ $? -ne 0 ]; then
        echo "Error: Commit failed. Ensure git is initialized and configured."
        exit 1
    fi

    # Increment date
    current_sec=$((current_sec + $(date -d "$COMMIT_INTERVAL" +%s -d "1970-01-01")))
done

echo "Done! Created history for $FILE_NAME."