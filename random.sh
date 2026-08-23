#!/bin/bash
START_DATE="2026-08-20"
END_DATE="2026-08-25"
# Logic to loop through your changes and assign dates
for i in {0..3}; do
    CURRENT_DATE=$(date -d "$START_DATE + $i days" -Iseconds)
    
    # Make your changes here
    echo "Change $i" > file$i.txt
    git add file$i.txt
    
    GIT_AUTHOR_DATE="$CURRENT_DATE" \
    GIT_COMMITTER_DATE="$CURRENT_DATE" \
    git commit -m "Commit for $CURRENT_DATE"
done