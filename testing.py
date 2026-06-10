import os
import subprocess
import random
from datetime import datetime, timedelta

# Configuration
REPO_URL = "https://github.com/Chirag-Kathuria-009/JobApplicationAutomation.git"  # e.g., "https://github.com/username/repo.git"
GIT_EMAIL = "chiragkathuria2000@gmail.com"
GIT_NAME = "Chirag-Kathuria-009"
DAYS_TO_FILL = 30

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    # Initialize repository if not already done
    if not os.path.exists(".git"):
        run_command("git init -b main")
    
    # Set local git config to ensure identity matches GitHub
    run_command(f'git config user.email "{GIT_EMAIL}"')
    run_command(f'git config user.name "{GIT_NAME}"')

    # Create a dummy file to modify
    filename = "activity_log.txt"
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("Initial commit\n")
        run_command("git add .")
        run_command('git commit -m "Initial commit"')

    today = datetime.now()-timedelta(days=11)  # Start from yesterday to avoid same-day commits

    print(f"Generating commits for the last {DAYS_TO_FILL} days...")
    
    # Loop backwards from yesterday to 7 days ago
    for i in range(1, DAYS_TO_FILL + 1):
        commit_date = today - timedelta(days=i)
        # Format: YYYY-MM-DD HH:MM:SS
        date_str = commit_date.strftime("%Y-%m-%d %H:%M:%S")
        
        # Randomize commits per day (1 to 5) for natural look
        commits_today = random.randint(1, 5)
        
        for c in range(commits_today):
            # Append content to file to create a diff
            with open(filename, "a") as f:
                f.write(f"Activity entry for {date_str} - commit {c+1}\n")
            
            run_command("git add .")
            
            # Critical: Set both author and committer dates
            cmd = f'git commit -m "Backdated activity: {date_str}" --date="{date_str}"'
            # Use environment variables for robustness across OS
            env = os.environ.copy()
            env["GIT_AUTHOR_DATE"] = date_str
            env["GIT_COMMITTER_DATE"] = date_str
            subprocess.run(cmd, shell=True, check=True, env=env)
            
        print(f"  -> Created {commits_today} commits for {commit_date.strftime('%Y-%m-%d')}")

    # Push to remote
    if REPO_URL != "https://github.com/Chirag-Kathuria-009/JobApplicationAutomation.git":
        print("Pushing to remote repository...")
        try:
            run_command(f"git remote add origin {REPO_URL}")
        except subprocess.CalledProcessError:
            # Remote might already exist
            run_command(f"git remote set-url origin {REPO_URL}")
        
        # Force push may be required if history was rewritten, 
        # but for simple appending, standard push works.
        run_command("git push -u origin main")
        print("Done! Check your GitHub graph in a few minutes.")
    else:
        print("\n[!] REPO_URL not set. Skipping push. Commit locally and push manually.")

if __name__ == "__main__":
    main()