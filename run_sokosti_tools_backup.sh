#!/usr/bin/env bash
set -e  # Stop immediately if any command fails

# Backs up the Sokosti_Tools GitHub repo into the 'sokosti_tools/' subfolder
# of the 'development' branch on the Aalto GitLab. It merges with (never
# overwrites) the 'measurement_archive/' backup that lives on the same
# branch, so both backup scripts can be run in any order.

# --- Logging: everything printed below is also appended to this file ---
LOG_DIR="${LOG_DIR:-$HOME/repos/Sokosti_tools/git_backup_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sokosti_tools_backup_$(date +'%Y-%m-%d_%H-%M-%S').log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Backup started: $(date '+%Y-%m-%d %H:%M:%S') ==="

UNIV_REPO="git@version.aalto.fi:bare/sokosti_emg_imu_wearable_system.git"
GH_REPO="git@github.com:riderson-1/Sokosti_Tools.git"

MAX_FILE_SIZE="100M"   # files larger than this are NOT backed up (find's M = MiB)

# Do not download heavy LFS files from GitHub
export GIT_LFS_SKIP_SMUDGE=1

# Setup temp directory
TEMP_DIR=$(mktemp -d -t git-backup-XXXXXX)
cleanup() {
    status=$?
    rm -rf "$TEMP_DIR"
    echo "=== Backup finished with exit code $status at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo -e "\n--> Temp files deleted. Press [ENTER] to close..."
    read -r
}
trap cleanup EXIT

# 1. Start from the current state of the Aalto 'development' branch, so the
#    other backup (measurement_archive/) on the same branch is preserved.
echo "--> Checking Aalto 'development' branch..."
if git ls-remote --exit-code --heads "$UNIV_REPO" development >/dev/null 2>&1; then
    echo "    'development' exists on remote, updating it"
    git clone --depth 1 -b development "$UNIV_REPO" "$TEMP_DIR/backup"
else
    echo "    'development' does not exist on remote yet, creating it"
    mkdir -p "$TEMP_DIR/backup"
    cd "$TEMP_DIR/backup"
    git init -b development
    git remote add origin "$UNIV_REPO"
fi

# 2. Clone the GitHub repo (fast & light) into a staging folder
echo "--> Downloading latest files from GitHub..."
git clone --depth 1 "$GH_REPO" "$TEMP_DIR/staging"

# 3. Strip Git metadata, LFS config, LFS pointer stubs and large files
STAGING_DIR="$TEMP_DIR/staging"
rm -rf "$STAGING_DIR/.git"
find "$STAGING_DIR" -name .gitattributes -delete

# LFS pointer stubs (tiny text files whose real content isn't here)
find "$STAGING_DIR" -type f -size -300c \
    -exec grep -q '^version https://git-lfs.github.com/spec' {} \; \
    -printf '  skip LFS pointer: %P\n' -delete

# Anything larger than MAX_FILE_SIZE
find "$STAGING_DIR" -type f -size +"$MAX_FILE_SIZE" \
    -printf '  skip large file:  %P (%s bytes)\n' -delete

# 4. Replace only our own subfolder in the backup tree
echo "--> Merging into 'sokosti_tools/' subfolder..."
rm -rf "$TEMP_DIR/backup/sokosti_tools"
mv "$STAGING_DIR" "$TEMP_DIR/backup/sokosti_tools"

# 5. Commit the changes
cd "$TEMP_DIR/backup"
git config user.name "Backup Script"
git config user.email "backup@aalto.fi"

# Disable the LFS pre-push hook, nothing here is LFS anymore
git config core.hooksPath /dev/null

git add -A .
if git diff --cached --quiet; then
    echo "    No changes since the last backup, nothing to push."
    echo "=== BACKUP SUCCESSFUL (nothing new) ==="
    exit 0
fi
git commit -m "Backup: Update sokosti_tools $(date +'%Y-%m-%d')"

# 6. Push to the 'development' branch (normal push, never force:
#    the measurement_archive/ backup lives on the same branch)
echo "--> Backup size: $(du -sh "$TEMP_DIR/backup" | cut -f1)"
echo "--> Uploading backup to GitLab..."
git push -u origin development

echo "=== BACKUP SUCCESSFUL ==="
