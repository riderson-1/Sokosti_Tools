#!/usr/bin/env bash
set -e  # Stop immediately if any command fails

# --- Logging: everything printed below is also appended to this file ---
LOG_DIR="${LOG_DIR:-$HOME/repos/Sokosti_tools/git_backup_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sokosti_backup_$(date +'%Y-%m-%d_%H-%M-%S').log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Backup started: $(date '+%Y-%m-%d %H:%M:%S') ==="

UNIV_REPO="git@version.aalto.fi:bare/sokosti_emg_imu_wearable_system.git"
GH_ARCHIVE="git@github.com:riderson-1/Sokosti_Measurement_Archive.git"

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
#    other backup (sokosti_tools/) on the same branch is preserved.
echo "--> Checking Aalto 'development' branch..."
if git ls-remote --exit-code --heads "$UNIV_REPO" development >/dev/null 2>&1; then
    echo "    'development' exists on remote, updating it"
    git clone --depth 1 -b development "$UNIV_REPO" "$TEMP_DIR/backup"
else
    echo "    'development' does not exist on remote yet, creating it"
    mkdir -p "$TEMP_DIR/backup"
    cd "$TEMP_DIR/backup"
    git init -b development
fi

# 2. Clone ONLY the GitHub archive (fast & light) into a staging folder
echo "--> Downloading latest files from GitHub..."
git clone --depth 1 "$GH_ARCHIVE" "$TEMP_DIR/staging"

# 3. Strip Git metadata, LFS config, LFS pointer stubs and large files
ARCHIVE_DIR="$TEMP_DIR/staging"
rm -rf "$ARCHIVE_DIR/.git"
find "$ARCHIVE_DIR" -name .gitattributes -delete

# LFS pointer stubs (tiny text files whose real content isn't here)
find "$ARCHIVE_DIR" -type f -size -300c \
    -exec grep -q '^version https://git-lfs.github.com/spec' {} \; \
    -printf '  skip LFS pointer: %P\n' -delete

# Anything larger than MAX_FILE_SIZE
find "$ARCHIVE_DIR" -type f -size +"$MAX_FILE_SIZE" \
    -printf '  skip large file:  %P (%s bytes)\n' -delete

# 4. Replace only our own subfolder in the backup tree
echo "--> Merging into 'measurement_archive/' subfolder..."
rm -rf "$TEMP_DIR/backup/measurement_archive"
mv "$ARCHIVE_DIR" "$TEMP_DIR/backup/measurement_archive"

# 5. Turn this local folder into a clean Git repository
cd "$TEMP_DIR/backup"
git config user.name "Backup Script"
git config user.email "backup@aalto.fi"

# Disable the LFS pre-push hook, nothing here is LFS anymore
git config core.hooksPath /dev/null

# 6. Commit the files
git add -A .
if git diff --cached --quiet; then
    echo "    No changes since the last backup, nothing to push."
    echo "=== BACKUP SUCCESSFUL (nothing new) ==="
    exit 0
fi
git commit -m "Backup: Update measurement archive $(date +'%Y-%m-%d')"

# 7. Push to the 'development' branch (normal push, never force:
#    the sokosti_tools/ backup lives on the same branch)
echo "--> Backup size: $(du -sh "$TEMP_DIR/backup" | cut -f1)"
echo "--> Uploading backup to GitLab..."
# The clone already set 'origin' when the branch existed; (re)add it for the fresh-init case
git remote remove origin 2>/dev/null || true
git remote add origin "$UNIV_REPO"
git push -u origin development

echo "=== BACKUP SUCCESSFUL ==="
