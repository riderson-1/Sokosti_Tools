#!/usr/bin/env bash
set -e  # Stop immediately if any command fails

# Backs up the two embedded Sokosti GitHub repos into subfolders of the
# 'development' branch on the Aalto GitLab:
#   - riderson-1/Sokosti               -> sokosti/            (device tree / board defs)
#   - riderson-1/Sokosti_System_Testing -> sokosti_system_integration_test/
# It merges with (never overwrites) the 'sokosti_tools/' and
# 'measurement_archive/' backups that live on the same branch, so all
# backup scripts can be run in any order.

# --- Logging: everything printed below is also appended to this file ---
LOG_DIR="${LOG_DIR:-$HOME/repos/Sokosti_tools/git_backup_logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sokosti_embedded_backup_$(date +'%Y-%m-%d_%H-%M-%S').log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Backup started: $(date '+%Y-%m-%d %H:%M:%S') ==="

UNIV_REPO="git@version.aalto.fi:bare/sokosti_emg_imu_wearable_system.git"
GH_BOARD_REPO="git@github.com:riderson-1/Sokosti.git"
GH_APP_REPO="git@github.com:riderson-1/Sokosti_System_Testing.git"

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
#    other backups (sokosti_tools/, measurement_archive/) on the same branch
#    are preserved.
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

# 2. Clone each GitHub repo (fast & light) into a staging folder and strip
#    Git metadata, LFS config, LFS pointer stubs and large files
stage_repo() {
    local gh_repo="$1" dest_subfolder="$2"
    echo "--> Downloading latest files from GitHub: $gh_repo"
    git clone --depth 1 "$gh_repo" "$TEMP_DIR/staging"

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

    # Replace only our own subfolder in the backup tree
    echo "    Merging into '$dest_subfolder/' subfolder..."
    rm -rf "$TEMP_DIR/backup/$dest_subfolder"
    mv "$STAGING_DIR" "$TEMP_DIR/backup/$dest_subfolder"
}

stage_repo "$GH_BOARD_REPO" "sokosti"
stage_repo "$GH_APP_REPO" "sokosti_system_integration_test"

# 3. Commit the changes
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
git commit -m "Backup: Update sokosti embedded repos $(date +'%Y-%m-%d')"

# 4. Push to the 'development' branch (normal push, never force:
#    the sokosti_tools/ and measurement_archive/ backups live on the same branch)
echo "--> Backup size: $(du -sh "$TEMP_DIR/backup" | cut -f1)"
echo "--> Uploading backup to GitLab..."
git push -u origin development

echo "=== BACKUP SUCCESSFUL ==="
