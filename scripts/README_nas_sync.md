# NAS Periodic Sync (WSL -> SynologyDrive)

## Files
- `scripts/nas_sync_projects.conf`: source project -> target folder mapping.
- `scripts/sync_projects_to_nas.sh`: main sync script (uses `.gitignore` + excludes `.git`).
- `scripts/install_nas_sync_cron.sh`: install/update a cron job in WSL.

## One-time setup
```bash
chmod +x /root/Cyclone_next/scripts/sync_projects_to_nas.sh
chmod +x /root/Cyclone_next/scripts/install_nas_sync_cron.sh
```

## Manual run
```bash
# Preview changes only
/root/Cyclone_next/scripts/sync_projects_to_nas.sh --dry-run

# Real sync
/root/Cyclone_next/scripts/sync_projects_to_nas.sh

# Sync one project only
/root/Cyclone_next/scripts/sync_projects_to_nas.sh --project Cyclone_next
```

## Install periodic sync in WSL (cron)
```bash
# Default: every 6 hours
/root/Cyclone_next/scripts/install_nas_sync_cron.sh

# Example: every 2 hours
/root/Cyclone_next/scripts/install_nas_sync_cron.sh "0 */2 * * *"
```

Check:
```bash
crontab -l
tail -n 100 /root/Cyclone_next/.state/nas-sync/cron.log
```

## Optional: trigger from Windows Task Scheduler (recommended)
If WSL is not always running, use Windows Task Scheduler to run this command periodically:

```powershell
wsl.exe -d Ubuntu -u root -- bash -lc '/root/Cyclone_next/scripts/sync_projects_to_nas.sh >> /root/Cyclone_next/.state/nas-sync/task_scheduler.log 2>&1'
```

This ensures sync still runs even when no active WSL terminal session is open.
