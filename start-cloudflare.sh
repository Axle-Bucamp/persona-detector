#!/bin/bash
# Start Cloudflare tunnel for sharing
# Make sure cloudflared is installed and authenticated

echo "[INFO] Starting Cloudflare tunnel..."
echo "[INFO] Make sure you have:"
echo "  1. Installed cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
echo "  2. Authenticated: cloudflared tunnel login"
echo "  3. Created tunnel: cloudflared tunnel create semantic-detector"
echo "  4. Updated cloudflared.yml with your tunnel ID and domain"
echo ""

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "[ERROR] cloudflared is not installed!"
    echo "[INFO] Install from: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
    exit 1
fi

# Start tunnel
cloudflared tunnel --config cloudflared.yml run

