# Cloudflare Tunnel Setup

This guide explains how to set up Cloudflare Tunnel to share your Semantic Detector application publicly.

## Prerequisites

1. A Cloudflare account (free tier works)
2. A domain managed by Cloudflare (or use Cloudflare's free tunnel service)

## Installation

### 1. Install cloudflared

**Windows:**
```powershell
# Download from: https://github.com/cloudflare/cloudflared/releases
# Or use Chocolatey:
choco install cloudflared
```

**macOS:**
```bash
brew install cloudflared
```

**Linux:**
```bash
# Download latest release from: https://github.com/cloudflare/cloudflared/releases
# Or use package manager:
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### 2. Authenticate

```bash
cloudflared tunnel login
```

This will open a browser window to authenticate with Cloudflare.

### 3. Create a Tunnel

```bash
cloudflared tunnel create semantic-detector
```

This will create a tunnel and save the credentials file. Note the tunnel ID from the output.

### 4. Configure the Tunnel

Edit `cloudflared.yml` and update:
- Replace `<tunnel-id>` with your actual tunnel ID
- Replace `semantic-detector.your-domain.com` with your desired subdomain

### 5. Configure DNS (Optional)

If using a custom domain, add a CNAME record in Cloudflare DNS:
- Name: `semantic-detector` (or your subdomain)
- Target: `<tunnel-id>.cfargotunnel.com`

### 6. Start the Tunnel

**Option A: Using the script**
```bash
chmod +x start-cloudflare.sh
./start-cloudflare.sh
```

**Option B: Manual start**
```bash
cloudflared tunnel --config cloudflared.yml run
```

**Option C: Run as a service (Linux)**
```bash
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

## Quick Start (No Domain Required)

If you don't have a domain, Cloudflare can provide a free `.trycloudflare.com` URL:

```bash
cloudflared tunnel --url http://localhost:8000
```

This will give you a temporary URL like `https://random-name.trycloudflare.com` that you can share.

## Troubleshooting

- **Port already in use**: Make sure port 8000 is available
- **Tunnel not connecting**: Check that the app is running on `0.0.0.0:8000`
- **DNS issues**: Wait a few minutes for DNS propagation, or use the trycloudflare.com URL

## Security Notes

- The tunnel provides HTTPS automatically
- Access is public unless you add Cloudflare Access rules
- Consider adding authentication for production use

