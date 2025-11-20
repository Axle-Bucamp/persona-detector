# Sharing Setup Guide

This document explains how to share your Semantic Detector application publicly.

## Features Added

✅ **0.0.0.0 Host Binding**: Application binds to `0.0.0.0` to accept connections from any interface  
✅ **Cloudflare Tunnel Support**: Easy sharing via Cloudflare tunnels  
✅ **App Icon**: SVG icon added and configured  
✅ **Sharable Media**: Open Graph and Twitter Card meta tags for social sharing  
✅ **Ollama Auto-Download**: Docker Compose automatically downloads the default embedding model

## Quick Start - Cloudflare Tunnel (No Domain Required)

The easiest way to share your app:

```bash
# Install cloudflared (if not already installed)
# Windows: Download from https://github.com/cloudflare/cloudflared/releases
# Mac: brew install cloudflared
# Linux: See README_CLOUDFLARE.md

# Quick tunnel (temporary URL)
make cloudflare-quick
# OR
cloudflared tunnel --url http://localhost:8000
```

This will give you a URL like `https://random-name.trycloudflare.com` that you can share immediately.

## Permanent Cloudflare Tunnel Setup

For a permanent tunnel with your own domain:

1. **Install cloudflared** (see README_CLOUDFLARE.md)
2. **Authenticate**: `cloudflared tunnel login`
3. **Create tunnel**: `cloudflared tunnel create semantic-detector`
4. **Edit `cloudflared.yml`** with your tunnel ID and domain
5. **Start**: `make cloudflare` or `./start-cloudflare.sh`

See `README_CLOUDFLARE.md` for detailed instructions.

## Docker Setup with Ollama

The Docker Compose setup now automatically downloads the default embedding model:

```bash
# Start everything
make docker-up

# Or in detached mode
make docker-up-detached

# Stop
make docker-down
```

The Ollama service will:
1. Start the Ollama server
2. Wait for it to be ready
3. Download `nomic-embed-text` model automatically
4. List available models

## App Icon & Social Sharing

- **Icon**: `/public/icon.svg` - Fingerprint icon in brand color (#6366f1)
- **Social Image**: `/public/sharable.PNG` - Used for Open Graph and Twitter Cards
- **Meta Tags**: Configured in `index.html` for proper social media previews

The icon and social image are automatically served from `/public/` when the app is running.

## Verification

1. **Check app is accessible**: Open `http://localhost:8000` (or your tunnel URL)
2. **Check icon**: Visit `http://localhost:8000/public/icon.svg`
3. **Check social image**: Visit `http://localhost:8000/public/sharable.PNG`
4. **Test sharing**: Use a tool like https://www.opengraph.xyz/ to preview how your link appears

## Troubleshooting

### Port Already in Use
```bash
# Use a different port
python run_web.py 8080
# Then update tunnel: cloudflared tunnel --url http://localhost:8080
```

### Ollama Model Not Downloading
- Check Ollama logs: `docker-compose logs ollama`
- Manually pull: `docker-compose exec ollama ollama pull nomic-embed-text`

### Cloudflare Tunnel Not Connecting
- Ensure app is running on `0.0.0.0:8000` (default)
- Check firewall settings
- Verify tunnel credentials are correct

## Security Notes

- Cloudflare tunnels provide HTTPS automatically
- Access is public unless you add Cloudflare Access rules
- Consider adding authentication for production use
- Don't expose sensitive data without proper security measures

