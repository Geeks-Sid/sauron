# Docker Usage with E: Drive

This guide explains how to run the AEGIS Docker container with your E: drive mounted.

## Quick Start

### Option 1: Using the Helper Script (Recommended)

**For WSL/Linux:**
```bash
./run_docker.sh
```

**For Windows (Command Prompt/PowerShell):**
```cmd
run_docker.bat
```

### Option 2: Using Docker Compose

```bash
docker-compose up -d
docker-compose exec aegis /bin/bash
```

To stop:
```bash
docker-compose down
```

### Option 3: Manual Docker Run

**For WSL:**
```bash
docker run -it --gpus all -v /mnt/e:/data aegis /bin/bash
```

**For Windows (native Docker Desktop):**
```bash
docker run -it --gpus all -v E:/:/data aegis /bin/bash
```

## Mount Points

When you run the container, your E: drive will be accessible at `/data` inside the container:

- **Host E: drive** → **Container `/data`**
- **Host `./results`** → **Container `/app/results`**
- **Host `./Data`** → **Container `/app/Data`**

## Example Usage

Once inside the container, you can access your E: drive data:

```bash
# List contents of E: drive
ls /data

# Access specific files
cat /data/myfile.txt

# Run AEGIS commands with data from E: drive
aegis --data_root_dir /data/features_uni_v2 ...
```

## Building the Image

If you need to rebuild the Docker image:

```bash
docker build -t aegis .
```

## Troubleshooting

### Permission Issues
If you encounter permission issues accessing files on E: drive:
- Ensure Docker has access to the E: drive in Docker Desktop settings
- On WSL, you may need to mount the drive: `sudo mount -t drvfs E: /mnt/e`

### GPU Not Available
Make sure you have:
- NVIDIA Docker runtime installed
- GPU drivers installed
- Docker Desktop with GPU support enabled (if using Docker Desktop)

### Path Issues
- **WSL**: Use `/mnt/e` format
- **Windows Native Docker**: Use `E:/` format
- The helper scripts auto-detect your environment

## Customizing Mount Points

Edit `docker-compose.yml` or the helper scripts to change mount points:

```yaml
volumes:
  - /mnt/e:/data:rw  # Change /data to your preferred path
  - ./results:/app/results:rw
```

