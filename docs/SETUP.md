# SETUP — Day 1-2 procedure

Alternative bring-up procedure for CellProbe on a local consumer GPU, kept as a reference. The active workflow is [`SETUP_AWS.md`](SETUP_AWS.md). This local path runs the BioNeMo container on a Windows + WSL2 host with an RTX 4070 SUPER for development and validation; production-scale fine-tuning would still need a cloud A100 / H100.

## Hardware target

| | Local dev | Final runs |
|---|---|---|
| GPU | RTX 4070 SUPER (12 GB) — pipeline validation, smoke tests | A100 40 GB (Lambda Labs) — fine-tuning |
| OS | Windows 11 + WSL2 Ubuntu 24.04 | Lambda Labs preconfigured Ubuntu + Docker |
| Network access | Tailscale (SSH `gpu` host, no public exposure) | Direct SSH from Lambda |

## 1. WSL2 + Docker + NVIDIA Container Toolkit

Done once on the local PC. Steps executed inside WSL2 Ubuntu 24.04.

```bash
# Docker Engine via the official repo (avoids the older docker.io)
sudo apt-get update -qq
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo usermod -aG docker $USER     # re-login (or `newgrp docker`) to take effect

# NVIDIA Container Toolkit — exposes the GPU to containers
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Validation (must print the RTX 4070 SUPER UUID):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
```

## 2. NGC account and registry login

1. Create a free account at <https://ngc.nvidia.com> and generate an API key under **Setup → Generate API Key**.
2. Log Docker into `nvcr.io`. Username is the literal string `$oauthtoken`; the password is the API key:

   ```bash
   echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

Credentials are stored unencrypted in `~/.docker/config.json` — file mode is user-only on WSL2, so the threat model is local-only. For a stricter setup, configure a credential helper.

## 3. Pull BioNeMo Framework container

Pinned to a specific tag for reproducibility (avoid `:latest`, `:nightly`).

```bash
docker pull nvcr.io/nvidia/clara/bionemo-framework:2.7.1
```

Image size ≈ **55 GB**. Plan disk accordingly. Available tags can be listed via the Docker Registry v2 API:

```bash
TOKEN=$(curl -s -u "\$oauthtoken:$NGC_API_KEY" \
  'https://nvcr.io/proxy_auth?scope=repository:nvidia/clara/bionemo-framework:pull&service=nvcr.io' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')
curl -s -H "Authorization: Bearer $TOKEN" \
  https://nvcr.io/v2/nvidia/clara/bionemo-framework/tags/list | python3 -m json.tool
```

## 4. Cache layout

A host-side cache directory persists model checkpoints and datasets between container runs (otherwise `--rm` destroys them on exit):

```
~/bionemo-cache/      # mounted to /root/.cache/bionemo inside the container
~/bionemo-workspace/  # mounted to /workspace/host  — code, results, scripts
```

Standard container invocation:

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/bionemo-cache:/root/.cache/bionemo \
  -v ~/bionemo-workspace:/workspace/host \
  nvcr.io/nvidia/clara/bionemo-framework:2.7.1 \
  <command>
```

The `--ipc=host` and `--ulimit` flags are NVIDIA's recommendation for PyTorch shared-memory workloads.

## 5. Download Geneformer 10M checkpoint and small CELLxGENE test data

```bash
# Inside the container
download_bionemo_data geneformer/10M_241113:2.0 --source ngc
download_bionemo_data single_cell/testdata-20240506 --source ngc
```

Sizes: 126 MB and 434 MB respectively. Both land under `/root/.cache/bionemo/` and persist to the host via the volume mount.

Available Geneformer pretrained tags:

| Tag | Params | Date |
|-----|--------|------|
| `geneformer/10M_240530:2.0` | 10M | May 2024 |
| `geneformer/10M_241113:2.0` | 10M | Nov 2024 (latest 10M, used here) |
| `geneformer/106M_240530:2.0` | 106M | May 2024 (production-grade, planned A100 run) |

## 6. Known compat shim — checkpoint class rename

The released 10M_241113 checkpoint serializes a Fiddle reference to `bionemo.geneformer.api.BERTMLMLossWithReductionNoForward`, but BioNeMo 2.7.1 renamed this class to `BERTMLMLossWithReduction`. Loading the checkpoint without an alias fails with:

```
AttributeError: module 'bionemo.geneformer.api' has no attribute
'BERTMLMLossWithReductionNoForward'
```

Workaround: run `scripts/infer_wrapper.py` instead of `infer_geneformer`. The wrapper installs the alias before importing the script entrypoint and then forwards all CLI args.

## 7. Smoke test (validates the full chain)

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/bionemo-cache:/root/.cache/bionemo \
  -v ~/bionemo-workspace:/workspace/host \
  nvcr.io/nvidia/clara/bionemo-framework:2.7.1 bash -c '
CKPT_DIR=$(find /root/.cache/bionemo -type d -path "*geneformer_10M*.untar" | head -1)
DATA_DIR=$(find /root/.cache/bionemo -type d -path "*scdl/test" | head -1)
mkdir -p /workspace/host/smoke-test-results
python /workspace/host/infer_wrapper.py \
  --data-dir $DATA_DIR \
  --checkpoint-path $CKPT_DIR \
  --results-path /workspace/host/smoke-test-results \
  --precision bf16-mixed \
  --include-hiddens \
  --micro-batch-size 4 \
  --seq-length 2048 \
  --num-gpus 1
'
```

Expected output: `predictions__rank_0.pt` in the results dir, model log reporting 10 300 032 parameters, vocab size 25 472. The `--include-hiddens` flag dumps full hidden states — the resulting `.pt` is ~8 GB; turn it off for routine use.

## 8. Remote control from the Mac

Local dev runs on a Mac, GPU lives on the Windows PC. All `docker` calls are tunneled via SSH over Tailscale:

```bash
ssh gpu "docker run ..."
```

`~/.ssh/config` snippet (Mac side):

```ssh
Host gpu
    HostName <your-tailscale-ip>     # Tailscale CGNAT IP of the WSL2 host (100.64.0.0/10)
    User <your-wsl2-username>
    IdentityFile ~/.ssh/id_ed25519
```

After a Windows sleep, the SSH connection may stall. To recover:

1. Wake the PC.
2. Open WSL2 (`wsl` from Windows).
3. `sudo service ssh start`.
4. If the WSL2 IP changed, reset the `netsh portproxy` from PowerShell (admin).

## What's next

- Decide target disease (cardiomyopathy benchmark vs cancer / IBD for drug-discovery narrative) — see project README, Open decisions.
- Replace the toy CELLxGENE test data with the chosen disease dataset.
- Set up an A100 instance on Lambda Labs for the actual fine-tuning runs (12 GB on the 4070 SUPER is enough for smoke tests, not enough for 106M Geneformer fine-tuning at useful batch sizes).
