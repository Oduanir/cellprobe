# SETUP — AWS EC2 procedure

Reproducible environment for fine-tuning Geneformer with NVIDIA BioNeMo on AWS. This is the path a NVIDIA Solutions Architect would recommend to an EMEA biopharma customer: containerized BioNeMo on an Ampere-generation EC2 instance, data-resident in `eu-west-3` (Paris), pay-per-second compute.

For the original local-GPU procedure (RTX 4070 SUPER on WSL2), see [`SETUP.md`](SETUP.md).

## Why this stack

- **Region `eu-west-3` (Paris)** — EMEA data residency. The data we use here is public (CELLxGENE), but a real pharma customer needs EU residency for any patient-derived omics. Defending region choice is part of the SA pitch.
- **Instance `g6.xlarge`** — 1× NVIDIA L4 (24 GB VRAM, Ada Lovelace), 4 vCPU, 16 GB RAM, ~$0.805/h on-demand. **Note**: `g6.xlarge` (A10G) is **not offered in `eu-west-3` as of 2026-05** — only `g4dn` (T4, Turing) and `g6` (L4, Ada Lovelace) are. L4 is the newer of the two, cheaper than A10G would have been, and ships fast FP8/INT8 tensor cores — strictly an upgrade. Step up to `g6.2xlarge` (same GPU, more RAM/CPU) only if data loading bottlenecks. Check availability in your region with `aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g6.xlarge,g6.xlarge,g4dn.xlarge`.
- **AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch — Ubuntu 22.04** — AWS-maintained AMI with the NVIDIA driver, CUDA, Docker, and the NVIDIA Container Toolkit pre-installed. Saves a full day of setup vs a vanilla Ubuntu AMI.
- **Storage `gp3` 100 GB** — BioNeMo container (~55 GB) + cache (~10 GB) + datasets (~5 GB) + working room. gp3 is the default modern EBS volume, 3000 IOPS / 125 MB/s baseline, plenty for our I/O.
- **Stop-when-idle billing** — `stop` (not `terminate`) the instance between sessions. Compute billing pauses; EBS storage continues at ~$8/month for 100 GB. Cheaper than keeping the instance running.

## 1. Quota check and request

> ⚠️ **Read this first.** AWS accounts default to 0 vCPU for GPU instance families. A "Service Quota Increase" request is required before any G-family instance can start. With **admin IAM rights this is a self-serve form but AWS Support still approves it** (typically 1–24 hours; instant for small asks). Submit the request on day 0 so the quota is open by the time you need it.

```bash
# Configure AWS CLI once (use admin credentials)
aws configure set region eu-west-3

# Check current G/VT quota — "L-DB2E81BA" is "Running On-Demand G and VT instances", measured in vCPUs
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA

# If the "Value" is 0 or below 4, request 4 (= one g6.xlarge):
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 4
```

Track the request in the AWS console → **Service Quotas → EC2 → Running On-Demand G and VT instances**.

**Fallback if G quota is stuck**: `g4dn.xlarge` (1× T4 16 GB, Turing-generation, $0.526/h Paris) is in a different quota family (often already open at 4–8 vCPUs by default). Same workflow, weaker NVIDIA optics for the blog. T4 16 GB is still enough for Geneformer 10M.

## 2. Launch the instance

### 2.1 SSH key pair

```bash
# Generate a dedicated key for this project (keep it local; never commit)
aws ec2 create-key-pair \
  --key-name rtx-code-key \
  --key-type ed25519 \
  --query 'KeyMaterial' \
  --output text > ~/.ssh/rtx-code-key.pem
chmod 600 ~/.ssh/rtx-code-key.pem
```

### 2.2 Security group (SSH from your IP only)

```bash
# Find your public IP
MYIP=$(curl -s https://checkip.amazonaws.com)/32

aws ec2 create-security-group \
  --group-name rtx-code-sg \
  --description "SSH to rtx-code dev instance"

aws ec2 authorize-security-group-ingress \
  --group-name rtx-code-sg \
  --protocol tcp --port 22 --cidr "$MYIP"
```

### 2.3 Pick the AMI

```bash
# Latest AWS Deep Learning AMI with PyTorch + NVIDIA driver, Ubuntu 22.04, x86_64
aws ec2 describe-images \
  --owners amazon \
  --filters 'Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch* (Ubuntu 22.04)*' \
            'Name=architecture,Values=x86_64' \
            'Name=state,Values=available' \
  --query 'sort_by(Images, &CreationDate)[-1].[ImageId,Name]' \
  --output text
```

Note the `ami-xxxxxxxx` ID returned.

### 2.4 Launch

```bash
aws ec2 run-instances \
  --image-id ami-XXXXXXXXXXXX \
  --instance-type g6.xlarge \
  --key-name rtx-code-key \
  --security-groups rtx-code-sg \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=rtx-code-dev},{Key=Project,Value=rtx-code}]' \
  --query 'Instances[0].InstanceId' \
  --output text
```

Wait for it to come up and fetch the public DNS:

```bash
INSTANCE_ID=i-XXXXXXXX
aws ec2 wait instance-running --instance-ids $INSTANCE_ID
aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

### 2.5 SSH

```bash
ssh -i ~/.ssh/rtx-code-key.pem ubuntu@<public-dns-from-above>
```

Add this to `~/.ssh/config` for convenience:

```ssh
Host aws-rtx
    HostName <public-dns>
    User ubuntu
    IdentityFile ~/.ssh/rtx-code-key.pem
```

Then just `ssh aws-rtx`.

## 3. Validate the GPU

```bash
# On the instance — the Deep Learning AMI has the driver + container toolkit pre-installed
nvidia-smi -L
# expected: GPU 0: NVIDIA A10G (UUID: GPU-...)

# Confirm docker + GPU passthrough work
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L
```

If the second command fails: `sudo systemctl restart docker` then retry.

## 4. NGC login and BioNeMo pull

NGC API key creation: see [`SETUP.md` §2](SETUP.md). Then:

```bash
# On the EC2 instance
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
docker pull nvcr.io/nvidia/clara/bionemo-framework:2.7.1
```

The pull is ~55 GB and runs at full ENI bandwidth on g6 (≈10 Gbps) — typically 5–10 min, occupies ~55 GB on the EBS root volume.

> 💡 **Disk sizing**: the default 100 GB EBS leaves only ~15–20 GB free after the BioNeMo image. Plan to resize to 200 GB before fine-tuning runs:
>
> ```bash
> # From the Mac (admin profile)
> VOL=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
>        --query 'Reservations[0].Instances[0].BlockDeviceMappings[0].Ebs.VolumeId' --output text)
> aws ec2 modify-volume --volume-id $VOL --size 200
>
> # On the EC2 instance, after AWS confirms "optimizing" state
> ROOT=$(findmnt -no SOURCE /); DISK=$(lsblk -no PKNAME $ROOT)
> sudo growpart /dev/$DISK 1 && sudo resize2fs $ROOT
> ```
>
> Cost: +$10/month for the extra 100 GB, prorated. The `g6.xlarge` AMI also exposes a separate ~232 GB ephemeral NVMe at `/opt/dlami/nvme` — useful for tokenization scratch (but ephemeral: data is lost on instance stop/start).

## 5. Repo + workspace layout on the instance

```bash
# On the EC2 instance
git clone https://github.com/<your-username>/rtx-code.git ~/rtx-code

mkdir -p ~/bionemo-cache ~/bionemo-workspace
ln -sf ~/rtx-code ~/bionemo-workspace/rtx-code
```

The cache directory persists across container runs (and across instance stop/start, because it sits on the EBS root volume, not container ephemeral storage).

## 6. Re-download datasets from CELLxGENE

The datasets we already pulled on the local PC are not accessible from the EC2 instance, but the source-of-truth is CELLxGENE Census — we just re-run our own script:

```bash
# Inside the container, mounted on the workspace
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/bionemo-cache:/root/.cache/bionemo \
  -v ~/rtx-code:/workspace/rtx-code \
  -w /workspace/rtx-code \
  nvcr.io/nvidia/clara/bionemo-framework:2.7.1 \
  python -u -m src.data.download --config configs/diseases.yaml --out data/
```

Expected: ~30–60 min total for the 3 diseases (~1.5M cells, ~5 GB on disk after gzip).

## 7. Smoke test

Same procedure as the local one. Pull the Geneformer 10M checkpoint and the CELLxGENE toy test data, then run the inference wrapper:

> ⚠️ **Test data version pitfall**: `single_cell/testdata-20240506` (v1.0, 434 MB) contains the **old** layout (`gene_expression_data.npy`, `_ind.npy`, `_ptr.npy` under `cellxgene_2023-12-15_small/processed_data/`). The `infer_geneformer` entrypoint expects the **v2.0 SCDL** layout (`data.npy`, `row_ptr.npy`, ... under `cellxgene_2023-12-15_small_processed_scdl/`). On first run, BioNeMo silently auto-downloads `singlecell-testdata:2.0` (224 MB) — just point `--data-dir` at `*scdl/test`, not at `*processed_data/test`. The `find ... -path "*scdl/test"` pattern below picks the right one.

```bash
docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/bionemo-cache:/root/.cache/bionemo \
  -v ~/rtx-code:/workspace/rtx-code \
  nvcr.io/nvidia/clara/bionemo-framework:2.7.1 bash -c '
download_bionemo_data geneformer/10M_241113:2.0 --source ngc
download_bionemo_data single_cell/testdata-20240506 --source ngc

CKPT_DIR=$(find /root/.cache/bionemo -type d -path "*geneformer_10M*.untar" | head -1)
DATA_DIR=$(find /root/.cache/bionemo -type d -path "*scdl/test" | head -1)
mkdir -p /workspace/rtx-code/results/smoke-test
python /workspace/rtx-code/scripts/infer_wrapper.py \
  --data-dir $DATA_DIR \
  --checkpoint-path $CKPT_DIR \
  --results-path /workspace/rtx-code/results/smoke-test \
  --precision bf16-mixed \
  --include-hiddens \
  --micro-batch-size 4 \
  --seq-length 2048 \
  --num-gpus 1
'
```

Expected output: `predictions__rank_0.pt`, model reports 10 300 032 parameters, vocab size 25 472. If this matches the local run from `SETUP.md`, the AWS environment is validated end-to-end.

## 8. Compat shim

The `BERTMLMLossWithReductionNoForward` → `BERTMLMLossWithReduction` rename in BioNeMo 2.7.1 still applies — `scripts/infer_wrapper.py` handles it. See [`SETUP.md` §6](SETUP.md) for the rationale.

## 9. Cost discipline

```bash
# Stop the instance when you're done for the session (preserves disk, pauses compute billing)
aws ec2 stop-instances --instance-ids $INSTANCE_ID

# Start again later (public DNS changes — re-fetch it; use an Elastic IP if you want a stable address)
aws ec2 start-instances --instance-ids $INSTANCE_ID
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Definitive cleanup at end-of-project (deletes the EBS volume — back up `data/` and `results/` first)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
aws ec2 delete-security-group --group-name rtx-code-sg
aws ec2 delete-key-pair --key-name rtx-code-key
rm ~/.ssh/rtx-code-key.pem
```

Estimated total project cost (Paris, on-demand):

| Item | Hours / GB | Unit cost | Total |
|---|---|---|---|
| `g6.xlarge` compute (dev + smoke + 3 fine-tunes + perturbation) | ~20 h | $0.805 / h | ~$16 |
| `gp3` EBS storage (100 GB, 2 weeks) | 50 GB-mo | $0.0928 / GB-mo | ~$5 |
| Data egress (only if you scp results back to the Mac) | ~1 GB | $0.085 / GB | ~$0 |
| **Project total estimate** | | | **~$20** |

Well within the $50–100 envelope.

## 10. What's next

- Wire `src/data/preprocess.py` (QC + tokenization + SCDL conversion) — same as the original plan, just running on AWS.
- Launch fine-tuning runs on the same `g6.xlarge` (or step up to `g6.2xlarge` if the data loader is CPU-bound).
- In silico perturbation analysis + OpenTargets/DGIdb validation.
- Blog post angle bonus: "deploying NVIDIA's BioNeMo stack on AWS — the customer reality for an EMEA pharma."
