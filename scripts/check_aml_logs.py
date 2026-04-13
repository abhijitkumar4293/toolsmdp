"""Fetch and print stdout/stderr logs from an AML job."""
import sys
import yaml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

job_name = sys.argv[1] if len(sys.argv) > 1 else "olden_pizza_l8bjtcmptb"

with open("aml.yaml") as f:
    cfg = yaml.safe_load(f)

ml = MLClient(DefaultAzureCredential(), cfg["subscription_id"], cfg["resource_group"], cfg["workspace_name"])
job = ml.jobs.get(job_name)
print(f"Job: {job_name} | Status: {job.status}")

# Download just the logs (user_logs/std_log.txt)
import tempfile, os
with tempfile.TemporaryDirectory() as tmpdir:
    ml.jobs.download(job_name, download_path=tmpdir, output_name="default")
    # Check for common log locations
    for root, dirs, files in os.walk(tmpdir):
        for f in files:
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, tmpdir)
            size = os.path.getsize(fpath)
            print(f"  {rel} ({size:,} bytes)")
