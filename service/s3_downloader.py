"""
Service Step 1: Download chunks from S3.
"""
import os
from pathlib import Path

try:
    import boto3
except ImportError:
    boto3 = None


def download_from_s3(s3_uri: str, local_dest: Path) -> Path:
    """
    Download a folder from an S3 bucket to a local destination directory.
    Uses boto3.
    """
    if boto3 is None:
        raise ImportError("boto3 is required to use the s3_downloader. Please install it.")

    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri must start with s3://")

    # Parse s3 uri
    uri_without_scheme = s3_uri[5:]
    bucket_name = uri_without_scheme.split("/")[0]
    prefix = uri_without_scheme[len(bucket_name) + 1:]

    # Ensure suffix slash for prefix semantics
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    print(f"[Service] Downloading from bucket: {bucket_name}, prefix: {prefix}")
    print(f"[Service] Destination: {local_dest}")

    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    count = 0
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):  # Skip "directories"
                continue
            
            relative_key = key[len(prefix):]
            local_file_path = local_dest / relative_key

            # Create folders if needed
            local_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Downloading {key} -> {local_file_path}")
            s3_client.download_file(bucket_name, key, str(local_file_path))
            count += 1

    print(f"[Service] Successfully downloaded {count} files.")
    return local_dest
