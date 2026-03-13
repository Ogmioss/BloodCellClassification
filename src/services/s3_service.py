"""
S3 Service

Single Responsibility: Handles S3-compatible (MinIO) object storage operations
for datasets and other files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError


class S3Service:
    """Service for S3-compatible (MinIO) storage operations."""

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ):
        self.endpoint_url = endpoint_url
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

    @staticmethod
    def from_config(config: Dict[str, Any]) -> "S3Service":
        """
        Create S3Service from configuration dictionary.

        Priority for credentials:
        1. Environment variables (AWS_ACCESS_KEY_ID, etc.)
        2. conf.yaml minio section
        """
        minio_config = config.get("minio", {})

        endpoint_url = os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL",
            minio_config.get("endpoint_url", "http://localhost:9000"),
        )
        access_key = os.environ.get(
            "AWS_ACCESS_KEY_ID",
            minio_config.get("access_key", "minio"),
        )
        secret_key = os.environ.get(
            "AWS_SECRET_ACCESS_KEY",
            minio_config.get("secret_key", "minio123"),
        )

        return S3Service(
            endpoint_url=endpoint_url,
            access_key=access_key,
            secret_key=secret_key,
        )

    def bucket_exists(self, bucket: str) -> bool:
        """Check if a bucket exists."""
        try:
            self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError:
            return False

    def list_objects(self, bucket: str, prefix: str = "") -> list[dict]:
        """List objects in a bucket with optional prefix."""
        result = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                result.append({"key": obj["Key"], "size": obj["Size"]})
        return result

    def upload_directory(
        self,
        local_dir: Path,
        bucket: str,
        prefix: str = "",
    ) -> int:
        """
        Upload a local directory recursively to S3.

        Args:
            local_dir: Local directory path
            bucket: Target S3 bucket
            prefix: Key prefix in the bucket

        Returns:
            Number of files uploaded
        """
        local_dir = Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {local_dir}")

        count = 0
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(local_dir)
            key = f"{prefix}/{relative}" if prefix else str(relative)
            # Use forward slashes for S3 keys
            key = key.replace("\\", "/")
            self._client.upload_file(str(file_path), bucket, key)
            count += 1

        return count

    def download_directory(
        self,
        bucket: str,
        prefix: str,
        local_dir: Path,
    ) -> int:
        """
        Download objects from S3 to a local directory.

        Args:
            bucket: Source S3 bucket
            prefix: Key prefix to download
            local_dir: Local target directory

        Returns:
            Number of files downloaded
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        objects = self.list_objects(bucket, prefix)
        count = 0
        for obj in objects:
            key = obj["key"]
            # Strip the prefix to get relative path
            relative = key[len(prefix):].lstrip("/")
            if not relative:
                continue
            target = local_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket, key, str(target))
            count += 1

        return count

    def dataset_exists(self, bucket: str, prefix: str) -> bool:
        """Check if a dataset exists in S3 (has at least one object)."""
        objects = self.list_objects(bucket, prefix)
        return len(objects) > 0

    def ensure_dataset_synced(
        self,
        local_path: Path,
        bucket: str,
        prefix: str,
    ) -> None:
        """
        Ensure dataset is synced to S3.
        Uploads from local if S3 is empty, skips if already present.

        Args:
            local_path: Local dataset directory
            bucket: Target bucket
            prefix: Key prefix for the dataset
        """
        if self.dataset_exists(bucket, prefix):
            print(f"Dataset already exists in s3://{bucket}/{prefix}, skipping upload.")
            return

        local_path = Path(local_path)
        if not local_path.is_dir():
            print(f"Local dataset not found at {local_path}, skipping S3 sync.")
            return

        print(f"Uploading dataset {local_path} -> s3://{bucket}/{prefix} ...")
        count = self.upload_directory(local_path, bucket, prefix)
        print(f"Uploaded {count} files to s3://{bucket}/{prefix}")

    def get_bucket_stats(self, bucket: str) -> Optional[dict]:
        """Get bucket statistics (object count and total size)."""
        if not self.bucket_exists(bucket):
            return None

        objects = self.list_objects(bucket)
        total_size = sum(obj["size"] for obj in objects)
        return {
            "bucket": bucket,
            "object_count": len(objects),
            "total_size_bytes": total_size,
        }
