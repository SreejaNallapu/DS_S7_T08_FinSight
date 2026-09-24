"""S3 transfers using boto3's standard environment/profile/role credentials."""

from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def upload_file(local_path: str | Path, bucket: str, key: str, *, client=None) -> None:
    """Upload a file; propagate credential, permission, and transfer errors."""
    path = Path(local_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    s3 = client if client is not None else boto3.client("s3")
    s3.upload_file(str(path), bucket, key)


def download_file(bucket: str, key: str, local_path: str | Path, *, client=None) -> Path:
    """Download to a local path, creating its parent directory as needed."""
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    s3 = client if client is not None else boto3.client("s3")
    s3.download_file(bucket, key, str(path))
    return path


def object_exists(bucket: str, key: str, *, client=None) -> bool:
    """Return False only for a missing object; propagate 403 and other errors."""
    s3 = client if client is not None else boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return True
