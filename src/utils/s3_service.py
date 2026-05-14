"""S3 service for uploading and retrieving media files using aioboto3."""

import logging
from pathlib import Path
from typing import Optional

import aioboto3
from pydantic import BaseModel

from src.config.config import Settings

logger = logging.getLogger(__name__)


class S3Config(BaseModel):
    """Configuration for S3 service extracted from Settings."""

    endpoint: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_name: Optional[str] = None
    region: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        """Check if all required S3 configuration is present."""
        return all(
            [
                self.endpoint,
                self.access_key,
                self.secret_key,
                self.bucket_name,
                self.region,
            ]
        )


class S3Service:
    """Async S3 service for media file operations.

    Provides methods to upload files to S3 and generate public URLs.
    Assumes the S3 bucket is configured with public-read access for media files.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize S3 service with application settings.

        Args:
            settings: Application settings containing S3 configuration.
        """

        self.config = S3Config(
            endpoint=settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket_name=settings.s3_bucket_name,
            region=settings.s3_region,
        )
        self._session: Optional[aioboto3.Session] = None

    async def __aenter__(self) -> "S3Service":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize aioboto3 session and S3 client."""
        if not self.config.is_configured:
            logger.warning(
                "S3 service is not fully configured. Some operations may fail."
            )
            return

        self._session = aioboto3.Session(
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            region_name=self.config.region,
        )
        logger.info(
            "S3 service initialized",
            extra={
                "bucket": self.config.bucket_name,
                "endpoint": self.config.endpoint,
                "region": self.config.region,
            },
        )

    async def close(self) -> None:
        """Close S3 client and session."""
        if self._session:
            self._session = None
            logger.info("S3 service closed")

    async def upload_file(self, local_path: Path | str, object_key: str) -> str:
        """Upload a file to S3 bucket.

        Args:
            local_path: Local file path to upload.
            object_key: S3 object key (path within bucket).

        Returns:
            Public URL of the uploaded file.

        Raises:
            ValueError: If S3 service is not configured or file not found.
            Exception: If upload fails.
        """

        if not self.config.is_configured:
            raise ValueError(
                "S3 service is not configured. Set S3_* environment variables."
            )

        local_path = Path(local_path)
        if not local_path.exists():
            raise ValueError(f"File not found: {local_path}")

        logger.info(
            "Uploading file to S3",
            extra={
                "local_path": str(local_path),
                "object_key": object_key,
                "bucket": self.config.bucket_name,
            },
        )

        try:
            async with self._session.client(
                "s3", endpoint_url=self.config.endpoint
            ) as s3_client:
                with open(local_path, "rb") as file_obj:
                    await s3_client.upload_fileobj(
                        file_obj,
                        self.config.bucket_name,
                        object_key,
                    )

            # Generate permanent public URL
            url = self.get_file_url(object_key)
            logger.info(
                "File uploaded successfully",
                extra={"object_key": object_key, "url": url},
            )
            return url

        except Exception as e:
            logger.error(
                "Failed to upload file to S3",
                extra={
                    "local_path": str(local_path),
                    "object_key": object_key,
                    "error": str(e),
                },
            )
            raise

    def get_file_url(self, object_key: str) -> str:
        """Generate permanent public URL for an S3 object.

        Assumes the bucket is configured for public-read access.

        Args:
            object_key: S3 object key (path within bucket).

        Returns:
            Permanent public URL to the S3 object.
        """
        
        if not self.config.endpoint or not self.config.bucket_name:
            raise ValueError(
                "S3 endpoint and bucket name must be configured to generate URL."
            )

        # Remove trailing slash from endpoint
        endpoint = self.config.endpoint.rstrip("/")

        # Construct URL: {endpoint}/{bucket_name}/{object_key}
        # For AWS S3, the standard format is:
        # https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}
        # For custom S3-compatible storage (MinIO, Cloudflare R2, etc.):
        # {endpoint}/{bucket_name}/{object_key}
        if "amazonaws.com" in endpoint:
            # AWS S3 standard URL format
            url = f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{object_key}"
        else:
            # Custom S3-compatible endpoint
            url = f"{endpoint}/{self.config.bucket_name}/{object_key}"

        return url
