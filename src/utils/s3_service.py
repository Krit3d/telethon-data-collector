"""S3 service for uploading and retrieving media files using aioboto3."""

import logging
from pathlib import Path

import aioboto3
import aiofiles
from botocore.exceptions import ClientError
from pydantic import BaseModel

from src.config.config import Settings

logger = logging.getLogger(__name__)


class S3Config(BaseModel):
    """Configuration for S3 service extracted from Settings."""

    endpoint: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    bucket_name: str | None = None
    region: str | None = None

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
        self._session: aioboto3.Session | None = None

    async def __aenter__(self) -> "S3Service":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """Initialize aioboto3 session."""
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
        """Upload a file to S3 bucket using async file operations.

        Args:
            local_path: Local file path to upload.
            object_key: S3 object key (path within bucket).

        Returns:
            Public URL of the uploaded file.

        Raises:
            ValueError: If S3 service is not configured or file not found.
            ClientError: If S3 upload fails.
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
            # Create session - aioboto3.Session().client() returns an async context manager
            session = aioboto3.Session(
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
            )
            # type: ignore[# type: ignore] - aioboto3 client is an async context manager but Pylance cannot infer it
            async with session.client("s3", endpoint_url=self.config.endpoint) as s3:  # type: ignore[attr-defined]
                # Read file asynchronously using aiofiles
                async with aiofiles.open(local_path, "rb") as file_obj:
                    file_data = await file_obj.read()
                    # Upload raw bytes using put_object
                    await s3.put_object(
                        Bucket=self.config.bucket_name,
                        Key=object_key,
                        Body=file_data,
                    )

            # Generate permanent public URL
            url = self.get_file_url(object_key)
            logger.info(
                "File uploaded successfully",
                extra={"object_key": object_key, "url": url},
            )
            return url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(
                "S3 upload failed",
                extra={
                    "local_path": str(local_path),
                    "object_key": object_key,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
            raise
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

    async def upload_bytes(self, data: bytes, object_key: str) -> str:
        """Upload raw bytes to S3 bucket without disk I/O.

        Useful for small files like Telegram photos that are already in memory.

        Args:
            data: Raw bytes to upload.
            object_key: S3 object key (path within bucket).

        Returns:
            Public URL of the uploaded file.

        Raises:
            ValueError: If S3 service is not configured.
            ClientError: If S3 upload fails.
        """
        if not self.config.is_configured:
            raise ValueError(
                "S3 service is not configured. Set S3_* environment variables."
            )

        logger.info(
            "Uploading bytes to S3",
            extra={
                "object_key": object_key,
                "bucket": self.config.bucket_name,
                "size": len(data),
            },
        )

        try:
            session = aioboto3.Session(
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
            )
            async with session.client("s3", endpoint_url=self.config.endpoint) as s3:  # type: ignore[attr-defined]
                await s3.put_object(
                    Bucket=self.config.bucket_name,
                    Key=object_key,
                    Body=data,
                )

            url = self.get_file_url(object_key)
            logger.info(
                "Bytes uploaded successfully",
                extra={"object_key": object_key, "url": url},
            )
            return url

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            logger.error(
                "S3 upload_bytes failed",
                extra={
                    "object_key": object_key,
                    "error_code": error_code,
                    "error_message": error_message,
                },
            )
            raise
        except Exception as e:
            logger.error(
                "Failed to upload bytes to S3",
                extra={
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

        # Construct URL based on endpoint type
        if "amazonaws.com" in endpoint:
            # AWS S3 standard URL format
            url = f"https://{self.config.bucket_name}.s3.{self.config.region}.amazonaws.com/{object_key}"
        else:
            # Custom S3-compatible endpoint (MinIO, Cloudflare R2, etc.)
            url = f"{endpoint}/{self.config.bucket_name}/{object_key}"

        return url
