import logging
import boto3
from botocore.exceptions import ClientError
from app.config.settings import settings

logger = logging.getLogger(__name__)

class S3Service:
    """
    Handles file storage to AWS S3.
    Stores files using path structure: uploads/{user_id}/{filename}
    """

    def __init__(self):
        # Initialize client ONLY if keys are set
        if not settings.AWS_ACCESS_KEY_ID or not settings.AWS_SECRET_ACCESS_KEY:
            logger.warning("AWS Credentials not fully set in settings. S3 Uploads will be skipped.")
            self.client = None
            return

        self.client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME

    def upload_file_bytes(self, file_bytes: bytes, filename: str, user_id: str, content_type: str) -> str:
        """
        Uploads file buffer directly to S3 under user workspace prefix.
        Returns the S3 URL string, or empty if skipped/failed.
        """
        if not self.client or not self.bucket_name:
            logger.info("S3 client not initialized. Skipping cloud storage upload.")
            return ""

        # Format user prefix: e.g. uploads/550e8400-e29b-41d4/document.pdf
        s3_key = f"uploads/{user_id}/{filename}"

        try:
            logger.info(f"Uploading {filename} to S3 in {self.bucket_name} for {user_id}")
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type
            )
            
            # Form standard S3 URL
            s3_url = f"https://{self.bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
            logger.info(f"S3 Upload Complete: {s3_key}")
            return s3_url

        except ClientError as e:
            logger.error(f"S3 Upload Failed for {filename}: {e}")
            raise Exception(f"S3 Upload Failed: {e}")

# Singleton
s3_service = S3Service()
