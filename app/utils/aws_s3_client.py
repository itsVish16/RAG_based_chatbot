import boto3
from boto3 import client
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from uuid import uuid7
from typing import Optional, BinaryIO
from io import BytesIO

logger  = logging.getLogger(__name__)

class S3Client:
    def __init__(self):
        """Initialize the AWS S3 Client"""
        try:
            self.s3_client = boto3.client ('s3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
            )
            self.bucket_name = settings.S3_BUCKET_NAME
            logger.info(f"s3 client initialized successfully for bucket: {self.bucket_name}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise ValueError("AWS credentials not configured peoperly")


    def upload_file(self, file : BinaryIO, file_name : str) -> str:
        """upload the file on s3 and return the path of it
           Args :
                file : file object to  be uplaoded
                file_name : name of the file
            Returns:
                str : path of the uploaded file
        """
        
        try:
            unique_id = uuid7()
            s3_key = f"raw-documents/{unique_id}/{file_name}"
            



for bucket in s3.buckets.all():
    print(bucket.name)