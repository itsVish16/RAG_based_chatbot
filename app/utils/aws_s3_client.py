import boto3
from boto3 import client
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from uuid import uuid4
from typing import Optional, BinaryIO
from io import BytesIO
from app.config.settings import settings

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
            unique_id = str(uuid4())
            s3_key = f"raw-documents/{unique_id}/{file_name}"

            self.s3_client.upload_fileobj(file, self.bucket_name, s3_key, ExtraArgs={"ContentType": content_type})
            logger.info(f"File uploaded successfully to s3 bucket: {s3_key}")
            return s3_key
        except ClientError as e:
            logger.error(f"failed to uplaod the file to s3 bucket: {e}")
            raise

    def download_file(self, s3_key : str) -> bytes:

        try:
            response = self.s3_client.get_object(Bucket = self.bucket_name, Key = s3_key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"failed to download the file from s3 bucket: {e}")
            raise


    def generate_presigned_url(self, s3_key :str, exception : int = 3600) -> str:

        try:
            url = self.s3_client.generate_presigned_url('get_object', Params  = {'Bucket' : self.bucket_name, 'Key' : s3_key}, 
            ExpiresIn  = expiration)
            return url
        except ClientError as e:
            logger.error(f"failed to generate presigned url : {e}")
            raise

    def delete_file(self, s3_key : str, ) -> bool:

        try:
            self.s3_client.delete_object(Bucket = self.bucket_name, Key = s3_key)
            logger.info(f"File deleted successfully from s3 bucket: {s3_key}")
            return True
        except ClientError as e:
            logger.error(f"failed to delete the file from s3 bucket: {e}")
            return False

s3_client =S3Client()
            


