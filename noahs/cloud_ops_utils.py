import pandas as pd
#import numpy as np
import ibm_boto3
from ibm_botocore.client import Config, ClientError

import os
import io

class CloudOps:

    def __init__(self, cos, bucket=None):
        '''
        cos: if cos is dict then funcs assume IBM cloud object storage api
        
        Issue:  add some test that cos works???
        Issue: separate args for creds and cos, maybe even creds as json file path
        '''
        if isinstance(cos, dict):
            keys = ['endpoint_url', 'ibm_api_key_id', 'ibm_auth_endpoint', 'ibm_service_instance_id']
            msg = 'creds dict must have valid keys: %s' %(keys)
            assert all(k in cos for k in keys), msg
            cos = ibm_boto3.client('s3', config=Config(signature_version="oauth"), **cos)
        
        # consider if NOT instance client if all other types have meta...
        elif isinstance(cos, ibm_boto3.resources.factory.s3.ServiceResource):
            cos = cos.meta.client

        else:
            raise TypeError('cos is not right')

        self.cos=cos
        self.bucket = bucket


    def file(self, cloud_file):
        '''
        return:
            BytesIO for the file obj

        Issue: some test to make sure it isn't too big for memory???
        Issue: does buffer need to be closed? no right?.. bc there will be no 
        more references to it once the function exists thus it will be deleted...
        Issue: only allow cloud_file==None inside context manager?
        Issue: manbe rename object, or obj (and overwrite builtin)
        '''
        
        print('Starting Download...')
        buffer = self.cos.get_object(Bucket=self.bucket, Key=cloud_file)['Body']
        return buffer

    def write(self, data, cloud_file=None):
        '''
        write str or bytes data to cloud
        self can be used as file-like obj .write() method

        no file format
        '''
        if cloud_file:
            self.cloud_file = cloud_file

        if isinstance(data, str):
            buffer = io.StringIO(data)
        
        elif isinstance(data, bytes):
            buffer = io.BytesIO(data)

        else:
            raise TypeError('data must be str or bytes type.')
 
        # insure closure no matter exceptions
        # Should try start before buffer is initiated??? doesn't seem necessary
        # but probs safer
        try:
            self.cos.upload_fileobj(buffer, self.bucket, self.cloud_file)
        finally: 
            buffer.close()
        
    # dangerous to mask open here???
    def open(self, cloud_file):
        self.cloud_file = cloud_file
        return self
    

    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        pass
        # do nothing bc buffer was already closed in .write()
        # no other clean up necessary


    def show_files(self):
        df = pd.DataFrame(self.cos.list_objects_v2(Bucket=self.bucket)['Contents'])
        return df[['Key', 'Size']]

    def show_buckets(self):
        return pd.DataFrame(self.cos.list_buckets()['Buckets'])
        



if __name__ == '__main__':
    from config import creds

    cos = ibm_boto3.resource("s3", config=Config(signature_version="oauth"), **creds)

    client = ibm_boto3.client("s3", config=Config(signature_version="oauth"), **creds)

    bucket_name = 'expert-id'
    cloudpath = 'stack_data.csv'
    localpath = '../data/Stack_py.csv'



    a = cos.ObjectSummary(bucket_name, cloudpath)

    dir(cos)

    dir(cos.meta)

    cos.meta.client

    cos.meta.service_name


    pd.DataFrame(client.list_objects_v2(Bucket='expert-id')['Contents'])

    

    type(cos)




    class test:
        def write():
            print('worked')


    with test() as f:
        f.write()