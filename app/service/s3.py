import os
import boto3
import shutil 
import requests
import botocore
import ffmpeg
import traceback
from PIL import Image
from loguru import logger
from app.base.error import Error
from app.base.media import get_mime_type_from_filepath, get_postfix_from_mime_type
from app.base.config import config

class S3:
    def __init__(self):
        self.clients = []
        pass
  
    def init(self):
        self.account_id      = config.get("r2.account_id")
        self.access_key      = config.get("r2.access_key")
        self.access_secret   = config.get("r2.access_secret")
        self.bucket_name     = config.get("r2.bucket_name")
        self.region          = "auto" 
        self.endpoint_url    = f"https://{self.account_id}.r2.cloudflarestorage.com"
        self.clients         = []
        logger.debug(f'account_id: {self.account_id}; access_key:{self.access_key}, access_secret: {self.access_secret}; bucket: {self.bucket_name}')
        
        proxy_count          = config.get_proxy_count()
        if proxy_count > 0:
            for i in range(0, proxy_count):
                proxy = config.get_proxy(i)
                logger.info(f's3 proxy[{i}]: {proxy} ')
                client = boto3.client(
                    service_name ="s3",
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.access_secret,
                    endpoint_url=self.endpoint_url,
                    region_name=self.region,
                    config=boto3.session.Config(proxies={'http': proxy, 'https': proxy})
                )
                self.clients.append(client)
        else:
            client = boto3.client(
                    "s3",
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.access_secret,
                    endpoint_url=self.endpoint_url,
                    region_name=self.region,
                )
            self.clients.append(client)
        
    def put_object(self, file_path, obj_key):
        for client in self.clients:
            result = self.do_put_object(client, file_path, obj_key)
            if result == Error.OK:
                return result
        return result
    
    def get_object(self, obj_key, task_path, name):
        for client in self.clients:
            tmp_file, content_type, err = self.do_get_object(client, obj_key, task_path, name)
            if err != Error.OK:
                if err == Error.NetworkError:
                    logger.error(f'get object({obj_key}) network error with proxy: {client._endpoint.host}')
                    continue
                return '', err
            format, video, err = get_postfix_from_mime_type(content_type)
            if err != Error.OK:
                logger.error(f'get object({obj_key}) unsupport file type({content_type})')
                return '', err
            
            if video == False:
                dst_file = os.path.join(task_path, f'{name}.jpg')
                if format == 'jpg':
                    os.rename(tmp_file, dst_file)
                elif format == 'png':
                    with Image.open(tmp_file) as img:
                        img = img.convert("RGB")
                        img.save(dst_file, "JPEG")
                    os.remove(tmp_file)
                else:
                    print(f'get object ({obj_key}) unknown file type({content_type})')
                    return '', Error.UnsupportFile
            else:
                dst_file = os.path.join(task_path, f'{name}.mp4')
                if format == 'mp4':
                    os.rename(tmp_file, dst_file)
                elif format == 'mov':
                    ffmpeg.input(tmp_file).output(dst_file, vcodec="copy", acodec="copy").run(quiet=True, overwrite_output=True)
                    os.remove(tmp_file)
                else:
                    print(f'get object ({obj_key}) unknown file type({content_type})')
                    return '', Error.UnsupportFile                    
                
            if os.path.isfile(dst_file): 
                return dst_file, Error.OK
            
        return '', Error.FileNotFound
    
    def do_put_object(self, client, file_path, obj_key):
        try:
            content_type = get_mime_type_from_filepath(file_path)
            if content_type == 'none':
                print(f'unknown support mime_type: {content_type}')
                return Error.UnsupportFile
            with open(file_path, "rb") as f:
                client.put_object(
                    Bucket=self.bucket_name,
                    Key=obj_key,
                    Body=f.read(),
                )
            logger.debug(f"put object >> {obj_key}")
            return Error.OK
        except requests.exceptions.ProxyError:
            logger.error("error: proxy error")
            return Error.NetworkError
        except requests.exceptions.ConnectionError:
            logger.error("error: connection error")
            return Error.NetworkError
        except requests.exceptions.Timeout:
            logger.error("error: timeout")
            return Error.NetworkError
        except botocore.exceptions.EndpointConnectionError:
            logger.error("error: endpoint connection error")
            return Error.NetworkError
        except botocore.exceptions.NoCredentialsError:
            logger.error("error: no credit")
            return Error.NetworkError
        except botocore.exceptions.PartialCredentialsError:
            logger.error("error: credit error")
            return Error.NetworkError
        except Exception as e:
            logger.error(f"put object fail: {e}")
            traceback.print_exc()
        return Error.Unknown
    
    # def do_get_object(self, client, obj_key, task_path, name):
    #     try:
    #         metadata = client.head_object(Bucket=self.bucket_name, Key=obj_key)
    #         content_type = metadata.get("ContentType", "").split(";")[0].strip().lower()
    #         content_length = metadata.get("ContentLength", 0)
    #         if content_length == 0:
    #             return '', '', Error.UnsupportFile
    #         temp_file = os.path.join(task_path, f'{name}.tmp')
            
    #         x = client.download_file(self.bucket_name, obj_key, temp_file)
    #         print(f'x: {x}, isfile: {os.path.isfile(temp_file)}')
    #         return temp_file, content_type, Error.OK
    def do_get_object(self, client, obj_key, task_path, name):
        try:
            response = client.get_object(Bucket=self.bucket_name, Key=obj_key)
            content_type = response.get("ContentType", "").lower()
            content_length = response.get("ContentLength", 0)
            if content_length == 0:
                return '', '', Error.UnsupportFile
            temp_file = os.path.join(task_path, f'{name}.tmp')
            with open(temp_file, 'wb') as f:
                shutil.copyfileobj(response['Body'], f)
                
            return temp_file, content_type, Error.OK
        
        except requests.exceptions.ProxyError:
            logger.error("error: proxy error")
            err = Error.NetworkError
        except requests.exceptions.ConnectionError:
            logger.error("error: connection error")
            err = Error.NetworkError
        except requests.exceptions.Timeout:
            logger.error("error: timeout")
            err = Error.NetworkError
        except botocore.exceptions.EndpointConnectionError:
            logger.error("error: endpoint connection error")
            err = Error.NetworkError
        except botocore.exceptions.ReadTimeoutError:
            logger.error("error: read timeout error")
            err = Error.NetworkError
        except botocore.exceptions.NoCredentialsError:
            logger.error("error: no credit")
            err = Error.NetworkError
        except botocore.exceptions.PartialCredentialsError:
            logger.error("error: credit error")
            err = Error.NetworkError
        except botocore.exceptions.ClientError:
            logger.error("error: client error")
            err = Error.FileNotFound
        except Exception as e:
            logger.error(f"get object fail: {e}")
            traceback.print_exc()
            err = Error.Unknown
        
        return '', '', err
    

        
    
 
    def do_del_object(self, client, obj_key):
        pass
  
s3 = S3()


if __name__ == '__main__':
    import sys
    
    def test_put_get():
        obj_key = "dev/task/1234567890ABCDEF/source"
        config.init("./deep.yaml")
        s3.init()
        #err = s3.put_object("/Users/wadahana/Desktop/test.mov", obj_key)
        err = s3.put_object("/Users/wadahana/Desktop/test10.png", obj_key)
        if err != Error.OK:
            print(f'test put obj fail, err: {err}')
            sys.exit(0)
        logger.info("test put obj success")
        path, err = s3.get_object(obj_key=obj_key, task_path="/Users/wadahana/Desktop", name="target")
        if err != Error.OK:
            print(f'test get obj fail, err: {err}')
            sys.exit(0)
        logger.info(f"test get obj success, save to {path}")
        
    def test_get():
        config.init("./deep.yaml")
        s3.init()
        obj_key = 'website/dev/task/2307b676c9a8856e5f69262ce00e5632/target' #video
        #obj_key = 'website/dev/task/06a33dfc3363dc37f61918e21e17f2bc/target' #image
        path, err = s3.get_object(obj_key=obj_key, task_path="/Users/wadahana/Desktop", name="target")
        if err != Error.OK:
            print(f'test get obj fail, err: {err}')
            sys.exit(0)
        logger.info(f"test get obj success, save to {path}")
    
    test_get()