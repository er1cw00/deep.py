import yaml

class Config:
    def __init__(self, config_file: str = "conv.yaml"):
        self.config = {}
        self.config_file = None
        self.video_codec = None
    
    def init(self, config_file: str):
        self.config_file = config_file
        with open(config_file, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)
            
    def get(self, key: str, default=None):
        """支持使用点号分隔获取嵌套键"""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if not isinstance(value, dict) or k not in value:
                return default
            value = value[k]
        return value
    
    def set(self, key: str, value: str):
        """支持点号路径方式修改配置"""
        keys = key.split(".")
        conf = self.config
        for k in keys[:-1]:
            if k not in conf or not isinstance(conf[k], dict):
                conf[k] = {}  # 自动创建嵌套字典
            conf = conf[k]
        conf[keys[-1]] = value  # 赋值

        # 保存到文件
        with open(self.config_file, "w", encoding="utf-8") as file:
            yaml.safe_dump(self.config, file, allow_unicode=True)    
         
    def get_video_codec(self) -> str:
        if self.video_codec is None:
            vcodec = self.get('deep.vcodec', 'libx265')
            if vcodec not in ['libx264', 'libx265', 'hevc_nvenc', 'h264_nvenc']:
                vcodec = 'libx265'
            self.video_codec = vcodec
            
        return self.video_codec
            
            
    def get_proxy(self, i: int) -> str:
        proxies = self.get('proxy', [])
        if not proxies or not isinstance(proxies, list):
            return None
        if i < 0 or i >= len(proxies):
            return None
        return proxies[i]
    def get_proxy_count(self) -> int:
        proxies = self.get('proxy', [])
        return len(proxies)
    
config = Config()

    
