import base64, logging

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def merge_configs(default_configs, envs, configs={}, required_keys=[]):
    ## from default
    for key in default_configs:
        if key not in configs or configs[key] is None:
            configs[key] = default_configs[key]
    ## from environment variables
    for key in envs:
        if key not in configs or configs[key] is None:
            configs[key] = envs[key]
    ## check required keys        
    for required_config_key in required_keys:
        if not configs.get(required_config_key):
            raise Exception(f"Missing required config {required_config_key}")
    return configs

def setup_logger():
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger