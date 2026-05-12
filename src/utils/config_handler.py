import yaml
from utils.path_tool import get_abs_path

def load_model_config(config_path:str=get_abs_path("config/model.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

def load_tools_config(config_path:str=get_abs_path("config/tools.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)

def load_prompts_config(config_path:str=get_abs_path("config/prompts.yml"),encoding:str="utf-8"):
    with open(config_path,"r",encoding=encoding) as f:
        #全量加载
        return yaml.load(f, Loader=yaml.FullLoader)



model_conf = load_model_config()
tools_conf = load_tools_config()
prompts_conf = load_prompts_config()



