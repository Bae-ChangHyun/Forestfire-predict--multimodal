import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    
    api_key = os.getenv("API_1")
    if(api_key==""):
        print("Please insert asos api key on env.json")
        exit()
        
    root_path = os.getenv("filepath")
    os.makedirs(f"{root_path}", exist_ok=True)
    if(root_path==""):
        print("Please set directory path path on env.json")
        exit()
        
    db_path = os.getenv("db_path")
    os.makedirs(f"{db_path}", exist_ok=True)
    if(db_path==""):
        print("Please set db path path on env.json")
        exit()
        
    image_size = os.getenv("image_size")
    image_size=int(image_size)
    

    return api_key,root_path,db_path,image_size
