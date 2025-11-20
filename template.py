import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,format='[%(asctime)s]: %(message)s:')


list_of_files=[
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/data.py",
    f"src/env.py",
    f"src/layers.py",
    f"src/model.py",
    f"src/trainer.py",
    "main.py",
    "solve.py",
    "visualize.py",
    "requirements.txt",
]

for filepath in list_of_files:
    file_path=Path(filepath)
    fileDir, fileName = os.path.split(file_path)

    if fileDir!="":
        os.makedirs(fileDir,exist_ok=True)
        logging.info(f"Creating Directory: {fileDir} for the file: {fileName}")
    if(not os.path.exists(file_path)) or (os.path.getsize(file_path)==0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file:{file_path}")
    else:
        logging.info(f"{fileName} already exist!")


