import tarfile
import os
import requests
from wasabi import msg

SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))


def download_scibert():
    os.makedirs(SCRIPT_PATH, exist_ok=True)
    model_path = os.path.join(SCRIPT_PATH, "scibert")

    if not os.path.exists(model_path):
        with msg.loading("   Downloading Scibert"):
            s = requests.get("https://s3-us-west-2.amazonaws.com/ai2-s2-research/"
                             "scibert/huggingface_pytorch/scibert_scivocab_uncased.tar")
        msg.good("Finished")

        tmp_tar = os.path.join(SCRIPT_PATH, "scibert_scivocab_uncased.tar")
        with open(tmp_tar, 'wb') as f:
            f.write(s.content)

        tar = tarfile.open(tmp_tar)
        tar.extractall(path=model_path)
        tar.close()
        os.remove(tmp_tar)
    else:
        msg.good("Found scibert cache!")
