#!/usr/bin/env python3

import torch
import numpy as np
import cv2
import subprocess

import sys, os

sys.path.append('/catkin_ws/src/yolov5')
sys.path.append('/catkin_ws/src/yolov5/yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, xyxy2xywh

from PIL import Image

import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import random
from PIL import Image, ImageDraw, ImageFont

# def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
#     """
#     Downloads a file from a URL (or alternate URL) to a specified path if file is above a minimum size.

#     Removes incomplete downloads.
#     """
#     from utils.general import LOGGER

#     file = Path(file)
#     assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
#     try:  # url1
#         LOGGER.info(f"Downloading {url} to {file}...")
#         torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
#         assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
#     except Exception as e:  # url2
#         if file.exists():
#             file.unlink()  # remove partial downloads
#         LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
#         # curl download, retry and resume on fail
#         curl_download(url2 or url, file)
#     finally:
#         if not file.exists() or file.stat().st_size < min_bytes:  # check
#             if file.exists():
#                 file.unlink()  # remove partial downloads
#             LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
#         LOGGER.info("")


# def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
#     """Downloads a file from GitHub release assets or via direct URL if not found locally, supporting backup
#     versions.
#     """
#     # from utils.general import LOGGER

#     def github_assets(repository, version="latest"):
#         """Fetches GitHub repository release tag and asset names using the GitHub API."""
#         if version != "latest":
#             version = f"tags/{version}"  # i.e. tags/v7.0
#         response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
#         return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

#     file = Path(str(file).strip().replace("'", ""))
#     if not file.exists():
#         # URL specified
#         name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
#         if str(file).startswith(("http:/", "https:/")):  # download
#             url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
#             file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
#             if Path(file).is_file():
#                 pass
#                 # LOGGER.info(f"Found {url} locally at {file}")  # file already exists
#             else:
#                 safe_download(file=file, url=url, min_bytes=1e5)
#             return file

#         # GitHub assets
#         assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
#         try:
#             tag, assets = github_assets(repo, release)
#         except Exception:
#             try:
#                 tag, assets = github_assets(repo)  # latest release
#             except Exception:
#                 try:
#                     tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
#                 except Exception:
#                     tag = release

#         if name in assets:
#             file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
#             safe_download(
#                 file,
#                 url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
#                 min_bytes=1e5,
#                 error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
#             )

#     return str(file)



def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class YoloWrapper:
    def __init__(self, model_param_path, device = DEVICE):
        self.model = attempt_load(model_param_path)
        self.model.to(device)
        print('model loadded using the file at ', model_param_path)
        self.color = color_list()

    def get_predictions(self, torch_images):
        print('torch_images size:', torch_images.size())
        result = self.model(torch_images)
        pred = non_max_suppression(result[0])   # Apply NMS
        return pred

    def draw_image_w_predictions(self, torch_images, show=False, wait=1000):
        np_images = torch_images.permute(0,2,3,1).cpu().numpy()
        np_images_uint8 = np.clip(np_images*255, 0, 255).astype(np.uint8)
        torch_images = torch_images.detach()
        pred = self.get_predictions(torch_images)
        for i in range(len(pred)):
            detection = pred[i]
            cv2_images_uint8 = np_images_uint8[i].copy()
            np_pred = []
            if len(detection)>0:
                for box in detection:
                    box = box.cpu().numpy()
                    xyxy = box[:4].astype(int)
                    class_int =  int(box[-1])
                    plot_one_box(xyxy, cv2_images_uint8, color=self.color[class_int%10], label=self.model.names[class_int])
                    np_pred.append(box)
            if show:    
                cv2.imshow('prediction'+str(i), cv2_images_uint8)
                cv2.waitKey(wait)
        return cv2_images_uint8, np.array(np_pred)

def example_use_yolo():
    cwd = os.getcwd()
    print(cwd)

    yolo_model = YoloWrapper('yolo_param/yolov5s.pt')

    ### Load the single image ###
    data = np.asarray(Image.open('yolo_sample_images/400.png').convert('RGB'))
    x_image = torch.FloatTensor(data).to(DEVICE).permute(2, 0, 1).unsqueeze(0)/255
    print('x_image', x_image.size(), 'min:', x_image.min(), 'max:', x_image.max())

    ### Use the model ### 
    pred = yolo_model.get_predictions(x_image)

    print(pred)

    ### Use the model to draw prediction ### 
    cv2_images, np_pred = yolo_model.draw_image_w_predictions(x_image, False)

    # print(type(yolo_model.model))
    # print(dir(yolo_model.model))


if __name__ == "__main__":

    example_use_yolo()
