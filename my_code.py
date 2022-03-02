import math
import os
import pathlib
import re
import urllib.request
from logging import DEBUG, INFO, StreamHandler, getLogger

import click
import cv2
import numpy as np
import pdf2image
import pyocr
import pyocr.builders
from cv2 import dnn_superres  # opencv-contrib-python
from PIL import Image
from scipy import ndimage


def set_logger(verbose_mode):
    logger = getLogger(__name__)
    stream = StreamHandler()
    logger.addHandler(stream)
    if verbose_mode is True:
        logger.setLevel(INFO)
    else:
        logger.setLevel(DEBUG)

    return logger


def get_degree(img):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)  #エッジ検出
    minLineLength = 10
    maxLineGap = 30
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength,
                            maxLineGap)
    sum_arg = 0
    count = 0
    HORIZONTAL = 0
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                arg = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)  #直線を削除
                DIFF = 20  #許容誤差
                if arg > HORIZONTAL - DIFF and arg < HORIZONTAL + DIFF:
                    sum_arg += arg
                    count += 1
        if count == 0:
            return HORIZONTAL
        else:
            return (sum_arg / count)
    else:
        return HORIZONTAL


def load_sr_model(model_name, scale, url):
    model_path = pathlib.Path("models").joinpath("{}_{}.pb".format(
        model_name, scale))
    if not model_path.exists():
        model_path.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(url, model_path)

    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_name, scale)

    return sr


def preprocess(img):
    #sr = load_sr_model("lapsrn", 4, "https://github.com/fannymonori/TF-LapSRN/raw/master/export#LapSRN_x4.pb")
    #img_sr = sr.upsample(image)
    arg = get_degree(img)  #文字列が水平からどのくらい傾いているか
    rotate_img = ndimage.rotate(img, arg)  #傾いている分だけ回転させる
    pil_image = Image.fromarray(rotate_img)  #PILに戻す

    return pil_image


def postprocess(path):
    with open(path) as ocr:
        Text = ocr.readlines()  #テキストを読み込み

    CleanText = []  #各ラインごとに正規表現制御により整えた分を格納するための空リスト
    for line in Text:
        noplus = re.sub('\+', '', line)  #+記号を削除
        nodots = re.sub('.(\.\.+)', '', noplus)  #連続するドットを削除
        comma = re.sub('@{2,3}', '', nodots)  #連続する@を削除
        new_line = re.sub('\n', '', comma)  #改行記号を削除

        CleanText.append(new_line)

    CleanText = ','.join(CleanText)
    CleanText = re.sub(',{2,3}', '', CleanText)  #連続するカンマを削除

    return CleanText


def save_file(dir_path, filename, file_content, mode='w'):
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), mode) as f:
        f.write(file_content)


@click.command()
@click.option('--input', type=str, help='input image', required=True)
@click.option('--output', type=str, help='output file name', required=True)
@click.option('--verbose', default='verbose_mode', required=False)
def extract_text(input, output, verbose):
    verbose_mode = verbose
    logger = set_logger(verbose_mode)

    img = input
    output_file_name = output

    tools = pyocr.get_available_tools()
    tool = tools[0]

    builder = pyocr.builders.TextBuilder()  #tesseract_layout=6
    ls_text = []

    if ".pdf" in img:
        images = pdf2image.convert_from_path(img, dpi=300, fmt='png')
        for image in images:
            img_array = np.asarray(image)  #オブジェクトをndarray化
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  #グレースケールに変換
            pil_image = preprocess(img_gray)  #前処理
            logger.info("complete preprocess")
            text = tool.image_to_string(pil_image, lang="eng", builder=builder)
            ls_text.append(text)
            text = ','.join(ls_text)
            logger.info("complete extract text from scanned image")
    else:
        image = cv2.imread(img, 0)  #グレースケールに変換
        pil_image = preprocess(image)
        logger.info("complete preprocess")
        text = tool.image_to_string(pil_image, lang="eng", builder=builder)
        logger.info("complete extract text from scanned image")

    OUTPUT_DIR = "/home/eikai/imageRecognition/nexus_code/outputs/"
    save_file_name = OUTPUT_DIR + output_file_name
    save_file(OUTPUT_DIR, output_file_name, text)
    CleanText = postprocess(save_file_name)  #後処理
    logger.info("complete postprocess")
    save_file(OUTPUT_DIR, output_file_name, CleanText)  #テキストファイルを保存
    logger.info("file is successfully saved")

    return CleanText


if __name__ == '__main__':
    extract_text()

#インストールしたTesseract-OCRのパスを環境変数「PATH」へ追記する。
#OS自体に設定してあれば以下の2行は不要
#path='C:\\Program Files\\Tesseract-OCR'
#os.environ['PATH'] = os.environ['PATH'] + path
