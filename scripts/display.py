from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import cv2
import click
import numpy as np

from pathlib import Path

BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / "data"
FIG_PATH = BASE_PATH / "figures"


def get_img_obj(fptr, color_scale):
    img = cv2.imread(fptr, color_scale)
    print(img.shape)
    raise Exception
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def display_in_table(all_img, sample_nums):
    ncols = 4
    nrows = len(all_img) // ncols
    fig, ax = plt.subplots(nrows, ncols, figsize=(18, 18))
    if isinstance(ax[0], Axes):
        ax = [ax]
    for r in range(nrows):
        for c in range(ncols):
            num = r * ncols + c
            img = all_img[num]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            s_num = sample_nums[num]
            ax[r][c].imshow(img)
            ax[r][c].title.set_text(f"Sample {s_num}")
    fig.tight_layout()
    return fig


def crop_xrf(img):
    return img[250:2880, 400:3350]


def crop(img):
    img = img[:-150, :]
    # print(img.shape)
    # raise Exception
    return img


def imgproc(img, proc):
    if proc == "canny":
        return cannyedge(img)
    elif proc == "erode":
        return erode(img)
    elif proc == "simpleTh":
        return simple_thresh(img)
    elif proc == "adapTh":
        return adapt_thresh(img)
    elif proc == "otsu":
        return otsu_thresh(img)
    elif proc == "bgsub":
        return bg_sub(img)
    elif proc == "hough":
        return hough(img)
    elif proc == "crop":
        return crop(img)
    else:
        return img


def cannyedge(img):
    return cv2.Canny(img, threshold1=10, threshold2=25)


def hough(img):
    return cv2.HoughLines(img, 1, np.pi/180, 200)


def create_from_hough(hough_lines):
    pass

def erode(img):
    k = 5
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(img, kernel, iterations=2)


def simple_thresh(img):
    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC)
    return img


def adapt_thresh(img):
    return cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5
    )


def otsu_thresh(img):
    return cv2.threshold(img, 100, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)[1]


def bg_sub(img):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    return fgbg.apply(img)


@click.command()
@click.option("--data")
@click.option("--proc")
def run(data, proc):
    proc = proc.split(",")
    if len(proc) == 1:
        proc = proc[0]
    run_base(data, proc)


def run_base(data, proc):
    # picked_nums = [1, 2, 9, 11, 47, 69, 70, 73, 79, 82, 85, 86]
    picked_nums = [11, 47, 69, 70]
    # color_scale = cv2.IMREAD_GRAYSCALE
    color_scale = None

    sample_nums = []
    path = DATA_PATH / data
    all_img = []
    for fptr in path.iterdir():
        if fptr.suffix[1:] == "png":
            name = fptr.stem
            if data == "XRF":
                s_num = int(name.split("_")[0])
            elif data == "Microscopy":
                s_num = int(name)
            else:
                raise ValueError(f"{data} not available as source")
            if s_num in picked_nums:
                sample_nums.append(s_num)
                if data == "XRF":
                    if name[-2] == "1":
                        continue
                img = get_img_obj(fptr, color_scale)
                # if data == "XRF":
                #     crop_xrf(img)

                all_img.append(img)
    sample_nums, all_img = zip(*sorted(zip(sample_nums, all_img), key=lambda s: s[0]))
    if proc == "og":
        pass
    else:
        if isinstance(proc, list):
            for p in proc:
                tmp = []
                for img in all_img:
                    tmp.append(imgproc(img, p))
            all_img = tmp[:]
            proc = "-".join(proc)
        else:
            all_img = [imgproc(img, proc) for img in all_img]

    fig = display_in_table(all_img, sample_nums)
    fig.savefig(FIG_PATH / f"{data}_{proc}.png")


if __name__ == "__main__":
    data = "Microscopy"
    # data = "XRF"
    # proc = "bgsub"
    proc = "og"
    # proc = "crop"
    run_base(data, proc)
    # (2056, 2464, 3)