import glob
import cv2
import numpy as np
from pdf2image import convert_from_path


def convert_pdf_to_image(directory):
    count = 0
    path = f"{directory}/*.pdf"

    for pdf in glob.iglob(path):
        pages = convert_from_path(pdf)
        for page in pages:
            count += 1
            page.save(f"jpegs/bb_page_{count}.jpg", "jpeg")


def crop_image(directory):
    path = f'{directory}/*.jpg'
    count = 1
    for image in glob.iglob(path):
        im = cv2.imread(image)
        grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh, thresh_img = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)
        points = np.argwhere(thresh_img == 0)
        points = np.fliplr(points)
        x, y, w, h = (cv2.boundingRect(points))
        crop = image[y:y+h, x:x+w]
        cv2.imwrite(f'jpegs/cropped/copped_page_{count}', crop)
        count += 1


def split_image(directory):
    path = f'{directory}/*.jpg'
    for image in glob.iglob(path):
        img = cv2.imread(image)
        tile_height = int(img.shape[0] / 3) + 1
        tile_width = int(img.shape[1] / 3) + 1
        for y in range(0, img.shape[0], tile_height):
            for x in range(0, img.shape[1], tile_width):
                tile = img[y:y+tile_height, x:x+tile_width]
                cv2.imwrite(f'cards/card_{y}_{x}.jpg', tile)


