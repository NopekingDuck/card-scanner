import glob
import cv2
import numpy as np
from pdf2image import convert_from_path

'''
converter.py contains functions to turn a pdf of cards into jpegs of each card:
    convert_pdf_to_image(directory) - takes the pdfs in a directory and converts each page to a jpeg
    crop_image(directory) - takes jpegs in a directory and crops out whitespace around the card grid
    split_image(directory) - takes cropped jpegs in a directory and slices them in 3 to get the individual cards as jpgs
'''


def convert_pdf_to_image(directory):
    count = 0
    path = f"{directory}/*.pdf"

    for pdf in glob.iglob(path):
        pages = convert_from_path(pdf)
        for page in pages:
            count += 1
            page.save(f"jpegs/page_{count}.jpg", "jpeg")


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
        crop = im[y:y+h, x:x+w]
        cv2.imwrite(f'jpegs/cropped/cropped_page_{count}.jpg', crop)
        count += 1


def split_image(directory):
    path = f'{directory}/*.jpg'
    count = 1
    for image in glob.iglob(path):
        img = cv2.imread(image)
        tile_height = int(img.shape[0] / 3) + 1
        tile_width = int(img.shape[1] / 3) + 1
        for y in range(0, img.shape[0], tile_height):
            for x in range(0, img.shape[1], tile_width):
                tile = img[y:y+tile_height, x:x+tile_width]
                cv2.imwrite(f'cards/card_{count}.jpg', tile)
                count += 1


