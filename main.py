import csv
import json
import re
import cv2
import pytesseract
import numpy as np
import glob
import converter
import constants





def process_image(image):
    filtered_im = filter2d(image)
    return filtered_im


def filter2d(image):
    kernel = np.array([[0, -2, 0], [-2, 4, 1], [0, 1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    kernel = np.ones((1, 2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image


def invert_colour(image):
    _, invert = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return invert


def draw_rects(image):
    ### split into get_name, get_level etc
    ### pull from json for coords
    im = cv2.imread(image)
    with open('card_coordinates.json') as json_file:
        card_coordinates = json.load(json_file)

    # return image part based on coordinates and pair with the best config to return good text results
    level = [get_level(im, card_coordinates['level']), 6]
    name = [get_rect_part(im, card_coordinates['name']), 11]
    health = [get_rect_part(im, card_coordinates['health']), 7]
    body = [get_rect_part(im, card_coordinates['body']), 6]
    body_inv = [invert_colour(body[0]), 6]

    health[0] = invert_colour(health[0])

    card_data = []
    components = (level, name, health, body, body_inv)
    for part in components:
        text = get_text(part[0], part[1])
        card_data.append(text)

    return card_data

# def separate_body(image):
#     pro = process_image(image)
#     grey = cv2.cvtColor(pro, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(grey, (7, 7), 0)
#     _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 8))
#     dilate = cv2.dilate(thresh, kernel, iterations=1)
#     cv2.imshow('conts', dilate)
#     cv2.waitKey(0)
#     cnts, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if area > 10060:
#             x, y, w, h = cv2.boundingRect(c)
#             print(x, y, w, h)
#             ha = cv2.rectangle(pro, (x, y), (w, h), (0, 255, 0), 3)
#             cv2.imshow("rect", ha)
#             cv2.waitKey(0)
#     # cv2.imshow('conts', wit)
#     # cv2.waitKey(0)
#     return 1


def get_level(image, coords):
    mask = np.zeros_like(image, np.uint8)
    center = (coords['x'], coords['y'])
    radius = coords['radius']
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    result = cv2.bitwise_and(image, mask)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    crop = result[y:y+h, x:x+w]
    return crop


def get_rect_part(image, coords):
    crop = image[coords['y']:coords['h'], coords['x']:coords['w']]
    return crop


def get_text(image, config):
    card_text = pytesseract.image_to_string(image, config=f'--psm {config}')
    return card_text


def export_card_to_csv(file):
    filename = file
    path = "cards/*.jpg"

    # for each card in directory
    for card in glob.iglob(path):
        card_data = draw_rects(card)
        trimmed_card_data = trim_text(card_data)
        with open(filename, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(trimmed_card_data)

    csvfile.close()


def make_csv(file):
    with open(file, mode='a', newline='') as csvfile:
        fieldnames = ('level', 'name', 'health', 'body_attacks', 'body_skills')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def trim_text(card_data):
    for level in ('1', '2', '3'):
        if level in card_data[0]:
            card_data[0] = level

    card_data[1] = card_data[1][:-2]
    card_data[2] = re.sub(r'\D', '', card_data[2]) # remove anything but digits
    return card_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    converter.convert_pdf_to_image(constants.PDF_PATH)
    converter.crop_image(constants.JPEGS_PATH)
    converter.split_image(constants.CROPPED_PATH)
    export_card_to_csv(constants.CSV_NAME)










