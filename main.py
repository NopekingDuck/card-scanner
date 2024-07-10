import csv
import json
import re
import cv2
import pytesseract
import numpy as np
import glob
import converter
import constants


'''
This will extract the text from individual playing cards.
converter.py is used to prepare each card from a pdf.
Each card is then broken into parts. Pytesseract then looks for text in each part
The resulting text is added to a csv.
'''


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


def invert_colour(image):
    _, invert = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)
    return invert


def get_text(image, config):
    card_text = pytesseract.image_to_string(image, config=f'--psm {config}')
    return card_text


def split_card(image):
    im = cv2.imread(image)
    with open('card_coordinates.json') as json_file:
        card_coordinates = json.load(json_file)

    # return image part based on coordinates and pair with the best config to return best pytesseract results
    level = [get_level(im, card_coordinates['level']), 6]
    name = [get_rect_part(im, card_coordinates['name']), 11]
    health = [get_rect_part(im, card_coordinates['health']), 7]
    body = [get_rect_part(im, card_coordinates['body']), 6]

    # invert parts that have white text
    body_inv = [invert_colour(body[0]), 6]
    health[0] = invert_colour(health[0])

    card_data = []
    components = (level, name, health, body, body_inv)
    for part in components:
        text = get_text(part[0], part[1])
        card_data.append(text)
    return card_data


def make_csv(file):
    with open(file, mode='a', newline='') as csvfile:
        fieldnames = ('level', 'name', 'health', 'body_attacks', 'body_skills')
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def trim_text(card_data):
    for level in ('1', '2', '3'):
        if level in card_data[0]:
            card_data[0] = level

    card_data[1] = card_data[1][:-2] # removes \n from name
    card_data[2] = re.sub(r'\D', '', card_data[2])  # remove anything but digits from health
    return card_data


def export_card_to_csv(file):
    filename = file
    path = "cards/*.jpg"

    for card in glob.iglob(path):
        card_data = split_card(card)
        # remove superfluous text and artifacts from card data
        trimmed_card_data = trim_text(card_data)

        with open(filename, mode='a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(trimmed_card_data)
    csvfile.close()


if __name__ == '__main__':
    # converter.convert_pdf_to_image(constants.PDF_PATH)
    # converter.crop_image(constants.JPEGS_PATH)
    converter.split_image(constants.CROPPED_PATH)
    make_csv(constants.CSV_NAME)
    export_card_to_csv(constants.CSV_NAME)










