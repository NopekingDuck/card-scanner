from PIL import Image
import pytesseract
import cv2
import numpy as np
import json
import converter

im_file = "cards/card_0_0.jpg"
p_img = "card_0_0_process.jpg"
nt = "bb_page_0_cropped_segment_1.jpg"


def process_image(image):
    # im = Image.open(image)
    im = cv2.imread(image)
    # red = get_red_channel(im)
    # revert = cv2.bitwise_not(no_noise)
    filtered_im = filter2d(im)
    # g_im = cv2.cvtColor(filtered_im, cv2.COLOR_BGR2GRAY)
    # thresh, bw_im = cv2.threshold(g_im, 90, 255, cv2.THRESH_BINARY)
    # cv2.imshow("filtered", bw_im)
    # cv2.waitKey(0)
    cv2.imwrite("card_0_0_process.jpg", filtered_im)


def filter2d(image):
    # kernel = np.ones((2, 2), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    # kernel = np.ones((1, 1), np.uint8)
    # image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.array([[0, -2, 0], [-2, 4, 1], [0, 1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    kernel = np.ones((1, 2), np.uint8)
    # image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    # image = cv2.medianBlur(image, 3)
    return image


def get_text(image):
    c_text = pytesseract.image_to_string(image, config='--psm 1')
    print(c_text)


def split_image(image):
    img = cv2.imread(image)
    tile_height = int(img.shape[0] / 3) + 1
    tile_width = int(img.shape[1] / 3) + 1

    for y in range(0, img.shape[0], tile_height):
        for x in range(0, img.shape[1], tile_width):
            tile = img[y:y+tile_height, x:x+tile_width]
            cv2.imwrite(f'cards/card_{y}_{x}.jpg', tile)


def crop_image(image):
    image = cv2.imread(image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, thresh_img = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)
    points = np.argwhere(thresh_img == 0)
    points = np.fliplr(points)
    x, y, w, h = (cv2.boundingRect(points))
    crop = image[y:y+h, x:x+w]
    cv2.imwrite('jpegs/cropped/bb_page_0_cropped.jpg', crop)


def draw_rects(image):
    ### split into get_name, get_level etc
    ### pull from json for coords
    im = cv2.imread(image)
    with open('card_coordinates.json') as json_file:
        card_coordinates = json.load(json_file)

    level = get_level(im, card_coordinates['level'])
    name = get_name(im, card_coordinates['name'])

    # crop = im[370:490, 40:460]
    # name_crop = im[12:56, 46:294]
    # cv2.imshow("level", level)
    # cv2.waitKey(0)
    # cv2.imwrite("bb_page_0_cropped_segment_1.jpg", crop)


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


def get_name(image, coords):
    crop = image[coords['y']:coords['h'], coords['x']:coords['w']]
    return crop

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # converter.convert_pdf_to_image("pdfs")
    # crop_image(im_file)
    # split_image('jpegs/bb_page_0_cropped.jpg')
    # process_image(im_file)
    draw_rects(p_img)

    # get_text(p_img)









