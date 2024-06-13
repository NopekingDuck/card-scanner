from PIL import Image
import pytesseract
import cv2
import numpy as np
import loader

im_file = "jpegs/bb_page_0.jpg"
p_img = "original/test_kai.png"


def process_image(image):
    # im = Image.open(image)
    im = cv2.imread(image)
    g_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh, bw_im = cv2.threshold(g_im, 200, 230, cv2.THRESH_BINARY)
    # red = get_red_channel(im)
    # revert = cv2.bitwise_not(no_noise)
    # no_noise = denoise(bw_im)
    cv2.imwrite("\insert filename here", bw_im)


def denoise(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image

# def get_red_channel(image):
#     r_c_g = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     (R, G, B) = cv2.split(image)
#     # cv2.namedWindow("red", cv2.WINDOW_NORMAL)
#     return B


def get_text(image):
    c_text = pytesseract.image_to_string(image)
    print(c_text)


def split_image(image):
    img = cv2.imread(image)
    tile_height = int(img.shape[0] / 3) + 1
    tile_width = int(img.shape[1] / 3) + 1

    for y in range(0, img.shape[0], tile_height):
        for x in range(0, img.shape[1], tile_width):
            tile = img[y:y+tile_height, x:x+tile_width]
            cv2.imwrite(f'jpegs/card_{y}_{x}.jpg', tile)


def crop_image(image):
    # for image in images:
        # filename = consider making each card an object with a file name and np.array
        image = cv2.imread(image)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, thresh_img = cv2.threshold(grey, 80, 255, cv2.THRESH_BINARY)
        points = np.argwhere(thresh_img == 0)
        points = np.fliplr(points)
        x, y, w, h = (cv2.boundingRect(points))
        crop = image[y:y+h, x:x+w]
        cv2.imwrite('jpegs/bb_page_0_cropped.jpg', crop)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # loader.convert_pdf_to_image("original")
    # images = loader.load_images("jpegs")
    crop_image(im_file)
    # process_image(im_file)
    # get_text(p_img)
    split_image('jpegs/bb_page_0_cropped.jpg')








