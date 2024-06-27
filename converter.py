import glob
import cv2
from pdf2image import convert_from_path


# consider broadening option instead of just jpegs . Possibly use a constant
# this doesn't go through all currently
def convert_pdf_to_image(directory):
    count = 0
    path = f"{directory}/*.pdf"

    for pdf in glob.iglob(path):
        pages = convert_from_path(pdf)
        for page in pages:
            count += 1
            page.save(f"jpegs/bb_page_{count}.jpg", "jpeg")


def load_images(directory):
    path = f"{directory}/*jpg"
    images = [cv2.imread(image) for image in glob.iglob(path)]
    return images


def load_card(path):
    card = cv2.imread(path)
    return card




