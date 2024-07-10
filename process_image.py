# def process_image(image):
#     filtered_im = filter2d(image)
#     return filtered_im
#
#
# def filter2d(image):
#     kernel = np.array([[0, -2, 0], [-2, 4, 1], [0, 1, 0]])
#     image = cv2.filter2D(image, -1, kernel)
#     kernel = np.ones((1, 2), np.uint8)
#     image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#     return image

### Code to try to find different blocks of text within a region. Not currently working well enough for the body of the card
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