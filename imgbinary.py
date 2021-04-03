
import numpy as np
import cv2
import time

def get_line_pos(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 160, 255, cv2.THRESH_BINARY)
    img_binary = cv2.erode(img_binary, None, iterations=1)
    color = []
    color.append(img_binary[:,60])
    color.append(img_binary[:,100])
    color.append(img_binary[:,160])
    color.append(img_binary[:,220])
    color.append(img_binary[:,260])

    result = []
    for i in range(0, 5):
        white_sum = np.sum(color[i] == 255)
        white_index = np.where(color[i] == 255)
        if white_sum > 6:
            # print(white_sum)
            # print(type(white_sum))
            # print(white_index)
            white_center = (white_index[0][white_sum - 1] + white_index[0][0]) / 2
            result.append([1, white_center - 120])
        else:
            result.append([0, 0])
    return result, img_binary

if __name__ == '__main__':
    img = cv2.imread("./img/img131.jpg")
    print(img.shape)
    t = time.time()
    ret, img_binary = get_line_pos(img)
    print('time: %f'%((time.time() - t)*1000))
    print(ret)

    cv2.imshow("BGR", img)
    cv2.imshow("BIN", img_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
