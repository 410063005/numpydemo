
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

def get_color(point, radius, arr, w, h):
    x, y = point
    left = max(y - radius, 0)
    right = min(y + radius, w)
    top = max(x - radius, 0)
    bottom = min(x + radius, h)

    r = 0
    g = 0
    b = 0
    count = 0
    for i in range(top, bottom):
        for j in range(left, right):
            if x == i and y == j:
                continue
            r += arr[i][j][0]    
            g += arr[i][j][1]    
            b += arr[i][j][2]    
            count += 1
    return (round(r / count), round(g / count), round(b / count))


if __name__ == '__main__':
    #s = mpimg.imread('sample.jpeg')
    s = mpimg.imread('images/1_swqqcs.jpg')
    blurred = s.copy()

    width = s.shape[1]
    height = s.shape[0]
    print("w=%d, h=%d" % (width, height))

    for i in range(height):
        for j in range(width):
            blurred[i][j] = get_color((i, j), 5, s, width, height)

    plt.imshow(blurred)
    plt.axis('off')
    plt.savefig('blurred.png')
    print('ok')

