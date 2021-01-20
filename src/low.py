import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


class FigureCloser:
    def __init__(self, fig, answer):
        self.fig = fig
        self.cid = fig.canvas.mpl_connect('key_press_event', self)
        self.answer = answer

    def __call__(self, event):
        if event.key in {'y', 'n'}:
            self.answer.append(event.key == 'y')
            plt.close(self.fig)


def threshold(img):
    return cv.threshold(img, 0b01111111, 0b11111111, cv.THRESH_BINARY)[1]

def img_int_to_bool(img):
    return 1 - img/0b11111111

def img_bool_to_int(img):
    return np.array((1-img) * 0b11111111, np.uint8)

def write_img(path, img):
    cv.imwrite(path + '.png', img_bool_to_int(img))

def show_projection(binarized_img, projections, horizontal=True):
    if horizontal:
        fig, axs = plt.subplots(ncols=len(projections)+1, sharey=True)
        for ax, projection in zip(axs[1:], projections):
            ax.plot(projection, np.arange(binarized_img.shape[0]))
        fig.subplots_adjust(wspace=0)
    else:
        fig, axs = plt.subplots(nrows=len(projections)+1, sharex=True)
        for ax, projection in zip(axs[1:], projections):
            ax.plot(np.arange(binarized_img.shape[1]), projection)
        fig.subplots_adjust(hspace=0)
    axs[0].imshow(img_bool_to_int(binarized_img), 'gray')
    plt.show()

def smooth(array, window):
    smoothed = np.zeros(array.size)
    for i in range(smoothed.size):
        smoothed[i] = np.average(array[max(0, i - window) : min(array.size, i + window + 1)])
    return smoothed

def show_smoothed(smoothed):
    fig, axes = plt.subplots(ncols=smoothed.shape[0], sharey=True)
    for i in range(smoothed.shape[0]):
        axes[i].plot(smoothed[i], np.arange(smoothed.shape[1]))
        axes[i].set_title(str(2**(i+2) + 1))
        axes[i].set_xticks([])
    plt.subplots_adjust(wspace=0)
    plt.show()

def find_minima(smoothed):
    minima = []
    for i in range(1, smoothed.size - 1):
        if smoothed[i] < smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
            minima.append(i)
    return np.array(minima)

def draw_minima_lines(img, minima, horizontal=True):
    for i in minima:
        if horizontal:
            img[i, :] = 1
        else:
            img[:, i] = 1

def extract_letter(img, minima, row_minima, i, j, dj=1):
    return img[minima[i]:minima[i + 1], row_minima[i][j]:row_minima[i][j + dj]]

def extract_cc(letter, connectivity8=False):
    if connectivity8:
        structure = np.ones((3, 3))
    else:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    letter = ndimage.label(letter, structure)[0]
    count = np.bincount(letter.flatten())
    if np.size(count) <= 1:
        return None
    mode = np.argmax(np.bincount(letter.flatten())[1:]) + 1
    letter[letter != mode] = 0
    letter = np.array(letter, bool)
    return letter

def clean_letter(letter, minima, row_minima, i, j, dj=1, connectivity8=False):
    if connectivity8:
        structure = np.ones((3, 3))
    else:
        structure = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    letter = ndimage.label(letter, structure)[0]
    count = np.bincount(letter.flatten())
    if np.size(count) <= 1:
        return letter, None
    mode = np.argmax(np.bincount(letter.flatten())[1:]) + 1
    letter[letter != mode] = 0
    letter = np.array(letter, bool)
    sum1 = np.sum(letter, 1)
    cut1 = 0
    while sum1[cut1] == 0:
        cut1 += 1
    letter = letter[sum1 != 0]
    sum0 = np.sum(letter, 0)
    cut0 = 0
    while sum0[cut0] == 0:
        cut0 += 1
    letter = letter[:, sum0 != 0]
    rectangle = np.array(((cut1, cut0), (cut1 + np.shape(letter)[0], cut0 + np.shape(letter)[1]))) + (minima[i], row_minima[i][j])
    return letter, rectangle

def find_center(letter):
    center = [0, 0]
    for axis in range(2):
        d_sum = np.sum(letter, axis=axis)
        for i, value in enumerate(d_sum):
            center[axis] += i * value
        denominator = np.sum(d_sum)
        if denominator == 0:
            return None
        center[axis] /= denominator
    return tuple((int(round(coord)) for coord in center))

def find_inertia_moment(letter):
    s = np.asarray([np.sum(letter, i) for i in range(2)])
    total = np.sum(s[0])
    if total == 0:
        return None, None
    shape = np.shape(letter)
    center = np.asarray([np.sum(np.arange(shape[1 - i]) * s[i]) for i in range(2)], int)
    moment = np.sum(np.asarray([np.sum(np.arange(shape[1 - i])**2 * s[i]) for i in range(2)]))
    moment -= np.sum(np.asarray([center[i] ** 2 for i in range(2)])) / total
    center = center / total
    return center, moment

def resize_letters(letters, centering=0):
    if centering == 0:
        centers = tuple(tuple(d//2 for d in l.shape) for l in letters)
        return letters, centers
    centers = np.zeros((2, 2), int)
    moments = np.ones(2)
    for i in range(2):
        if centering > 1:
            centers[i], moments[i] = find_inertia_moment(letters[i])
        elif centering > 0:
            centers[i] = find_center(letters[i])
        if centers[i] is None:
            return None
    small = np.argmin(moments)
    if int(moments[small]) == 0:
        return None
    ratio = (moments[1 - small]/moments[small]) ** (1/4)
    new_shape = tuple(int(round(x * ratio)) for x in np.shape(letters[small]))
    resized_letter = img_bool_to_int(letters[small])
    new_shape = (new_shape[1], new_shape[0])
    resized_letter = cv.resize(resized_letter, new_shape)
    resized_letter = img_int_to_bool(threshold(resized_letter))
    centers[small] = find_center(resized_letter)
    return (resized_letter, letters[1 - small]), centers

def pad_letters(letters, centers):
    letters = list(letters)
    for axis in range(2):
        small = np.argmin((centers[0][axis], centers[1][axis]))
        delta = centers[1 - small][axis] - centers[small][axis]
        padding = tuple((0, 0) if i != axis else (delta, 0) for i in range(2))
        letters[small] = np.lib.pad(letters[small], padding, 'constant')
        small = np.argmin((np.shape(letters[0])[axis], np.shape(letters[1])[axis]))
        delta = np.shape(letters[1 - small])[axis] - np.shape(letters[small])[axis]
        padding = tuple((0, 0) if i != axis else (0, delta) for i in range(2))
        letters[small] = np.lib.pad(letters[small], padding, 'constant')
    return tuple(letters)

def compare_letters(letters):
    intersection = np.logical_and(letters[0], letters[1])
    union = np.logical_or(letters[0], letters[1])
    return np.sum(intersection) / np.sum(union)

def compare_letters_dt(letters, metric='chessboard'):
    letters_dt = tuple(ndimage.distance_transform_cdt(letter, metric) for letter in letters)
    return 1 / np.sum(abs(letters_dt[0] - letters_dt[1]))

def draw_rectangle(img, coordinates, color):
    img[coordinates[0][0] : coordinates[1][0], coordinates[0][1]] = color
    img[coordinates[0][0] : coordinates[1][0], coordinates[1][1]] = color
    img[coordinates[0][0], coordinates[0][1] : coordinates[1][1]] = color
    img[coordinates[1][0], coordinates[0][1] : coordinates[1][1]] = color

def generate_test_image(n, corner):
    assert type(n) == int
    assert type(corner) == bool
    img = np.zeros((n, n))
    if corner:
        img[:n//2, :n//2] = 1
        img[0, :] = 1
        img[:, 0] = 1
    else:
        img[n//2:, n//2:] = 1
        img[n-1, :] = 1
        img[:, n-1] = 1
    return img

def visual_jaccard(letters):
    resized_img = tuple(img_bool_to_int(letters[i]) for i in range(len(letters)))
    resized_img = tuple(cv.cvtColor(resized_img[i], cv.COLOR_GRAY2BGR) for i in range(len(resized_img)))
    for i1 in range(np.shape(resized_img[0])[0]):
        for j1 in range(np.shape(resized_img[0])[1]):
            if np.array_equal(resized_img[0][i1, j1], (0, 0, 0)):
                resized_img[0][i1, j1] = np.array([255, 0, 1])
            if np.array_equal(resized_img[1][i1, j1], (0, 0, 0)):
                resized_img[1][i1, j1] = np.array([1, 0, 255])
    jaccard = (resized_img[0] * resized_img[1]) / 255
    return jaccard

def in_rectangle(point, rectangle):
    point = np.array((point[1], point[0]))
    return np.all(point > rectangle[0]) and np.all(point < rectangle[1])

def ask_interactive(letter, reference_name, similarity):
    fig, ax = plt.subplots()
    im = ax.imshow(letter, cmap='Greys')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(f'press "y" if this is a {reference_name}, "n" otherwise')
    ax.set_title(similarity)
    answer = []
    figure_closer = FigureCloser(fig, answer)
    plt.show(fig)
    return answer[0]

def rectangle_to_img(rectangle, page):
    return page[rectangle[0][0] : rectangle[1][0], rectangle[0][1] : rectangle[1][1]]

def jaccard_average(data):
    product = 1
    for d in data:
        product *= 1 - d
    return 1 - product**(1/len(data))

def interpolate_precision(recall, precision):
    interpolated_recall, interpolated_precision = [1], [0]
    max_precision = 0
    for i in range(1, len(recall)):
        if precision[i] > max_precision:
            interpolated_recall.append(recall[i])
            interpolated_precision.append(max_precision)
            max_precision = precision[i]
            interpolated_recall.append(recall[i])
            interpolated_precision.append(precision[i])
    interpolated_recall.append(0)
    interpolated_precision.append(1)
    return interpolated_recall, interpolated_precision

def area_interpolated(x, y):
    return sum((x[i]-x[i+1]) * y[i] for i in range(len(x) - 1))

def average_graph(xs, ys, weights):
    counters = [0] * len(xs)
    x, y = [], []
    while any(counters[i] < len(xs[i]) for i in range(len(xs))):
        # find max
        i_max = -1
        for i in range(len(xs)):
            if counters[i] < len(xs[i]) and (i_max < 0 or xs[i][counters[i]] > xs[i_max][counters[i_max]]):
                i_max = i
        # calculate average
        x.append(xs[i_max][counters[i_max]])
        average = 0
        for i in range(len(xs)):
            if counters[i] < len(xs[i]):
                average += ys[i][counters[i]] * weights[i]
            else:
                average += weights[i]
        average /= sum(weights)
        y.append(average)
        counters[i_max] += 1
    return x, y
