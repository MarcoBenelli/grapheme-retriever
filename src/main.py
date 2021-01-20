import os
import json
import random
from collections import Counter

from high import *


np.set_printoptions(threshold=np.inf)

connectivity8 = True
input_path = 'input_images/'
output_path = 'output_images/'
if connectivity8:
    letters_path = 'letters8/'
else:
    letters_path = 'letters/'
json_path = 'output_json/'
modes = ('static', 'baricenter', 'inertia')
letter_names = ('D_dritta', 'D_tonda', 'S_dritta', 'S_tonda')
similarities = np.linspace(0, 1, 1 + 2**8)
bisection = False
interactive = True
main_mode = 0
main_similarity = 0
windows = (16, 4)
n_choises = 2**0


#weights_total = [[0] for i in range(len(modes))]
for img_file in os.scandir(input_path):
    img_name = img_file.name
    print(img_name)
    img = cv.imread(input_path + img_name, cv.IMREAD_GRAYSCALE)

    thresh = img_int_to_bool(threshold(img))
    write_img(output_path + img_name + '_binary', thresh)

    minima, row_minima = section_img(thresh, windows, output_path + img_name)

    basename = os.path.splitext(img_name)[0]
    with open(json_path + basename + '.json') as f:
        solution = json.load(f)
    xs_mode = [[] for i in range(len(modes))]
    ys_mode = [[] for i in range(len(modes))]
    x_mode, y_mode = [], []
    weights = []
    for reference_n, reference_name in enumerate(letter_names):
        print(4*' ' + reference_name, len(solution[reference_name]))
        weights.append(len(solution[reference_name]))
        #weights_total[reference_n] += len(solution[reference_name])
        reference = img_int_to_bool(cv.imread(letters_path + reference_name + '.png', cv.IMREAD_GRAYSCALE))
        fig, ax = plt.subplots()
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        fig_interpolated, ax_interpolated = plt.subplots()
        ax_interpolated.set_xlabel('recall')
        ax_interpolated.set_ylabel('precision')
        for mode in range(len(modes)):
            print(8*' ' + modes[mode])
            rectangles = find_letter_jaccard(thresh, minima, row_minima, [reference], reference_name, 0, mode)
            true_positives = []
            positives = []
#            rectangles = sorted(rectangles, reverse=True)
            for similarity in similarities:
                #print(11*' ', similarity)
                rectangles = [r for r in rectangles if r[0] >= similarity]
                true_positives.append(0)
                positives.append(len(rectangles))
                points = solution[reference_name].copy()
                for rectangle in rectangles:
                    for point in points:
                        if in_rectangle(point, rectangle[1]):
                            true_positives[-1] += 1
                            points.remove(point)
                            break
            true_positives = np.array([len(solution[reference_name])] + true_positives)
            positives = np.array([np.inf] + positives)
            x = np.append(true_positives / len(solution[reference_name]), 0)
            y = np.append(true_positives / positives, 1)
            y[np.isnan(y)] = 1
            ax.plot(x, y, label=modes[mode])
            interpolated = interpolate_precision(x, y)
            area = area_interpolated(interpolated[0], interpolated[1])
            ax_interpolated.plot(interpolated[0], interpolated[1], label=modes[mode])
            print(11*' ', area)
            xs_mode[mode].append(interpolated[0])
            ys_mode[mode].append(interpolated[1])
        ax.legend()
        fig.savefig('figures/' + reference_name + str(connectivity8))
        ax_interpolated.legend()
        fig_interpolated.savefig('figures/' + reference_name + str(connectivity8) + 'interpolated')
#        plt.show(fig)
    fig, ax = plt.subplots()
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    for mode in range(len(modes)):
        average = average_graph(xs_mode[mode], ys_mode[mode], weights)
        x_mode.append(average[0])
        y_mode.append(average[1])
        ax.plot(average[0], average[1], label=modes[mode])
        area = area_interpolated(average[0], average[1])
        print(4*' ' + modes[mode])
        print(7*' ', area)
        if mode == main_mode:
            x_simple = average[0]
            y_simple = average[1]
    ax.legend()
    fig.savefig('figures/' + str(connectivity8))


xs = []
ys = []
weights = []
for reference_name in letter_names:
    weights.append(len(solution[reference_name]))
    reference = img_int_to_bool(cv.imread(letters_path + reference_name + '.png', cv.IMREAD_GRAYSCALE))
    pages = os.listdir(input_path)
    choices = random.choices(pages, k=n_choises)
    counter = Counter(choices)
    new_references = [reference]
    for img_name in counter:
        print(img_name)
        img = cv.imread(input_path + img_name, cv.IMREAD_GRAYSCALE)
       
        thresh = img_int_to_bool(threshold(img))
        write_img(output_path + img_name + '_binary', thresh)
       
        minima, row_minima = section_img(thresh, windows, output_path + img_name, False)
        rectangles = find_letter_jaccard(thresh, minima, row_minima, [reference], reference_name, main_similarity, main_mode, connectivity8)
        rectangles = random.sample(rectangles, counter[img_name])
        with open(json_path + basename + '.json') as f:
            solution = json.load(f)
        points = solution[reference_name].copy()
        for rectangle in rectangles:
            letter = rectangle_to_img(rectangle[1], thresh)
            letter = extract_cc(letter, connectivity8)
#            if ask_interactive(letter, reference_name, rectangle[0]):
#                new_references.append(letter)
            for point in points:
                if in_rectangle(point, rectangle[1]):
                    new_references.append(letter)
                    break

    for img_file in os.scandir(input_path):
        img_name = img_file.name
        img = cv.imread(input_path + img_name, cv.IMREAD_GRAYSCALE)
        thresh = img_int_to_bool(threshold(img))
        minima, row_minima = section_img(thresh, windows, output_path + img_name, False)
        rectangles = find_letter_jaccard(thresh, minima, row_minima, new_references, reference_name, main_similarity, main_mode)
# -----------
        basename = os.path.splitext(img_name)[0]
        with open(json_path + basename + '.json') as f:
            solution = json.load(f)
        true_positives = []
        positives = []
#        rectangles = sorted(rectangles, reverse=True)
        for similarity in similarities:
            rectangles = [r for r in rectangles if r[0] >= similarity]
            true_positives.append(0)
            positives.append(len(rectangles))
            points = solution[reference_name].copy()
            for rectangle in rectangles:
                for point in points:
                    if in_rectangle(point, rectangle[1]):
                        true_positives[-1] += 1
                        points.remove(point)
                        break
        true_positives = np.array([len(solution[reference_name])] + true_positives)
        positives = np.array([np.inf] + positives)
        x = np.append(true_positives / len(solution[reference_name]), 0)
        y = np.append(true_positives / positives, 1)
        y[np.isnan(y)] = 1
        x, y = interpolate_precision(x, y)
        xs.append(x)
        ys.append(y)

weights = []
for reference_name in letter_names:
    weights.append(len(solution[reference_name]))
fig, ax = plt.subplots()
ax.set_xlabel('recall')
ax.set_ylabel('precision')
average = average_graph(xs, ys, weights)
ax.plot(average[0], average[1], label='interactive')
ax.plot(x_simple, y_simple, label='not interactive')
area = area_interpolated(average[0], average[1])
print(area)
ax.legend()
fig.savefig('figures/total' + modes[main_mode] + str(connectivity8))
