from low import *

def section_img(img, windows, path=None, interactive=False):
    horizontal_sum = np.sum(img, axis=1)
    if interactive:
        show_projection(img, (horizontal_sum,))
    
        all_smoothed = []
        i = 1
        while i < windows[0]:
            i *= 2
            all_smoothed.append(smooth(horizontal_sum, i))
        all_smoothed = np.asarray(all_smoothed)
        show_smoothed(all_smoothed)
    smoothed = smooth(horizontal_sum, windows[0])
    if interactive:
        show_projection(img, (smoothed,))
        show_projection(img, (horizontal_sum, smoothed))

    minima = find_minima(smoothed)
    if interactive:
        sectioned_img = img.copy()
        draw_minima_lines(sectioned_img, minima)
        show_projection(sectioned_img, (smoothed,))
        write_img(path + '_sectioned', sectioned_img)

    row_imgs = []
    for i in range(minima.size - 1):
        row_img = img[minima[i]:minima[i + 1], :]
        row_imgs.append(row_img)
        if interactive:
            write_img(path + '_row_' + str(i), row_img)

    row_minima = []
    for i, row_img in enumerate(row_imgs):
        vertical_sum = np.sum(row_img, axis=0)
        smoothed_vertical_sum = smooth(vertical_sum, windows[1])
        row_minima.append(find_minima(smoothed_vertical_sum))
        if interactive:
            sectioned_row = row_img.copy()
            draw_minima_lines(sectioned_row, row_minima[-1], False)
            write_img(path + '_sectioned_row_' + str(i), sectioned_row)
            show_projection(row_img, (smoothed_vertical_sum,), False)

    return minima, row_minima

def find_letter_jaccard(thresh, minima, row_minima, references, reference_name,
                        min_similarity, mode, connectivity8=False,
                        interactive=False, central=jaccard_average):
    rectangles = set()
    for dj in range(1, 3):
        for i in range(len(minima) - 1):
#            print(i/(len(minima)-1)/2 + (dj-1)/2)
            for j in range(len(row_minima[i]) - dj):
                letter = extract_letter(thresh, minima, row_minima, i, j, dj)
                letter, rectangle = clean_letter(letter, minima, row_minima, i, j, dj, connectivity8)
                if rectangle is None:
                    continue
                similarities = []
                for reference in references:
                    resized = resize_letters((letter, reference), mode)
                    if resized is None:
                        continue
                    letters, centers = resized
                    letters = pad_letters(letters, centers)
                    similarities.append(compare_letters(letters)) # dt
                if len(similarities) == 0:
                    continue
                similarity = central(similarities)

                if similarity >= min_similarity:
                    if not interactive:
                        rectangles.add((similarity, tuple(tuple(i) for i in rectangle)))
                    else:
                        if ask_interactive(letters[0], reference_name, similarity):
                            rectangles.add((similarity, tuple(tuple(i) for i in rectangle)))
    return rectangles

def section_and_find(input_path, output_path, windows, letter_names, test=False, json_path=None):
    # windows 16 4
    for img_file in os.scandir(input_path):
        img_name = img_file.name
        print(img_name)
        img = cv.imread(input_path + img_name, cv.IMREAD_GRAYSCALE)

        thresh = img_int_to_bool(threshold(img))
        write_img(output_path + img_name + '_binary', thresh)

        minima, row_minima = section_img(thresh, windows[0], windows[1], output_path + img_name)

        if test:
            basename = os.path.splitext(img_name)[0]
            with open(json_path + basename + '.json') as f:
                solution = json.load(f)
        for reference_name in letter_names:
            print(4*' ' + reference_name, len(solution[reference_name]))
            reference = img_int_to_bool(cv.imread(letters_path + reference_name + '.png', cv.IMREAD_GRAYSCALE))
            if test:
                fig, ax = plt.subplots()
                ax.set_xlabel('recall')
                ax.set_ylabel('precision')
            for mode in range(len(modes)):
                print(8*' ' + modes[mode])
                rectangles = find_letter_jaccard(thresh, minima, row_minima, reference, reference_name, 0, mode)
                if test:
                    true_positives = []
                    positives = []
#                rectangles = sorted(rectangles, reverse=True)
                for similarity in similarities:
                    print(11*' ', similarity)
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
                ax.legend()
                fig.savefig('figures/' + reference_name + str(connectivity8))
#                plt.show(fig)
