from high import *

connectivity8 = False
if connectivity8:
    output_path = 'letters8/'
else:
    output_path = 'letters/'

img = cv.imread('input_images/Bodmer 30_012v_large.jpg', cv.IMREAD_GRAYSCALE)

thresh = img_int_to_bool(threshold(img))

minima, row_minima = section_img(thresh, 16, 4)

i_list = (13, 7, 15, 11)
j_list = (0, 56, 48, 33)
dj_list = (2, 2, 2, 2)

letters = []

for i in range(4):
    letter = extract_letter(thresh, minima, row_minima, i_list[i], j_list[i], dj_list[i])
    letter = clean_letter(letter, minima, row_minima, i_list[i], j_list[i], dj_list[i], connectivity8)[0]
    letters.append(letter)

write_img(output_path + 'D_tonda', letters[0])
write_img(output_path + 'D_dritta', letters[1])
write_img(output_path + 'S_dritta', letters[2])
write_img(output_path + 'S_tonda', letters[3])
