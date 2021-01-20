from low import *


n = 5

imgs = tuple(generate_test_image(2**(n+i), bool(i)) for i in range(2))
for i in range(len(imgs)):
    write_img('output_images/visual-test-input-' + str(i), imgs[i])
for mode in range(3):
    resized, centers = resize_letters(imgs, mode)
    resized = pad_letters(resized, centers)
    jaccard = visual_jaccard(resized)
    write_img('output_images/visual-test-' + str(mode), jaccard)
