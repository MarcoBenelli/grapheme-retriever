import os
import json

from low import *

json_input_path = 'input_json/'
json_output_path = 'output_json/'
img_input_path = 'input_images/'
img_output_path = 'highlight_images/'

for json_direntry in os.scandir(json_input_path):
    json_name = json_direntry.name
    with open(json_input_path + json_name) as f:
        shapes = json.load(f)['shapes']
    basename = os.path.splitext(json_name)[0]
    img = cv.imread(img_input_path + basename + '.jpg')
    positions = {}
    for shape in shapes:
        if shape['label'] in positions:
            positions[shape['label']].append([0, 0])
        else:
            positions[shape['label']] = [[0, 0]]
        for point in shape['points']:
            for i in range(len(point)):
                positions[shape['label']][-1][i] += point[i]
        for i in range(len(point)):
            positions[shape['label']][-1][i] /= len(shape['points'])
            positions[shape['label']][-1][i] = round(positions[shape['label']][-1][i])
    with open(json_output_path + json_name, 'w') as f:
        json.dump(positions, f)
