import math
from svgpathtools import svg2paths
from PIL import Image

import matplotlib.path as mplPath
import numpy as np

def cut_path(filename, ground_truth_dir):
    paths, attributes = svg2paths(ground_truth_dir + filename + '.svg')
    image = Image.open('../data/images/' + filename + '.jpg')
    image_data = np.array(image)

    multiple_cut_data = []
    # iterate over words
    for path in paths:
        coordinates = []
        # iterate over edges in word path
        for line in path:
            start_cord = line.start
            # the x and y coordinates are represented as the real and imaginary part of a complex number
            coordinates.append([math.floor(start_cord.real), math.floor(start_cord.imag)])

        # create polygon from path
        poly_path = mplPath.Path(np.array(coordinates))

        # find bounding box of the polygon
        max_x = max(co[0] for co in coordinates)
        min_x = min(co[0] for co in coordinates)
        max_y = max(co[1] for co in coordinates)
        min_y = min(co[1] for co in coordinates)

        # cut out bounding box of jpg
        # pixels that are inside the path are copied from the jpg, the others are filled in as zero
        cut_data = np.empty(shape=(max_y - min_y, max_x - min_x))
        for i in range(0, max_y - min_y - 1):
            y = min_y + i
            for k in range(0, max_x - min_x - 1):
                x = min_x + k
                if poly_path.contains_point((x, y)):
                    cut_data[i, k] = image_data[y, x]
                else:
                    cut_data[i, k] = 255

        multiple_cut_data.append(cut_data)

    result = np.array(multiple_cut_data)
    return result
