The cutting out of the words from the jpg files is done with a python script.
Following dependencies need to be present:
- numpy
- matplotlib
- Pillow (/PIL)
- svgpathtools
- svgwrite

The script exports the words to a .mat file called cut_words.mat in the folder preprocessing. This
file is then loaded by the matlab script.
