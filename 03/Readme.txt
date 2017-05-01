The cutting out of the words from the jpg files is done with a python script.
Following dependencies need to be present:
- numpy
- matplotlib
- Pillow (/PIL)
- svgpathtools
- svgwrite

The script exports the words to a .mat file called cut_words.mat in the folder preprocessing. This
file is then loaded by the matlab script.

For convenience, the .mat file can also be downloaded manually from https://1drv.ms/u/s!Aqm6mJk7Rt7knftvxP_3fs0fUMc3sg
and added to the preprocessing folder. Then the python script does not need to be run. Unfortunately, github does not allow
the upload of files of that size.
