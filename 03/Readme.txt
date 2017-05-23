The cutting out of the words from the jpg files is done with a python script.
Following dependencies need to be present:
- numpy
- matplotlib
- Pillow (/PIL)
- svgpathtools
- svgwrite

The script exports the words and the IDs to a .mat file called cut_words.mat in the folder preprocessing. This
file is then loaded by the matlab script.
Unfortunately, github does not allow the upload of files of that size, so you have to run the script yourselves.
You have to change in the script which data you want to export.

Afterwards the main.m file can be run to execute KWS on the test files and producing the output as result.txt file.
