import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
from glob import glob

""" This script is for converting xml files from LabelImg to a folder containing the coordinates of bounding box """
def xml_to_csv(xml_filepath):
    cols = ["xmin", "ymin", "xmax", "ymax"]
    rows = []

    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    filename = (root[1].text)[:-5]
    for i in range(6, len(root)):
        for j in range(0, 4):
            rows.append(root[i][4][j].text)

    df = pd.DataFrame(np.array(rows).reshape(-1, 4), columns=None)
    df.to_csv('detections2/' + filename + '.csv', index=False, header=False)

if __name__ == '__main__':
    xmls = sorted(glob(os.path.curdir + '/tensorflow/workspace/images/img_set/*.xml'))
    
    for i in range(len(xmls)):
        xml_to_csv(xmls[i])