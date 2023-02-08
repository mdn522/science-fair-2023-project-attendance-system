import cv2
import imutils
import requests
import numpy as np
import re
import config


# TODO resize
def get_cam_ip_frame():
    img_resp = requests.get(config.cam_ip_url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    frame = img
    # frame = imutils.resize(img, width=1200, height=720)

    return frame


def get_frame(cap):
    if not config.use_cam_ip:
        ret, frame = cap.read()
    else:
        frame = get_cam_ip_frame()

    return frame


'''
Parses depth, encoded names into a JSON'ish dictionary structure
'''


def parse_to_dict_val(key, value, dict={}):
    patt = re.compile(r'(?P<name>.*?)[\[](?P<key>.*?)[\]](?P<remaining>.*?)$')
    matcher = patt.match(key)
    tmp = (matcher.groupdict() if not matcher == None else {"name": key, "remaining": ''})
    if tmp['remaining'] == '':
        dict[tmp['name']] = value
    else:  # looks like we have more to do
        fwdDict = (dict[tmp['name']] if tmp['name'] in dict else {})
        tmp2 = parse_to_dict_val(tmp['key'], value, fwdDict)
        dict[tmp['name']] = tmp2
    return dict


'''
Parses dictionary for encoded keys signifying depth
'''


def parse_to_dict_vals(dictin):
    dictout = {}
    for key, value in dictin.items():
        parse_to_dict_val(key, value, dictout)
    return dictout
