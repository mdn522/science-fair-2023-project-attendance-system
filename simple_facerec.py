from collections import OrderedDict
from pathlib import Path
from typing import Optional

import face_recognition
import cv2
import os
import glob
import numpy as np
import diskcache as dc
from dataclasses import dataclass

import config
from config import base_path


@dataclass
class FaceDataItem:
    name: str
    user_id: int
    filesize: int
    filename: str
    occupation: str = 'student'
    encoding: Optional[object] = None


class SimpleFaceRec:
    def __init__(self):
        self.known_face_data = dc.Index(str(config.dc_path / 'known_face_data'))
        # self.known_face_encodings = []
        # self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.5

        print('known_face_data_len', len(self.known_face_data))

    @property
    def valid_known_face_data(self):
        return list(filter(lambda e: e.encoding is not None, list(self.known_face_data.values())))

    @property
    def known_face_encodings(self):
        return [e.encoding for e in self.valid_known_face_data]

    @property
    def known_face_ids(self):
        return [e.user_id for e in self.valid_known_face_data]

    def load_encoding_images(self, images_path, force: bool = False):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(str(images_path), "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        known_face_data_keys = list(self.known_face_data.keys())
        known_face_data = OrderedDict(self.known_face_data.items())
        base_names = []
        for img_path in images_path:
            print(img_path)
            img_ext = img_path.split('.')[-1]
            if img_ext not in ['jpg', 'jpeg', 'png']:
                continue
            img = cv2.imread(img_path)
            img_size = os.path.getsize(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            base_names.append(basename)
            (filename, ext) = os.path.splitext(basename)
            # print(filename)
            user_id, person_name, *_ = filename.split('_', maxsplit=2)
            user_id = int(user_id)
            if _:
                occupation = _[0]
            else:
                occupation = 'student'
            # Get encoding
            # TODO rgb -> frame
            if force \
                    or user_id not in known_face_data_keys \
                    or known_face_data[user_id].encoding is None \
                    or known_face_data[user_id].filesize != img_size:
                print('Encoding')
                img_encoding = face_recognition.face_encodings(rgb_img)[0]

                # Store file name and file encoding
                # self.known_face_encodings.append(img_encoding)
                # self.known_face_names.append(filename)
                self.known_face_data[user_id] = FaceDataItem(
                    name=person_name,
                    user_id=user_id,
                    encoding=img_encoding,
                    filesize=img_size,
                    filename=basename,
                    occupation=occupation
                )
                print('Encoded', person_name)

        print("Encoding images loaded")

        for k, v in list(self.known_face_data.items()):
            if v.filename not in base_names:
                print('Removing', v.filename, 'from saved encodings')
                del self.known_face_data[k]

    def face_locations(self, frame):
        # small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # # Find all the faces and face encodings in the current frame of video
        # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        return face_locations

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        # rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = self.face_locations(small_frame)
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        # print(face_locations, face_encodings)

        # face_names = []
        face_user_ids = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            # name = "Unknown"
            user_id = None

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                # name = self.known_face_names[first_match_index]
                user_id = self.known_face_ids[first_match_index]

            # if True in matches:
            #     #Find positions at which we get True and store them
            #     matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            #     counts = {}
            #     # loop over the matched indexes and maintain a count for
            #     # each recognized face face
            #     for i in matchedIdxs:
            #         #Check the names at respective indexes we stored in matchedIdxs
            #         name = data["names"][i]
            #         #increase count for the name we got
            #         counts[name] = counts.get(name, 0) + 1
            #     #set name which has highest count
            #     name = max(counts, key=counts.get)

            # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            # if not face_distances:
            #     continue
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index]:
            #     name = self.known_face_names[best_match_index]
            # face_names.append(name)
            face_user_ids.append(user_id)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_user_ids


if __name__ == '__main__':
    s = SimpleFaceRec()
    s.load_encoding_images(images_path=base_path / 'static/faces')
    pass
