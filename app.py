import base64
import time
import uuid
from collections import OrderedDict
from typing import List

# from livereload import Server
import cv2

import face_recognition
from flask import Flask, request, render_template
from datetime import date
from datetime import datetime
import pandas as pd
from pathlib import Path
import diskcache as dc

import config
import utils
from simple_facerec import SimpleFaceRec, FaceDataItem

# Defining Flask App
app = Flask(__name__)

add_view_cache = dc.Cache(str(config.dc_path / 'add_view_cache'))


sfr = SimpleFaceRec()
sfr.load_encoding_images(images_path=config.faces_path)


@app.template_filter()
def np_base64(arr):
    retval, buffer = cv2.imencode('.jpg', arr)
    jpg_as_text = base64.b64encode(buffer)

    # print(type(jpg_as_text), jpg_as_text)
    return jpg_as_text.decode('ascii')


# Saving Date today in 2 different formats
def get_todays_attendance_csv_filepath():
    file_date_today = date.today().strftime("%Y_%m_%d")
    path = Path('Attendance') / f'Attendance-{file_date_today}.csv'
    if not path.exists():
        with path.open('w') as f:
            f.write('Name,Roll,Occupation,Time')

    return path


# A function which trains the model on all the faces available in faces folder
def train_model():
    sfr.load_encoding_images(images_path=config.faces_path, force=True)


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(get_todays_attendance_csv_filepath())
    names = df['Name']
    rolls = df['Roll']
    occupations = df['Occupation']
    times = df['Time']
    l = len(df)
    return names, rolls, occupations, times, l


# Add Attendance of a specific user
def add_attendance(person: FaceDataItem, dt=None):
    user_name = person.name
    user_id = person.user_id
    occupation = person.occupation

    if not dt:
        dt = datetime.now()

    current_time = dt.strftime("%H:%M:%S")

    path = get_todays_attendance_csv_filepath()
    df = pd.read_csv(path)

    already_attended = int(user_id) in list(df['Roll'])
    if not already_attended:
        with path.open('a') as f:
            f.write(f'\n{user_name},{user_id},{occupation.title()},{current_time}')
    return already_attended


@app.route('/')
def home():
    # TODO zip it
    names, rolls, occupations, times, l = extract_attendance()

    return render_template('index.html', names=names, rolls=rolls, occupations=occupations, times=times, l=l)  # totalreg=totalreg(), datetoday2=datetoday2


@app.route('/train')
def force_train():
    train_model()

    return 'Forcefully Trained'


@app.route('/add', methods=['GET', 'POST'])
def add():
    # TODO retry button
    # TODO show borders around faces
    if request.method == 'POST':
        # print(request.form)
        # print(request.form.to_dict())

        from nested_multipart_parser import NestedParser
        parser = NestedParser(request.form.to_dict(), {"separator": "dot"})

        # print(utils.parse_to_dict_vals(request.form.to_dict()))
        # print(parser.is_valid())
        # print(parser.validate_data)
        parser.is_valid()
        data = parser.validate_data

        print(data.get('uuid'))
        ctx = add_view_cache.get(data.get('uuid'))
        if not ctx:
            return render_template('index.html', msg='Data may have processed already')

        for face_id, face_inputs in data.get('faces', {}).items():
            if face_inputs.get('enabled') != 'on':
                continue

            face = next(filter(lambda x: x['uuid'] == face_id, ctx['faces']))

            cv2.imwrite(str(config.faces_path / '{}_{}_{}.jpg'.format(face_inputs['id'], face_inputs['name'], face_inputs['type'].lower())), face['image'])
            print('Saved image')

            train_model()
            print('Trained model')

        return render_template('index.html', msg='Added New Faces')

        # {'uuid': 'e1bc91e6-12bf-4288-b032-83ac55300914', 'faces': {
        #     '6220bd06-bba5-4d06-9221-1959eb6a7a89': {'enabled': 'on', 'name': '12', 'id': '12', 'type': 'Student'},
        #     '1f1126a3-0109-452f-b63b-26c76bbac56c': {'enabled': 'on', 'name': '12', 'id': '12', 'type': 'Student'}}}

    # Capture image. find faces
    # Show faces in webpage
    # Let people
    TIMER = config.add_capture_timer - 1
    timer_key = 'q'
    captured_frame = None

    if config.test_use_static_image:
        captured_frame = cv2.imread(config.test_static_image_name)  # TODO test. delete it
    else:
        cap = cv2.VideoCapture(config.camera_index)
        while True:
            frame = utils.get_frame(cap)

            cv2.putText(
                frame, 'Press {} to start timer'.format(timer_key), (30, 50),
                cv2.FONT_HERSHEY_DUPLEX, 1,
                (100, 255, 100), 2, cv2.LINE_AA
            )

            cv2.imshow('Add new User', frame)
            k = cv2.waitKey(1)

            if k == ord(timer_key):
                prev = time.time()

                while TIMER >= 0:
                    frame = utils.get_frame(cap)
                    cv2.putText(
                        frame, str(TIMER + 1), (75, 200),
                        cv2.FONT_HERSHEY_DUPLEX, 7, (0, 0, 255),
                        4, cv2.LINE_AA
                    )
                    cv2.imshow('Add new User', frame)
                    cv2.waitKey(1)

                    cur = time.time()

                    # Update and keep track of Countdown
                    # if time elapsed is one second
                    # then decrease the counter
                    if cur - prev >= 1:
                        prev = cur
                        TIMER = TIMER - 1
                else:
                    captured_frame = utils.get_frame(cap)
                    break
            elif k == 27:
                break

        # close the camera
        cap.release()
        # close all the opened windows
        cv2.destroyAllWindows()

    #
    if captured_frame is None:
        return render_template('index.html')

    frame = captured_frame

    height, width, channels = frame.shape
    pad = True  # TODO check
    # TODO relative padding depending on (x,y) coordinate
    # 40% up, 15% bottom, 15% lr
    pad_x, pad_y = 40, 40
    face_locations = face_recognition.face_locations(frame)
    faces = []

    frame_m = frame.copy()
    for face_location in face_locations:
        y1, x2, y2, x1 = face_location[0], face_location[1], face_location[2], face_location[3]
        if pad:
            face_image = frame[max(y1 - pad_y, 0):min(y2 + pad_y, height - 1), max(x1 - pad_x, 0):min(x2 + pad_x, width - 1)]
        else:
            face_image = frame[y1:y2, x1:x2]

        # print(y1, y2, x1, x2)
        # print(max(y1 - pad_y, 0), min(y2 + pad_y, height - 1), max(x1 - pad_x, 0), min(x2 + pad_x, width - 1))
        # face_image = frame[y1 - pad_y:y2, x1:x2]

        frame_id = str(uuid.uuid4())

        faces.append({
            'image': face_image,
            'loc': face_location,
            'uuid': str(uuid.uuid4()),
            'frame_uuid': frame_id,
        })

        cv2.rectangle(frame_m, (x1, y1), (x2, y2), (0, 0, 200), 2)

    ctx = {
        'uuid': str(uuid.uuid4()),
        'original_frame': frame,
        'frame': frame_m,
        'faces': faces,
    }

    add_view_cache.set(ctx['uuid'], {
        'uuid': ctx['uuid'],
        'faces': ctx['faces'],
    }, expire=60 * 60 * 24 * 7)

    # print(ctx)

    return render_template('add.html', **ctx)


@app.route('/take-attendance', methods=['GET', 'POST'])
def take_attendance():
    if len(sfr.valid_known_face_data) == 0:
        return render_template(
            'index.html',
            msg='There are no faces in the system. Please add a face/user before attending.'
        )

    cancelled = False
    captured_frame = None

    dt = None

    if config.test_use_static_image:
        captured_frame = cv2.imread('camera.jpg')
    else:
        cap = cv2.VideoCapture(config.camera_index)

        while True:
            frame = utils.get_frame(cap)
            original_frame = frame.copy()

            face_locations = sfr.face_locations(frame)

            for face_loc in face_locations:
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 20), 2)

            cv2.putText(frame, f'Press q to take picture', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            cv2.imshow('Take Attendance', frame)
            k = cv2.waitKey(1)

            if k == ord('q'):
                captured_frame = original_frame
                dt = datetime.now()
                break
            elif k == 27:
                cancelled = True
                break

        cap.release()
        cv2.destroyAllWindows()

    if cancelled:
        return render_template('index.html', msg='Cancelled')

    if captured_frame is None:
        return render_template('index.html', msg='Cancelled')

    frame = captured_frame
    frame_m = frame.copy()

    known_face_data = OrderedDict(sfr.known_face_data.items())
    face_locations, face_user_ids = sfr.detect_known_faces(frame)
    faces = []

    attended_persons: List[FaceDataItem] = []

    height, width, channels = frame.shape
    pad = True  # TODO check
    pad_x, pad_y = 40, 40

    for face_loc, face_user_id in zip(face_locations, face_user_ids):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        if pad:
            face_image = frame[max(y1 - pad_y, 0):min(y2 + pad_y, height - 1), max(x1 - pad_x, 0):min(x2 + pad_x, width - 1)]
        else:
            face_image = frame[y1:y2, x1:x2]

        person: FaceDataItem = known_face_data.get(face_user_id)

        cv2.rectangle(frame_m, (x1, y1), (x2, y2), (0, 0, 200), 2)

        attended = False
        has_already_attended_today = False
        if person:
            attended_persons.append(person)
            has_already_attended_today = add_attendance(person, dt)
            attended = True
            # cv2.putText(frame, "{}".format(name), (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.putText(frame_m, "Name: {}".format(person.name), (x1, y1 - 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 2)
            cv2.putText(frame_m, "ID: {} | {}".format(person.user_id, person.occupation.title()), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 230, 0), 2)
            # cv2.putText(frame, "ID: {}".format(), (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 230, 0), 2)
        else:
            cv2.putText(frame_m, "Unknown Person", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

        faces.append({
            'image': face_image,
            'loc': face_loc,
            'person': person,
            'attended': attended,
            'has_already_attended_today': has_already_attended_today,
        })

    ctx = {
        'frame': frame_m,
        'original_frame': frame,
        'faces': faces,
        # 'attended_persons': attended_persons,
    }
    return render_template('attend.html', **ctx)


@app.route('/take-attendance', methods=['GET'])
def attendance_history():
    # TODO zip it
    names, rolls, occupations, times, l = extract_attendance()

    return render_template('attendance_history.html.html', names=names, rolls=rolls, occupations=occupations, times=times, l=l)  # totalreg=totalreg(), datetoday2=datetoday2


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['TESTING'] = True
    app.debug = True
    app.run(debug=True)

    # server = Server(app.wsgi_app)
    # server.serve(port=5000)
