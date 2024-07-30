from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from email import encoders
from email.mime.base import MIMEBase
from PIL import Image
from datetime import datetime
import dlib
import ssl
# import face_recognition
import cv2
import os
import numpy as np
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mask_detected = False
not_proceed = False
proceed = False
visitor_id = 0
promt = None


class VideoCamera(object):
    def __init__(self):

        """
        self.known_face_names = []
        self.known_face_encodings = []
        self.images = []
        self.i = 0
        self.source = './known_images'for root, dirs, filenames in os.walk(self.source):
            for f in filenames:
                picture = (os.path.splitext(f))[0]
                self.known_face_names.append(picture)
                self.known_face_encodings.append('{}_face_encoding'.format(picture))
                self.images.append('{}_image'.format(picture))

                self.images[self.i] = face_recognition.load_image_file(self.source + "/{}.jpg".format(picture))
                self.known_face_encodings[self.i] = face_recognition.face_encodings(self.images[self.i])[0]

                self.i += 1
        # Initializes some variables

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.face_detected_array = []
        self.process_this_frame = True
        """

        self.send_email = os.environ.get('SEND_EMAIL')

        self.video = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.mid = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        self.model = load_model('models/mask_detector.h5')

        self.mask_detection_completed = False
        self.mask_count = 0

        self.operation_status_failed = False

        if self.send_email == 'TRUE':
            self.sender_email = 'younes.azouagh@gmail.com'
            self.receiver_email = 'ezouagh.youness@gmail.com'
            self.password = 'myapplication.'

            self.message = MIMEMultipart("alternative")
            self.message["Subject"] = "Alert: A New Person Entered the Premises"
            self.message["From"] = self.sender_email
            self.message["To"] = self.receiver_email

    def detect_mask(self, image):
        copy_img = image.copy()

        resized = cv2.resize(copy_img, (254, 254))

        resized = img_to_array(resized)
        resized = preprocess_input(resized)

        resized = np.expand_dims(resized, axis=0)

        mask, _ = self.model.predict([resized])[0]

        return mask

    def email(self, img_path, temp, mask):
        with open(img_path, 'rb') as f:
            # set attachment mime and file name, the image type is png
            mime = MIMEBase('image', 'png', filename='img1.png')
            # add required header data:
            mime.add_header('Content-Disposition', 'attachment', filename='img1.png')
            mime.add_header('X-Attachment-Id', '0')
            mime.add_header('Content-ID', '<0>')
            # read attachment file content into the MIMEBase object
            mime.set_payload(f.read())
            # encode with base64
            encoders.encode_base64(mime)
            # add MIMEBase object to MIMEMultipart object
            self.message.attach(mime)

        body = MIMEText('''
        <html>
            <body>
                <h1>Alert</h1>
                <h2>A new  Person has entered the Premises</h2>
                <h2>Mask: {}</h2>
                <h2>Time: {}</h2>
                <p>
                    <img src="cid:0">
                </p>
            </body>
        </html>'''.format(mask, datetime.now()), 'html', 'utf-8')

        self.message.attach(body)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(self.sender_email, self.password)
            server.sendmail(
                self.sender_email, self.receiver_email, self.message.as_string()
            )

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, img = self.video.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        im_p = Image.fromarray(img)
        global mask_detected, not_proceed, proceed, visitor_id, promt
        promt = None

        if success:
            if self.mask_detection_completed is False:
                mask_prob = self.detect_mask(img)

                if mask_prob > 0.5:
                    self.mask_count += 1

                    if self.mask_count >= 5:
                        self.mask_detection_completed = True

                elif mask_prob < 0.5:
                    mask_detected = False

            elif self.mask_detection_completed:
                mask_detected = True

                faces = self.detector(img_gray, 0)

                if len(faces) > 0:

                    for face in faces:

                        landmarks = self.predictor(img_gray, face)

                        im_n = np.array(im_p)

                        landmarks_list = []
                        for i in range(0, landmarks.num_parts):
                            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

                            cv2.circle(im_n, (landmarks.part(i).x, landmarks.part(i).y), 4, (255, 255, 255), -1)

                        dist = np.sqrt((landmarks.part(21).x - landmarks.part(22).x) ** 2 + (
                                landmarks.part(21).y - landmarks.part(22).y) ** 2)

                        face_ptx, face_pty = (int((landmarks.part(21).x + landmarks.part(22).x) / 2),
                                              int((landmarks.part(21).y + landmarks.part(22).y) / 2) - int(dist))

                        cv2.circle(im_n, (face_ptx, face_pty), 4, (0, 255, 0), -1)
                        im_p = Image.fromarray(im_n)
                else:
                    promt = 'No Face Detected Please remove mask'

            visitor_id = self.mid + 1
            img = np.array(im_p)

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
