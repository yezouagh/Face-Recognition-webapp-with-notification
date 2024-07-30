from flask import Blueprint, request
from flask_login import login_user
from webapp.FrontendMicroservice.UserMicroservice.user import User
import json
from datetime import datetime
from flask_login import LoginManager
from webapp.FrontendMicroservice.setup_app import app
from webapp.FrontendMicroservice.UserMicroservice.user_action import UserAction
from webapp.FrontendMicroservice.NotificationMicroservice.notification_action import NotificationAction
import secrets

secret = secrets.token_urlsafe(32)

app.secret_key = secret
n_action = NotificationAction(app)

action = UserAction(app)

user_api = Blueprint('user_api', __name__)


@user_api.route("/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    text, status_code = action.signup(data)

    return text, status_code


@user_api.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    text, status_code = action.login(data)

    if status_code == 200:
        user = get_user(data['email'])
        if user is None:
            return 'User not found', 404

        login_user(user, remember=True)

    return text, status_code


@app.route("/notification_read", methods=["PUT"])
def notification_read():
    return n_action.notification_read(request)


@app.route("/get_notifications/<user_id>", methods=["GET"])
def get_notifications(user_id):
    return n_action.get_notifications(user_id)


@app.route("/get_unread_notifications/<user_id>", methods=["GET"])
def get_unread_notifications(user_id):
    return n_action.get_unread_notifications(user_id)


def get_user(email):
    text, status_code = action.get_user_by_email(email)

    if status_code == 200:
        user_dict = json.loads(text.data)
        user = User(**user_dict)
        user.created_date = datetime.strptime(user.created_date, '%a, %d %b %Y %H:%M:%S %Z')

        return user
    else:
        return None


login_manager = LoginManager(app)
login_manager.session_protection = 'basic'


@login_manager.user_loader
def load_user(id_):
    text, status_code = action.get_user_by_user_id(id_)
    if status_code == 200:
        user_dict = json.loads(text.data)
        user = User(**user_dict)
        user.created_date = datetime.strptime(user.created_date, '%a, %d %b %Y %H:%M:%S %Z')

        return user
    else:
        return None


