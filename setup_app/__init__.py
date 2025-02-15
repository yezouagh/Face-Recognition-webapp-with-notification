import os
from sqlalchemy import create_engine
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import jinja2

# Finding all our directories for this template
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
template_dir = os.path.join(base_dir, 'templates')
static_dir = os.path.join(base_dir, 'static')
base_template_dir = os.path.join(template_dir, 'base_templates')

# Making the Flask app
app = Flask(__name__,
            static_url_path='',
            static_folder=static_dir)

# Adding some of the directories to Jinja (for loading templates)
my_loader = jinja2.ChoiceLoader([
    app.jinja_loader,
    jinja2.FileSystemLoader([template_dir,
                             base_template_dir,
                             static_dir]),
])
app.jinja_loader = my_loader

# Load the config file
app.config['USER_PORT'] = os.getenv('USER_PORT', '5003')
app.config['BASE_PORT'] = os.getenv('BASE_PORT', '5003')
app.config['BASE_URL'] = os.getenv('BASE_URL', 'localhost')
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_PORT'] = os.getenv('MYSQL_PORT', '3306')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', '1234567890yasdfghjkl')
app.config['MYSQL_DB_NAME'] = os.getenv('MYSQL_DB_NAME', 'users')

# Make database connection strings
CONN_STR = "mysql://{0}:{1}@{2}:{3}" \
    .format(app.config['MYSQL_USER'], app.config['MYSQL_PASSWORD'], app.config['MYSQL_HOST'], app.config['MYSQL_PORT'])
CONN_STR_W_DB = CONN_STR + '/' + app.config['MYSQL_DB_NAME']
app.config['CONN_STR'] = CONN_STR
app.config['CONN_STR_W_DB'] = CONN_STR_W_DB

# Create the database if it does not exist yet
mysql_engine = create_engine(app.config['CONN_STR'])
mysql_engine.execute("CREATE DATABASE IF NOT EXISTS {0}".format(app.config['MYSQL_DB_NAME']))

# Setup the final connection string for SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = app.config['CONN_STR_W_DB']
db = SQLAlchemy(app)

# Import user after setup (important)
from webapp.FrontendMicroservice.UserMicroservice.user import User
from webapp.FrontendMicroservice.NotificationMicroservice.notifications import Notifications

# Within our app context, create all missing tables
db.create_all()


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    # If you want all HTTP converted to HTTPS
    # response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

    return response


print('>>>App is setup')
