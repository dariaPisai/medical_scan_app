import os
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required, current_user
from flask_migrate import Migrate


#variabile globale, nu au nevoie de contextul aplicatiei
db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate()

#configuram login manager
login_manager.login_view = 'auth.login' #numele rutei pentru login page
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    from db_models import User
    return User.query.get(int(user_id))

def create_app():
    app = Flask(__name__, instance_relative_config=True) # flask se uita pt config in folderul instance

    app.config['SECRET_KEY'] = 'random-string123'
    db_path = os.path.join(app.instance_path, 'database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = 'uploads'

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)
    login_manager.init_app(app)
    migrate.init_app(app, db)

    from auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    @app.route('/')
    @login_required
    def index():


        return render_template('index.html', username=current_user.username)
    
    @app.route('/analyze', methods=['POST'])
    @login_required
    def analyze():
        if 'scan_file' not in request.files:
            return redirect(url_for('index'))
        file = request.files['scan_file']

        report_output = "Analysis complete "
        visualization_url = None
        uploaded_filename = file.filename

        return render_template('index.html', 
                               username=current_user.username,
                               report=report_output,
                               image_visualization_url=visualization_url,
                               uploaded_filename=uploaded_filename)
    return app
