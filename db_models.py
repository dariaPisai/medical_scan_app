from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db
from datetime import datetime

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256))


    history_records = db.relationship('AnalysisHistory', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}'
    

class AnalysisHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False) # The link
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    original_filename = db.Column(db.String(200), nullable=False)
    stored_filename = db.Column(db.String(100), nullable=False, unique=True)
    predicted_class = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    severity_level = db.Column(db.Integer, nullable=False)

    # Note: The 'backref='user'' in the User model's relationship automatically
    # creates the 'user' attribute on this AnalysisHistory model.
    # So, you don't necessarily need another relationship here.

    def __repr__(self):
        return f'<AnalysisHistory {self.id} for User {self.user_id} - {self.original_filename}>'

    