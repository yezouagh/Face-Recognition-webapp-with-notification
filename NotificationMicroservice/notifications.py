from webapp.FrontendMicroservice.setup_app import db
from sqlalchemy import Column, Integer, DateTime, String, Boolean, ForeignKey, Text
from sqlalchemy.sql import func


class Notifications(db.Model):
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), unique=False, nullable=False)

    color = Column(String(50))
    icon = Column(String(50))
    message_preview = Column(String(160))
    message = Column(Text)
    created_date = Column(DateTime, server_default=func.now())
    isRead = Column(Boolean, default=False, nullable=False)

    def __repr__(self):
        return 'id: '.join([str(id)])

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)