from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange


class CreateHistoryForm(FlaskForm):
    year = IntegerField('Year', validators=[
        DataRequired(),
        NumberRange(min=1900, max=9999, message="Year must be between 1900 and 9999")
    ])

    student = IntegerField('Student Count', validators=[
        DataRequired(),
        NumberRange(min=0, message="Student count must be a non-negative number")
    ])

    submit = SubmitField('Simpan')

