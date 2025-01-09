import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for, jsonify
)
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
from prediksipmb.db import get_db
from .forms import CreateHistoryForm

bp = Blueprint('history', __name__, url_prefix='/histories')


def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('auth.login'))

        return view(**kwargs)

    return wrapped_view

# -- ROUTES --


@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM users WHERE id = ?', (user_id,)
        ).fetchone()


@bp.route('/', methods=['GET'])
def index():
    form = CreateHistoryForm()
    db = get_db()
    histories = db.execute(
        """
        SELECT * FROM histories
        """
    ).fetchall()

    return render_template('history/index.html', histories=histories, form=form)


# AJAX
@bp.route('/store-history', methods=['POST'])
def storeHistory():
    form = CreateHistoryForm()
    if form.validate_on_submit():
        # Process the data (e.g., save to the database)
        year = form.year.data
        student = form.student.data
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        db = get_db()
        try:
            db.execute(
                """
                INSERT INTO histories (year,student, created_at, updated_at)
                VALUES ( ?, ?, ?, ?);
                """,
                (year, student, timestamp, timestamp)
            )
            db.commit()

            # Flash messages won't work well with AJAX; instead, return a JSON response
            return jsonify({
                'success': True,
                'message': f"Data saved: Year {year}, Student Count {student}"
            })

        except db.IntegrityError:
            return jsonify({
                'success': False,
                'message': "Failed to save data"
            })
    else:
        # Collect error messages
        errors = {field: error[0] for field, error in form.errors.items()}
        return jsonify({'success': False, 'errors': errors})
