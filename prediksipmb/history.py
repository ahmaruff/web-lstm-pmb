import functools
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
from prediksipmb.db import get_db


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
    db = get_db()
    histories = db.execute(
        """
        SELECT * FROM histories
        """
    ).fetchall()

    return render_template('history/index.html', histories=histories)
