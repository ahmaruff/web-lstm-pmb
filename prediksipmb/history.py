import functools
from flask import (
    Blueprint, g, redirect, render_template, request, session, url_for, jsonify
)
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
        SELECT * FROM histories ORDER BY year DESC
        """
    ).fetchall()

    return render_template('history/index.html', histories=histories,)


# AJAX
@bp.route('/store-history', methods=['POST'])
def storeHistory():

    data = request.json
    year = data.get('year')
    student = data.get('student')
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    db = get_db()
    try:
        db.execute(
            """
            INSERT INTO histories (year, student, created_at, updated_at)
            VALUES ( ?, ?, ?, ?);
            """,
            (year, student, timestamp, timestamp)
        )
        db.commit()

        return jsonify({
            'success': True,
            'message': f"Data saved: Year {year}, Student Count {student}"
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': "Failed to save data",
            'error': str(e)

        }), 500


@bp.route('/update-history/<int:id>', methods=['PUT'])
def updateHistory(id):
    data = request.json
    year = data.get('year')
    student = data.get('student')

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        db = get_db()
        db.execute(
            """
            UPDATE histories SET year = ?, student = ?, updated_at = ? WHERE id = ?
            """,
            (year, student, timestamp, id)
        )
        db.commit()

        return jsonify({
            'success': True,
            'message': f"Data updated: Year {year}, Student Count {student}"
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'message': "Failed to save data",
            'error': str(e)

        }), 500


@bp.route('/delete-history/<int:id>', methods=['DELETE'])
def delete_history(id):
    try:
        db = get_db()
        db.execute('DELETE FROM histories WHERE id = ?', (id,))
        db.commit()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
