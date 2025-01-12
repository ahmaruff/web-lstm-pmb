import functools
from datetime import datetime
import os
from flask import (
    Blueprint, g, redirect, render_template, request, session, url_for, jsonify
)
from prediksipmb.db import get_db
from .LSTMModel import LSTMModel

bp = Blueprint('prediction', __name__, url_prefix='/predictions')


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
@login_required
def index():
    db = get_db()
    data = db.execute(
        """
        SELECT *
        FROM histories ORDER BY year ASC
        """
    ).fetchall()

    # Convert the fetched data to a dictionary format
    data_dict = {'tahun': [row['year'] for row in data], 'jml_mhs': [row['student'] for row in data]}

    # Load model
    lstm_model = LSTMModel(input_shape=(2, 1))

    # Check if the model exists
    model_exists = os.path.exists(lstm_model.model_path)
    plot_url = None
    evaluation_metrics = None
    training_logs = None

    if model_exists:
        # Load the pre-trained model
        lstm_model.load()

        # Generate the plot if the model exists
        plot_url = lstm_model.plot_predictions(data_dict, sequence_length=2, num_predictions=3)

        # Evaluate the model
        evaluation_metrics = lstm_model.evaluate(data_dict, sequence_length=2)

        # Ambil 5 log training terakhir
        training_logs = db.execute(
            """
            SELECT loss, accuracy, training_date
            FROM training_logs
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

    return render_template('prediction/index.html', plot_url=plot_url, model_exists=model_exists, evaluation_metrics=evaluation_metrics, training_logs=training_logs)


@bp.route('/train', methods=['POST'])
@login_required
def train_model():
    db = get_db()
    data = db.execute(
        """
        SELECT *
        FROM histories ORDER BY year ASC
        """
    ).fetchall()

    data_dict = {'tahun': [row['year'] for row in data], 'jml_mhs': [row['student'] for row in data]}
    lstm_model = LSTMModel(input_shape=(2, 1))

    # Train the model and get training metrics
    result = lstm_model.train(data_dict, sequence_length=2, epochs=200, batch_size=1)

    # Simpan hasil training ke database
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    db.execute(
        """
        INSERT INTO training_logs (loss, accuracy, training_date, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (result['loss'], result['accuracy'], result['training_date'], timestamp, timestamp)
    )
    db.commit()

    return jsonify({
        'message': 'Model trained successfully!',
        'status': 'success',
        'result': result
    })


@bp.route('/predict', methods=['POST'])
@login_required
def predict():
    db = get_db()
    data = db.execute(
        """
        SELECT *
        FROM histories ORDER BY year ASC
        """
    ).fetchall()

    data_dict = {'tahun': [row['year'] for row in data], 'jml_mhs': [row['student'] for row in data]}
    lstm_model = LSTMModel(input_shape=(2, 1))

    # Check if the model exists
    model_exists = os.path.exists(lstm_model.model_path)
    plot_url = None

    if model_exists:
        # Load the pre-trained model
        lstm_model.load()

        # Generate the prediction plot
        plot_url = lstm_model.plot_predictions(data_dict, sequence_length=2, num_predictions=3)
        return jsonify({'plot_url': plot_url})
    else:
        return jsonify({'message': 'Model not trained yet', 'status': 'error'}), 400
