import functools
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

    if model_exists:
        # Load the pre-trained model
        lstm_model.load()

        # Generate the plot if the model exists
        plot_url = lstm_model.plot_predictions(data_dict, sequence_length=2, num_predictions=3)

        # Evaluate the model
        evaluation_metrics = lstm_model.evaluate(data_dict, sequence_length=2)

    return render_template('prediction/index.html', plot_url=plot_url, model_exists=model_exists, evaluation_metrics=evaluation_metrics)


@bp.route('/train', methods=['POST'])
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

    # Train the model and save it
    lstm_model.train(data_dict, sequence_length=2, epochs=200, batch_size=1)

    return jsonify({'message': 'Model trained successfully!', 'status': 'success'})


@bp.route('/predict', methods=['POST'])
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
