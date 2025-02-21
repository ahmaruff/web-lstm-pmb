import os
from flask import Flask
from flask_wtf.csrf import CSRFProtect


csrf = CSRFProtect()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'prediksipmb.sqlite'),
    )

    # Initialize CSRF
    csrf.init_app(app)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # database
    from . import db
    db.init_app(app)

    # BLUEPRINT
    from . import auth
    app.register_blueprint(auth.bp)

    from . import home
    app.register_blueprint(home.bp)

    from . import history
    app.register_blueprint(history.bp)

    from . import prediction
    app.register_blueprint(prediction.bp)

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app
