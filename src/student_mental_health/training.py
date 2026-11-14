
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.student_mental_health import config
import pandas as pd


def build_preprocessor(X_train):
    # Build numerical and categorical preprocessing pipelines.

    numerical_features = config.get_columns(X_train, 'number')

    categorical_features = config.get_columns(X_train, 'object' )

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), 
                                         ('scaler', StandardScaler()) ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                           ('encoder', OneHotEncoder(handle_unknown='ignore')) ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
                                    transformers=[('num', numerical_pipeline, numerical_features), 
                                                  ('cat', categorical_pipeline, categorical_features) ],
                                    remainder='drop' )

    return preprocessor


def get_models():
    # Define classification models.

    models = {
              'Logistic Regression': LogisticRegression(
                  max_iter=1000,
                  random_state=config.RANDOM_SEED
              ),
      
              'Random Forest': RandomForestClassifier(
                  random_state=config.RANDOM_SEED
              ),
      
              'SVM': SVC(
                  probability=True,
                  random_state=config.RANDOM_SEED
              ) }

    return models


def build_model_pipelines(preprocessor, models):
    # Create model pipelines dynamically.

    model_pipelines = {}

    for name, model in models.items():

        model_pipeline = Pipeline(steps=[('preprocessor', clone(preprocessor)),
                                         ('classifier', model) ])

        model_pipelines[name] = model_pipeline

    return model_pipelines


def train_models(model_pipelines, X_train, y_train):
    # Train baseline classification models.

    trained_models = {}

    for name, pipeline in model_pipelines.items():

        print(f'Training {name}')

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

        print(f'Training {name}: Done')

    print('All models training: Done')

    return trained_models


def get_param_grids():
    # Define hyperparameter grids for selected models.

    param_grids = {
                    'Logistic Regression': {
                        'classifier__C': [0.01, 0.1, 1, 10, 100],
                        'classifier__penalty': ['l2'],
                        'classifier__solver': ['lbfgs'],
                        'classifier__class_weight': [None, 'balanced']
                    },

                    'SVM': [
                        {
                            'classifier__kernel': ['linear'],
                            'classifier__C': [0.1, 1, 10, 100],
                            'classifier__class_weight': [None, 'balanced']
                        },
                        {
                            'classifier__kernel': ['rbf'],
                            'classifier__C': [0.1, 1, 10, 100],
                            'classifier__gamma': ['scale', 'auto'],
                            'classifier__class_weight': [None, 'balanced']
                        }
                    ]
                    }

    return param_grids


def optimize_models(model_pipelines, X_train, y_train, selected_models=None ):
    # Optimize selected models using GridSearchCV.

    if selected_models is None:
        selected_models = ['Logistic Regression', 'SVM']

    param_grids = get_param_grids()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED )

    grid_searches = {}
    best_estimators = {}

    for name in selected_models:

        print(f'\nOptimizing {name}')

        grid_search = GridSearchCV(estimator=model_pipelines[name], 
                                   param_grid=param_grids[name], 
                                   scoring='f1_weighted', 
                                   cv=cv, 
                                   n_jobs=1, 
                                   refit=True, 
                                   verbose=1 )

        grid_search.fit(X_train, y_train)

        grid_searches[name] = grid_search
        best_estimators[name] = grid_search.best_estimator_

        print(f'\n{name} fine-tuning completed')
        print(f'Best parameters: {grid_search.best_params_}')
        print(f'Best CV F1-score: {grid_search.best_score_:.4f}')

    print('\nAll fine-tuning completed')

    return grid_searches, best_estimators


def save_model(model, model_name='best_of_the_best_model'):
    # Save the trained model pipeline.

    models_dir = config.PROJECT_ROOT / 'models'

    models_dir.mkdir(parents=True, 
                     exist_ok=True )

    filename = model_name.lower().replace(' ', '_') + '.joblib'

    model_path = models_dir / filename

    joblib.dump(model, model_path)

    print(f'Model saved: {model_path}')

    return model_path



def train_model():

    # Load processed training data
    train_path = config.PROCESSED_DATA_DIR / 'train.csv'
    train_df = pd.read_csv(train_path)

    # Separate features and target
    X_train = train_df.drop(columns=[config.TARGET_COLUMN])
    y_train = train_df[config.TARGET_COLUMN]

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)

    # Get models
    models = get_models()

    # Build model pipelines
    model_pipelines = build_model_pipelines(preprocessor, models)

    # Train baseline models
    trained_models = train_models(
        model_pipelines,
        X_train,
        y_train
    )

    # Optimize selected models
    param_grids = get_param_grids()

    grid_searches, best_estimators = optimize_models(
        model_pipelines,
        X_train,
        y_train
    )

    # Select final model
    final_model_name = 'Logistic Regression'
    final_model = best_estimators[final_model_name]

    # Save final model
    model_path = save_model(final_model, final_model_name)

    print(f'Final model saved successfully: {model_path}')

    return str(model_path)
