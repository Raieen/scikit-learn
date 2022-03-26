import numpy as np
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.base import BaseEstimator, ClassifierMixin, check_X_y
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state


class TemplateEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, random_state=None):
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X, y):
        self.random_state_ = check_random_state(self.random_state)
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        y_hat = self.random_state_.choice(self.classes_, size=X.shape[0])
        return y_hat

    def _more_tags(self):
        return {
            "non_deterministic": True,
            "no_validation": True,
            "poor_score": True,
        }


@parametrize_with_checks([TemplateEstimator()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)