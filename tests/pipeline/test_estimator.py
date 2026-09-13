"""
Tests for medpipe.pipeline.estimator.DistributionalPipeline.
"""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone

from medpipe.pipeline.estimator import DistributionalPipeline


class _PredDistEstimator(BaseEstimator):
    """Dummy estimator mimicking NGBoost's `pred_dist` naming convention."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def pred_dist(self, X):
        return f"pred_dist:{len(X)}"


class _PredictDistEstimator(BaseEstimator):
    """Dummy estimator mimicking OrdBoost's presumed `predict_dist` naming."""

    def fit(self, X, y=None):
        return self

    def predict_dist(self, X):
        return f"predict_dist:{len(X)}"


class _NoDistEstimator(BaseEstimator):
    """Dummy estimator implementing neither distributional method."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class _Passthrough(BaseEstimator):
    """Dummy preprocessing step that records the data it transformed."""

    def fit(self, X, y=None):
        # Set a trailing-underscore attribute so sklearn's check_is_fitted
        # recognizes this transformer (and any Pipeline slice containing
        # only it) as fitted.
        self.n_features_in_ = getattr(X, "shape", (None, None))[1]
        return self

    def transform(self, X):
        return X * 2


class TestDistributionalPipelinePredictDist:
    """Unit tests for DistributionalPipeline.predict_dist dispatch."""

    def test_dispatches_to_pred_dist(self):
        """Test dispatch to the estimator's `pred_dist` method (NGBoost-style)."""
        pipeline = DistributionalPipeline([("estimator", _PredDistEstimator())])
        X = np.array([[1.0], [2.0], [3.0]])
        pipeline.fit(X)

        assert pipeline.predict_dist(X) == "pred_dist:3"

    def test_dispatches_to_predict_dist(self):
        """Test dispatch to the estimator's `predict_dist` method
        (OrdBoost-style)."""
        pipeline = DistributionalPipeline([("estimator", _PredictDistEstimator())])
        X = np.array([[1.0], [2.0], [3.0]])
        pipeline.fit(X)

        assert pipeline.predict_dist(X) == "predict_dist:3"

    def test_raises_attribute_error_when_neither_method_exists(self):
        """Test that a clear AttributeError is raised when the final
        estimator supports neither distributional method."""
        pipeline = DistributionalPipeline([("estimator", _NoDistEstimator())])
        X = np.array([[1.0], [2.0]])
        pipeline.fit(X)

        with pytest.raises(
            AttributeError,
            match="implements neither 'predict_dist' nor 'pred_dist'",
        ):
            pipeline.predict_dist(X)

    def test_applies_preprocessing_before_dispatch(self):
        """Test that preceding pipeline steps still transform X before the
        final estimator's distributional method is called."""
        pipeline = DistributionalPipeline(
            [("scale", _Passthrough()), ("estimator", _PredDistEstimator())]
        )
        X = np.array([[1.0], [2.0]])
        pipeline.fit(X)

        # _Passthrough doubles values but does not change sample count,
        # so only the sample count is observable through pred_dist here.
        assert pipeline.predict_dist(X) == "pred_dist:2"

    def test_no_preprocessing_step_skips_transform(self):
        """Test that a single-step pipeline (estimator only) does not
        attempt to call transform() on an empty sub-pipeline."""
        pipeline = DistributionalPipeline([("estimator", _PredDistEstimator())])
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        pipeline.fit(X)

        assert pipeline.predict_dist(X) == "pred_dist:4"


class TestDistributionalPipelineCloneSafety:
    """Tests ensuring DistributionalPipeline survives sklearn.base.clone()."""

    def test_clone_preserves_subclass_and_params(self):
        """Test that clone() returns a DistributionalPipeline with
        equivalent (unfitted) parameters, as required for GridSearchCV/
        cross_validate to work with this subclass."""
        pipeline = DistributionalPipeline(
            [("scale", _Passthrough()), ("estimator", _PredDistEstimator())]
        )

        cloned = clone(pipeline)

        assert type(cloned) is DistributionalPipeline
        assert cloned.get_params(deep=False).keys() == pipeline.get_params(
            deep=False
        ).keys()

    def test_cloned_pipeline_still_exposes_predict_dist(self):
        """Test that a cloned, freshly-fitted pipeline still dispatches
        predict_dist correctly."""
        pipeline = DistributionalPipeline([("estimator", _PredDistEstimator())])
        cloned = clone(pipeline)

        X = np.array([[1.0], [2.0], [3.0]])
        cloned.fit(X)

        assert cloned.predict_dist(X) == "pred_dist:3"
