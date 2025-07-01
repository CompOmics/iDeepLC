from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline
import numpy as np
import logging


LOGGER = logging.getLogger(__name__)

class SplineTransformerCalibration:
    """Spline Transformer Calibration for Retention Time Prediction."""

    def __init__(self):
        """Initialize the Spline Transformer Calibration."""
        self._fit = False
        self._calibrate_min = None
        self._calibrate_max = None
        self._linear_model_left = None
        self._spline_model = None
        self._linear_model_right = None

    def fit(self, measured_tr: np.ndarray, predicted_tr: np.ndarray, simplified: bool = False):
        """
        Fit a SplineTransformer model to the measured and predicted retention times.

        Parameters
        ----------
        measured_tr: np.ndarray
            The actual measured retention times.
        predicted_tr: np.ndarray
            The predicted retention times from the model.
        simplified: bool, optional
            If True, use a simplified model. Default is False.
        """
        LOGGER.info("Fitting Spline Transformer Calibration...")

        # Ensure inputs are numpy arrays
        measured_tr = np.asarray(measured_tr)
        predicted_tr = np.asarray(predicted_tr)

        # Check if the lengths match
        if len(measured_tr) != len(predicted_tr):
            raise ValueError("Measured and predicted retention times must have the same length.")

        # Fit a SplineTransformer model
        if simplified:
            spline = SplineTransformer(degree=2, n_knots=10)
            linear_model = LinearRegression()
            linear_model.fit(predicted_tr.reshape(-1, 1), measured_tr)

            linear_model_left = linear_model
            spline_model = linear_model
            linear_model_right = linear_model
        else:
            spline = SplineTransformer(degree=4, n_knots=int(len(measured_tr) / 500) + 5)
            spline_model = make_pipeline(spline, LinearRegression())
            spline_model.fit(predicted_tr.reshape(-1, 1), measured_tr)

            # Linear extrapolation models for the extremes
            n_top = int(len(predicted_tr) * 0.1)

            # Fit linear model for the bottom 10% (left-side extrapolation)
            X_left = predicted_tr[:n_top]
            y_left = measured_tr[:n_top]
            linear_model_left = LinearRegression()
            linear_model_left.fit(X_left.reshape(-1, 1), y_left)

            # Fit linear model for the top 10% (right-side extrapolation)
            X_right = predicted_tr[-n_top:]
            y_right = measured_tr[-n_top:]
            linear_model_right = LinearRegression()
            linear_model_right.fit(X_right.reshape(-1, 1), y_right)

        # Store calibration information
        self._calibrate_min = min(predicted_tr)
        self._calibrate_max = max(predicted_tr)
        self._linear_model_left = linear_model_left
        self._spline_model = spline_model
        self._linear_model_right = linear_model_right

        self._fit = True


    def transform(self, tr: np.ndarray) -> np.ndarray:
        """
        Transform the predicted retention times using the fitted SplineTransformer model.

        Parameters
        ----------
        tr: np.ndarray
            The predicted retention times to be calibrated.

        Returns
        -------
        np.ndarray
            The calibrated retention times.
        """
        if not self._fit:
            print("The model has not been fitted yet. Please call fit() before transform().")

        # if tr.shape[0] == 0:
        #     return np.array([])
        tr_array = np.array(tr)
        tr = tr_array.reshape(-1,1)# Ensure tr is a 2D array for prediction

        # Get spline predictions and linear extrapolation predictions
        y_pred_spline = self._spline_model.predict(tr)
        y_pred_left = self._linear_model_left.predict(tr)
        y_pred_right = self._linear_model_right.predict(tr)

        # Use spline model within the range of the calibration
        within_range = (tr >= self._calibrate_min) & (tr <= self._calibrate_max)
        within_range = within_range.ravel()  # Make it 1D for indexing

        # Create the final predictions, replacing out-of-range values with linear extrapolation
        cal_preds = np.copy(y_pred_spline)
        cal_preds[~within_range & (tr.ravel() < self._calibrate_min)] = y_pred_left[
            ~within_range & (tr.ravel() < self._calibrate_min)
            ]
        cal_preds[~within_range & (tr.ravel() > self._calibrate_max)] = y_pred_right[
            ~within_range & (tr.ravel() > self._calibrate_max)
            ]

        return np.array(cal_preds)