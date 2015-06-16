import math
import traceback

from graphite_api.functions import _fetchWithBootstrap
from graphite_api.render.datalib import TimeSeries
from itertools import izip, izip_longest
from scipy import stats

LEAST_SQUARED_DAYS = 60


class FittedLine(object):
    """
    Ordinary Least Squares(OLS) linear regression model

    """
    def __init__(self, x_data, y_data):
        """
        Inits the the model

        @param x_data - collection of datapoints for the x-axis or
            independent variable
        @param y_data - collection of datapoints for the y-axis or
            response variable

        """
        self.x_data = x_data
        self.y_data = y_data
        self.prep_data()
        self.slope, \
            self.intercept, \
            self.r_value, \
            self.p_value, \
            self.std_error = stats.linregress(self.x_data, self.y_data)

        if self.slope == 0:
            raise Exception("0 Slope")

        self.t = stats.t.ppf(1 - 0.025, self.n - 2)

        self._sums()
        self._errors()
        self.slope_range = [self.slope - (self.t * self.error_slope),
                            self.slope + (self.t * self.error_slope)]

        self.intercept_range = [
            self.intercept - (self.t * self.error_intercept),
            self.intercept + (self.t * self.error_intercept)
        ]

    def prep_data(self):
        """
        Prepares the data. Ensures the same dimensions and looks for
        None values.

        """
        # Trim to same dimensions
        self.n = min(len(self.x_data), len(self.y_data))
        self.x_data = self.x_data[:self.n]
        self.y_data = self.y_data[:self.n]

        # Take care of None values. Must remove pairs

        # First find indexes
        nones = [i for i, p in
                 enumerate(izip(self.x_data, self.y_data))
                 if p[0] is None or p[1] is None]

        # Nones are in ascending order, iterate backwards
        for i in reversed(nones):
            del self.x_data[i]
            del self.y_data[i]

        # Set new n to account for reduced data points
        self.n = self.n - len(nones)

    def _sums(self):
        """
        Compute sums of various parts to be reused in calculation later

        """
        squared = lambda l: l ** 2
        mult = lambda x, y: x * y
        self.sum_x = sum(self.x_data)
        self.sum_y = sum(self.y_data)
        self.sum_xx = sum(map(squared, self.x_data))
        self.sum_yy = sum(map(squared, self.y_data))
        self.sum_xy = sum(map(mult, self.x_data, self.y_data))

    def _errors(self):
        """
        Compute estimation of error in the slope and in the intercept

        """
        self.error_sigma = (
            (
                (self.n * self.sum_yy) -
                (self.sum_y ** 2) -
                (
                    (self.slope ** 2) *
                    ((self.n * self.sum_xx) - (self.sum_x ** 2))
                )
            ) / (self.n * (self.n - 2))
        )
        self.error_sigma = math.sqrt(self.error_sigma)

        self.error_slope = (
            (self.n * (self.error_sigma ** 2)) /
            ((self.n * self.sum_xx) - (self.sum_x ** 2))
        )
        self.error_slope = math.sqrt(self.error_slope)

        self.error_intercept = ((self.error_slope ** 2) * self.sum_xx) / self.n
        self.error_intercept = math.sqrt(self.error_intercept)

    def line_generator(self):
        """
        Returns a lambda to generate a line for the mean response

        @return lambda

        """
        return lambda x: (self.slope * x) + self.intercept

    def prediction_band_lower(self, q):
        """
        Returns a lambda to generate a line for the lower prediction band

        @return lambda

        """
        alpha = (1 - q) / 2
        t = stats.t.ppf(1 - alpha, self.n - 2)
        return lambda x: (
            (self.slope * x) +
            self.intercept -
            (t * self.error_sigma)
        )

    def prediction_band_upper(self, q):
        """
        Returns a lambda to generate a line for the upper prediction band

        @return lambda

        """
        alpha = (1 - q) / 2
        t = stats.t.ppf(1 - alpha, self.n - 2)
        return lambda x: (
            (self.slope * x) +
            self.intercept +
            (t * self.error_sigma)
        )

    def predict_mean_response(self, value):
        """
        Predicts the mean value of the independent variable when the response
        variable is value

        @param value - Float
        @return Float

        """
        return (value - self.intercept) / self.slope

    def predict_lower(self, q, value):
        """
        Predicts the lower prediction value of the independent varia1ble when
        the response variable is value

        @param q - Float - 0.95 = 95% likelyhood
        @param value - Float
        @return Float

        """
        alpha = (1 - q) / 2
        t = stats.t.ppf(1 - alpha, self.n - 2)
        return (value - self.intercept + (t * self.error_sigma)) / self.slope

    def predict_upper(self, q, value):
        """
        Predicts the upper prediction value of the independent variable when
        the response variable is value

        @param q - Float - 0.95 = 95% likelyhood
        @param value - Float
        @return Float

        """
        alpha = (1 - q) / 2
        t = stats.t.ppf(1 - alpha, self.n - 2)
        return (value - self.intercept - (t * self.error_sigma)) / self.slope


def debug(msg):
    with open('/tmp/debug', 'a') as f:
        f.write(msg)


def least_squared_line(series):
    """
    Calculates the line of best fit from a TimeSeries. Returns
    the slop and y intercept of a line of the form mx + b.

    @param series - graphite_api.render.datalib.Timeseries
    @return tuple - Two tuple (float, float)

    """
    time_range = range(series.start, series.end, series.step)
    return FittedLine(time_range, series)


def last_value(series):
    """
    Grabs the last known non None value

    @param series - TimeSeries

    """
    for v in reversed(series):
        if v is not None:
            return v
    return None


def leastSquaresIntercept(requestContext, seriesList, threshold,
                          days=None, id=None):
    """
    Calculates the time at which the line of best fit created by
    the least squared method will have the expected value.

    @param requestContext
    @param seriesList - List of time series
    @param threshold - Value to find intercept for
    @param days - Number of days back to fetch data
    @param id - Special identifier to send back if specified

    """
    if days is None:
        days = LEAST_SQUARED_DAYS
    result = []
    bootstrapList = _fetchWithBootstrap(requestContext, seriesList,
                                        days=int(days))
    for bootSeries, series in izip_longest(bootstrapList, seriesList):
        try:
            line = least_squared_line(bootSeries)
        except:
            debug("%s\n" % traceback.format_exc())
            continue

        m, _, r_squared = (line.slope, line.intercept, line.r_value ** 2)
        t_trend = int(line.predict_mean_response(threshold))
        t_low = int(line.predict_lower(0.95, threshold))
        t_high = int(line.predict_upper(0.95, threshold))

        obj = {
            'intercepts': {
                'lower': t_low,
                'trend': t_trend,
                'upper': t_high
            },
            'r_squared': r_squared,
            'slope': m,
            'trend_now': line.line_generator()(series.end),
            'threshold': threshold,
            'last': last_value(series)
        }

        if id is not None:
            obj['id'] = id

        result.append(TimeSeries(
            series.name,
            series.start, series.start + 1, series.step,
            [obj]
        ))
    return result


def leastSquares(requestContext, seriesList, days=None):
    """
    Creates a new time series with extrapolated values based on a least
    squares approximation of a line of best fit

    @param - requestContext
    @param - seriesList
    @return - List

    """
    if days is None:
        days = LEAST_SQUARED_DAYS
    result = []
    bootstrapList = _fetchWithBootstrap(requestContext, seriesList,
                                        days=int(days))
    for oldSeries, series in izip_longest(bootstrapList, seriesList):
        time_range = range(series.start, series.end, series.step)
        line = least_squared_line(oldSeries)

        # Add modeled mean response line (trend)
        result.append(TimeSeries(
            series.name,
            series.start, series.end, series.step,
            map(line.line_generator(), time_range)
        ))

        # Add modeled lower prediction band
        result.append(TimeSeries(
            series.name,
            series.start, series.end, series.step,
            map(line.prediction_band_lower(0.95), time_range)
        ))

        # Add modeled upper prediction band
        result.append(TimeSeries(
            series.name,
            series.start, series.end, series.step,
            map(line.prediction_band_upper(0.95), time_range)
        ))
    return result


def removeTrendByDifferences(requestContext, seriesList):
    """
    Experimental whitening. Computes new line by subtracting
    the value at t-1 from the value at t.

    @param requestContext
    @param seriesList
    @return - List

    """
    result = []
    for series in seriesList:
        result.append(TimeSeries(
            series.name,
            series.start + series.step, series.end, series.step,
            [series[i] - series[i - 1] for i in xrange(1, len(series))]
        ))
    return result


def removeTrendByLine(requestContext, seriesList):
    """
    Experimental whitening. Computes new series by subtracting the value
    of the trend line from the actual value.

    @param requestContext
    @param seriesList - List
    @return - List

    """
    result = []
    for series in seriesList:
        line = least_squared_line(series)
        trend = line.line_generator()
        diff = lambda x, y: y - trend(x)
        result.append(TimeSeries(
            series.name,
            series.start, series.end, series.step,
            map(diff, line.x_data, line.y_data)
        ))
    return result

CustomFunctions = {
    'leastSquares': leastSquares,
    'leastSquaresIntercept': leastSquaresIntercept,
}
