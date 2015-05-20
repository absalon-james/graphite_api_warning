import datetime
import dateutil
import math
import pprint
import traceback

from graphite_api.functions import \
    _fetchWithBootstrap,  \
    safeMul, \
    safeSum
from graphite_api.render.datalib import TimeSeries
from itertools import izip_longest
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
        self.n = min(len(x_data), len(y_data))
        self.x_data = x_data[:self.n]
        self.y_data = y_data[:self.n]
        self.slope, \
            self.intercept, \
            self.r_value, \
            self.p_value, \
            self.std_error = stats.linregress(x_data, y_data)

        self.t = stats.t.ppf(1 - 0.025, self.n - 2)

        self._sums()
        self._errors()
        self.slope_range = [self.slope - (self.t * self.error_slope),
                            self.slope + (self.t * self.error_slope)]

        self.intercept_range = [
            self.intercept - (self.t * self.error_intercept),
            self.intercept + (self.t * self.error_intercept)
        ]

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
        debug("R-squares: %s\n" % self.r_value ** 2)
        debug("Error sigma squared: %s\n" % self.error_sigma)
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
    debug("series start: %s\n" % series.start)
    debug("series end: %s\n" % series.end)
    debug("series step: %s\n" % series.step)
    n = len(time_range)
    debug("n: %s\n" % n)
    if len(series) > n:
        series = series[:n]
    debug("Number of values: %s\n" % len(series))

    line = FittedLine(time_range, series)

    debug("scipy_slope: %s\n" % line.slope)
    debug("scipy_int: %s\n" % line.intercept)
    debug("scipy_r_squared: %s\n" % line.r_value ** 2)
    debug("scipy_p_value: %s\n" % line.r_value ** 2)
    debug("scipy_stderr: %s\n" % line.std_error)
    debug("scipy_slope_range: %s\n" % line.slope_range)
    debug("scipy_int_range: %s\n" % line.intercept_range)
    return line


def time_delta(a, b):
    debug("Time delta %s, %s\n" % (a, b))
    datetime_a = datetime.datetime.fromtimestamp(int(a))
    debug("Time a: %s\n" % datetime_a)
    datetime_b = datetime.datetime.fromtimestamp(int(b))
    debug("Time b %s\n" % datetime_b)
    try:
        delta = dateutil.relativedelta.relativedelta(datetime_b, datetime_a)
    except Exception as e:
        debug("%s" % e)
        raise
    debug("Finished delta: %s\n" % pprint.pformat(delta))
    return delta


def time_delta_string(delta):
    debug("Delta string %s\n" % pprint.pformat(delta))
    parts = []
    if delta.years != 0:
        parts.append("%s years" % delta.years)
    if delta.months != 0:
        parts.append("%s months" % delta.months)
    if delta.days != 0:
        parts.append("%s days" % delta.days)
    if delta.hours != 0:
        parts.append("%s hours" % delta.hours)
    if parts:
        return ', '.join(parts)
    return None


def leastSquaresIntercept(requestContext, seriesList, value, days=None):
    """
    Calculates the time at which the line of best fit created by
    the least squared method will have the expected value.

    @param requestContext
    @param seriesList - List of time series
    @param value - Value to find intercept for

    """
    debug("Least squares intercept. looking for %s\n" % value)
    if days is None:
        days = LEAST_SQUARED_DAYS
    result = []
    bootstrapList = _fetchWithBootstrap(requestContext, seriesList,
                                        days=int(days))
    for bootSeries, series in izip_longest(bootstrapList, seriesList):
        line = least_squared_line(bootSeries)
        m, b, r_squared = (line.slope, line.intercept, line.r_value ** 2)
        debug("Least Squares Intercept:\n")
        debug("slope: %s\n" % m)
        debug("y-intercept: %s\n" % b)
        debug("r_squared: %s\n" % r_squared)
        debug(pprint.pformat(requestContext))
        debug("\n")

        debug("Starting string time trends\n")

        t_trend = int(line.predict_mean_response(value))
        delta_trend = time_delta(series.end, t_trend)
        string_trend = time_delta_string(delta_trend)
        debug("Delta trend: %s\n" % string_trend)
        result.append(TimeSeries(
            series.name,
            series.start, series.start + 1, series.step,
            [string_trend]
        ))

        t_low = int(line.predict_lower(0.95, value))
        delta_low = time_delta(series.end, t_low)
        string_low = time_delta_string(delta_low)
        debug("Delta low: %s\n" % string_low)
        result.append(TimeSeries(
            "%s: low" % series.name,
            series.start, series.start + 1, series.step,
            [string_low]
        ))

        t_high = int(line.predict_upper(0.95, value))
        delta_high = time_delta(series.end, t_high)
        string_high = time_delta_string(delta_high)
        debug("Delta high: %s\n" % string_high)
        result.append(TimeSeries(
            "%s: high" % series.name,
            series.start, series.start + 1, series.step,
            [string_high]
        ))

        debug("Times for %s\n" % value)
        debug("low: %s\n" % string_low)
        debug("mean: %s\n" % string_trend)
        debug("high: %s\n" % string_high)
    return result


def leastSquaresTest(requestContext, seriesList):
    debug("Starting least squares test for %s\n" % requestContext)
    debug("Length of input series: %s\n" % len(seriesList))
    result = []
    for series in seriesList:
        debug("Looking at series %s\n" % series.name)
        time_range = range(series.start, series.end, series.step)
        debug("Created time range\n")
        src_time = time_range[:len(time_range) / 2]
        debug("Created src range\n")
        dest_time = time_range[len(time_range) / 2:]
        debug("Created dest range\n")
        src_series = TimeSeries(
            series.name,
            src_time[0], src_time[-1], series.step,
            series[:len(src_time)]
        )
        debug("Created the src series.  Length of %s\n" % len(src_series))

        try:
            line = least_squared_line(src_series)
        except Exception as e:
            debug("%s\n" % traceback.format_exc())
            return []

        debug("Created the test line\n")
        result.append(TimeSeries(
            series.name,
            dest_time[0], dest_time[-1], series.step,
            map(line.line_generator(), dest_time)
        ))
        debug("Created the trend line\n")

        result.append(TimeSeries(
            series.name,
            dest_time[0], dest_time[-1], series.step,
            map(line.prediction_band_lower(0.95), dest_time)
        ))
        debug( "Created the lower line\n")

        result.append(TimeSeries(
            series.name,
            dest_time[0], dest_time[-1], series.step,
            map(line.prediction_band_upper(0.95), dest_time)
        ))
        debug("Created the upper line\n")
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
    result = []
    for series in seriesList:
        result.append(TimeSeries(
            series.name,
            series.start + series.step, series.end, series.step,
            [series[i] - series[i - 1] for i in xrange(1, len(series))]
        ))
    return result


def removeTrendByLine(requestContext, seriesList):
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
    'leastSquaresTest': leastSquaresTest,
    'leastSquaresIntercept': leastSquaresIntercept,
    'removeTrendByDifferences': removeTrendByDifferences,
    'removeTrendByLine': removeTrendByLine
}
