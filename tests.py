import unittest

import functions
import numpy


class TestFittedLine(unittest.TestCase):

    def gen_data(self):
        """
        Returns some sample data to test with

        """
        x_data = [
            1.47, 1.50, 1.52,
            1.55, 1.57, 1.60,
            1.63, 1.65, 1.68,
            1.70, 1.73, 1.75,
            1.78, 1.80, 1.83
        ]
        y_data = [
            52.21, 53.12, 54.48,
            55.84, 57.20, 58.57,
            59.93, 61.29, 63.11,
            64.47, 66.28, 68.10,
            69.92, 72.19, 74.46
        ]
        return (x_data, y_data)

    def test_data_size(self):
        """
        Test that data of similar size is used.

        """
        x_data, y_data = self.gen_data()
        # Add to y data. These elements are expected to be ignored
        additional_y_data = [14.15, 16.12, 21.98]
        y_data += additional_y_data
        expected_len = min(len(x_data), len(y_data))
        line = functions.FittedLine(x_data, y_data)
        self.assertTrue(line.n == expected_len)
        self.assertTrue(len(line.x_data) == expected_len)
        self.assertTrue(len(line.y_data) == expected_len)
        # Assert that additional data is not included in line data
        for data in additional_y_data:
            self.assertTrue(data not in line.y_data)

    def test_wikipedia_example(self):
        """
        Tests that a sample provided on wikipedia works as expected.
        The article can be found at
            http://en.wikipedia.org/wiki/Simple_linear_regression

        """
        x_data, y_data = self.gen_data()
        line = functions.FittedLine(x_data, y_data)
        self.assertTrue(numpy.isclose(line.slope, 61.272))
        self.assertTrue(numpy.isclose(line.intercept, -39.0619))
        self.assertTrue(numpy.isclose(line.slope_range[0], 57.4355))
        self.assertTrue(numpy.isclose(line.slope_range[1], 65.1088))
        self.assertTrue(numpy.isclose(line.intercept_range[0], -45.4091))
        self.assertTrue(numpy.isclose(line.intercept_range[1], -32.7149))

    def test_nones(self):
        """
        Tests the handling of Nones in data in that they should be removed.
        If nones are removed from y data, the corresponding values in
        x_data should also be remove and vice versa.

        """
        none_indexes = [1, 5, 8]

        # Try Nones in y_data first
        x_data, y_data = self.gen_data()
        expected_len = len(y_data) - len(none_indexes)
        values_to_be_cut = [x_data[i] for i in none_indexes]
        for i in none_indexes:
            y_data[i] = None
        line = functions.FittedLine(x_data, y_data)
        self.assertTrue(len(line.x_data) == expected_len)
        self.assertTrue(len(line.y_data) == expected_len)
        for data in values_to_be_cut:
            self.assertTrue(data not in line.x_data)
        self.assertTrue(None not in line.y_data)

        # Try Nones in x_data next
        x_data, y_data = self.gen_data()
        expected_len = len(y_data) - len(none_indexes)
        values_to_be_cut = [y_data[i] for i in none_indexes]
        for i in none_indexes:
            x_data[i] = None
        line = functions.FittedLine(x_data, y_data)
        self.assertTrue(len(line.x_data) == expected_len)
        self.assertTrue(len(line.y_data) == expected_len)
        for data in values_to_be_cut:
            self.assertTrue(data not in line.y_data)
        self.assertTrue(None not in line.x_data)

        # Try Nones in both
        x_data, y_data = self.gen_data()
        x_none_indexes = [1, 2]
        y_none_indexes = [3, 4]
        none_indexes = set(x_none_indexes).union(y_none_indexes)
        expected_len = len(x_data) - len(none_indexes)
        x_cut_values = []
        for i in x_none_indexes:
            x_cut_values.append(x_data[i])
            x_data[i] = None
        y_cut_values = []
        for i in y_none_indexes:
            y_cut_values.append(y_data[i])
            y_data[i] = None
        line = functions.FittedLine(x_data, y_data)
        self.assertTrue(len(line.x_data) == expected_len)
        self.assertTrue(None not in line.x_data)
        self.assertTrue(len(line.y_data) == expected_len)
        self.assertTrue(None not in line.y_data)
        for v in x_cut_values:
            self.assertTrue(v not in line.x_data)
        for v in y_cut_values:
            self.assertTrue(v not in line.y_data)


if __name__ == '__main__':
    unittest.main()
