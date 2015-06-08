Description
===========

A collection of custom processing functions for graphite-api intended for use
with the custom warning panel plugin written for grafana. The included
functions will make a line of best fit in an attempt to estimate a trend.
Seasonality is not accounted for.

Requirements
============
* [numpy](http://www.numpy.org/)
* [scipy](http://www.scipy.org/)
* [graphite-api](http://graphite-api.readthedocs.org/en/latest/)

Installation
============

* Clone this repo
* Add it to the python path
* Update the graphite-api configuration to include the following:

```yaml
...
functions:
  - graphite_api.functions.SeriesFunctions
  - graphite_api.functions.PieFunctions
  - graphite_api_warning.functions.CustomFunctions
...
```

Custom Functions
================

**leastSquares(target, days)**  
Will generate a lower prediction line, a trend line, and an upper prediction
line.

Example graphite-api target: leastSquares(*.*.*.*.checks.agent.cpu.*.usage_average, 60)

| Parameter | Description |
| --------- | ----------- |
| target    | graphite-api target|
| days      | Number of days back to fetch data when making the estimation. Defaults to 60 days. |


**leastSquaresIntercept(target, threshold, days, id)**  
Will calculate the time at which the threshold will be crossed on the lower
and upper prediction lines. This information will be relayed back as a simple
json object instead of a proper time series. The grafana warning panel plugin
expects this object.

| Parameter | Description |
| --------- | ----------- |
| target    | graphite-api target |
| threshold | Value to find time intercepts for |
| days      | Number of days back to fetch data when making the estimation. Defaults to 60 days. |
| id        | Optional value. This value is included in the result. |
