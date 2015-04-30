import functions

x_data = [1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78 , 1.80, 1.83]
y_data = [52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46]

line = functions.FittedLine(x_data, y_data)

print "Length x_data: %s" % len(x_data)
print "Length y_data: %s" % len(y_data)

print "Slope: %s" % line.slope
print "Intercept: %s" % line.intercept
print "Slope range: %s" % line.slope_range
print "Intercept range: %s" % line.intercept_range
