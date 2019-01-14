# The user can modify the values of the weight w
# as well as bias_value_1 and bias_value_2 to observe
# how this plots to different step functions

import matplotlib.pyplot as plt
import numpy

weight_value = 1000

# modify to change where the step function starts
bias_value_1 = 5000

# modify to change where the step function ends
bias_value_2 = -5000

# plot the
plt.axis([-10, 10, -1, 10])

print("The step function starts at {0} and ends at {1}"
      .format(-bias_value_1 / weight_value,
              -bias_value_2 / weight_value))

inputs = numpy.arange(-10, 10, 0.01)
outputs = list()

# iterate over a range of inputs
for x in inputs:
    y1 = 1.0 / (1.0 + numpy.exp(-weight_value * x - bias_value_1))
    y2 = 1.0 / (1.0 + numpy.exp(-weight_value * x - bias_value_2))

    # modify to change the height of the step function
    w = 7

    # network output
    y = y1 * w - y2 * w

    outputs.append(y)

plt.plot(inputs, outputs, lw=2, color='black')
plt.show()
