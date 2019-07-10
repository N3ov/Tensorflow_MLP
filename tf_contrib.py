import tensorflow as tf
import numpy as np

# void features
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# define model, logicistic regression
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# define data, batch size, epochs size
x_train = np.array([i for i in range(1, 5)]).astype(float)
y_train = np.array([j for j in range(0, -4, -1)]).astype(float)

# print(type(x_train))
# print(type(np.array([1., 2., 3., 4.])))
# x_train = np.array([1., 2., 3., 4.])
# y_train = np.array([0., -1., -2., -3.])

x_eval = np.array([2.0, 5., 8., 1])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000)


# train model
estimator.fit(input_fn=input_fn, steps=1000)

# test model
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print(train_loss, eval_loss)

# {'loss': 1.3029852e-05, 'global_step': 1000} {'loss': 0.0029144106, 'global_step': 1000}


