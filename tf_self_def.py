import tensorflow as tf
import numpy as np

# self define model
def model(features, labels, mode):

    # create a linear model
    W = tf.get_variable("W", [1], dtype=tf.float64)
    # print(W)
    b = tf.get_variable("b", [1], dtype=tf.float64)
    y = W * features["x"] + b

    # create loss function
    loss = tf.reduce_sum(tf.square(y - labels))

    # train sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    return tf.contrib.learn.ModelFnOps(mode=mode, predictions=y, loss=loss, train_op=train)

# def model
estimator = tf.contrib.learn.Estimator(model_fn=model)

# define data, batch size, epochs size
x_train = np.array([i for i in range(1, 5)]).astype(float)
y_train = np.array([j for j in range(0, -4, -1)]).astype(float)

x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_eval}, y_eval, 4, num_epochs=1000)

# model training
estimator.fit(input_fn=input_fn, steps=1000)

# test model
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)

print(train_loss, eval_loss)
# {'loss': 4.994929e-11, 'global_step': 1000} {'loss': 0.010100719, 'global_step': 1000}
