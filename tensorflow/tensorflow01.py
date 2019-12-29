import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# create data
x_data = np.random.rand(100).astype(np.float32)
print(x_data)
#x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.005, x_data.shape)

y_data = x_data**2 + 0.3
#y_data = x_data*0.1 + 0.3 


# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1],1.0, 3.0))
biases = tf.Variable(tf.zeros([1]))

y = x_data*Weights + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
#optimizer = tf.train.AdamOptimizer(0.5)
#optimizer = tf.train.MomentumOptimizer(0.4,0.5)

#optimizer = tf.train.RMSPropOptimizer(0.1)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### create tensorflow structure end ###


with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction = x_data*sess.run(Weights) + sess.run(biases)
        lines = ax.plot(x_data,prediction, 'r-', lw=1)
        plt.pause(0.1)
        if step % 20 == 0:
            print(step, sess.run(Weights), sess.run(biases))
#            try:
#                 ax.lines.remove(lines[0])
#            except Exception:
#                 pass
        # plot the prediction
           
#            lines = ax.plot(x_data,sess.run(y), 'r-', lw=5)
#            ax.lines.remove(lines[0])
#            plt.pause(0.1)
#            ax.lines.remove(lines[0])
    
    
