import tensorflow as tf
import numpy as np
import os
from PIL import Image as pImage

pathDir = "./image"
fileList = os.listdir(pathDir)
print(fileList)

X = tf.placeholder(tf.float32, [None, 300, 300, 1])
Y = tf.placeholder(tf.float32, [None, 2])
isTraining = tf.placeholder(tf.bool)
global_step = tf.Variable(0, trainable=False, name='global_step')

L1 = tf.layers.conv2d(X, 128, [3, 3], activation=tf.nn.relu, padding='SAME')
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
L1 = tf.layers.dropout(L1, 0.7, isTraining)

L2 = tf.layers.conv2d(L1, 256, [3, 3], activation=tf.nn.relu, padding='SAME')
L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
L2 = tf.layers.dropout(L2, 0.7, isTraining)

L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, isTraining)

model = tf.layers.dense(L3, 2, activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost, global_step)

imgs = []
i = 0
for imgName in fileList:
    imgTemp = pImage.open(pathDir + '/' + imgName)
    imgTemp = imgTemp.resize((300, 300))
    imgs.append(imgTemp.convert("L"))

print(imgs)

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("branch1")
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    print("branch2")
    sess.run(tf.global_variables_initializer())
print("variables init complete")

for epoch in range(1):
    totalCost = 0

    for img in imgs:
        print(img)
        data = np.array(img)
        img.show()
        print(data)
        target = []
        if imgName.split('_')[1] == "i":
            target = [0, 1]
        else:
            target = [1, 0]

        _, costVal = sess.run([optimizer, cost], feed_dict={X: data, Y: target, isTraining: True})
        print(target)
        totalCost += costVal

    print('Epoch: ', '%04d' % sess.run(global_step),
          'Avg. Cost: %3f' % (totalCost / len(fileList)))

saver.save(sess, './model/fald.ckpt', global_step=global_step)