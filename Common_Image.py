import os
import re
import glob
import numpy as np
import PIL.Image as img
import random
from sklearn import metrics
from sklearn.utils.linear_assignment_ import linear_assignment
import tensorflow as tf

# Option
mode = 'Training'
num_cluster = 153
eps = 1e-10
height = 300
width = 300
channel = 3

def img_array(path):
    image = img.open(path)
    tmp = np.array(image)
    image.close()
    return tmp

# Regex
re_model = re.compile("^(\d+)_")

train = glob.glob('train/*.jpg')
train_path = os.listdir('train')
train_img_paths = [filename for filename in train_path if filename.endswith('jpg')]

# Image Resizing (HeightxWeightxChannel) and Get numpy array
train_X = np.array([np.array(img.open(fname).resize((height, width), img.ANTIALIAS)) for fname in train])
train_Y = np.array([re_model.match(img_path).group(1) for img_path in train_img_paths])
print('train load finished')

test = glob.glob('test/*.jpg')
test_path = os.listdir('test')
test_img_paths = [filename for filename in test_path if filename.endswith('jpg')]

# Image Resizing (HeightxWeightxChannel) and Get numpy array
test_X = np.array([np.array(img.open(fname).resize((height, width), img.ANTIALIAS)) for fname in test])
test_Y = np.array([re_model.match(img_path).group(1) for img_path in test_img_paths])
print('test load finished')

# Get Datas and Labels as batch_size
def get_batch(batch_size, img_data, imgt_labels):
    batch_index = random.sample(range(len(imgt_labels)), batch_size)

    batch_data = np.empty([batch_size, height, width, channel], dtype=np.float32)
    batch_label = np.empty([batch_size], dtype=np.int32)

    for n, i in enumerate(batch_index):
        batch_data[n, ...] = img_data[i, ...]
        batch_label[n] = imgt_labels[i]
    return batch_data, batch_label

# Get Datas and Labels as batch_size for Testing
def get_batch_test(batch_size, img_data, i):
    batch_data = np.copy(img_data[batch_size * i:batch_size * (i + 1), ...])
    return batch_data

def clustering_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def NMI(y_true, y_pred):
    return metrics.normalized_mutual_info_score(y_true, y_pred)

def ARI(y_true, y_pred):
    return metrics.adjusted_rand_score(y_true, y_pred)

tf.reset_default_graph()

def ConvNetwork(in_img, num_cluster, name='ConvNetwork', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        # 64 x 3 -> pooling
        conv1 = tf.layers.conv2d(in_img, 64, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv1 = tf.layers.batch_normalization(conv1, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv2d(conv1, 64, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv2 = tf.layers.batch_normalization(conv2, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv2d(conv2, 64, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2])
        conv3 = tf.layers.batch_normalization(conv3, axis=-1, epsilon=1e-5, training=True, trainable=False)

        # 128 x 3 -> pooling
        conv4 = tf.layers.conv2d(conv3, 128, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv4 = tf.layers.batch_normalization(conv4, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv4 = tf.nn.relu(conv4)
        conv5 = tf.layers.conv2d(conv4, 128, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv5 = tf.layers.batch_normalization(conv5, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv5 = tf.nn.relu(conv5)
        conv6 = tf.layers.conv2d(conv5, 128, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv6 = tf.nn.relu(conv6)
        conv6 = tf.layers.max_pooling2d(conv6, [2, 2], [2, 2])
        conv6 = tf.layers.batch_normalization(conv6, axis=-1, epsilon=1e-5, training=True, trainable=False)

        # 256 x 3 -> pooling
        conv7 = tf.layers.conv2d(conv6, 256, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv7 = tf.layers.batch_normalization(conv7, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv7 = tf.nn.relu(conv7)
        conv8 = tf.layers.conv2d(conv7, 256, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv8 = tf.layers.batch_normalization(conv8, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv8 = tf.nn.relu(conv8)
        conv9 = tf.layers.conv2d(conv8, 256, [3, 3], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        conv9 = tf.layers.batch_normalization(conv9, axis=-1, epsilon=1e-5, training=True, trainable=False)
        conv9 = tf.nn.relu(conv9)
        conv9 = tf.layers.max_pooling2d(conv9, [2, 2], [2, 2])
        conv9 = tf.layers.batch_normalization(conv9, axis=-1, epsilon=1e-5, training=True, trainable=False)

        # 1x1 kernel
        one = tf.layers.conv2d(conv9, num_cluster, [1, 1], [1, 1], padding='valid', activation=None, kernel_initializer=tf.keras.initializers.he_normal())
        one = tf.layers.batch_normalization(one, axis=-1, epsilon=1e-5, training=True, trainable=False)
        one = tf.nn.relu(one)
        one = tf.layers.average_pooling2d(one, [6, 6], [2, 2])
        flat = tf.layers.flatten(one)

        # two Full-Connected Layer
        # dense 0
        x0 = tf.layers.dense(flat, num_cluster, kernel_initializer=tf.initializers.identity())
        x0 = tf.layers.batch_normalization(x0, axis=-1, epsilon=1e-5, training=True, trainable=False)
        x0 = tf.nn.relu(x0)
        # dense 1
        x1 = tf.layers.dense(x0, num_cluster, kernel_initializer=tf.initializers.identity())
        x1 = tf.layers.batch_normalization(x1, axis=-1, epsilon=1e-5, training=True, trainable=False)
        x1 = tf.nn.relu(x1)

        out = tf.nn.softmax(x1)

    return out

image_pool_input = tf.placeholder(shape=[None, height, width, channel], dtype=tf.float32, name='image_pool_input')
u_thres = tf.placeholder(shape=[], dtype=tf.float32, name='u_thres')
l_thres = tf.placeholder(shape=[], dtype=tf.float32, name='l_thres')
lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

label_feat = ConvNetwork(image_pool_input, num_cluster, name='ConvNetwork', reuse=False)
label_feat_norm = tf.nn.l2_normalize(label_feat, dim=1)
sim_mat = tf.matmul(label_feat_norm, label_feat_norm, transpose_b=True)

pos_loc = tf.greater(sim_mat, u_thres, name='greater')
neg_loc = tf.less(sim_mat, l_thres, name='less')
pos_loc_mask = tf.cast(pos_loc, dtype=tf.float32)
neg_loc_mask = tf.cast(neg_loc, dtype=tf.float32)

pred_label = tf.argmax(label_feat, axis=1)

# Deep Adaptive Image Clustering Cost Function Optimize
pos_entropy = tf.multiply(-tf.log(tf.clip_by_value(sim_mat, eps, 1.0)), pos_loc_mask)
neg_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-sim_mat, eps, 1.0)), neg_loc_mask)

loss_sum = tf.reduce_mean(pos_entropy) + tf.reduce_mean(neg_entropy)
train_op = tf.train.RMSPropOptimizer(lr).minimize(loss_sum)

image_data = np.concatenate([train_X, test_X], axis=0)
image_label = np.concatenate([train_Y, test_Y], axis=0)

print(len(image_data))

# Compress Dimension
mapping = {}
mapped_label = 0
for index,data in enumerate(image_label):
  if(data in mapping):
    image_label[index] = mapping[data]
  else:
    image_label[index] = mapping[data] = mapped_label+1
    mapped_label += 1

saver = tf.train.Saver()
if mode == 'Training':
    batch_size = 12
    test_batch_size = 64
    base_lr = 0.001
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        lamda = 0
        epoch = 1
        u = 0.95
        l = 0.455

        while u > l:
            u = 0.95 - lamda
            l = 0.455 + 0.1 * lamda
            print(u, l)
            for i in range(1, int(1001)):  # 1000 iterations is roughly 1 epoch
                data_samples, _ = get_batch(batch_size, image_data, image_label)
                feed_dict = {image_pool_input: data_samples, u_thres: u, l_thres: l, lr: base_lr}
                train_loss, _ = sess.run([loss_sum, train_op], feed_dict=feed_dict)
                if i % 20 == 0:
                    print('training loss at iter %d is %f' % (i, train_loss))

            lamda += 1.1 * 0.009
            print(lamda)
            # run testing every epoch
            data_samples, data_labels = get_batch(test_batch_size, image_data, image_label)
            feed_dict = {image_pool_input: data_samples}
            pred_cluster = sess.run(pred_label, feed_dict=feed_dict)

            acc = c(data_labels, pred_cluster)
            nmi = NMI(data_labels, pred_cluster)
            ari = ARI(data_labels, pred_cluster)
            print('testing NMI, ARI, ACC at epoch %d is %f, %f, %f.' % (epoch, nmi, ari, acc))

            if epoch % 5 == 0:  # save model at every 5 epochs
                model_name = 'DAC_ep_' + str(epoch) + '.ckpt'
                save_path = saver.save(sess, 'DAC_models/' + model_name)
                print("Model saved in file: %s" % save_path)

            epoch += 1

elif mode == 'Testing':
    test_batch_size = 64
    with tf.Session() as sess:
        saver.restore(sess, "DAC_models/DAC_ep_45.ckpt")
        print('model restored!')
        all_predictions = np.zeros([len(image_label)], dtype=np.float32)
        for i in range(len(image_datsa) // test_batch_size):
            data_samples = get_batch_test(test_batch_size, image_data, i)
            feed_dict = {image_pool_input: data_samples}
            pred_cluster = sess.run(pred_label, feed_dict=feed_dict)
            all_predictions[i * test_batch_size:(i + 1) * test_batch_size] = pred_cluster

        acc = clustering_acc(image_label.astype(int), all_predictions.astype(int))
        nmi = NMI(image_label.astype(int), all_predictions.astype(int))
        ari = ARI(image_label.astype(int), all_predictions.astype(int))
        print('testing NMI, ARI, ACC are %f, %f, %f.' % (nmi, ari, acc))