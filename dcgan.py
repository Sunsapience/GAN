'''
Deep Convolutional Generative Adversarial Networks(DCGAN).
https://arxiv.org/pdf/1511.06434.pdf

The coder refers to :
        https://github.com/sugyan/tf-dcgan  (main refer)
        https://github.com/carpedm20/DCGAN-tensorflow
        https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN 
'''
import tensorflow as tf  

class generator:
    def __init__(self,filters=[1024,512,256,128],s_size=4):
        # 每次反卷积时的通道个数
        self.filters = filters + [3]
        # 初始噪声图像大小
        self.s_size = s_size
        self.reuse = False

    def __call__(self,inputs,training=False):
        # 以初始随机噪音输入为例， shape:[128,100]
        inputs = tf.convert_to_tensor(inputs)
        with tf.variable_scope('g',reuse=self.reuse):
            # [batch_size,1024*4*4]
            outputs = tf.layers.dense(inputs, self.filters[0] * self.s_size * self.s_size)
            # [batch_size,4,4,1024]
            outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.filters[0]])
            # 批正则化和激活
            outputs = tf.nn.relu(
                tf.layers.batch_normalization(outputs, training=training), name='outputs')
        
        # 反卷积
        with tf.variable_scope('deconv1'):
            # [batch_size,4,4,1024] --> [batch_size,8,8,512]
            outputs = tf.layers.conv2d_transpose(
                outputs, self.filters[1], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(
                tf.layers.batch_normalization(outputs, training=training), name='outputs')

        with tf.variable_scope('deconv2'):
            # [batch_size,8,8,512] --> [batch_size,16,16,256]
            outputs = tf.layers.conv2d_transpose(
                outputs, self.filters[2], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(
                tf.layers.batch_normalization(outputs, training=training), name='outputs')
        
        with tf.variable_scope('deconv3'):
            # [batch_size,16,16,256] --> [batch_size,32,32,128]
            outputs = tf.layers.conv2d_transpose(
                outputs, self.filters[3], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.relu(
                tf.layers.batch_normalization(outputs, training=training), name='outputs') 

        with tf.variable_scope('deconv4'):
            # [batch_size,32,32,128] --> [batch_size,64,64,3]
            outputs = tf.layers.conv2d_transpose(
                outputs, self.filters[4], [5, 5], strides=(2, 2), padding='SAME')

        with tf.variable_scope('tanh'):
                outputs = tf.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs     # [batch_size,64,64,3]

class discriminator:
    def __init__(self, filters=[64, 128, 256, 512]):
        self.filters = [3] + filters
        self.reuse = False

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        # [batch_size,64,64,3]
        outputs = tf.convert_to_tensor(inputs)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            with tf.variable_scope('conv1'):
                #  [batch_size,64,64,3]--> [batch_size,32,32,64]
                outputs = tf.layers.conv2d(
                    outputs, self.filters[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(
                    tf.layers.batch_normalization(outputs, training=training), name='outputs')

            with tf.variable_scope('conv2'):
                # [batch_size,32,32,64] --> [batch_size,16,16,128]
                outputs = tf.layers.conv2d(
                    outputs, self.filters[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(
                    tf.layers.batch_normalization(outputs, training=training), name='outputs')

            with tf.variable_scope('conv3'):
                # [batch_size,16,16,128] --> [batch_size,8,8,256]
                outputs = tf.layers.conv2d(
                    outputs, self.filters[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(
                    tf.layers.batch_normalization(outputs, training=training), name='outputs')

            with tf.variable_scope('conv4'):
                # [batch_size,8,8,256] --> [batch_size,4,4,512]
                outputs = tf.layers.conv2d(
                    outputs, self.filters[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(
                    tf.layers.batch_normalization(outputs, training=training), name='outputs')

            with tf.variable_scope('classify'):
                batch_size = outputs.get_shape()[0].value
                reshape = tf.reshape(outputs, [batch_size, -1]) 
                outputs = tf.layers.dense(reshape, 2, name='outputs')   # [batch_size,2]
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs

class DCGAN:
    def __init__(self,
                 batch_size=128, s_size=4, z_dim=100,
                 g_filters=[1024, 512, 256, 128],
                 d_filters=[64, 128, 256, 512]):
        self.batch_size = batch_size
        self.s_size = s_size
        self.z_dim = z_dim
        self.g = generator(filters=g_filters, s_size=self.s_size)
        self.d = discriminator(filters=d_filters)
        self.z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)

    def loss(self, traindata):
        """build models, calculate losses.
        Args:
            traindata: 4-D Tensor of shape `[batch, height, width, channels]`.
        Returns:
            dict of each models' losses.
        """
        generated = self.g(self.z, training=True)  # 噪声生成的图片
        g_outputs = self.d(generated, training=True, name='g')  # 判别生成器生成的图片
        t_outputs = self.d(traindata, training=True, name='t')  # 判别训练数据的图片
        # add each losses to collection

        # 生成器的损失函数只有一部分 
        # 期望生成器生成的图片的标签都是1(都是真实的)
        tf.add_to_collection( 
            'g_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))

        # 判别器的损失函数两部分 
        # 期望判别器 判别真实数据时标签都为1
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([self.batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        # 期望判别器 判别来自生成器生成的图片 ，其标签都为0
        tf.add_to_collection(
            'd_losses',
            tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([self.batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def train(self, losses, learning_rate=0.0002, beta1=0.5):
        """
        Args:
            losses dict.
        Returns:
            train op.
        """
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)

        # 这两行代码参考 http://www.cnblogs.com/reaptomorrow-flydream/p/9492191.html
        with tf.control_dependencies([g_opt_op, d_opt_op]):
            return tf.no_op(name='train')

    def sample_images(self, row=8, col=8, inputs=None):
        if inputs is None:
            inputs = self.z
        images = self.g(inputs, training=True) #[128,64,64,3]
        images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8) #由于生成器最后采用了tanh,再[-1,1]之间，所以先加1，再除2，使之落到[0,1]之间
        images = [image for image in tf.split(images, self.batch_size, axis=0)] # 128 个[1,64,64,3]
        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col], 2)) #8个[1，64，512，3]
        image = tf.concat(rows, 1)  # [1,512,512,3]
        return tf.image.encode_jpeg(tf.squeeze(image, [0])) #[512,512,3]

'''
# Train
dcgan = DCGAN()
train_images = <images batch>
losses = dcgan.loss(train_images)
train_op = dcgan.train(losses)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(max_steps):
        _, g_loss_value, d_loss_value = sess.run([train_op, losses[dcgan.g], losses[dcgan.d]])
    
    # saved generator variables
    # saved discriminator variables
'''

'''
# Generate
dcgan = DCGAN()
images = dcgan.sample_images()

with tf.Session() as sess:
    # restore generator variables

    generated = sess.run(images)
    with open('<filename>', 'wb') as f:
        f.write(generated)
'''

