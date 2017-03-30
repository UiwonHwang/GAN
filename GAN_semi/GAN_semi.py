import numpy as np
from scipy.stats import norm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns
import yaml
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# load parameters
with open("GAN_semi.yaml") as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

seed = config['seed']

# plot 색상
sns.set(color_codes=True)

# random seed setting
np.random.seed(seed)
tf.set_random_seed(seed)

# real data distribution
class DataDistribution(object):
    def __init__(self):  # mu=4, sigma=0.5 인 normal distribution
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):  # real data distribution에서 N개의 sample을 추출
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples

# generated data distribution
class GeneratorDistribution(object):
    def __init__(self, range):  # z: (-range) ~ (range) 에서 +-0.01의 noise가 있는 uniform distribution
        self.range = range

    def sample(self, N, data_dim):
        z=[]
        for i in range(N):
            z_i = np.linspace(-self.range, self.range, data_dim) + np.random.random(data_dim) * 0.01  # 범위 내에서 uniform하게 뽑은 후 +-0.01의 noise를 줌
            z = np.append(z, z_i)
        return np.reshape(z, (N,data_dim))

# output = x * w + b
def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)  # weight initialize value
    const = tf.constant_initializer(0.0)  # bias initialize value
    with tf.variable_scope(scope or 'linear'):  # scope명도 linear의 input으로 줄 수 있고 없으면 'linear' 사용
        w = tf.get_variable('w', [input.get_shape()[1], output_dim],initializer=norm)  # stddev=1인 normal distr로 initialize된 weight, liner/w
        b = tf.get_variable('b', [output_dim], initializer=const)  # 0으로 initialize된 bias, linear/b
        return tf.matmul(input, w) + b

def generator(input, hidden_size):
    h0 = tf.nn.softplus(linear(input, hidden_size, 'g0')) # noise를 nonlinearlity(softplus)에 통과시킴
    h1 = linear(h0, 1, 'g1') # linear transformation
    return h1

# minibatch discrimination : 각 example이 통과할 때 minibatch 내 다른 example의 data에 대한 feature 생성
def minibatch(input, num_kernels=config['num_kernels'], kernel_dim=config['kernel_dim']):
    fxT = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev = 0.02) # f(x)(N*A) * T(A*B*C) (reshape 필요)
    # x_i 의 각 dimension에 대해 kernel_dim 차원의 num_kernels 개의 kernel을 곱함
    M = tf.reshape(fxT, (-1,num_kernels, kernel_dim)) # M ⊃ M_i (∈ (num_kernels(B), kernel_dim(C)))
    diffs = tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1, 2, 0]), 0) # M_i,b - M_j,b
    # diffs.shape = [num_input,num_kernels,kernel_dim,1] - [1,num_kernels,kernel_dim,num_input] = [num_input,num_kernels,kernel_dim,num_input]
    # 즉 input 간의 M_i = [num_kernels, kernel_dim] 끼리 서로 뺀 모든 경우의 수를 가진 tensor가 됨
    c = tf.exp(-tf.reduce_sum(tf.abs(diffs), 2)) # 다른 example이 같은 kernel을 통과한 결과 간 L1 distance를 계산 후 exponential 취함
    # c.shape = [num_input, num_kernels, num_input] = c_b(x_i, y_i)
    o = tf.reduce_sum(c, 2) # o(x_i)_b
    # o.shape = [num_input, num_kernels]
    return tf.concat(1, [input, o])

def discriminator(input, hidden_size, minibatch_layer):  # D가 G보다 더 powerful해야 진짜와 가짜를 구분가능하므로 더 깊은 network를 만듬
    h0 = tf.tanh(linear(input, hidden_size * 2, 'd0'))  # nonlinearity로 tanh 사용
    h1 = tf.tanh(linear(h0, hidden_size * 2, 'd1'))

    if minibatch_layer:  # minibatch discrimination 사용시 세번째 layer는 minibatch layer로
        h2 = minibatch(h1)
    else:  # minibatch discrimination 미사용시 일반 tanh layer로
        h2 = tf.tanh(linear(h1, hidden_size * 2, 'd2'))

    h3 = tf.sigmoid(linear(h2, 1, 'd3'))  # 0~1의 probability로 내보내기 위해 sigmoid 사용
    return h3

def optimizer(loss, var_list, init_learning_rate, name):
    with tf.variable_scope(name) as scope:
        step = tf.Variable(0, name='step') # 현재 iteration을 표시해 줄 변수
        learning_rate = tf.train.exponential_decay(init_learning_rate, step, config['num_decay_steps'], config['decay'], staircase=True) # learning_rate = (initial_learning_rate)*decay^(int(step/num_decay_step))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=step, var_list=var_list) # var_list의 변수들을 update
    return optimizer


class GAN(object):
    def __init__(self, data, data_dim, gen, init_learning_rate, num_pretrain_steps, num_steps, batch_size, log_every,
                 hidden_size, minibatch_layer, anim_path):

        self.data = data
        self.data_dim = data_dim
        self.gen = gen  # generator distribution (z)

        self.learning_rate = init_learning_rate
        self.num_pretrain_steps = num_pretrain_steps
        self.num_steps = num_steps
        self.batch_size = batch_size

        self.log_every = log_every
        self.hidden_size = hidden_size
        self.minibatch = minibatch_layer

        self.anim_path = anim_path
        self.anim_frames = []

        self._create_model()

    def _create_model(self):
        # 초기에 D를 pre-training
        with tf.variable_scope("pre_D"):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, self.data_dim))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.hidden_size, self.minibatch)

        # generator
        with tf.variable_scope("G") as scope:
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, self.data_dim), name='z')  # generator의 input인 noise
            self.G = generator(self.z, self.hidden_size)

        # discriminator
        with tf.variable_scope("D") as scope:  # scope명을 D로 하면 D로 시작하는 D_pre의 변수도 같이 D_params로 들어감
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, self.data_dim), name='x')
            self.D1 = discriminator(self.x, self.hidden_size, self.minibatch)  # D(x)
            scope.reuse_variables()  # 함수가 같은 scope 내에서 두번 호출될 때 함수 내의 변수(ex. weights, bias..)를 공유하도록 함
            self.D2 = discriminator(self.G, self.hidden_size, self.minibatch)  # D(G(z))
            # accuracy
            self.y_D1 = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name='y_D1')
            self.y_D2 = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name='y_D2')
        with tf.variable_scope("Acc") as scope:
            TF_D1 = tf.equal(tf.round(self.D1),self.y_D1)
            TF_D2 = tf.equal(tf.round(self.D2),self.y_D2)
            TF = tf.concat(0,[TF_D1, TF_D2])
            self.acc = tf.reduce_mean(tf.cast(TF, tf.float32))

        with tf.variable_scope("pre_loss_D") as scope:
            self.pre_loss_D = tf.reduce_mean(tf.square(D_pre - self.pre_labels))  # maximum likelihood loss function
        with tf.variable_scope("loss_D") as scope:
            self.loss_D = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))  # x를 input으로 한 D의 output을 높게, z를 input으로 한 D의 output을 낮게 만듬
        with tf.variable_scope("loss_G") as scope:
            self.loss_G = tf.reduce_mean(-tf.log(self.D2))  # 가짜를 잘 못맞추게 만듬

        # D, G scope에서 각각 training 가능한 parameter들을 모음 (optimizer에 넣을거임)
        self.pre_D_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pre_D')  # pre-train한 weight를 복사해 넣으려고 만든거
        self.D_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')
        self.G_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')

        self.pre_opt_D = optimizer(self.pre_loss_D, self.pre_D_params, self.learning_rate, name='pre_opt_D')
        self.opt_D = optimizer(self.loss_D, self.D_params, self.learning_rate, name='opt_D')
        self.opt_G = optimizer(self.loss_G, self.G_params, self.learning_rate, name='opt_G')

        # tensorboard graph
        self.pre_loss_D_summary = tf.summary.scalar("pre_loss_D", self.pre_loss_D)
        self.loss_D_summary = tf.summary.scalar("loss_D", self.loss_D)
        self.loss_G_summary = tf.summary.scalar("loss_G", self.loss_G)
        self.acc_summary = tf.summary.scalar("acc", self.acc)
        self.D_merged = tf.summary.merge([self.loss_D_summary, self.acc_summary])

    def train(self):
        config = tf.ConfigProto(device_count={'GPU': 0})
        config.gpu_options.allow_growth = True
        session = tf.InteractiveSession(config=config)

        tf.global_variables_initializer().run()
        writer = tf.summary.FileWriter("C:/tmp/GAN_semi", session.graph)  # tensorboard를 위해 log를 기록

        # D를 pre-training
        print('========== pre-training start ==========')
        for step in range(1,self.num_pretrain_steps+1):
            pre_input = 10.0 * (np.random.random(self.batch_size) - 0.5)  # -5~+5 사이의 random한 수를 batch size개 만큼 생성
            # label을 real data distribution 상에서의 y값으로 설정하여 input이 mu에 가까운 값일수록 진짜일 확률이 높게 label을 줌
            pre_labels = norm.pdf(pre_input, loc=self.data.mu, scale=self.data.sigma)

            pre_D_feed = {self.pre_input: np.reshape(pre_input, (self.batch_size, 1)),
                          self.pre_labels: np.reshape(pre_labels, (self.batch_size, 1))}
            pre_D_summary, _ = session.run([self.pre_loss_D_summary,self.pre_opt_D], feed_dict=pre_D_feed)
            writer.add_summary(pre_D_summary, step)
            # pre_training 된 weight를 저장
            self.pre_D_weights = session.run(self.pre_D_params)
            if (step % self.log_every) == 0:
                print('\rstep : {}, pre_loss_D : {:.6f}'.format(step, self.pre_loss_D.eval(pre_D_feed)), end=' ')
        print('\n=========== pre-training end ===========\n')

        # D의 parameter에 pre_D_weights를 대입 : i=0,1,2,3... , v=D/d0/w, D/d0/b, D/d1/w ...
        # assign 연산 : 표현 그래프 -> run 이전에 연산을 수행하지 않음
        for i, v in enumerate(self.D_params):
            assign_D = v.assign(self.pre_D_weights[i])
            session.run(assign_D)

        # D와 G를 training
        print('=========== training start ===========')
        print('step:   accuracy,     loss_D,     loss_G')
        for step in range(1,self.num_steps+1):
            # D를 update
            x = self.data.sample(self.batch_size)
            z = self.gen.sample(self.batch_size, self.data_dim)
            D_feed = {self.x: np.reshape(x, (self.batch_size, self.data_dim)),
                      self.z: z,
                      self.y_D1: np.ones([self.batch_size, self.data_dim]),
                      self.y_D2: np.zeros([self.batch_size, self.data_dim])}
            D_merged, _ = session.run([self.D_merged, self.opt_D], feed_dict=D_feed)
            writer.add_summary(D_merged, step)

            # G를 update
            z = self.gen.sample(self.batch_size, self.data_dim)
            G_feed = {self.z: z}
            G_summary, _ = session.run([self.loss_G_summary, self.opt_G], feed_dict=G_feed)
            writer.add_summary(G_summary, step)

            # 진행상황 print
            if (step % self.log_every) == 0:
                print('{:4d}: {:10.6f},\t{:10.6f},\t{:10.6f}'.format(step, self.acc.eval(D_feed), self.loss_D.eval(D_feed), self.loss_G.eval(G_feed)))

            if self.anim_path:
                self.anim_frames.append(self._samples(session))

        if self.anim_path:
            self._save_animation()
        else:
            self._plot_distributions(session)
        tf.summary.FileWriter.close(writer)
        print('=========== training end ===========')

    # 현재 parameter에 대한 decision boundary, data distribution, generated distribution을 만들어줌
    def _samples(self, session, num_points=10000, num_bins=100):

        # decision boundary : 작은 값부터 천천히 D에 넣으면서 나오는 output을 db에 저장
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db_feed = {self.x: np.reshape(xs[self.batch_size * i: self.batch_size * (i + 1)], (self.batch_size, 1))}
            db[self.batch_size * i: self.batch_size * (i + 1)] = session.run(self.D1, feed_dict=db_feed)

        # data distribution : data distr에서 num_points회 sampling 후, np.histogram 함수를 사용
        d = self.data.sample(num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated distribution : G(z)를 num_point개 만들어서 np.histogram
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            pg_feed = {self.z: np.reshape(zs[self.batch_size * i: self.batch_size * (i + 1)], (self.batch_size, 1))}
            g[self.batch_size * i: self.batch_size * (i + 1)] = session.run(self.G, feed_dict=pg_feed)
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    # db, pd, pg를 plotting
    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()

    # 동영상 저장
    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return (line_db, line_pd, line_pg, frame_number)

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])

def main(config):
    model = GAN(
        data = DataDistribution(),
        data_dim = config['data_dim'],
        gen = GeneratorDistribution(range=config['z_range']),
        init_learning_rate = config['init_learning_rate'],
        num_pretrain_steps = config['num_pretrain_steps'],
        num_steps = config['num_steps'],
        batch_size = config['batch_size'],
        log_every = config['log_every'],
        hidden_size = config['hidden_size'],
        minibatch_layer = config['minibatch_layer'],
        anim_path = config['anim_path']
    )
    model.train()

if __name__ =='__main__':
    main(config)