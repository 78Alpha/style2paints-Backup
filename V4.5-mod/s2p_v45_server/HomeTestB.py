import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
import datetime
import pickle
import base64
import json
import gzip
import re
import shutil
import argparse
import random
import glob

from tqdm import tqdm
from cv2.ximgproc import l0Smooth, createGuidedFilter, guidedFilter
#from bottle import route, run, static_file, request, BaseRequest

#BaseRequest.MEMFILE_MAX = 10000 * 1000
import logging
logging.getLogger("tensorflow").propagate = False
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow
import rtree
import scipy
import trimesh
from scipy.spatial import ConvexHull
from cv2.ximgproc import createGuidedFilter
from PIL import Image
import threading
import multiprocessing

tensorflow.compat.v1.disable_v2_behavior()
tf = tensorflow.compat.v1

#tf.logging.set_verbosity(tf.logging.ERROR)
ID = datetime.datetime.now().strftime('H%HM%MS%S')

default_color_file= random.choice(glob.glob("./colors/*.jpg"))

apm = argparse.ArgumentParser(prog='Style2PaintsV4.5')
apm.add_argument('--input', '-i', default='./INPUT', metavar='DIR', type=str, help='Directory of input file(s)')
apm.add_argument('--color', '-c', default=str(default_color_file), metavar='IMG', type=str, help='File to use for Color base')
apm.add_argument('--output', '-o', default='./OUTPUT', metavar='DIR', type=str, help='Output directory')
apm.add_argument('--style', '-s', default=0, metavar='INT', type=int, help='style choice: 0 for all, 1 for flat, 2 for smooth, 3 for blend flat, 4 for blend smooth')
apm.add_argument('--grade', '-g', default=1, metavar='INT', type=int, help='Grade of the output image')
apm.add_argument('--grade-all', '-ga', default=False, action='store_true', help='Grade all input files')
apm.add_argument('--lighting', '-l', default=False, action='store_true', help='Use lighting')
apm.add_argument('--version', '-v', action='version', version=f"%(prog)s 4.5")
args = vars(apm.parse_args())
_INPUT_: str = args['input']
_COLOR_IMAGE_ = args['color']
_OUTPUT_: str = args['output']
_GRADE_: int = args['grade']
_GRADE_ALL_: bool = args['grade_all']
_STYLE_: int = args['style']
_LIGHTING_: bool = args['lighting']


def ToGray(x):
    R = x[:, :, :, 0:1]
    G = x[:, :, :, 1:2]
    B = x[:, :, :, 2:3]
    return 0.30 * R + 0.59 * G + 0.11 * B


def VGG2RGB(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1]


def norm_feature(x, core):
    cs0 = tf.shape(core)[1]
    cs1 = tf.shape(core)[2]
    small = tf.image.resize_area(x, (cs0, cs1))
    avged = tf.nn.avg_pool(tf.pad(small, [[0, 0], [2, 2], [2, 2], [0, 0]], 'REFLECT'), [1, 5, 5, 1], [1, 1, 1, 1],
                           'VALID')
    return tf.image.resize_bicubic(avged, tf.shape(x)[1:3])


def blur(x):
    def layer(op):
        def layer_decorated(self, *args, **kwargs):
            # Automatically set a name if not provided.
            name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
            # Figure out the layer inputs.
            if len(self.terminals) == 0:
                raise RuntimeError('No input variables found for layer %s.' % name)
            elif len(self.terminals) == 1:
                layer_input = self.terminals[0]
            else:
                layer_input = list(self.terminals)
            # Perform the operation and get the output.
            layer_output = op(self, layer_input, *args, **kwargs)
            # Add to layer LUT.
            self.layers[name] = layer_output
            # This output is now the input for the next layer.
            self.feed(layer_output)
            # Return self for chained calls.
            return self

        return layer_decorated

    class Smoother(object):
        def __init__(self, inputs, filter_size, sigma):
            self.inputs = inputs
            self.terminals = []
            self.layers = dict(inputs)
            self.filter_size = filter_size
            self.sigma = sigma
            self.setup()

        def setup(self):
            (self.feed('data')
             .conv(name='smoothing'))

        def get_unique_name(self, prefix):
            ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
            return '%s_%d' % (prefix, ident)

        def feed(self, *args):
            assert len(args) != 0
            self.terminals = []
            for fed_layer in args:
                if isinstance(fed_layer, str):
                    try:
                        fed_layer = self.layers[fed_layer]
                    except KeyError:
                        raise KeyError('Unknown layer name fed: %s' % fed_layer)
                self.terminals.append(fed_layer)
            return self

        def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
            out_filter = np.load('./nets/gau.npy')
            return out_filter

        def make_gauss_var(self, name, size, sigma, c_i):
            kernel = self.gauss_kernel(size, sigma, c_i)
            var = tf.Variable(tf.convert_to_tensor(kernel), name=name)
            return var

        def get_output(self):
            '''Returns the smoother output.'''
            return self.terminals[-1]

        @layer
        def conv(self,
                 input,
                 name,
                 padding='SAME'):
            # Get the number of channels in the input
            c_i = input.get_shape().as_list()[3]
            # Convolution for a given input and kernel
            convolve = lambda i, k: tf.nn.depthwise_conv2d(i, k, [1, 1, 1, 1],
                                                           padding=padding)
            with tf.variable_scope(name) as scope:
                kernel = self.make_gauss_var('gauss_weight', self.filter_size,
                                             self.sigma, c_i)
                output = convolve(input, kernel)
                return output

    return Smoother({'data': tf.pad(x, [[0, 0], [9, 9], [9, 9], [0, 0]], 'SYMMETRIC')}, 7, 2).get_output()[:, 9: -9,
           9: -9, :]


def downsample(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def nts(x):
    return (x + [103.939, 116.779, 123.68])[:, :, :, ::-1] / 255.0


def np_expand_image(x):
    p = np.pad(x, ((1, 1), (1, 1), (0, 0)), 'symmetric')
    r = []
    r.append(p[:-2, 1:-1, :])
    r.append(p[1:-1, :-2, :])
    r.append(p[1:-1, 1:-1, :])
    r.append(p[1:-1, 2:, :])
    r.append(p[2:, 1:-1, :])
    return np.stack(r, axis=2)


def build_sketch_sparse(x, abs):
    x = x[:, :, None].astype(np.float32)
    expanded = np_expand_image(x)
    distance = x[:, :, None] - expanded
    if abs:
        distance = np.abs(distance)
    weight = 8 - distance
    weight[weight < 0] = 0.0
    weight /= np.sum(weight, axis=2, keepdims=True)
    return weight


def build_repeat_mulsep(x, m, i):
    a = m[:, :, 0]
    b = m[:, :, 1]
    c = m[:, :, 2]
    d = m[:, :, 3]
    e = m[:, :, 4]
    y = x
    for _ in range(i):
        p = tf.pad(y, [[1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        y = p[:-2, 1:-1, :] * a + p[1:-1, :-2, :] * b + y * c + p[1:-1, 2:, :] * d + p[2:, 1:-1, :] * e
    return y


session = tf.Session()
tf.keras.backend.set_session(session)

ip1 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 1))
ip3 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
ip4 = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 4))
ipsp9 = tf.placeholder(dtype=tf.float32, shape=(None, None, 5, 1))
ipsp3 = tf.placeholder(dtype=tf.float32, shape=(None, None, 3))

tf_sparse_op_H = build_repeat_mulsep(ipsp3, ipsp9, 64)
tf_sparse_op_L = build_repeat_mulsep(ipsp3, ipsp9, 16)


def make_graph():
    with gzip.open('./nets/refs.net', 'rb') as fp:
        refs_img = pickle.load(fp)

    tail = tf.keras.models.load_model('./nets/tail.net', compile=True)
    reader = tf.keras.models.load_model('./nets/reader.net', compile=True)
    head = tf.keras.models.load_model('./nets/head.net', compile=True)
    neck = tf.keras.models.load_model('./nets/neck.net', compile=True)
    inception = tf.keras.models.load_model('./nets/inception.net', compile=True)
    render_head = tf.keras.models.load_model('./nets/render_head.net', compile=True)
    render_neck = tf.keras.models.load_model('./nets/render_neck.net', compile=True)

    tail_op = tail(ip3)
    features = reader(ip3 / 255.0)
    feed = [1 - ip1 / 255.0, (ip4[:, :, :, 0:3] / 127.5 - 1) * ip4[:, :, :, 3:4] / 255.0]
    for _ in range(len(features)):
        feed.append(tf.reduce_mean(features[_], axis=[1, 2]))
    nil0, nil1, head_temp = head(feed)
    feed[0] = tf.clip_by_value(1 - tf.image.resize_bilinear(ToGray(VGG2RGB(head_temp) / 255.0), tf.shape(ip1)[1:3]),
                               0.0, 1.0)
    nil4, nil5, head_temp = neck(feed)
    head_op = VGG2RGB(head_temp)
    features_render = inception((ip3 + (downsample(ip1) - blur(downsample(ip1))) * 2.0) / 255.0)
    precessed_feed = [(ip4[:, :, :, 0:3] / 127.5 - 1) * ip4[:, :, :, 3:4] / 255.0] + [
        norm_feature(item, features_render[-1]) for item in features_render]
    nil6, nil7, render_A = render_head([1 - ip1 / 255.0] + precessed_feed)
    nil8, nil9, render_B = render_neck(
        [1 - tf.image.resize_bilinear(ToGray(nts(render_A)), tf.shape(ip1)[1:3])] + precessed_feed)
    render_op = nts(render_B) * 255.0
    session.run(tf.global_variables_initializer())

    tail.load_weights('./nets/tail.net')
    head.load_weights('./nets/head.net')
    neck.load_weights('./nets/neck.net')
    reader.load_weights('./nets/reader.net')
    inception.load_weights('./nets/inception.net')
    render_head.load_weights('./nets/render_head.net')
    render_neck.load_weights('./nets/render_neck.net')
    return tail_op, head_op, render_op, refs_img


tail_op_g, head_op_g, render_op_g, refs_img_g = make_graph()


def go_tail(x):
    def srange(l, s):
        result = []
        iters = int(float(l) / float(s))
        for i in range(iters):
            result.append([i * s, (i + 1) * s])
        result[len(result) - 1][1] = l
        return result

    H, W, C = x.shape
    padded_img = np.pad(x, ((20, 20), (20, 20), (0, 0)), 'symmetric').astype(np.float32) / 255.0
    lines = []
    for hs, he in srange(H, 64):
        items = []
        for ws, we in srange(W, 64):
            items.append(padded_img[hs:he + 40, ws:we + 40, :])
        lines.append(items)
    iex = 0
    result_all_lines = []
    for line in lines:
        result_one_line = []
        for item in line:
            ots = session.run(tail_op_g, feed_dict={ip3: item[None, :, :, :]})[0]
            result_one_line.append(ots[41:-41, 41:-41, :])
            iex += 1
        result_one_line = np.concatenate(result_one_line, axis=1)
        result_all_lines.append(result_one_line)
    result_all_lines = np.concatenate(result_all_lines, axis=0)
    return (result_all_lines * 255.0).clip(0, 255).astype(np.uint8)


def go_head(sketch, global_hint, local_hint):
    return session.run(head_op_g, feed_dict={
        ip1: sketch[None, :, :, None], ip3: global_hint[None, :, :, :], ip4: local_hint[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def go_render(sketch, segmentation, points):
    return session.run(render_op_g, feed_dict={
        ip1: sketch[None, :, :, None], ip3: segmentation[None, :, :, :], ip4: points[None, :, :, :]
    })[0].clip(0, 255).astype(np.uint8)


def k_resize(x, k):
    if x.shape[0] < x.shape[1]:
        s0 = k
        s1 = int(x.shape[1] * (k / x.shape[0]))
        s1 = s1 - s1 % 64
        _s0 = 16 * s0
        _s1 = int(x.shape[1] * (_s0 / x.shape[0]))
        _s1 = (_s1 + 32) - (_s1 + 32) % 64
    else:
        s1 = k
        s0 = int(x.shape[0] * (k / x.shape[1]))
        s0 = s0 - s0 % 64
        _s1 = 16 * s1
        _s0 = int(x.shape[0] * (_s1 / x.shape[1]))
        _s0 = (_s0 + 32) - (_s0 + 32) % 64
    new_min = min(_s1, _s0)
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (_s1, _s0), interpolation=interpolation)
    return y


def d_resize(x, d, fac=1.0):
    new_min = min(int(d[1] * fac), int(d[0] * fac))
    raw_min = min(x.shape[0], x.shape[1])
    if new_min < raw_min:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (int(d[1] * fac), int(d[0] * fac)), interpolation=interpolation)
    return y


def min_resize(x, m):
    if x.shape[0] < x.shape[1]:
        s0 = m
        s1 = int(float(m) / float(x.shape[0]) * float(x.shape[1]))
    else:
        s0 = int(float(m) / float(x.shape[1]) * float(x.shape[0]))
        s1 = m
    new_max = min(s1, s0)
    raw_max = min(x.shape[0], x.shape[1])
    if new_max < raw_max:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LANCZOS4
    y = cv2.resize(x, (s1, s0), interpolation=interpolation)
    return y


def cli_norm(sketch):
    light = np.max(min_resize(sketch, 64), axis=(0, 1), keepdims=True)
    intensity = (light - sketch.astype(np.float32)).clip(0, 255)
    line_intensities = np.sort(intensity[intensity > 16])[::-1]
    line_quantity = float(line_intensities.shape[0])
    intensity /= line_intensities[int(line_quantity * 0.1)]
    intensity *= 0.9
    return (255.0 - intensity * 255.0).clip(0, 255).astype(np.uint8)


def from_png_to_jpg(map):
    if map.shape[2] == 3:
        return map
    color = map[:, :, 0:3].astype(np.float) / 255.0
    alpha = map[:, :, 3:4].astype(np.float) / 255.0
    reversed_color = 1 - color
    final_color = (255.0 - reversed_color * alpha * 255.0).clip(0, 255).astype(np.uint8)
    return final_color


def s_enhance(x, k=2.0):
    p = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float)
    p[:, :, 1] *= k
    p = p.clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(p, cv2.COLOR_HSV2RGB).clip(0, 255)


def ini_hint(x):
    r = np.zeros(shape=(x.shape[0], x.shape[1], 4), dtype=np.uint8)
    return r


def opreate_normal_hint(gird, points, length):
    h = gird.shape[0]
    w = gird.shape[1]
    for point in points:
        x, y, r, g, b = point
        x = int(x * w)
        y = int(y * h)
        l_ = max(0, x - length)
        b_ = max(0, y - length)
        r_ = min(w, x + length + 1)
        t_ = min(h, y + length + 1)
        gird[b_:t_, l_:r_, 2] = r
        gird[b_:t_, l_:r_, 1] = g
        gird[b_:t_, l_:r_, 0] = b
        gird[b_:t_, l_:r_, 3] = 255.0
    return gird


def get_hdr(x):
    def get_hdr_g(x):
        img = x.astype(np.float32)
        mean = np.mean(img)
        h_mean = mean.copy()
        l_mean = mean.copy()
        for i in range(2):
            h_mean = np.mean(img[img >= h_mean])
            l_mean = np.mean(img[img <= l_mean])
        for i in range(2):
            l_mean = np.mean(img[img <= l_mean])
        return l_mean, mean, h_mean

    l_mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    h_mean = np.zeros(shape=(1, 1, 3), dtype=np.float32)
    for c in range(3):
        l, m, h = get_hdr_g(x[:, :, c])
        l_mean[:, :, c] = l
        mean[:, :, c] = m
        h_mean[:, :, c] = h
    return l_mean, mean, h_mean


def f2(x1, x2, x3, y1, y2, y3, x):
    A = y1 * ((x - x2) * (x - x3)) / ((x1 - x2) * (x1 - x3))
    B = y2 * ((x - x1) * (x - x3)) / ((x2 - x1) * (x2 - x3))
    C = y3 * ((x - x1) * (x - x2)) / ((x3 - x1) * (x3 - x2))
    return A + B + C


def refine_image(image, sketch, origin):
    sketch = sketch.astype(np.float32)
    sparse_matrix = build_sketch_sparse(sketch, True)
    bright_matrix = build_sketch_sparse(sketch - cv2.GaussianBlur(sketch, (0, 0), 3.0), False)
    guided_matrix = createGuidedFilter(sketch.clip(0, 255).astype(np.uint8), 1, 0.01)
    HDRL, HDRM, HDRH = get_hdr(image)

    def go_guide(x):
        y = x + (x - cv2.GaussianBlur(x, (0, 0), 1)) * 2.0
        for _ in range(4):
            y = guided_matrix.filter(y)
        return y

    def go_refine_sparse(x):
        return session.run(tf_sparse_op_H, feed_dict={ipsp3: x, ipsp9: sparse_matrix})

    def go_refine_bright(x):
        return session.run(tf_sparse_op_L, feed_dict={ipsp3: x, ipsp9: bright_matrix})

    def go_flat(x):
        pia = 32
        y = x.clip(0, 255).astype(np.uint8)
        y = cv2.resize(y, (x.shape[1] // 2, x.shape[0] // 2), interpolation=cv2.INTER_AREA)
        y = np.pad(y, ((pia, pia), (pia, pia), (0, 0)), 'reflect')
        y = l0Smooth(y, None, 0.01)
        y = y[pia:-pia, pia:-pia, :]
        y = cv2.resize(y, (x.shape[1], x.shape[0]), interpolation=cv2.INTER_CUBIC)
        return y

    def go_hdr(x):
        xl, xm, xh = get_hdr(x)
        y = f2(xl, xm, xh, HDRL, HDRM, HDRH, x)
        return y.clip(0, 255)

    def go_blend(BGR, X, m):
        BGR = BGR.clip(0, 255).astype(np.uint8)
        X = X.clip(0, 255).astype(np.uint8)
        YUV = cv2.cvtColor(BGR, cv2.COLOR_BGR2YUV)
        s_l = YUV[:, :, 0].astype(np.float32)
        t_l = X.astype(np.float32)
        r_l = (s_l * t_l / 255.0) if m else np.minimum(s_l, t_l)
        YUV[:, :, 0] = r_l.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)

    smoothed, flat, blended_smoothed, blended_flat = None, None, None, None
    smoothed = d_resize(image, sketch.shape)
    sparse_smoothed = go_refine_sparse(smoothed)
    smoothed = go_guide(sparse_smoothed)
    smoothed = go_hdr(smoothed)
    if _STYLE_ == 4 or _STYLE_ == 0:
        blended_smoothed = go_blend(smoothed, origin, False)

    flat = sparse_smoothed.copy()
    flat = go_refine_bright(flat)
    flat = go_flat(flat)
    flat = go_refine_sparse(flat)
    flat = go_guide(flat)
    flat = go_hdr(flat)
    if _STYLE_ == 3 or _STYLE_ == 0:
        blended_flat = go_blend(flat, origin, True)
    return smoothed, flat, blended_smoothed, blended_flat


def get_image_gradient(dist):
    cols = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, 0, +1], [-2, 0, +2], [-1, 0, +1]]))
    rows = cv2.filter2D(dist, cv2.CV_32F, np.array([[-1, -2, -1], [0, 0, 0], [+1, +2, +1]]))
    return cols, rows


def generate_lighting_effects(stroke_density, content):

    # Computing the coarse lighting effects
    # In original paper we compute the coarse effects using Gaussian filters.
    # Here we use a Gaussian pyramid to get similar results.
    # This pyramid-based result is a bit better than naive filters.
    h512 = content
    h256 = cv2.pyrDown(h512)
    h128 = cv2.pyrDown(h256)
    h64 = cv2.pyrDown(h128)
    h32 = cv2.pyrDown(h64)
    h16 = cv2.pyrDown(h32)
    c512, r512 = get_image_gradient(h512)
    c256, r256 = get_image_gradient(h256)
    c128, r128 = get_image_gradient(h128)
    c64, r64 = get_image_gradient(h64)
    c32, r32 = get_image_gradient(h32)
    c16, r16 = get_image_gradient(h16)
    c = c16
    c = d_resize(cv2.pyrUp(c), c32.shape) * 4.0 + c32
    c = d_resize(cv2.pyrUp(c), c64.shape) * 4.0 + c64
    c = d_resize(cv2.pyrUp(c), c128.shape) * 4.0 + c128
    c = d_resize(cv2.pyrUp(c), c256.shape) * 4.0 + c256
    c = d_resize(cv2.pyrUp(c), c512.shape) * 4.0 + c512
    r = r16
    r = d_resize(cv2.pyrUp(r), r32.shape) * 4.0 + r32
    r = d_resize(cv2.pyrUp(r), r64.shape) * 4.0 + r64
    r = d_resize(cv2.pyrUp(r), r128.shape) * 4.0 + r128
    r = d_resize(cv2.pyrUp(r), r256.shape) * 4.0 + r256
    r = d_resize(cv2.pyrUp(r), r512.shape) * 4.0 + r512
    coarse_effect_cols = c
    coarse_effect_rows = r

    # Normalization
    EPS = 1e-10
    max_effect = np.max((coarse_effect_cols**2 + coarse_effect_rows**2)**0.5)
    coarse_effect_cols = (coarse_effect_cols + EPS) / (max_effect + EPS)
    coarse_effect_rows = (coarse_effect_rows + EPS) / (max_effect + EPS)

    # Refinement
    stroke_density_scaled = (stroke_density.astype(np.float32) / 255.0).clip(0, 1)
    coarse_effect_cols *= (1.0 - stroke_density_scaled ** 2.0 + 1e-10) ** 0.5
    coarse_effect_rows *= (1.0 - stroke_density_scaled ** 2.0 + 1e-10) ** 0.5
    refined_result = np.stack([stroke_density_scaled, coarse_effect_rows, coarse_effect_cols], axis=2)

    return refined_result


def run(image, output_name):
    x = 0.0
    y = 0.0
    image = cv2.imread(image)
    mask = None

    ambient_intensity = 0.45
    light_intensity = 0.85
    light_source_height = 1.0
    gamma_correction = 1.0
    stroke_density_clipping = 1.2
    enabling_multiple_channel_effects = True

    light_color_red = 1.0
    light_color_green = 1.0
    light_color_blue = 1.0
    # Some pre-processing to resize images and remove input JPEG artifacts.
    # raw_image = min_resize(image, 512)
    # raw_image = run_srcnn(raw_image)
    light_dtype = np.float32
    raw_image = min_resize(image, 1024)
    raw_image = raw_image.astype(light_dtype)
    unmasked_image = raw_image.copy()

    if mask is not None:
        alpha = np.mean(d_resize(mask, raw_image.shape).astype(np.float32) / 255.0, axis=2, keepdims=True)
        raw_image = unmasked_image * alpha

    # Compute the convex-hull-like palette.
    h, w, c = raw_image.shape
    flattened_raw_image = raw_image.reshape((h * w, c))
    raw_image_center = np.mean(flattened_raw_image, axis=0)
    hull = ConvexHull(flattened_raw_image)

    # Estimate the stroke density map.
    intersector = trimesh.Trimesh(faces=hull.simplices, vertices=hull.points).ray
    start = np.tile(raw_image_center[None, :], [h * w, 1])
    direction = flattened_raw_image - start
    # print('Begin ray intersecting ...')
    index_tri, index_ray, locations = intersector.intersects_id(start, direction, return_locations=True, multiple_hits=True)
    # print('Intersecting finished.')
    intersections = np.zeros(shape=(h * w, c), dtype=light_dtype)
    intersection_count = np.zeros(shape=(h * w, 1), dtype=light_dtype)
    CI = index_ray.shape[0]
    for c in range(CI):
        i = index_ray[c]
        intersection_count[i] += 1
        intersections[i] += locations[c]
    intersections = (intersections + 1e-10) / (intersection_count + 1e-10)
    intersections = intersections.reshape((h, w, 3))
    intersection_count = intersection_count.reshape((h, w))
    intersections[intersection_count < 1] = raw_image[intersection_count < 1]
    intersection_distance = np.sqrt(np.sum(np.square(intersections - raw_image_center[None, None, :]), axis=2, keepdims=True))
    pixel_distance = np.sqrt(np.sum(np.square(raw_image - raw_image_center[None, None, :]), axis=2, keepdims=True))
    stroke_density = ((1.0 - np.abs(1.0 - pixel_distance / intersection_distance)) * stroke_density_clipping).clip(0, 1) * 255

    # A trick to improve the quality of the stroke density map.
    # It uses guided filter to remove some possible artifacts.
    # You can remove these codes if you like sharper effects.
    guided_filter = createGuidedFilter(pixel_distance.clip(0, 255).astype(np.uint8), 1, 0.01)
    for _ in range(4):
        stroke_density = guided_filter.filter(stroke_density)

    # Visualize the estimated stroke density.
    # cv2.imwrite('stroke_density.png', stroke_density.clip(0, 255).astype(np.uint8))

    # Then generate the lighting effects
    raw_image = unmasked_image.copy()
    lighting_effect = np.stack([
        generate_lighting_effects(stroke_density, raw_image[:, :, 0]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 1]),
        generate_lighting_effects(stroke_density, raw_image[:, :, 2])
    ], axis=2)
    # x = w / 2.0
    # y = h / 2.0
    # # Using a simple user interface to display results.
    # gx = - float(x % w) / float(w) * 2.0 + 1.0
    # gy = - float(y % h) / float(h) * 2.0 + 1.0
    gx = 0.0
    gy = 0.0
    light_source_color = np.array([light_color_blue, light_color_green, light_color_red])

    light_source_location = np.array([[[light_source_height, gy, gx]]], dtype=light_dtype)
    light_source_direction = light_source_location / np.sqrt(np.sum(np.square(light_source_location)))
    final_effect = np.sum(lighting_effect * light_source_direction, axis=3).clip(0, 1)
    if not enabling_multiple_channel_effects:
        final_effect = np.mean(final_effect, axis=2, keepdims=True)
    rendered_image = (ambient_intensity + final_effect * light_intensity) * light_source_color * raw_image
    rendered_image = ((rendered_image / 255.0) ** gamma_correction) * 255.0
    canvas = rendered_image.clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_name, canvas)


def filter_image_name(path):
    file_name = os.path.basename(path)
    file_name = file_name.replace(' ', '_')
    file_name = re.sub(r'[^a-zA-Z0-9_]+', '_', file_name)
    ext_list = ['.jpg', '.png', '.bmp', '.jpeg', '.webp', '.tiff', '.tif']
    for ext in ext_list:
        if file_name.endswith(ext):
            file_name = file_name[:-len(ext)]
            break
    new_path = os.path.dirname(path) + '/' + file_name
    return file_name, new_path


def get_request_image(filepath):
    img = open(filepath, 'rb').read()
    img = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(img, -1)
    return img


def upload_sketch(_INPUT_=_INPUT_, ID=ID):
    filtered_name, new_path = filter_image_name(_INPUT_)
    origin = from_png_to_jpg(get_request_image(_INPUT_))
    room_path = './game/rooms/' + ID + '/'
    os.makedirs(room_path, exist_ok=True)
    cv2.imwrite(room_path + filtered_name + '.origin.png', origin)
    sketch = min_resize(origin, 512)
    sketch = np.min(sketch, axis=2)
    sketch = cli_norm(sketch)
    sketch = np.tile(sketch[:, :, None], [1, 1, 3])
    sketch = go_tail(sketch)
    sketch = np.mean(sketch, axis=2)
    cv2.imwrite(room_path + filtered_name + '.sketch.png', sketch)
    return ID + '_' + ID


def request_result(_INPUT_=_INPUT_, _COLOR_IMAGE_=_COLOR_IMAGE_, ID=ID):
    filtered_name, new_path = filter_image_name(_INPUT_)
    room = ID
    room_path = './game/rooms/' + room + '/'
    points = f'[]'
    with open(room_path + '/points.' + ID + '.txt', 'wt') as fp:
        fp.write(points)
    points = json.loads(points)
    for _ in range(len(points)):
        points[_][1] = 1 - points[_][1]
    sketch = cv2.imread(room_path + filtered_name + '.sketch.png', cv2.IMREAD_UNCHANGED)
    origin = cv2.imread(room_path + filtered_name + '.origin.png', cv2.IMREAD_GRAYSCALE)
    origin = d_resize(origin, sketch.shape).astype(np.float32)
    low_origin = cv2.GaussianBlur(origin, (0, 0), 3.0)
    high_origin = origin - low_origin
    low_origin = (low_origin / np.median(low_origin) * 255.0).clip(0, 255)
    origin = (low_origin + high_origin).clip(0, 255).astype(np.uint8)
    face = from_png_to_jpg(get_request_image(_COLOR_IMAGE_))
    face = s_enhance(face, 2.0)
    sketch_1024 = k_resize(sketch, 64)
    hints_1024 = opreate_normal_hint(ini_hint(sketch_1024), points, length=2)
    careless = go_head(sketch_1024, k_resize(face, 14), hints_1024)
    smooth, flat, bsmooth, bflat = None, None, None, None
    for i in range(_GRADE_ + 1):
        if _GRADE_ALL_:
            smooth, flat, bsmooth, bflat = refine_image(careless, sketch, origin)
            # cv2.imwrite(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(i) + '.png', smooth) if _STYLE_ == 2 or _STYLE_ == 0 else None
            # cv2.imwrite(room_path + '/' + filtered_name + ID + '.flat.grade' + str(i) + '.png', flat) if _STYLE_ == 1 or _STYLE_ == 0 else None
            # cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(i) + '.png', bsmooth) if _STYLE_ == 4 or _STYLE_ == 0 else None
            # cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(i) + '.png', bflat) if _STYLE_ == 3 or _STYLE_ == 0 else None
            if _STYLE_ == 2 or _STYLE_ == 0:
                cv2.imwrite(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(i) + '.png', smooth)
                if _LIGHTING_:
                    run(image=str(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(i) + '.png'),
                        output_name=str(
                            room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(i) + 'lighting.png'))
            if _STYLE_ == 1 or _STYLE_ == 0:
                cv2.imwrite(room_path + '/' + filtered_name + ID + '.flat.grade' + str(i) + '.png', flat)
                if _LIGHTING_:
                    run(image=str(room_path + '/' + filtered_name + ID + '.flat.grade' + str(i) + '.png'),
                        output_name=str(
                            room_path + '/' + filtered_name + ID + '.flat.grade' + str(i) + 'lighting.png'))
            if _STYLE_ == 4 or _STYLE_ == 0:
                cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(i) + '.png', bsmooth)
                if _LIGHTING_:
                    run(image=str(
                        room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(i) + '.png'),
                        output_name=str(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(
                            _GRADE_) + 'lighting.png'))
            if _STYLE_ == 3 or _STYLE_ == 0:
                cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(i) + '.png', bflat)
                if _LIGHTING_:
                    run(image=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(i) + '.png'),
                        output_name=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(
                            _GRADE_) + 'lighting.png'))
            if i == _GRADE_:
                break
            careless = go_render(sketch_1024, d_resize(flat, sketch_1024.shape, 0.5), hints_1024)
            # print('grade ' + str(i) + ' finished.')
        else:
            smooth, flat, bsmooth, bflat = refine_image(careless, sketch,origin)
            careless = go_render(sketch_1024, d_resize(flat, sketch_1024.shape, 0.5), hints_1024)
    if _STYLE_ == 2 or _STYLE_ == 0:
        cv2.imwrite(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + '.png', smooth)
        if _LIGHTING_:
            run(image=str(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + '.png'),
                output_name=str(
                    room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + 'lighting.png'))
    if _STYLE_ == 1 or _STYLE_ == 0:
        cv2.imwrite(room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + '.png', flat)
        if _LIGHTING_:
            run(image=str(room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + '.png'),
                output_name=str(
                    room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + 'lighting.png'))
    if _STYLE_ == 4 or _STYLE_ == 0:
        cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(_GRADE_) + '.png', bsmooth)
        if _LIGHTING_:
            run(image=str(
                room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(_GRADE_) + '.png'),
                output_name=str(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(
                    _GRADE_) + 'lighting.png'))
    if _STYLE_ == 3 or _STYLE_ == 0:
        cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(_GRADE_) + '.png', bflat)
        if _LIGHTING_:
            run(image=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(_GRADE_) + '.png'),
                output_name=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(
                    _GRADE_) + 'lighting.png'))
    # cv2.imwrite(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + '.png', smooth) if _STYLE_ == 2 or _STYLE_ == 0 else None
    # cv2.imwrite(room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + '.png', flat) if _STYLE_ == 1 or _STYLE_ == 0 else None
    # cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(_GRADE_) + '.png', bsmooth) if _STYLE_ == 4 or _STYLE_ == 0 else None
    # cv2.imwrite(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(_GRADE_) + '.png', bflat) if _STYLE_ == 3 or _STYLE_ == 0 else None
    # # print('grade ' + str(_GRADE_) + ' finished.')
    # run(image=str(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + '.png'), output_name=str(room_path + '/' + filtered_name + ID + '.smoothed.grade' + str(_GRADE_) + 'lighting.png'))
    # run(image=str(room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + '.png'), output_name=str(room_path + '/' + filtered_name + ID + '.flat.grade' + str(_GRADE_) + 'lighting.png'))
    # run(image=str(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(_GRADE_) + '.png'), output_name=str(room_path + '/' + filtered_name + ID + '.blended_smooth.grade' + str(_GRADE_) + 'lighting.png'))
    # run(image=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(_GRADE_) + '.png'), output_name=str(room_path + '/' + filtered_name + ID + '.blended_flat.grade' + str(_GRADE_) + 'lighting.png'))

    os.remove(room_path + filtered_name + '.sketch.png')
    os.remove(room_path + filtered_name + '.origin.png')

    return room_path


def main():
    print("Style2PaintsV4.5 Starting...")
    room_path = ''
    input_files = glob.glob(f"{_INPUT_}/*.*")
    ext = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif']
    threads_ = []
    thread_batches = []
    max_threads = 24
    current_step = 0
    # remainder = len(input_files) % max_threads
    for file in tqdm(input_files, desc='Threads'):
        if not file.endswith(tuple(ext)):
            pass
        else:
            # create a new thread for each file and add the thread to the list
            upload_sketch(_INPUT_=file)
            threads_.append(threading.Thread(target=request_result, args=(file,)))
            current_step += 1

            if current_step == max_threads:
                thread_batches.append(threads_)
                current_step = 0
                for thread in threads_:
                    thread.start()
                for thread in tqdm(threads_, desc='Join'):
                    thread.join()
                threads_ = []

    # thread_batches.append(threads_) if len(threads_) > 0 else None
    # for batch in tqdm(thread_batches, desc='Batch'):
    #     for thread in batch:
    #         thread.start()
    #     for thread in tqdm(batch, desc='Join'):
    #         thread.join()
    print("Style2PaintsV4.5 Finished.")



    # for file in glob.glob(f"{room_path}/*.png"):
    #     shutil.copy(file, _OUTPUT_)


# if __name__ == '__main__':
main()
