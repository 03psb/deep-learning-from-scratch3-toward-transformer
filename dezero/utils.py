import os
import urllib.request
import numpy as np
from dezero import as_variable
from dezero import Variable
from dezero import cuda


# =============================================================================
# Visualize for computational graph
# =============================================================================
def _dot_var(v):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'
    name = ''
    if hasattr(v, 'name') and v.name is not None:
        name = v.name

    return dot_var.format(id(v), name)


def _dot_func(f):
    # for function
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # for edge
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:  # y is weakref
        ret += dot_edge.format(id(f), id(y()))
    return ret


def get_dot_graph(y):
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.priority)
            seen_set.add(f)

    add_func(y.creator)
    txt = _dot_var(y)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'


# =============================================================================
# Utility functions (numpy magic)
# =============================================================================
def sum_to(x, shape):
    """x が shape の形状になるように和を求める。

    Parameters
    ----------
    x : numpy.ndarray
    shape : None or int or tuple of ints

    Returns
    -------
    y : numpy.ndarray
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(gy, x_shape, axis, keepdims):
    """dezero.functions.sum関数の逆伝播で伝わる勾配を適切な形状に変換する。

    Parameters
    ----------
    gy : dezero.Variable
        逆伝播で出力側から伝わる勾配
    x_shape : tuple
        順伝播のsum関数で使用した入力変数の形状
    axis : None or int or tuple of ints
        順伝播のsum関数の引数で指定した axis
    keepdims : bool
        順伝播のsum関数の引数で指定した keepdims

    Returns
    -------
    gy : dezero.Variable
        形状変換後の勾配
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not hasattr(axis, 'len'):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x, axis=1):
    xp = cuda.get_array_module(x)
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


def get_conv_outsize(input_size, kernel_size, stride, pad):
    return (input_size + pad * 2 - kernel_size) // stride + 1


def im2col(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride)
    PH, PW = _pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
                     mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = _pair(kernel_size)
    SH, SW = _pair(stride)
    PH, PW = _pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, img_shape, kernel_size, stride, pad)
    else:
        img = np.zeros((N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1),
                       dtype=col.dtype)
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]

    return img[:, :, PH:H + PH, PW:W + PW]


def _pair(x):
    if hasattr(x, '__iter__'):
        return x
    return (x, x)


def _im2col_gpu(img, kernel_size, stride, pad):
    """
    based on the code from
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = _pair(kernel_size)
    sy, sx = _pair(stride)
    ph, pw = _pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(col, img_shape, kernel_size, stride, pad):
    n, c, kh, kw, out_h, out_w = col.shape
    n, c, h, w = img_shape
    kh, kw = _pair(kernel_size)
    sy, sx = _pair(stride)
    ph, pw = _pair(pad)
    dx, dy = 1, 1

    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    oh = get_conv_outsize(h, kh, sy, ph)
    ow = get_conv_outsize(w, kw, sx, pw)
    col = col.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    cuda.cupy.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(f, x, eps=0.001, atol=1e-5, rtol=1e-4):
    y = f(x)
    y.backward()
    grad_num = numerical_grad(f, x, eps=eps)
    grad = x.grad if isinstance(x.grad, np.ndarray) else x.grad.data
    flg = np.allclose(grad, grad_num, atol=atol, rtol=rtol)
    return flg


def check_backward(func, x_data, y_grad=None, eps=0.001,
                   atol=1e-5, rtol=1e-4, verbose=True):
    x_data = _as_tuple(x_data)
    x_data = tuple([x.astype(np.float64) for x in x_data])
    if y_grad is not None:
        y_grad = y_grad.astype(np.float64)

    def f(inputs):
        inputs = _as_tuple(inputs)
        inputs = [as_variable(x) for x in inputs]
        y = func(*inputs)
        return y.data

    num_grads = numerical_grad(f, x_data, y_grad, eps)
    inputs = [as_variable(x) for x in x_data]
    y = func(*inputs)
    if y_grad is not None:
        y.grad = Variable(y_grad)
    y.backward()
    bp_grads = [x.grad.data for x in inputs]

    results = []
    for num_grad, bp_grad in zip(num_grads, bp_grads):
        assert bp_grad.shape == num_grad.shape
        res = np.allclose(num_grad, bp_grad, atol=atol, rtol=rtol)
        results.append(res)
        if not res and verbose:
            diff = abs(num_grad - bp_grad)
            print('-------------------------')
            print('diff', diff)
            print('diff mean', np.array(diff).mean())
            # print('num_grad:', num_grad.shape, num_grad)
            # print('bp_grad:', bp_grad.shape, bp_grad)

    return all(results)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def numerical_grad(f, inputs, grad_output=None, eps=0.001):
    h = eps
    inputs = _as_tuple(inputs)
    grads = [np.zeros_like(x) for x in inputs]

    for x, grad in zip(inputs, grads):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]

            x[idx] = tmp_val + h
            fxh1 = f(*inputs)  # f(x+h)
            fxh1 = np.copy(fxh1)

            x[idx] = tmp_val - h
            fxh2 = f(*inputs)  # f(x-h)
            fxh2 = np.copy(fxh2)

            if grad_output is None:
                diff = (fxh1 - fxh2).sum()
            else:
                diff = ((fxh1 - fxh2) * grad_output).sum()

            grad[idx] = diff / (2 * h)

            x[idx] = tmp_val  #
            it.iternext()

    return _as_tuple(grads)


# =============================================================================
# download function
# =============================================================================
def show_progress(block_num, block_size, total_size):
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0: p = 100.0
    if i >= 30: i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end='')

cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')

def get_file(url, file_name=None):
    """ファイルをダウンロードする。
    すでにダウンロード済みの場合は、そのファイルを使用する。

    Parameters
    ----------
    url : str
        ダウンロード先のURL
    file_name : str
        保存するファイル名
    """
    if file_name is None:
        file_name = url[url.rfind('/') + 1:]
    file_path = os.path.join(cache_dir, file_name)


    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path