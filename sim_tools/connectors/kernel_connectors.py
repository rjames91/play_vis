from ..common import *

def full_kernel_connector(layer_width, layer_height, kernel, delay=1.,
                          col_step=1, row_step=1, 
                          col_start=0, row_start=0, min_w = 0.001):
    '''Create connection list based on a convolution kernel, the format
       for the lists is to be used with PyNN 0.7. 
       (Pre neuron index, Post neuron index, weight, delay)
       
       :param layer_width: Pre layer width
       :param layer_height: Pre layer height
       :param kernel: Convolution kernel
       :param col_step: Skip this many columns (will reduce Post layer size)
       :param row_step: Skip this many rows (will reduce Post layer size)
       :param delay: How many time units will it take the spikes to
                     get from Pre to Post

       :return exc_conns: Excitatory connections list
       :return inh_conns: Inhibitory connections list
    '''
    exc_conns = []
    inh_conns = []
    kh, kw = kernel.shape
    half_kh, half_kw = kh//2, kw//2
    
    for dr in range(row_start, layer_height, row_step):
        for dc in range(col_start, layer_width, col_step):
            sr0 = dr - half_kh
            sc0 = dc - half_kw

            for kr in range(kh):
                sr = sr0 + kr
                if sr < 0 or sr >= layer_height:
                    continue

                for kc in range(kw):
                    sc = sc0 + kc
                    if sc < 0 or sc >= layer_width:
                        continue

                    w = kernel[kr, kc]
                    if np.abs(w) < min_w:
                        continue
                    
                    src = sr*layer_width + sc
                    # divide values so that indices match the size of the
                    # Post (destination) next layer
                    dst = (dr//row_step)*layer_width//col_step + (dc//col_step)
                    
                    src = int(src); dst = int(dst); w = float(w);
                    delay = float(delay)
                    
                    if w < 0:
                        inh_conns.append((src, dst, w, delay))
                    elif w > 0:
                        exc_conns.append((src, dst, w, delay))

    return exc_conns, inh_conns
