from ..common import *
from standard_connectors import *


def prob_connector(pre, post, conn_prob, weight=2., delay=1.):
    pass

def convolution_wta(layer_width, layer_height, convolution_width,
                    neurons_per_zone, input_weight, 
                    wta_exc_weight, wta_inh_weight, 
                    all2all=True, conn_prob=0.1,
                    col_start=0, row_start=0,
                    col_step=1, row_step=1):

    convw = convolution_width
    lh, lw = layer_height, layer_width
    zone_count = 0
    nrn_pz = neurons_per_zone
    
    exc_conns = []
    wta_e_conns = []
    wta_i_conns = []
    
    for r in range(row_start, lh-convw, row_step):
        for c in range(col_start, lw-convw, col_step):
            
            src = [i*lw + j for i in range(r, convw) \
                            for j in range(c, convw)]
            dst = [zone_count*nrn_pz + i for i in range(neurons_per_zone)]

            zone_count += 1 
            
            if all2all:
                exc_conns += list_all2all(src, dst, input_weight, delay=1)
            else:
                exc_conns += prob_connector(src, dst, input_weight, delay=1)
            
            # assume there's a population equal to dst for interneurons
            tmp = list_wta_interneuron(dst, dst, wta_exc_weight, 
                                       wta_inh_weight, delay=1.)
            
            wta_e_conns += tmp[0]
            wta_i_conns += tmp[1]

    return exc_conns, wta_e_conns, wta_i_conns
