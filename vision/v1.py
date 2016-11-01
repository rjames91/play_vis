from sim_tools.common import *
from column import V1MultiColumn
import sys

defaults = {'kernel_width': 3,
            'kernel_exc_delay': 2.,
            'kernel_inh_delay': 1.,
            'gabor': {'num_divs': 7., 'freq': 5., 'std_dev': 1.1},
            'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} ,
            'w2s': 1.7,
            'pix_in_weight': 0.001,
            'context_in_weight': 0.3,
            'context_to_context_weight': 0.5, 
            'context_to_simple_weight': 1., 
            'inh_cell': {'cell': "IF_curr_exp",
                         'params': {'cm': 0.3,  # nF
                                    'i_offset': 0.0,
                                    'tau_m': 10.0,
                                    'tau_refrac': 2.0,
                                    'tau_syn_E': 2.,
                                    'tau_syn_I': 8.,
                                    'v_reset': -70.0,
                                    'v_rest': -65.0,
                                    'v_thresh': -55.4
                               }
                        }, 
            'exc_cell': {'cell': "IF_curr_exp",
                         'params': {'cm': 0.3,  # nF
                                    'i_offset': 0.0,
                                    'tau_m': 10.0,
                                    'tau_refrac': 2.0,
                                    'tau_syn_E': 2.,
                                    'tau_syn_I': 2.,
                                    'v_reset': -70.0,
                                    'v_rest': -65.0,
                                    'v_thresh': -55.4
                               }
                        },
            'record': {'voltages': False, 
                       'spikes': False,
                      },
            'lat_inh': False,
            'stdp': {'tau_plus': 15,
                     'tau_minus': 20,
                     'w_max': 0.25,
                     'w_min': 0.,
                     'a_plus': 0.01,
                     'a_minus': 0.02,
                    },
           }

class V1():
    
    def __init__(self, sim, lgn, learning_on,
                 in_width, in_height, unit_receptive_width, 
                 unit_group_size, cfg=defaults):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]

        self.sim = sim
        self.cfg = cfg
        self.lgn = lgn
        self.learn_on = learning_on
        self.in_width = in_width
        self.in_height = in_height
        self.unit_recpt_width = unit_receptive_width
        self.unit_group_size = unit_group_size
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']
        self.num_in_ctx = len(self.feat_keys)
        
        print("Building V1...")
        self.build_units()
        self.connect_units()


    def build_units(self):
        cfg = self.cfg
        cols = []
        hlf_col_w = self.unit_recpt_width//2
        r_start = hlf_col_w
        r_end = self.in_height - hlf_col_w
        r_step = hlf_col_w# + 1
        
        c_start = hlf_col_w
        c_end = self.in_width  - hlf_col_w
        c_step = hlf_col_w# + 1
        total_cols = (r_end//r_step)*(c_end//c_step)
        num_steps = 20
        cols_to_steps = float(num_steps)/total_cols
        
        print("\t%d columns..."%(total_cols))
        prev_step = 0
        curr_col = 0
        units = {}
        sys.stdout.write("\t\t")
        sys.stdout.flush()
        # sys.stdout.write("[%s]" % (" " * num_steps))
        # sys.stdout.flush()
        # sys.stdout.write("\b"*(num_steps + 1)) # return to start of line, after '['
        # sys.stdout.flush()
        for r in range(r_start, r_end, r_step):
            units[r] = {}
            for c in range(c_start, c_end, c_step):
                # print("\t\t Row, Col = (%d, %d)"%(r, c))
                
                mc = V1MultiColumn(self.sim, self.lgn, self.learn_on,
                                   self.in_width, self.in_height, 
                                   [r, c], self.unit_recpt_width, 
                                   self.unit_group_size, cfg=self.cfg)
                units[r][c] = mc
                
                curr_col += 1
                curr_step = int(curr_col*cols_to_steps)
                if curr_step > prev_step:
                    prev_step = curr_step
                    sys.stdout.write("#")
                    sys.stdout.flush()
                
        sys.stdout.write("\n")
        self.units = units
        # print(self.cols.keys())
        # print(self.cols[self.cols.keys()[0]].keys())
        self.num_rows = len(units.keys())
        self.units_per_row = len(units[units.keys()[0]])
        self.num_units = self.num_rows*self.units_per_row


    def connect_units(self):
        pass


