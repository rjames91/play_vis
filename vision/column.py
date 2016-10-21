from sim_tools.common import *


defaults = {'kernel_width': 3,
            'kernel_exc_delay': 2.,
            'kernel_inh_delay': 1.,
            'row_step': 1, 'col_step': 1,
            'start_row': 0, 'start_col': 0,
            'gabor': {'num_divs': 7., 'freq': 5., 'std_dev': 1.1},
            'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} ,
            'w2s': 1.7,
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
           }


class CorticalColumn():
    
    def __init__(self, sim, in_width, in_height, in_location, in_receptive_width, 
                 group_size, cfg=defaults):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        self.cfg = cfg
        self.sim = sim
        self.in_width  = in_width
        self.in_height = in_height
        self.in_location = in_location
        self.in_receptive_width = in_receptive_width
        self.group_size = group_size
        
        self.build_input_indices()
        self.build_connectors()
        self.build_populations()
        
        
    def build_input_indices(self):
        indices = []
        hlf_in_w = self.in_receptive_width//2
        fr_r = max(0, self.location[ROW] - hlf_in_w)
        to_r = min(self.in_height, self.location[ROW] + hlf_in_w)
        fr_c = max(0, self.location[COL] - hlf_in_w)
        to_c = min(self.in_width, self.location[COL] + hlf_in_w)
        
        for r in range(fr_r, to_r):
            for c in range(fr_c, to_c):
                indices.append(r*self.in_width + c)
        
        self.in_indices = indices
    
    def update_weights(self, new_weights):
        pass
    
    def build_connectors(self):
        conns = {}
        
    
    def build_populations(self):
        def loc2lbl(pop, loc):
            return "column (%d, %d) - %s"(loc[ROW], loc[COL], pop)

        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        
        pops = {}
        pops['simple'] = sim.Population(self.group_size,
                                        exc_cell, exc_parm,
                                        label=loc2lbl(self.location, 'simple') )
                                                            
        pops['context'] = sim.Population(self.group_size,
                                         exc_cell, exc_parm,
                                         label=loc2lbl(self.location, 'context') )
                                                             
        pops['wta_inh'] = sim.Population(self.group_size,
                                         inh_cell, inh_parm,
                                         label=loc2lbl(self.location, 'wta') )
                                                
        pops['complex'] = sim.Population(self.group_size,
                                         exc_cell, exc_parm,
                                         label=loc2lbl(self.location, 'output') )
        
        self.pops = pops
        
        
