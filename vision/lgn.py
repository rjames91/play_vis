from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std
from scipy.signal import convolve2d, correlate2d

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



class LGN():
    def __init__(self, simulator, retina, width, height, cfg=defaults):
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        
        self.cfg = cfg
        self.width = width
        self.height = height
        self.sim = simulator
        self.retina = retina
        self.popsize = width*height

        
        self.kernels()
        self.connections()
        self.populations()
        self.projections()


    def kernels(self):
        cfg = self.cfg
        self.cs = krn_cs.center_surround_kernel(cfg['kernel_width'],
                                                cfg['ctr_srr']['std_dev'], 
                                                cfg['ctr_srr']['sd_mult'])
        self.cs *= cfg['w2s']
        
        self.split_cs = krn_cs.split_center_surround_kernel(cfg['kernel_width'],
                                                            cfg['ctr_srr']['std_dev'], 
                                                            cfg['ctr_srr']['sd_mult'])
        for i in range(len(self.split_cs)):
            self.split_cs[i] *= cfg['w2s']


    def connections(self):
        cfg = self.cfg
        conns = {}
        conns['cs'] = conn_krn.full_kernel_connector(self.width,
                                                     self.height,
                                                     self.cs,
                                                     cfg['kernel_exc_delay'],
                                                     cfg['kernel_inh_delay'],
                                                     cfg['col_step'], 
                                                     cfg['row_step'],
                                                     cfg['start_col'], 
                                                     cfg['start_row'])
                                                     
        exc, inh = conn_krn.full_kernel_connector(self.width,
                                                        self.height,
                                                        self.split_cs[EXC],
                                                        cfg['kernel_exc_delay'],
                                                        cfg['kernel_inh_delay'],
                                                        cfg['col_step'], 
                                                        cfg['row_step'],
                                                        cfg['start_col'], 
                                                        cfg['start_row'])
        
        tmp, inh[:] = conn_krn.full_kernel_connector(self.width,
                                                     self.height,
                                                     self.split_cs[INH],
                                                     cfg['kernel_exc_delay'],
                                                     cfg['kernel_inh_delay'],
                                                     cfg['col_step'], 
                                                     cfg['row_step'],
                                                     cfg['start_col'], 
                                                     cfg['start_row'],
                                                     remove_inh_only=False)
        

        conns['split'] = [exc, inh]
        self.conns = conns
        
    def populations(self):
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']
        
        pops = {}
        
        for k in self.retina.get_output_keys():
            pops[k] = {}
            pops[k]['inter'] = sim.Population(self.popsize,
                                              inh_cell, inh_parm,
                                              label='LGN inter %s'%k)
            pops[k]['output'] = sim.Population(self.popsize,
                                               exc_cell, exc_parm,
                                               label='LGN output %s'%k)

            if cfg['record']['voltages']:
                pops[k]['inter'].record_v()
                pops[k]['output'].record_v()

            if cfg['record']['spikes']:
                pops[k]['inter'].record()
                pops[k]['output'].record()
            
            
        self.pops = pops


    def projections(self):
        sim = self.sim
        cfg = self.cfg
        projs = {}
        for k in self.retina.get_output_keys():
            projs[k] = {}
            o2o = sim.OneToOneConnector(weights=cfg['w2s'],
                                        delays=cfg['kernel_inh_delay'])
            projs[k]['inter'] = sim.Projection(self.retina.pops['off'][k]['ganglion'],
                                               self.pops[k]['inter'], o2o,
                                               target='excitatory')
            
            flc = sim.FromListConnector(self.conns['split'][INH]) #conns['cs']?
            projs[k]['inh'] = sim.Projection(self.pops[k]['inter'], 
                                             self.pops[k]['output'], flc,
                                             target='inhibitory')
                                             
            flc = sim.FromListConnector(self.conns['split'][EXC])
            projs[k]['inh'] = sim.Projection(self.retina.pops['on'][k]['ganglion'], 
                                             self.pops[k]['output'], flc,
                                             target='excitatory')
                                             
                                             
            

        
