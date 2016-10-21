from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std
from scipy.signal import convolve2d, correlate2d
MERGED, SPLIT = 0, 1
dvs_modes = ['merged', 'split']
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


class Retina():
    
    def __init__(self, simulator, camera_pop, width, height, dvs_mode, 
                 cfg=defaults):
        
        for k in defaults.keys():
            if k not in cfg.keys():
                cfg[k] = defaults[k]
        
        self.width = width
        self.height = height
        self.dvs_mode = dvs_mode
        if self.dvs_mode == dvs_modes[0]:
            self.on_idx = 0
            self.off_idx = width*height
        else:
            self.on_idx = 0
            self.off_idx = 0
        
        self.filter_size = ((width  - cfg['start_col'])//cfg['col_step'])*\
                           ((height - cfg['start_row'])//cfg['row_step'])
        
        self.cam = {'on':  camera_pop if dvs_mode==dvs_modes[0] else camera_pop[ON],
                    'off': camera_pop if dvs_mode==dvs_modes[0] else camera_pop[OFF],
                   }
        self.sim = simulator
        
        self.cfg = cfg
        
        self.ang_div = deg2rad(180./cfg['gabor']['num_divs'])
        self.angles = [i*self.ang_div for i in range(cfg['gabor']['num_divs'])]
        
        self.build_kernels()
        self.build_connectors()
        self.build_populations()
        self.build_projections()
    
    def get_output_keys(self):
        return [k for k in self.pops['on'] if k is not 'cam_inter']
        
    def build_kernels(self):
        def a2k(a):
            return 'gabor_%d'%( int( a ) )
            
        cfg = self.cfg
        angles = self.angles
        self.cs = krn_cs.center_surround_kernel(cfg['kernel_width'],
                                                cfg['ctr_srr']['std_dev'], 
                                                cfg['ctr_srr']['sd_mult'])
        self.cs *= cfg['w2s']
         
        gab = krn_gbr.multi_gabor(cfg['kernel_width'], 
                                  angles, 
                                  cfg['gabor']['std_dev'], 
                                  cfg['gabor']['freq'])
        self.gab = {a2k(k): gab[k]*cfg['w2s'] for k in gab.keys()}
        # self.gab = gab
        
        # self.cs_correlation =  convolve2d(self.cs, self.cs, mode='same')
        # self.cs_correlation = -self.cs_correlation*(self.cs_correlation > 0)

        # self.gab_correlation = []
        # for g in self.gab:
            # corr = convolve2d(g, g, mode='same')
            # corr = -corr*(corr > 0)
            # self.gab_correlation.append(corr)
            

    def build_connectors(self):
        cfg = self.cfg
        self.conns = {'off': {}, 'on':{}}

        self.conns['off']['cs'] = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.cs,
                                                                 cfg['kernel_exc_delay'],
                                                                 cfg['kernel_inh_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['start_col'], 
                                                                 cfg['start_row'], 
                                                                 self.off_idx)
        
        self.conns['on']['cs']  = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.cs,
                                                                 cfg['kernel_exc_delay'],
                                                                 cfg['kernel_inh_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['start_col'], 
                                                                 cfg['start_row'], 
                                                                 self.on_idx)

        for k in self.gab.keys():
            
            self.conns['off'][k] = conn_krn.full_kernel_connector(self.width,
                                                                  self.height,
                                                                  self.gab[k],
                                                                  cfg['kernel_exc_delay'],
                                                                  cfg['kernel_inh_delay'],
                                                                  cfg['col_step'], 
                                                                  cfg['row_step'],
                                                                  cfg['start_col'], 
                                                                  cfg['start_row'], 
                                                                  self.off_idx)

            self.conns['on'][k] = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.gab[k],
                                                                 cfg['kernel_exc_delay'],
                                                                 cfg['kernel_inh_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['start_col'], 
                                                                 cfg['start_row'], 
                                                                 self.on_idx)

        if self.dvs_mode == dvs_modes[0]:
            conns = conn_std.one2one(self.width*self.height*2,
                                     weight=cfg['w2s'], 
                                     delay=cfg['kernel_inh_delay'])
        else:
            conns = conn_std.one2one(self.width*self.height,
                                     weight=cfg['w2s'], 
                                     delay=cfg['kernel_inh_delay'])
        
        self.extra_conns = {'o2o': conns}
    
    def build_populations(self):
        self.pops = {}
        sim = self.sim
        cfg = self.cfg
        exc_cell = getattr(sim, cfg['exc_cell']['cell'], None)
        exc_parm = cfg['exc_cell']['params']
        inh_cell = getattr(sim, cfg['inh_cell']['cell'], None)
        inh_parm = cfg['inh_cell']['params']

        if self.dvs_mode == dvs_modes[0]:
            cam_inter = sim.Population(self.width*self.height*2,
                                       inh_cell, inh_parm,
                                       label='cam_inter')
            if cfg['record']['voltages']:
                cam_inter.record_v()

            if cfg['record']['spikes']:
                cam_inter.record()

            for k in self.conns.keys():
                self.pops[k] = {}
                self.pops[k]['cam_inter'] = cam_inter
        else:
            for k in self.conns.keys(): 
                self.pops[k] = {}
                self.pops[k]['cam_inter'] = sim.Population(self.width*self.height,
                                                           inh_cell, inh_parm,
                                                           label='cam_inter_%s'%k)
                if cfg['record']['voltages']:
                   self.pops[k]['cam_inter'].record_v()

                if cfg['record']['spikes']:
                    self.pops[k]['cam_inter'].record()

        for k in self.conns.keys():
            for p in self.conns[k].keys():
                self.pops[k][p] = {'bipolar': sim.Population(self.filter_size,
                                                             exc_cell, exc_parm,
                                                             label='bipolar_%s_%s'%(k, p)),
                                                             
                                   'inter':   sim.Population(self.filter_size,
                                                             inh_cell, inh_parm,
                                                             label='inter_%s_%s'%(k, p)),
                                                             
                                   'ganglion':  sim.Population(self.filter_size,
                                                               exc_cell, exc_parm,
                                                               label='ganglion_%s_%s'%(k, p)),
                                  } 
                if cfg['record']['voltages']:
                   self.pops[k][p]['bipolar'].record_v()
                   self.pops[k][p]['inter'].record_v()
                   self.pops[k][p]['ganglion'].record_v()

                if cfg['record']['spikes']:
                    self.pops[k][p]['bipolar'].record()
                    self.pops[k][p]['inter'].record()
                    self.pops[k][p]['ganglion'].record()

                
    def build_projections(self):
        self.projs = {}
        cfg = self.cfg
        sim = self.sim
        
        if self.dvs_mode == dvs_modes[0]:
            conn = self.extra_conns['o2o']
            exc = sim.Projection(self.cam['on'], 
                                 self.pops['on']['cam_inter'],
                                 sim.FromListConnector(conn),
                                 target='excitatory')
            
            for k in self.conns.keys():
                self.projs[k] = {}
                self.projs[k]['cam_inter'] = {}
                self.projs[k]['cam_inter']['cam2intr'] = [exc]
        else:
            for k in self.conns.keys():
                self.projs[k] = {}
                self.projs[k]['cam_inter'] = {}
                
                conn = self.extra_conns['o2o']
                exc = sim.Projection(self.cam[k], 
                                     self.pops[k]['cam_inter'],
                                     sim.FromListConnector(conn),
                                     target='excitatory')            
                self.projs[k]['cam_inter']['cam2intr'] = [exc]

        for k in self.conns.keys():
            for p in self.conns[k].keys():
                self.projs[k][p] = {}
                exc = sim.Projection(self.cam[k], 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(self.conns[k][p][EXC]),
                                     target='excitatory')
                inh = sim.Projection(self.pops[k]['cam_inter'], 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(self.conns[k][p][INH]),
                                     target='inhibitory')
                
                self.projs[k][p]['cam2bip'] = [exc, inh]
                
                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['inter'],
                                     sim.OneToOneConnector(weights=cfg['w2s'], 
                                                           delays=cfg['kernel_inh_delay']),
                                     target='excitatory')
                
                self.projs[k][p]['bip2intr'] = [exc]

                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(self.conns['on']['cs'][EXC]),
                                     target='excitatory')
                inh = sim.Projection(self.pops[k][p]['inter'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(self.conns['on']['cs'][INH]),
                                     target='inhibitory')
                
                self.projs[k][p]['bip2gang'] = [exc, inh]

                

                
