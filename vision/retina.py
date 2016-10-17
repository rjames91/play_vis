from sim_tools.common import *
from sim_tools.kernels import center_surround as krn_cs, gabor as krn_gbr
from sim_tools.connectors import kernel_connectors as conn_krn 
from scipy.signals import convolve2d

dvs_modes = ['merged', 'split']
defaults = {'imgw': 160, 'imgh': 128, 'dvs': dvs_modes[0],
            'kernel_width': 3,
            'row_step': 1, 'col_step': 1,
            'start_row': 0, 'start_col': 0,
            'gabor': {'num_divs': 7., 'freq': 4., 'std_dev': 1.1}
            'ctr_srr': {'std_dev': 0.8, 'sd_mult': 6.7} 
            'kernel_delay': 1.,
            'w2s': 1.6,
            'inh_cell': {'cell': self.sim.IF_curr_exp,
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
                        } 
            'exc_cell': {'cell': self.sim.IF_curr_exp,
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
                        }
            'record': {'voltage': False, 
                       'spikes': False,
                       'weights': False
                      }
           }

class Retina():
    
    def __init__(simulator, camer_pop, width, height, dvs_mode, cfg=defaults):
        self.width = width
        self.height = height
        self.dvs_mode = dvs_mode
        if self.dvs_mode == dvs_modes[0]:
            self.on_idx = 0
            self.off_idx = width*height
        else:
            self.on_idx = 0
            self.off_idx = 0
        
        self.filter_size = ((width  - cfg['col_start'])//cfg['col_step'])*\
                           ((height - cfg['row_start'])//cfg['row_step'])
        
        self.cam = {'on':  camera_pop if dvs_mode==dvs_modes[0] else camera_pop[ON],
                    'off': camera_pop if dvs_mode==dvs_modes[0] else camera_pop[OFF],
                   }
        self.sim = simulator
        
        self.cfg = cfg
        
        self.ang_div = deg2rad(360./cfg['gabor']['num_divs'])
        self.angles = [i*self.ang_div for i in range(cfg['gabor']['num_divs'])]
        
        self.kernels()
        self.connectors()
        self.populations()
        self.projections()
    
    
    def kernels(self):
        def a2k(a):
            return 'gabor_%d'%( int(rad2deg(angles[i])) )
            
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
        self.gab = {a2k(angles[i]): gab[i]*cfg['w2s'] \
                                    for i in range(len(gab))}
        
        # self.cs_correlation =  convolve2d(self.cs, self.cs, mode='same')
        # self.cs_correlation = -self.cs_correlation*(self.cs_correlation > 0)

        # self.gab_correlation = []
        # for g in self.gab:
            # corr = convolve2d(g, g, mode='same')
            # corr = -corr*(corr > 0)
            # self.gab_correlation.append(corr)
            

    def connectors(self):
        cfg = self.cfg
        self.conns = {'off': {}, 'on':{}}
        
        self.conns['off']['cs'] = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.cs,
                                                                 cfg['kernel_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['col_start'], 
                                                                 cfg['row_start'], 
                                                                 self.off_idx)
        
        self.conns['on']['cs']  = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.cs,
                                                                 cfg['kernel_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['col_start'], 
                                                                 cfg['row_start'], 
                                                                 self.on_idx)

        for k in self.gab.keys():
            
            self.conns['on'][k] = conn_krn.full_kernel_connector(self.width,
                                                                 self.height,
                                                                 self.gab[k],
                                                                 cfg['kernel_delay'],
                                                                 cfg['col_step'], 
                                                                 cfg['row_step'],
                                                                 cfg['col_start'], 
                                                                 cfg['row_start'], 
                                                                 self.off_idx) )

            self.conns['off'][k] = conn_krn.full_kernel_connector(self.width,
                                                                  self.height,
                                                                  self.gab[k],
                                                                  cfg['kernel_delay'],
                                                                  cfg['col_step'], 
                                                                  cfg['row_step'],
                                                                  cfg['col_start'], 
                                                                  cfg['row_start'], 
                                                                  self.on_idx) )

    
    
    def populations(self):
        self.pops = {}
        sim = self.sim
        exc_cell = self.cfg['exc_cell']['cell']
        exc_parm = self.cfg['exc_cell']['params']
        inh_cell = self.cfg['inh_cell']['cell']
        inh_parm = self.cfg['inh_cell']['params']

        for k in self.conns.keys():
            self.pops[k] = {}
            self.pops[k]['cam_inter'] = sim.Population(self.width*self.height,
                                                       inh_cell, inh_param,
                                                       label='cam_inter_%s'%k)
            if self.cfg['record']['voltages']:
               self.pops[k]['cam_inter'].record_v()

            if self.cfg['record']['spikes']:
                self.pops[k]['cam_inter'].record()
            
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
                if self.cfg['record']['voltages']:
                   self.pops[k][p]['bipolar'].record_v()
                   self.pops[k][p]['inter'].record_v()
                   self.pops[k][p]['ganglion'].record_v()

                if self.cfg['record']['spikes']:
                    self.pops[k][p]['bipolar'].record()
                    self.pops[k][p]['inter'].record()
                    self.pops[k][p]['ganglion'].record()

                
    def projections(self):
        self.projs = {}
        cfg = self.cfg
        sim = self.sim
        for k in self.conns.keys():
            self.projs[k] = {}
            
            self.projs[k]['cam_inter'] = {}
            exc = sim.Projection(self.cam[k], 
                                 self.cam[k]['cam_inter'],
                                 sim.OneToOneConnector(weights=cfg['w2s'], 
                                                       delay=cfg['kernel_delay']),
                                 target='excitatory')            
            self.projs[k]['cam_inter']['cam2intr'] = [exc]
            
            for p in self.conns[k].keys():
                self.projs[k][p] = {}
                exc = sim.Projection(self.cam[k], 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(self.conns[k][p][EXC]),
                                     target='excitatory')
                inh = sim.Projection(self.cam[k]['cam_inter'], 
                                     self.pops[k][p]['bipolar'],
                                     sim.FromListConnector(self.conns[k][p][INH]),
                                     target='inhibitory')
                
                self.projs[k][p]['cam2bip'] = [exc, inh]
                
                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['inter'],
                                     sim.OneToOneConnector(weights=cfg['w2s'], 
                                                           delay=cfg['kernel_delay']),
                                     target='excitatory')
                
                self.projs[k][p]['bip2intr'] = [exc]

                exc = sim.Projection(self.pops[k][p]['bipolar'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(self.conns[k]['cs'][EXC]),
                                     target='excitatory')
                inh = sim.Projection(self.pops[k][p]['inter'], 
                                     self.pops[k][p]['ganglion'],
                                     sim.FromListConnector(self.conns[k]['cs'][INH]),
                                     target='inhibitory')
                
                self.projs[k][p]['bip2gang'] = [exc, inh]

                

                
