from sim_tools.common import *
from sim_tools.connectors import kernel_connectors as conn_krn, \
                                 standard_connectors as conn_std




class MultiColumn():
    
    def __init__(self, sim, lgn, learning_on,
                 in_width, in_height, in_location, in_receptive_width, 
                 group_size, cfg=defaults):
        
        self.cfg = cfg
        self.sim = sim
        self.learn_on = learning_on
        
        self.lgn = lgn
        self.in_width  = in_width
        self.in_height = in_height
        self.in_location = in_location
        self.in_receptive_width = in_receptive_width
        self.group_size = group_size
        
        self.pix_key   = 'cs'
        self.feat_keys = [k for k in lgn.pops.keys() if k != 'cs']

        self.build_input_indices()
        self.build_connectors()
        self.build_populations()
        

    def update_weights(self, new_weights):
        pass
        
        
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
    
    
    def build_connectors(self):
        conns = {}
        cfg = self.cfg
        size = self.group_size
        in_idx = self.in_indices
        spl_idx = [i in range(size)]
        conns['in2sipl'] = conn_std.list_all2all(in_idx, spl_idx, 
                                            weight=cfg['pix_in_weight'], 
                                            delay=1.)
        
        conns['in2cnt'] = [conn_std.list_all2all(in_idx, [i], \
                                      weight=cfg['context_in_weight'],\
                                      delay=1.) for i in range(size) ]

        conns['sipl2intr'] = conn_std.list_wta_interneuron(spl_idx, spl_idx, 
                                                    ff_weight=cfg['w2s'], 
                                                    fb_weight=-cfg['w2s'], 
                                                    delay=1.)

        self.conns = conns


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
        
        if cfg['record']['spikes']:
            for k in pops:
                pops[k].record()

        if cfg['record']['voltages']:
            for k in pops:
                pops[k].record_v()
        
        self.pops = pops
    
    
    def get_synapse_dynamics(self):
        cfg = self.cfg['stdp']
        
        stdp_model = STDPMechanism(
            timing_dependence = SpikePairRule(tau_plus=cfg['tau_plus'], 
                                              tau_minus=cfg['tau_minus']),
            weight_dependence = AdditiveWeightDependence(w_min=cfg['w_min'], 
                                                         w_max=cfg['w_max'], 
                                                         A_plus=cfg['a_plus'], 
                                                         A_minus=cfg['a_minus']),
        )
        syn_dyn = SynapseDynamics(slow=stdp_model)
        
        return syn_dyn
    
    
    def build_projections(self):
        sim = self.sim
        cfg = self.cfg
        in_pop = self.lgn.pops['cs']['output']
        
        projs = {}
        
        conn = sim.FromListConnector(self.conns['in2sipl'])
        syn_dyn = self.get_synapse_dynamics() if self.learn_on else None
        projs['in2sipl'] = sim.Projection(in_pop, self.pops['simple'],
                                          conn, synapse_dynamics=syn_dyn,
                                          label='input to simple')
        
        conn = sim.FromListConnector(self.conns['sipl2intr'][0])
        projs['sipl2intr'] = sim.Projection(self.pops['simple'],
                                            self.pops['wta_inh'],
                                            conn, 
                                            label='simple to inter')

        conn = sim.FromListConnector(self.conns['sipl2intr'][1])
        projs['intr2sipl'] = sim.Projection(self.pops['wta_inh'],
                                            self.pops['simple'],
                                            conn, 
                                            label='inter to simple')
    
        projs['ctxt2wta'] = self.build_input_to_context_projections()
        
        self.projs = projs

    def rand_context_pops(self):
        feat_keys = self.feat_keys
        lgn = self.lgn
        np.random.seed(np.uint32( time.time()*(10**6) ))
        pop_idx = np.random.choice(len(feat_keys), size=2,
                                   replace=False)
                                   
        k0 = feat_keys[pop_idx[0]]
        k1 = feat_keys[pop_idx[1]]

        return k0, k1, lgn[k0]['output'], lgn[k1]['output'] 
    
    
    def build_input_to_context_projections(self):
        cfg = self.cfg
        projs = []
        
        for i in range(size):
            k0, k1, pop0, pop1 = self.rand_context_pops()
            conn = sim.FromListConnector(self.conns['in2cnt'][i])
            projs.append(sim.Projection(pop0, self.pops['wta_inh'],
                                        conn, label='%s to inter'%k0) )
            projs.append(sim.Projection(pop1, self.pops['wta_inh'],
                                        conn, label='%s to inter'%k1) )

                                        
        return projs
