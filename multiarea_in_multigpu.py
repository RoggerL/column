import brainpy as bp
import brainpy.math as bm
import jax 
import numpy as np
import threading
import matplotlib.pyplot as plt
import time
import datetime
import threading
# bm.disable_gpu_memory_preallocation() 

devices = jax.devices()
devices = [devices[2],devices[3]]
class LIF(bp.NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., Cm=1., tau=10., t_ref=5., device=devices[0],**kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # initialize parameters
    self.V_rest = jax.device_put(V_rest,device)
    self.V_reset = jax.device_put(V_reset,device)
    self.V_th = jax.device_put(V_th,device)
    self.Cm = jax.device_put(Cm,device)
    self.tau = jax.device_put(tau,device)
    self.t_ref = jax.device_put(t_ref,device)

    # initialize variables
    self.V = bm.Variable(jax.device_put(bm.random.randn(self.num) + V_reset,device))
    self.input = bm.Variable(jax.device_put(bm.zeros(self.num),device))
    self.t_last_spike = bm.Variable(jax.device_put(bm.ones(self.num) * -1e7,device))
    self.refractory = bm.Variable(jax.device_put(bm.zeros(self.num, dtype=bool),device))
    self.spike = bm.Variable(jax.device_put(bm.zeros(self.num, dtype=bool),device))

    # integral function
    self.integral = bp.odeint(f=self.derivative, method='exp_auto')

  def derivative(self, V, t, Iext):
    dvdt = (-V + self.V_rest + Iext/self.Cm) / self.tau
    return dvdt

  def update(self, _t, _dt):
    # Whether the neurons are in the refractory period
    refractory = (_t - self.t_last_spike) <= self.t_ref
    
    # compute the membrane potential
    V = self.integral(self.V, _t, self.input, dt=_dt)
    
    # computed membrane potential is valid only when the neuron is not in the refractory period 
    V = bm.where(refractory, self.V, V)
    
    # update the spiking state
    spike = self.V_th <= V
    self.spike.value = spike
    
    # update the last spiking time
    self.t_last_spike.value = bm.where(spike, _t, self.t_last_spike)
    
    # update the membrane potential and reset spiked neurons
    self.V.value = bm.where(spike, self.V_reset, V)
    
    # update the refractory state
    self.refractory.value = bm.logical_or(refractory, spike)
    
    # reset the external input
    self.input[:] = 0.
    
_default_g_max = dict(type='homo', value=1., prob=0.1, seed=123)
class Exponential(bp.TwoEndConnNS):
  def __init__(
      self,
      pre: bp.NeuGroup,
      post: bp.NeuGroup,
      output: bp.SynOut = bp.synouts.CUBA(),
      g_max_par=_default_g_max,
      E = 0.,
      delay_step=None,
      tau=8.0,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
      device=devices[0],
  ):
    super().__init__(pre, post, None, output=output, name=name, mode=mode)
    self.tau = tau
    self.g_max_par = g_max_par
    self.g = bm.Variable(jax.device_put(bm.zeros(self.post.num),device))
    self.E = E
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)
    # print(type(self.delay_step))
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)
    # self.output = output
  def reset_state(self, batch_size=None):
    self.g.value = bp.init.variable_(bm.zeros, self.post.num, batch_size)

  def update(self,t,dt):
    # t = bp.share.load('t')
    # dt = bp.share.load('dt')
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    if self.g_max_par['type'] == 'homo':
      f = lambda s: bm.event_matvec_prob_conn_homo_weight(s,
                                                          self.g_max_par['value'],
                                                          conn_prob=self.g_max_par['prob'],
                                                          shape=(self.pre.num, self.post.num),
                                                          seed=self.g_max_par['seed'],
                                                          transpose=True)
    elif self.g_max_par['type'] == 'uniform':
      f = lambda s: bm.event_matvec_prob_conn_uniform_weight(s,
                                                             w_low=self.g_max_par['w_low'],
                                                             w_high=self.g_max_par['w_high'],
                                                             conn_prob=self.g_max_par['prob'],
                                                             shape=(self.pre.num, self.post.num),
                                                             seed=self.g_max_par['seed'],
                                                             transpose=True)
    elif self.g_max_par['type'] == 'normal':
      f = lambda s: bm.event_matvec_prob_conn_normal_weight(s,
                                                            w_mu=self.g_max_par['w_mu'],
                                                            w_sigma=self.g_max_par['w_sigma'],
                                                            conn_prob=self.g_max_par['prob'],
                                                            shape=(self.pre.num, self.post.num),
                                                            seed=self.g_max_par['seed'],
                                                            transpose=True)
    else:
      raise ValueError
    if isinstance(self.mode, bm.BatchingMode):
      f = jax.vmap(f)
    post_vs = f(pre_spike)
    self.g.value = self.integral(self.g.value, t, dt) + post_vs
    # print(self.g.value.device_buffer.device())
    return self.g.value*(self.E - self.post.V.value)

class InputNeurons(bp.NeuGroup):
      def __init__(
      self,
      spike,
      keep_size: bool = False,
      name: str = None,
      mode: bm.Mode = None,
      device = devices[0],
  ):    
        self.device = device
        self.spike = bm.Variable(jax.device_put(spike,device))
        super().__init__(size=spike.shape,keep_size=keep_size,name=name,mode=mode)
        
      def update(self, spike,*args, **kwargs):
        self.spike.value = spike


class CorticalMicrocircuit(bp.Container):
  # Names for each layer:
  layer_name = ['L23e', 'L23i', 'L4e', 'L4i', 'L5e', 'L5i', 'L6e', 'L6i', 'Th']

  # Population size per layer:
  #            2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
  layer_num = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]

  # Layer-specific background input [nA]:
  #                             2/3e  2/3i  4e    4i    5e    5i    6e    6i
  layer_specific_bg = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100]) / 1000

  # Layer-independent background input [nA]:
  #                                2/3e  2/3i  4e    4i    5e    5i    6e    6i
  layer_independent_bg = np.array([2000, 1850, 2000, 1850, 2000, 1850, 2000, 1850]) / 1000

  # Prob. connection table
  conn_table = np.array([[0.101, 0.169, 0.044, 0.082, 0.032, 0.0000, 0.008, 0.000, 0.0000],
                         [0.135, 0.137, 0.032, 0.052, 0.075, 0.0000, 0.004, 0.000, 0.0000],
                         [0.008, 0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.000, 0.0983],
                         [0.069, 0.003, 0.079, 0.160, 0.003, 0.0000, 0.106, 0.000, 0.0619],
                         [0.100, 0.062, 0.051, 0.006, 0.083, 0.3730, 0.020, 0.000, 0.0000],
                         [0.055, 0.027, 0.026, 0.002, 0.060, 0.3160, 0.009, 0.000, 0.0000],
                         [0.016, 0.007, 0.021, 0.017, 0.057, 0.0200, 0.040, 0.225, 0.0512],
                         [0.036, 0.001, 0.003, 0.001, 0.028, 0.0080, 0.066, 0.144, 0.0196]])

  def __init__(self, scale=1,bg_type=0, stim_type=0, conn_type=0, poisson_freq=8., has_thalamus=False,method='exp_auto'):
    super(CorticalMicrocircuit, self).__init__()

    self.device = devices[0]
    
    # parameters
    self.bg_type = bg_type
    self.stim_type = stim_type
    self.conn_type = conn_type
    self.poisson_freq = poisson_freq
    self.has_thalamus = has_thalamus

    # NEURON: populations
    self.populations = bp.Collector()
    for i in range(8):
      l_name = self.layer_name[i]
      print(f'Creating {l_name} ...')
      self.populations[l_name] = LIF(self.layer_num[i])

    # SYNAPSE: synapses
    self.synapses = bp.Collector()
    for c in range(8):  # from
      for r in range(8):  # to
        if self.conn_table[r, c] > 0.:
          print(f'Creating Synapses from {self.layer_name[c]} to {self.layer_name[r]} ...')
          syn = Exponential(pre=self.populations[self.layer_name[c]],
                            post=self.populations[self.layer_name[r]],
                            g_max_par=dict(type='homo', value=0.6 / scale, prob=self.conn_table[r, c], seed=123),
                            tau=5., method=method, output=bp.synouts.COBA(E=0.),device=self.device)
          self.synapses[f'{self.layer_name[c]}_to_{self.layer_name[r]}'] = syn
    # Synaptic weight from L4e to L2/3e is doubled
    self.synapses['L4e_to_L23e'].g_max_par['value'] *= 2.

    # # NEURON & SYNAPSE: poisson inputs
    # if stim_type == 0:
    #   # print(f'Creating Poisson noise group ...')
    #   # self.populations['Poisson'] = PoissonInput2(
    #   #   freq=poisson_freq, pops=[self.populations[k] for k in self.layer_name[:-1]])
    #   for r in range(0, 8):
    #     l_name = self.layer_name[r]
    #     print(f'Creating Poisson group of {l_name} ...')
    #     N = PoissonInput(freq=poisson_freq, post=self.populations[l_name])
    #     self.populations[f'Poisson_to_{l_name}'] = N
    # elif stim_type == 1:
    #   bg_inputs = self._get_bg_inputs(bg_type)
    #   assert bg_inputs is not None
    #   for i, current in enumerate(bg_inputs):
    #     self.populations[self.layer_name[i]].Iext = 0.3512 * current

    # # NEURON & SYNAPSE: thalamus inputs
    # if has_thalamus:
    #   thalamus = bp.dyn.PoissonInput(self.layer_num[-1], freqs=15.)
    #   self.populations[self.layer_name[-1]] = thalamus
    #   for r in range(0, 8):
    #     l_name = self.layer_name[r]
    #     print(f'Creating Thalamus projection of {l_name} ...')
    #     S = ThalamusInput(pre=thalamus,
    #                       post=self.populations[l_name],
    #                       conn_prob=self.conn_table[r, 8])
    #     self.synapses[f'{self.layer_name[-1]}_to_{l_name}'] = S

    # finally, compose them as a network
    self.register_implicit_nodes(self.populations)
    self.register_implicit_nodes(self.synapses)

net = CorticalMicrocircuit()


def get_excute_time(scale=0.5,type=2):
  t1 = time.time()
  num = 20000
  num = int(num*scale)

  spikes = [jax.device_put(bm.ones(num,dtype=bool),devices[0]) for i in range(2)]
  if type ==0:
    #single gpu
    nets = [CorticalMicrocircuit(num=num,scale=scale,spike=spikes[i],device=devices[0],x = i) for i in range(2)]
  else:
    #multi gpu
    nets = [CorticalMicrocircuit(num=num,scale=scale,spike=spikes[i],device=devices[i],x = i) for i in range(2)]
    


  during = 1e2
  t = 0
  dt = 0.4
  t_all = []
  spike_all = []

  while t < during:
    if type ==2:
      print(type)
      #multi processing
      thread1 = threading.Thread(target=nets[0].update,args = (t,dt,nets[1].neurons.spike.value))
      thread2 = threading.Thread(target=nets[1].update,args = (t,dt,nets[0].neurons.spike.value))
      thread1.start()
      thread2.start()
      thread1.join()
      thread2.join()
    else:
      nets[0].update(t,dt,nets[1].neurons.spike.value)
      nets[1].update(t,dt,nets[0].neurons.spike.value)
      
    t = t + dt
    t_all.append(t)
    spike_all.append(np.hstack([np.array(nets[0].neurons.spike.value),np.array(nets[1].neurons.spike.value)]))
    print('程序完成度：{:.2f}%'.format(float(t)/float(during)*100.))

  spike_all = np.array(spike_all)

  t2 = time.time()
  bp.visualize.raster_plot(t_all, spike_all, show=True)
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d_%H%M%S")
  plt.savefig('multigpu/figure/scatter{}.png'.format(timestamp))

  t3 = time.time()
  print('程序运行时间:%s秒' %(t2 - t1))
  print('画图所用的时间：%s秒' %(t3-t2))
  return t2 - t1

scales = [0.5,5,25,50]
run_times_types = []
# for type in range(3):
#   run_times = []
#   print(type)
#   for scale in scales:
#     print(scale)
#     run_times.append(get_excute_time(scale=scale,type=type))
#   run_times_types.append(run_times)

get_excute_time(scale=25,type=3)
# run_times = []
# print(type)
# for scale in scales:
#   print(scale)
#   run_times.append(get_excute_time(scale=scale,type=type))
# run_times_types.append(run_times)

# print(run_times_types)
# np.save('multigpu/time/run_times_types.txt',run_times_types)