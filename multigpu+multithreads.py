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

class SUBNet(bp.Container):
  def __init__(self, spike,scale=1.0,num=20,method='exp_auto',device = devices[1],x = 1):
    super().__init__()
    self.device = device
    self.falseN = InputNeurons(spike,device=self.device)
    self.neurons = LIF(num,device=self.device)
    self.E = Exponential(self.falseN, self.neurons,
                         g_max_par=dict(type='homo', value=0.6 / scale, prob=0.02, seed=123),
                         tau=5., method=method, output=bp.synouts.COBA(E=0.),device=self.device)
    self.I = Exponential(self.falseN, self.neurons,
                         g_max_par=dict(type='homo', value=6.7 / scale, prob=0.02, seed=12345),
                         tau=10., method=method, output=bp.synouts.COBA(E=-80.),device=self.device)
    
    self.x = x
  
  def update(self,t,dt,spike,*args, **kwargs):
    spike = jax.device_put(spike,self.device)
    self.falseN.update(spike)
    input = 20.
    self.neurons.input+=self.x*self.E.update(t,dt)+(1-self.x)*self.I.update(t,dt)+input
    self.neurons.input+=self.E.update(t,dt)+input
    self.neurons.update(t,dt)
  
def get_excute_time(scale=0.5,type=1):
  t1 = time.time()
  num = 20000
  num = int(num*scale)

  spikes = [jax.device_put(bm.ones(num,dtype=bool),devices[0]) for i in range(2)]
  if type ==0:
    #single gpu
    nets = [SUBNet(num=num,scale=scale,spike=spikes[i],device=devices[0],x = i) for i in range(2)]
  else:
    #multi gpu
    nets = [SUBNet(num=num,scale=scale,spike=spikes[i],device=devices[i],x = i) for i in range(2)]
    


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

get_excute_time(scale=25,type=2)
# run_times = []
# print(type)
# for scale in scales:
#   print(scale)
#   run_times.append(get_excute_time(scale=scale,type=type))
# run_times_types.append(run_times)

# print(run_times_types)
# np.save('multigpu/time/run_times_types.txt',run_times_types)