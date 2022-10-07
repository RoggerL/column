import brainpy as bp
import brainpy.math as bm
import matplotlib.pylab as plt
from matplotlib.pyplot import axes
from netParams import *

bm.set_platform("cpu")
bm.arange(10).value.device()

class LIF(bp.dyn.
          NeuGroup):
  def __init__(self, size, V_rest=0., V_reset=-5., V_th=20., Cm=1., tau=10., t_ref=5., **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # initialize parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.Cm = Cm
    self.tau = tau
    self.t_ref = t_ref

    # initialize variables
    self.V = bm.Variable(bm.random.randn(self.num) + V_reset)
    self.input = bm.Variable(bm.zeros(self.num))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)
    self.refractory = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))

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

#定义8层神经元模型
pop = []
popdict = dict()

pars = dict(V_rest=-65., V_th=-50., V_reset=-65., tau=10., tau_ref=2.)
for r in range(0, 8):
	pop.append(bp.dyn.LIF(n_layer[r], **pars))
	pop[-1].V[:] = bm.random.random(n_layer[r]) * (pop[-1].V_th - pop[-1].V_rest) + pop[-1].V_rest
popnet = bp.dyn.Network(*pop)
# print(popnet.nodes)

#计算突触延迟时间
av_delay = np.array([d_ex,d_in,d_ex,d_in,d_ex,d_in,d_ex,d_in])
std_delay = np.array([std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in])
delay = [av_delay[i] + std_delay[i]*np.random.normal() for i in range(8)]
print(delay)

#定义突触权重
def gmax(c,r):
 	# Excitatory connections
	if (c % 2) == 0:
		# Synaptic weight from L4e to L2/3e is doubled
		if c == 2 and r == 0:
			return 2.0
		else:
			return 1.
	# Inhibitory connections
	else:
		return -4.

tau = 10
w_ex=87.8*tau/250
w_ex_std = 8.8*tau/250

#定义8层神经元之间的突触连接
conn = []
# print(table)
for c in range(0,8):
    for r in range(0,8):
        K = n_layer[r]*table[r][c]*n_layer[c]
        # conmatrix = (np.random.random(size=(n_layer[c],n_layer[r]))<table[r][c])
        conn.append(bp.dyn.ExpCUBA(pop[c], pop[r], bp.conn.FixedProb(table[r][c]), g_max=(w_ex+w_ex_std*np.random.normal())*gmax(c,r),tau=tau_syn))
        # conn.append(bp.dyn.ExpCUBA(pop[c], pop[r], bp.connect.MatConn(conmatrix), g_max=(w_ex+w_ex_std*np.random.normal())*gmax(c,r), tau=tau_syn))

# Creating poissonian background inputs
input_type =3

bg_in  = []
bg_con = []
fre = 8
for c in range(0, 8):
	if input_type == 1:
		bg_in.append(bp.dyn.PoissonGroup(int(bg_layer_specific[c]), fre))
		bg_con.append(bp.dyn.ExpCUBA(bg_in[c], pop[c], bp.connect.All2All(),  g_max=(w_ex+w_ex_std*np.random.normal()), tau=tau_syn))
		# bg_con.append(bp.dyn.DeltaSynapse(bg_in[c], pop[c], bp.connect.All2All(),  weights=(w_ex+w_ex_std*np.random.normal())))
	# if input_type == 2:
		# bg_in.append([bp.dyn.PoissonGroup(int(n_layer[c]), fre) for i in range(0,bg_layer_specific)])
        # bg_con.append(bp.dyn.ExpCUBA(bg_in[c], pop[c], bp.connect.One2One(),  g_max=(w_ex+w_ex_std*np.random.normal()), tau=tau_syn))

spikelist = [pop[i].name+".spike" for i in range(0,8)]

if input_type == 1 or input_type == 2:
    net = bp.dyn.Network(*pop,*conn,*bg_in, *bg_con)
    runner = bp.dyn.DSRunner(net, monitors=spikelist)

if input_type ==3:
    bg_in  = [bg_layer_specific[i]*tau_syn*fre*0.001*w_ex for i in range(0,8)]
    
    # bg_in = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    inputlist = [pop[i].name+".input" for i in range(0,8)]
    inputpartlist = [(inputlist[i], bg_in[i]) for i in range(0,8)]

    net = bp.dyn.Network(*pop,*conn)
    runner = bp.dyn.DSRunner(net, monitors=spikelist, inputs = inputpartlist)
if input_type ==4:
    bg_in  = [bg_layer_specific[i]*tau_syn*fre*0.001*w_ex for i in range(0,8)]
    # bg_in = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    inputlist = [pop[i].name+".input" for i in range(0,8)]
    inputpartlist = [(inputlist[i], bg_in[i]) for i in range(0,8)]
    print(inputpartlist)
    net = bp.dyn.Network(*pop)
    runner = bp.dyn.DSRunner(net, monitors=spikelist, inputs = inputpartlist)

biotime = 500.
t = runner.run(biotime)

spikeunit = []
for spike_i in spikelist:
    spikeunit.append(runner.mon[spike_i])

spikeall = np.hstack(spikeunit)


col_layer = n_layer[:-1]
cumn_layer = np.cumsum(col_layer[::-1])

# fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

# fig.add_subplot(gs[:3, 0])
# plt.figure(figsize=(5,15))
bp.visualize.raster_plot(runner.mon.ts, spikeall[:,::-1], xlim = (100,biotime),ylim=(0, cumn_layer[-1]),markersize=0.05)


plt.yticks(cumn_layer,['L6i','L6e','L5i','L5e','L4i','L4e','L2/3i','L2/3e'])

plt.style.use("bmh")
plt.savefig("/home/liugangqiang/brainpy2/202206190100{}spike.png".format(input_type))

plt.cla()


rate = np.array([bm.sum(runner.mon[spikelist[i]])/float(n_layer[i])*1000./biotime for i in range(0,8)])

# rate_all = []
# for spike_i in spikeunit:
# 	rate_all.append(bp.measure.firing_rate(spike_i, 5.))
# rate_all_av = np.array([np.mean(rate_i[100:]) for rate_i in rate_all])*1000
# rate_eve_av = rate_all_av/np.array(n_layer[0:8])
# print(rate_eve_av)

name_list = ['L6i','L6e','L5i','L5e','L4i','L4e','L2/3i','L2/3e']
# fig.add_subplot(gs[3, 0])

plt.bar(name_list[::-1], rate)

plt.ylabel("rate",fontsize=14)
plt.xlabel("Time(ms)",fontsize=14)
plt.style.use("bmh")
plt.savefig("/home/liugangqiang/brainpy2/202206190100{}rate.png".format(input_type))