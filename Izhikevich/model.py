import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

bm.set_platform("cpu")

class Izhmodel(bp.dyn.NeuGroup):
    def __init__(self, 
                size=1, 
                capacitance = 100,
                peak = 30.,
                k = 3.,
                vr = -60.,
                vt = -50.,
                a = 0.01,
                b = 5,
                c = -60.,
                d = 400, 
                u_max = None, 
                method = 'exp_auto',           
                 **kwargs):
        super(Izhmodel, self).__init__(size=size, **kwargs)
        
        v_reset = 33
        u_reset = 1.
        #define parameter
        self.capacitance = capacitance
        self.peak = peak
        self.k = k
        self.vr = vr
        self.vt = vt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.u_max = u_max
        self.method = method        

        #initial dynamic Variable                
        self.V = bm.Variable(bm.zeros(self.num)+v_reset)
        # self.bm.random.random(self.num) * (pop[-1].V_th - pop[-1].V_rest) + pop[-1].V_rest
        self.u = bm.Variable(bm.ones(self.num)+u_reset)
        self.input = bm.Variable(bm.zeros(self.num))
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))        
        
        #define function
        self.integral = bp.odeint(f=bp.JointEq([self.dV, self.dU]),method = self.method)
        
    def dV(self, V, t, U, I_ext):
        dV = (self.k*(V-self.vr) * (V - self.vt) - U + I_ext)/self.capacitance 
        return dV
    
    def dU(self, U, t, V):
        dU = self.a * (self.b * (V-self.vr) - U)
        return dU 

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        v, u = self.integral(self.V, self.u, t, self.input, dt)
        
        #spike
        spike = self.peak <= v
        v = bm.where(spike, self.c, v)
        u = bm.where(spike, u + self.d, u) 
        if not self.u_max == None:        
            u = bm.where(self.u_max<=u, self.u_max, u)
        # finally
        self.V.value = v
        self.u.value = u
        self.spike.value = spike
        self.input[:] = 0.


# Basic Model to define the exponential synapse model. This class 
# defines the basic parameters, variables, and integral functions. 


class BaseExpSyn(bp.dyn.SynConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0., method='exp_auto'):
    super(BaseExpSyn, self).__init__(pre=pre, post=post, conn=conn)

    # check whether the pre group has the needed attribute: "spike"
    self.check_pre_attrs('spike')

    # check whether the post group has the needed attribute: "input" and "V"
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max

    # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
    self.delay_step = int(delay/bm.get_dt())
    self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step)

    # integral function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)
    
    
# Basic Model to define the AMPA synapse model. This class 
# defines the basic parameters, variables, and integral functions. 


class BaseAMPASyn(bp.dyn.SynConn):
  def __init__(self, pre, post, conn, delay=0., g_max=0.42, E=0., alpha=0.98,
               beta=0.18, T=0.5, T_duration=0.5, method='exp_auto'):
        super(BaseAMPASyn, self).__init__(pre=pre, post=post, conn=conn)

        # check whether the pre group has the needed attribute: "spike"
        self.check_pre_attrs('spike')

        # check whether the post group has the needed attribute: "input" and "V"
        self.check_post_attrs('input', 'V')

        # parameters
        self.delay = delay
        self.g_max = g_max
        self.E = E
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.T_duration = T_duration

        # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
        self.delay_step = int(delay/bm.get_dt())
        self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step)

        # store the arrival time of the pre-synaptic spikes
        self.spike_arrival_time = bm.Variable(bm.ones(self.pre.num) * -1e7)

        # integral function
        self.integral = bp.odeint(self.derivative, method=method)

  def derivative(self, g, t, TT):
        dg = self.alpha * TT * (1 - g) - self.beta * g
        return dg

# for more details of how to run a simulation please see the tutorials in "Dynamics Simulation"

class AMPAConnMat(BaseAMPASyn):
    def __init__(self, *args, **kwargs):
        super(AMPAConnMat, self).__init__(*args, **kwargs)

        # connection matrix
        self.conn_mat = self.conn.require('conn_mat')

        # synapse gating variable
        # -------
        # NOTE: Here the synapse shape is (num_pre, num_post),
        #       in contrast to the ExpConnMat
        self.g = bm.Variable(bm.zeros((self.pre.num, self.post.num)))

    def update(self, tdi, x=None):
        _t, _dt = tdi.t, tdi.dt
        # pull the delayed pre spikes for computation
        delayed_spike = self.pre_spike(self.delay_step)
        # push the latest pre spikes into the bottom
        self.pre_spike.update(self.pre.spike)
        # get the time of pre spikes arrive at the post synapse
        self.spike_arrival_time.value = bm.where(delayed_spike, _t, self.spike_arrival_time)
        # get the neurotransmitter concentration at the current time
        TT = ((_t - self.spike_arrival_time) < self.T_duration) * self.T
        # integrate the synapse state
        TT = TT.reshape((-1, 1)) * self.conn_mat  # NOTE: only keep the concentrations
                                                  #       on the invalid connections
        self.g.value = self.integral(self.g, _t, TT, dt=_dt)
        # get the post-synaptic current
        g_post = self.g.sum(axis=0)
        self.post.input += self.g_max * g_post * (self.E - self.post.V)
        
class AMPA(BaseExpSyn):
    
    def __init__(self, pre, post, conn, g_max=0.42, alpha = 0.98, delay=0., t_step = 0.01, tau=5.0, E=0., method='exp_auto'):
        super(BaseExpSyn, self).__init__(pre=pre, post=post, conn=conn)

        # check whether the pre group has the needed attribute: "spike"
        self.check_pre_attrs('spike')
        # check whether the post group has the needed attribute: "input" and "V"
        self.check_post_attrs('input', 'V')

        # parameters
        self.E = E
        self.tau = tau
        self.delay = delay
        self.g_max = g_max
        self.alpha = alpha
        self.conn_mat = self.conn.require('conn_mat')
        
        
        # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
        self.delay_step = int(delay/t_step)
        self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step)

        
        # integral function
        self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

        # variable
        self.g = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi, x=None):
        _t, _dt = tdi.t, tdi.dt
        # pull the delayed pre spikes for computation
        delayed_spike = self.pre_spike(self.delay_step)
        # push the latest pre spikes into the bottom
        self.pre_spike.update(self.pre.spike)
        # integrate the synapse state
        g_minus = self.g
        g = self.integral(self.g, _t, dt=_dt)
        self.g.value = g
        
        #integrate scalar factor
        # x = self.stp(self.x,_t,dt=_dt)
        # p = 0.7
        # self.x.value = (1.- (1.-p)*post_sps)*x
        # get the post-synaptic current
        # self.post.input += self.g_max * self.g*self.x* self.post.V
        # update synapse states according to the pre spikes
        post_sps = bm.dot(delayed_spike, self.conn_mat)
        self.g += self.alpha*post_sps*(1-g_minus)
        self.post.input += -self.g_max * self.g*self.post.V
        
class GABAa(BaseExpSyn):
    
    def __init__(self, pre, post, conn, g_max=0.04, delay=0., t_step = 0.01, tau=6.0, E=-80., method='exp_auto'):
        super(BaseExpSyn, self).__init__(pre=pre, post=post, conn=conn)

        # check whether the pre group has the needed attribute: "spike"
        self.check_pre_attrs('spike')
        # check whether the post group has the needed attribute: "input" and "V"
        self.check_post_attrs('input', 'V')

        # parameters
        self.E = E
        self.tau = tau
        self.delay = delay
        self.g_max = g_max
        self.conn_mat = self.conn.require('conn_mat')
        
        # use "LengthDelay" to store the spikes of the pre-synaptic neuron group
        self.delay_step = int(delay/t_step)
        self.pre_spike = bm.LengthDelay(pre.spike, self.delay_step)

        
        # integral function
        self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

        # variable
        self.g = bm.Variable(bm.zeros(self.post.num))

    def update(self, tdi, x=None):
        _t, _dt = tdi.t, tdi.dt
        # pull the delayed pre spikes for computation
        delayed_spike = self.pre_spike(self.delay_step)
        # push the latest pre spikes into the bottom
        self.pre_spike.update(self.pre.spike)
        # integrate the synapse state
        g_minus = self.g
        g = self.integral(self.g, _t, dt=_dt)
        self.g.value = g
        # update synapse states according to the pre spikes
        post_sps = bm.dot(delayed_spike, self.conn_mat)
        self.g += post_sps*(1-g_minus)
        self.post.input += -self.g_max * self.g*(self.post.V - self.E)       

# pre = Izhmodel()
# # post = bp.neurons.LIF(1, V_rest=-60., V_reset=-60., V_th=-40.)
# post = Izhmodel()
# syn = AMPA(pre, post, conn=bp.conn.One2One(),g_max= 100.)
# net = bp.dyn.Network(pre=pre, post=post,syn=syn)
# runner = bp.dyn.DSRunner(
#     net, 
#     monitors=['pre.V','post.V'],
#     dt = 0.01,
#     inputs = [('pre.input',10.),('post.input',10.)]
# )

# runner.run(200)

# bp.visualize.line_plot(runner.mon.ts, runner.mon['pre.V'], legend='pre.V')
# bp.visualize.line_plot(runner.mon.ts, runner.mon['post.V'], legend='post.V', show=True)
# plt.xlabel('t (ms)')
# plt.ylabel('V (mV)')

# plt.show()
# plt.savefig("test202209280001.png")
 

  
  
  