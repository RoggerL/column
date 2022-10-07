import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
from model import * 
from netParams import *
from multiprocessing import Process

bm.set_platform("cpu")


def main(w_ex, epoch):
    #定义各层神经元
    pops = []
    for i,cortex_soma_par in enumerate(cortex_soma_pars):
        pars = neuron_pars_soma[neuron_types.index(cortex_soma_par["neuron_type"])]
        pops.append(Izhmodel(size = neuron_part_number[i], **pars))
        pops[-1].V[:] = pops[-1].c

    #计算突触延迟时间
    av_delay = np.array([d_ex,d_in,d_ex,d_in,d_ex,d_in,d_ex,d_in])
    std_delay = np.array([std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in])
    delay = [av_delay[i] + std_delay[i]*np.random.normal() for i in range(8)]
    print(delay)

    # w_ex=87.8


    #定义突触连接
    w_ex_std = 0.1*w_ex
    synapses = []
    for pre_index in range(len(pops)):
        pre_type = cortex_soma_pars[pre_index]["neuron_type"]
        pre_pop = pops[pre_index]

        if pre_type in neuron_excited:
            print("excited")
            for post_index in range(len(pops)):
                post_pop = pops[post_index]
                synapses.append(bp.dyn.synapses.AMPA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))``
                # synapses.append(AMPA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))
                synapses.append(bp.dyn.synapses.NMDA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))            
                # synapses.append(NMDA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))            
        if pre_type in neuron_inhibitory:
            print("inhibitory")
            for post_index in range(len(pops)):
                post_pop = pops[post_index]
                # synapses.append(bp.dyn.synapses.GABAa(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=-8.*(w_ex+w_ex_std*np.random.normal())))
                synapses.append(bp.dyn.synapses.GABAa(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))

        pass


    # net = bp.dyn.Network(*pops)
    net = bp.dyn.Network(*pops,*synapses)
    len_pops = len(pops)
    # V_list = [pops_i.name+".V" for pops_i in pops]
    inputlist = [pop.name+".input" for pop in pops]
    bg_in = []
    inputpartlist = [(inputlist[i], 150.) for i in range(len_pops)]
    V_list = [pops_i.name+".V" for pops_i in pops]
    spike_list = [pops_i.name+".spike" for pops_i in pops]
    runner = bp.dyn.DSRunner(net, monitors=V_list+spike_list,dt = 0.1, inputs = inputpartlist)

    #running
    biotime = 1000.
    t = runner.run(biotime)

    dir_path = "/home/liugangqiang/Izhikevich/figure/"
    date = "20221002"
    run_time = "04"
    
    
    V_all = []
    for V_i in V_list:
        V_all.append(runner.mon[V_i])

    #visualization
    fig_ncol = 6;fig_nrow = 6;pix_width = 3;pix_heigth = 4
    fig, gs = bp.visualize.get_figure(fig_ncol, fig_nrow, pix_width, pix_heigth)
    for fig_i in range(len_pops):
    # for fig_i in range(12):    
        a = int(fig_i/fig_nrow)
        b = int(fig_i - a*fig_nrow)
        fig.add_subplot(gs[a, b])
        bp.visualize.line_plot(runner.mon.ts, V_all[fig_i], legend=cortex_soma_pars[fig_i]["name"])
    # plt.style.use("bmh")
    plt.savefig(dir_path+date+run_time+"00{}V.png".format(epoch))
    plt.cla()
    
    
    
    
    spikeunit = []
    for spike_i in spike_list:
        spikeunit.append(runner.mon[spike_i])

    spikeall = np.hstack(spikeunit)


    col_layer = neuron_part_number[:-1]
    cumneuron_part_number = np.cumsum(col_layer[::-1])

    # fig, gs = bp.visualize.get_figure(1, 1, 3, 4)

    # fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, spikeall[:,::-1], xlim = (100,biotime),ylim=(0, cumneuron_part_number[-1]),markersize=0.1)


    plt.yticks(cumneuron_part_number,cortex_name_list[::-1])

    plt.style.use("bmh")
    plt.savefig(dir_path+date+run_time+"00{}spike.png".format(epoch))

    plt.cla()


    rate = np.array([bm.sum(runner.mon[spike_list[i]])/float(neuron_part_number[i])*1000./biotime for i in range(len_pops)])

    # rate_all = []
    # for spike_i in spikeunit:
    # 	rate_all.append(bp.measure.firing_rate(spike_i, 5.))
    # rate_all_av = np.array([np.mean(rate_i[100:]) for rate_i in rate_all])*1000
    # rate_eve_av = rate_all_av/np.array(neuron_part_number[0:8])
    # print(rate_eve_av)


    # fig.add_subplot(gs[3, 0])

    plt.bar(cortex_name_list, rate)

    plt.ylabel("rate",fontsize=14)
    plt.xlabel("Time(ms)",fontsize=14)
    plt.style.use("bmh")
    plt.savefig(dir_path+date+run_time+"00{}rate.png".format(epoch))    
    

w_ex = 87.8
# epoch = 1

# main(w_ex,epoch)
 
for i in range(10):
    p = Process(target=main, args=((i+5)*0.1*w_ex,i))  # target传入目标函数，args传入目标函数所需参数
    p.start()  # 启动进程