import brainpy as bp
import brainpy.math as bm
import matplotlib
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from model import * 
from netParams import *
import utils
# from matplotlib import font_manager

# font_path = '/home/liugangqiang/fonts/SimHei.ttf' # ttf的路径 最好是具体路径
# font_manager.fontManager.addfont(font_path)

bm.set_platform("cpu")

def main(w_ex, epoch):
    
    #定义各层神经元
    pops = []
    for i,cortex_soma_par in enumerate(cortex_soma_pars):
        pars = neuron_pars_soma[neuron_types.index(cortex_soma_par["neuron_type"])]
        pops.append(Izhmodel(size = cortex_part_number[i], **pars))
        pops[-1].V[:] = pops[-1].c

    #计算突触延迟时间
    av_delay = np.array([d_ex,d_in,d_ex,d_in,d_ex,d_in,d_ex,d_in])
    std_delay = np.array([std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in,std_d_ex,std_d_in])
    # delay = [av_delay[i] + std_delay[i]*np.random.normal() for i in range(8)]
    # print(delay)

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
                synapses.append(bp.dyn.synapses.AMPA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))
                # synapses.append(AMPA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))
                synapses.append(bp.dyn.synapses.NMDA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))            
                # synapses.append(NMDA(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))            
        if pre_type in neuron_inhibitory:
            print("inhibitory")
            for post_index in range(len(pops)):
                post_pop = pops[post_index]
                # synapses.append(bp.dyn.synapses.GABAa(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=-8.*(w_ex+w_ex_std*np.random.normal())))
                synapses.append(bp.dyn.synapses.GABAa(pre_pop, post_pop, bp.conn.FixedProb(connection_matrix[pre_index,post_index]), g_max=(w_ex+w_ex_std*np.random.normal())))

    # net = bp.dyn.Network(*pops)
    net = bp.dyn.Network(*pops,*synapses)
    len_pops = len(pops)
    inputlist = [pop.name+".input" for pop in pops]
    bg_in = []
    inputpartlist = [(inputlist[i], 150.) for i in range(len_pops)]
    V_list = [pops_i.name+".V" for pops_i in pops]
    spike_list = [pops_i.name+".spike" for pops_i in pops]
    runner = bp.dyn.DSRunner(net, monitors=V_list+spike_list,dt = 0.1, inputs = inputpartlist)

    #running
    biotime = 1000.
    runner.run(biotime)    
    plt.clf()
    V_all = []
    for V_i in V_list:
        V_all.append(runner.mon[V_i])

    spikeunit = []
    for spike_i in spike_list:
        spike_i_array = runner.mon[spike_i]
        # print(spike_i_array.shape)
        # spike_i_arry = utils.sample_matrix_axon2(spike_i_array,1)
        spikeunit.append(spike_i_array)

    #visualization
    # dir_path = "/home/liugangqiang/columnmodel/Izhikevich/figure/"
    dir_path = "/home/liugangqiang/figure/column_model/"
    run_time = "04"  
    
    utils.check_paths(dir_path)
    date = utils.get_datetime()
    dir_run = dir_path + date +"/" + run_time+ "/"
    # print(dir_run)
    utils.check_paths(dir_run)    
    
    # 将全局的字体设置为黑体
    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    
    #plot spike dot
    plt.style.use("bmh")
    # plt.rc('font',family=['SimHei'])
    # fig = plt.figure()
    # gs = GridSpec(3,4)
    # fig.add_subplot(gs[0:17,0:2])
    
    spikeall = np.hstack(spikeunit)
    col_layer = cortex_part_number
    cumcortex_part_number = np.cumsum(col_layer[::-1])
    (spike_point_x, spike_point_y) = utils.spike_to_coordinate(runner.mon.ts,spikeall)
    
    color_list = ['#7e1e9c','#15b01a','#0343df','#ff81c0','#653700','#e50000','#95d0fc','#029386','#f97306','#96f97b','#c20078','#ffff14','#75bbfd','#929591','#89fe05','#bf77f6','#9a0eea','#033500','#06c2ac','#c79fef','#00035b','#d1b26f','#13eac9','#06470c','#ae7181']
    
    #classify spike point
    left_number = np.array([0] + list(cumcortex_part_number[:-1]))
    right_number = cumcortex_part_number
    spike_point_list = []
    plt.figure(figsize=(5, 20))
    for i in range(len(left_number)):
        left = left_number[i]
        right = right_number[i]
        print(left," ",right)
        spike_point_in = [(spike_point_y_i<right and spike_point_y_i>=left) for spike_point_y_i in spike_point_y] 
        spike_point_y_in = np.asarray([spike_point_y[i] for i in range(len(spike_point_y)) if spike_point_in[i]])
        spike_point_x_in = np.asarray([spike_point_x[i] for i in range(len(spike_point_x)) if spike_point_in[i]]) 
        plt.plot(spike_point_x_in, spike_point_y_in,'o',markersize=1,color = color_list[i]) 
    
    # plt.plot(spike_point_x, spike_point_y,'o',markersize=1)
    # plt.scatter(spike_point_x, spike_point_y)
    plt.xlim(0,biotime)
    plt.ylim(0, cumcortex_part_number[-1])
    label_place = ( left_number + right_number)/2.
    # plt.yticks(cumcortex_part_number,cortex_name_list[::-1],fontsize=14)
    plt.yticks(label_place,cortex_name_list[::-1],fontsize=14)
    plt.xlabel("Time(ms)")
    # plt.ylabel("Neuron name")
    plt.savefig(dir_run+"00{}spike.png".format(epoch)) 
    plt.clf()
    # bp.visualize.raster_plot(runner.mon.ts, spikeall[:,::-1], xlim = (0,biotime),ylim=(0, cumcortex_part_number[-1]),markersize=2)
    # plt.axis('off')
    # plt.plot(runner.mon.ts, spikeall[:,::-1],  markersize=1)
    # fig.add_subplot(gs[0:17,2:4])
    
    rate = np.array([bm.sum(runner.mon[spike_list[i]])/float(cortex_part_number[i])*1000./biotime for i in range(len_pops)])
    plt.figure(figsize=(4,3))
    plt.barh(cortex_name_list[::-1], rate[::-1])
    plt.xlabel("rate")
    plt.savefig(dir_run+"00{}rate.png".format(epoch))    
    plt.clf()
    # plt.xlabel("rate",fontsize=14)
    # plt.xticks([])
    # plt.axis('off')
    # plt.ylabel("name",fontsize=14)
    
    #plot 
    # rate = np.array([bm.sum(runner.mon[spike_list[i]])/float(cortex_part_number[i])*1000./biotime for i in range(len_pops)])
    # plt.barh(cortex_name_list[::-1], rate[::-1])
    # plt.xlabel("rate",fontsize=14)
    # plt.ylabel("Time(ms)",fontsize=14)
    # plt.style.use("bmh")
    # plt.savefig(dir_run+"00{}rate.png".format(epoch))    
    # plt.clf()

    fig_ncol = 6;fig_nrow = 6;pix_width = 3;pix_heigth = 4
    fig, gs = bp.visualize.get_figure(fig_ncol, fig_nrow, pix_width, pix_heigth)
    for fig_i in range(len_pops):
    # for fig_i in range(12):    
        a = int(fig_i/fig_nrow)
        b = int(fig_i - a*fig_nrow)
        fig.add_subplot(gs[a, b])
        bp.visualize.line_plot(runner.mon.ts, V_all[fig_i], xlim = (0,biotime),ylim=(-80,60),legend=cortex_soma_pars[fig_i]["name"])
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        # plt.xlabel('Time (ms)',fontsize=14)
        # plt.ylabel("mV",fontsize=14)
    plt.style.use("bmh")
    plt.savefig(dir_run+"00{}v.png".format(epoch))
    plt.clf()

w_ex = 87.8
# epoch = 1

main(w_ex, 0)
# main(w_ex,epoch)
