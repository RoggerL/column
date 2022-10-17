import numpy as np

n_layer = [20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948, 902]
# Total cortical Population
N = np.sum(n_layer[:-1])

# Number of neurons accumulated
nn_cum = [0]
nn_cum.extend(np.cumsum(n_layer))

# Prob. connection table
table = np.array(
              [[0.101,  0.169, 0.044, 0.082, 0.032, 0.,     0.008, 0.,     0.    ],
               [0.135,  0.137, 0.032, 0.052, 0.075, 0.,     0.004, 0.,     0.    ],
               [0.008,  0.006, 0.050, 0.135, 0.007, 0.0003, 0.045, 0.,     0.0983],
               [0.069,  0.003, 0.079, 0.160, 0.003, 0.,     0.106, 0.,     0.0619],
               [0.100,  0.062, 0.051, 0.006, 0.083, 0.373,  0.020, 0.,     0.    ],
               [0.055,  0.027, 0.026, 0.002, 0.060, 0.316,  0.009, 0.,     0.    ],
               [0.016,  0.007, 0.021, 0.017, 0.057, 0.020,  0.040, 0.225,  0.0512],
               [0.036,  0.001, 0.003, 0.001, 0.028, 0.008,  0.066, 0.144,  0.0196]])

# Synapses parameters

d_ex = 1.5      	# Excitatory delay
std_d_ex = 0.75 	# Std. Excitatory delay
d_in = 0.80      # Inhibitory delay
std_d_in = 0.4  	# Std. Inhibitory delay
tau_syn = 0.5    # Post-synaptic current time constant

# Layer-specific background input
bg_layer_specific = np.array([1600, 1500 ,2100, 1900, 2000, 1900, 2900, 2100])

# Layer-independent background input
bg_layer_independent = np.array([2000, 1850 ,2000, 1850, 2000, 1850, 2000, 1850])

#神经元参数列表
neuron_C =  [20,100,20,100,100,100,100,200,20,40]
neuron_k =  [0.3,3,1,1,3,3,3,1.6,0.5,0.25]
neuron_vr = [-66,-60,-55,-56,-60,-60,-60,-60,-60,-65]
neuron_vt = [-40,-50,-40,-42,-50,-50,-50,-50,-50,-45]
neuron_v_peak_soma = [30,50,25,40,50,50,50,40,20,0.0]
neuron_v_peak_dendr = [100,30,25,40,30,50,30,40,20,0.0]
neuron_g_up =  [0.6,3.0,0.5,1.0,3.0,3.0,3.0,2.0,5.0,5.0]
neuron_g_down = [2.5,5.0,1.0,1.0,5.0,5.0,5.0,2.0,5.0,5.0]
neuron_a = [0.17,0.01,0.15,0.03,0.01,0.01,0.01,0.1,0.05,0.015]
neuron_b =  [5,5,8,8,5,5,5,15,7,10]
neuron_c_soma = [-45,-60,-55,-50,-60,-60,-60,-60,-65,-55]
neuron_c_dendr = [-45,-55,-55,-50,-50,-50,-50,-60,-65,-55]
neuron_d = [100,400,200,20,400,400,400,10,50,50]
neuron_u_max = [None, None, None, 670., None, None, None, None, 530., None, ]

neuron_pars_soma = [dict(capacitance = float(neuron_C[i]), peak = float(neuron_v_peak_soma[i]),k = float(neuron_k[i]),vr = float(neuron_vr[i]),vt = float(neuron_vt[i]),a = float(neuron_a[i]),b = float(neuron_b[i]),c = float(neuron_c_soma[i]),d = float(neuron_d[i]),u_max = neuron_u_max[i]) for i in range(10)]
neuron_pars_dendr = [dict(capacitance = float(neuron_C[i]), peak = float(neuron_v_peak_dendr[i]),k = float(neuron_k[i]),vr = float(neuron_vr[i]),vt = float(neuron_vt[i]),a = float(neuron_a[i]),b = float(neuron_b[i]),c = float(neuron_c_dendr[i]),d = float(neuron_d[i]),u_max = neuron_u_max[i]) for i in range(10)]

neuron_types = ["nb1(LS)","p23(RS)","b(FS)","nb(LTS)","ss4(RS)","p4(RS)","p5p6(RS)","TC","TI","TRN"]
neuron_excited = ["nb1(LS)","b(FS)","nb(LTS)","TC"]
neuron_inhibitory = ["p23(RS)","ss4(RS)","p4(RS)","p5p6(RS)","TI","TRN"]
###############################################################################
# Network parameters
###############################################################################
# Population per layer
#          2/3e   2/3i   4e    4i    5e    5i    6e     6i    Th
layers_pars = [
    [dict(name = "nb1",neuron_type = "nb1(LS)",type = 0)],
    [dict(name = "p2/3",neuron_type = "p23(RS)",type = 0),
     dict(name = "p2/3 L1",neuron_type = "p23(RS)",type = 1)],
    [dict(name = "b2/3",neuron_type = "b(FS)" ,type = 0)],
    [dict(name = "nb2/3",neuron_type = "nb(LTS)",type = 0)],
    [dict(name = "ss4(L4)",neuron_type = "ss4(RS)",type = 0)],
    [dict(name = "ss4(L2/3)",neuron_type = "ss4(RS)",type = 0)],
    [dict(name = "p4",neuron_type = "p4(RS)",type = 0),
     dict(name = "p4 L2/3",neuron_type = "p4(RS)",type = 1),
     dict(name = "p4 L1",neuron_type = "p4(RS)",type = 2)], 
    [dict(name = "b4",neuron_type = "b(FS)",type = 0)],
    [dict(name = "nb4",neuron_type = "nb(LTS)",type = 0)],
    [dict(name = "p5(L2/3)",neuron_type = "p5p6(RS)",type = 0),
     dict(name = "p5(L2/3) L4",neuron_type = "p5p6(RS)",type = 1),
     dict(name = "p5(L2/3) L2/3",neuron_type = "p5p6(RS)",type = 2), 
     dict(name = "p5(L2/3) L1",neuron_type = "p5p6(RS)",type = 3)],
    [dict(name = "p5(L5/6)",neuron_type = "p5p6(RS)",type = 0),
     dict(name = "p5(L5/6) L4",neuron_type = "p5p6(RS)",type = 1),
     dict(name = "p5(L5/6) L2/3",neuron_type = "p5p6(RS)",type = 2), 
     dict(name = "p5(L5/6) L1",neuron_type = "p5p6(RS)",type = 3)],    
    [dict(name = "b5",neuron_type = "b(FS)",type = 0)], 
    [dict(name = "nb5",neuron_type = "nb(LTS)",type = 0)], 
    [dict(name = "p6(L4)",neuron_type = "p5p6(RS)",type = 0),
     dict(name = "p6(L4) L5",neuron_type = "p5p6(RS)",type = 1),
     dict(name = "p6(L4) L4",neuron_type = "p5p6(RS)",type = 2),
     dict(name = "p6(L4) L2/3",neuron_type = "p5p6(RS)",type = 3)],
    [dict(name = "p6(L5/6)",neuron_type = "p5p6(RS)",type = 0),
     dict(name = "p6(L5/6) L5",neuron_type = "p5p6(RS)",type = 1), 
     dict(name = "p6(L5/6) L4",neuron_type = "p5p6(RS)",type = 2),
     dict(name = "p6(L5/6) L2/3",neuron_type = "p5p6(RS)",type = 3),
     dict(name = "p6(L5/6) L1",neuron_type = "p5p6(RS)",type = 4)],
    [dict(name = "b6",neuron_type = "p5p6(RS)",type = 0)],
    [dict(name = "nb6",neuron_type = "nb(LTS)",type = 0)], 
    [dict(name = "TCs",neuron_type = "TC",type = 0)],     
    [dict(name = "TCn",neuron_type = "TC",type = 0)],   
    [dict(name = "TIs",neuron_type = "TI",type = 0)],   
    [dict(name = "TIn",neuron_type = "TI",type = 0)],   
    [dict(name = "TRN",neuron_type = "TRN",type = 0)] ]

pops_pars = sum(layers_pars,[])

matrix=[
[[15000 , 8890 , 10.1, 6.3 , 0.6 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 77.6, 0   , 4.1 , 0   , 0   , 0   ]],
[[260000, 5800 , 0   , 59.9, 9.1 , 4.4 , 0.6 , 6.9 , 7.7 , 0   , 0.8 , 7.4 , 0   , 0   , 0   , 2.3 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ],
 [260000, 1306 , 10.2, 6.3 , 0.1 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 78  , 0   , 4.1 , 0   , 0   , 0   ]],
[[31000 , 3854 , 1.3 , 51.6, 10.6, 3.4 , 0.5 , 5.8 , 6.6 , 0   , 0.8 , 6.3 , 0   , 0   , 0   , 2.1 , 0   , 0   , 0.7 , 9.8 , 0   , 0.5 , 0   , 0   , 0   ]],
[[42000 , 3307 , 1.7 , 48.6, 11.4, 3.3 , 0.5 , 5.5 , 6.2 , 0   , 0.8 , 5.9 , 0   , 0   , 0   , 1.8 , 0   , 0   , 0.6 , 13  , 0   , 0.7 , 0   , 0   , 0   ]],
[[92000 , 5792 , 0   , 2.7 , 0.2 , 0.6 , 11.9, 3.7 , 4.1 , 7.1 , 2   , 0.8 , 0.1 , 0   , 0   , 32.7, 0   , 0   , 5.8 , 25.3, 1.7 , 1.3 , 0   , 0   , 0   ]],
[[92000 , 4989 , 0   , 5.6 , 0.4 , 0.8 , 11.3, 3.8 , 4.3 , 7.2 , 2.1 , 1.1 , 0.1 , 0   , 0   , 31.1, 0   , 0   , 5.5 , 23.9, 1.7 , 1.3 , 0   , 0   , 0   ]],
[[92000 , 5031 , 0   , 4.3 , 0.2 , 0.6 , 11.5, 3.6 , 4.2 , 7.2 , 2.1 , 1.2 , 0.1 , 0   , 0   , 31.4, 0.1 , 0   , 5.9 , 24.5, 1.7 , 1.3 , 0   , 0   , 0   ],
 [92000 , 866  , 0   , 63.1, 5.1 , 4.1 , 0.6 , 7.2 , 8.1 , 0   , 0.6 , 7.8 , 0   , 0   , 0   , 2.5 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ],
 [92000 , 806  , 10.2, 6.3 , 0.1 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 78  , 0   , 4.1 , 0   , 0   , 0   ]],
[[54000 , 3230 , 0   , 5.8 , 0.5 , 0.8 , 11  , 3.8 , 4.2 , 8.4 , 2.4 , 1.1 , 0   , 0   , 0   , 30.3, 0   , 0   , 5.4 , 23.3, 1.6 , 1.2 , 0   , 0   , 0   ]],
[[15000 , 3688 , 0   , 2.7 , 0.2 , 0.6 , 11.7, 3.6 , 4   , 8.2 , 2.3 , 0.8 , 0.1 , 0   , 0   , 32.2, 0   , 0   , 5.7 , 24.9, 1.7 , 1.3 , 0   , 0   , 0   ]],
[[48000 , 4316 , 0   , 45.9, 1.8 , 0.3 , 3.3 , 2   , 7.5 , 0   , 0.9 , 11.7, 1   , 0.8 , 1.1 , 2.3 , 2.1 , 0   , 11.5, 7.2 , 0.1 , 0.4 , 0   , 0   , 0   ],
 [48000 , 283  , 0   , 2.8 , 0.1 , 0.7 , 12.2, 3.8 , 4.2 , 5.2 , 1.5 , 0.8 , 0.1 , 0   , 0   , 33.7, 0   , 0   , 5.9 , 26  , 1.8 , 1.4 , 0   , 0   , 0   ],
 [48000 , 412  , 0   , 63.1, 5.1 , 4.1 , 0.6 , 7.2 , 8.1 , 0   , 0.6 , 7.8 , 0   , 0   , 0   , 2.5 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ],
 [48000 , 185  , 10.2, 6.3 , 0.1 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 78  , 0   , 4.1 , 0   , 0   , 0   ]],
[[13000 , 5101 , 0   , 44.3, 1.7 , 0.2 , 3.2 , 2   , 7.3 , 0   , 0.8 , 11.3, 1.2 , 0.8 , 1.1 , 2.3 , 2.5 , 0.3 , 11.3, 9.2 , 0.2 , 0.5 , 0   , 0   , 0   ],
 [13000 , 949  , 0   , 2.8 , 0.1 , 0.7 , 12.2, 3.8 , 4.2 , 5.2 , 1.5 , 0.8 , 0.1 , 0   , 0   , 33.7, 0   , 0   , 5.9 , 26  , 1.8 , 1.4 , 0   , 0   , 0   ],
 [13000 , 1367 , 0   , 63.1, 5.1 , 4.1 , 0.6 , 7.2 , 8.1 , 0   , 0.6 , 7.8 , 0   , 0   , 0   , 2.5 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ],
 [13000 , 5658 , 10.2, 6.3 , 0.1 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 78  , 0   , 4.1 , 0   , 0   , 0   ]],
[[6000  , 2981 , 0   , 45.5, 2.3 , 0.2 , 3.3 , 2   , 7.5 , 0   , 1.1 , 11.6, 1   , 0.9 , 1.3 , 2.3 , 2   , 0   , 11.4, 7.2 , 0.1 , 0.4 , 0   , 0   , 0   ]],
[[8000  , 2981 , 0   , 45.5, 2.3 , 0.2 , 3.3 , 2   , 7.5 , 0   , 1.1 , 11.6, 1   , 0.9 , 1.3 , 2.3 , 2   , 0   , 11.4, 7.2 , 0.1 , 0.4 , 0   , 0   , 0   ]],
[[136000, 3261 , 0   , 2.5 , 0.1 , 0.1 , 0.7 , 0.9 , 1.3 , 0   , 0.1 , 0.1 , 4.9 , 0   , 0.3 , 1.2 , 13.2, 7.7 , 7.7 , 55.7, 0.6 , 2.9 , 0   , 0   , 0   ],
 [136000, 1066 , 0   , 46.8, 0.8 , 0.3 , 3.4 , 2.1 , 7.7 , 0   , 0.6 , 11.9, 1   , 0.6 , 0.8 , 2.3 , 2.1 , 0   , 11.7, 7.4 , 0.1 , 0.4 , 0   , 0   , 0   ],
 [136000, 1915 , 0   , 2.8 , 0.1 , 0.7 , 12.2, 3.8 , 4.2 , 5.2 , 1.5 , 0.8 , 0.1 , 0   , 0   , 33.7, 0   , 0   , 5.9 , 26  , 1.8 , 1.4 , 0   , 0   , 0   ],
 [136000, 121  , 0   , 63.1, 5.1 , 4.1 , 0.6 , 7.2 , 8.1 , 0   , 0.6 , 7.8 , 0   , 0   , 0   , 2.5 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ]],
[[45000 , 5573 , 0   , 2.5 , 0.1 , 0.1 , 0.7 , 0.9 , 1.3 , 0   , 0.1 , 0.1 , 4.9 , 0   , 0.3 , 1.2 , 13.2, 7.8 , 7.8 , 55.7, 0.6 , 2.9 , 0   , 0   , 0   ],
 [45000 , 257  , 0   , 46.8, 0.8 , 0.3 , 3.4 , 2.1 , 7.7 , 0   , 0.6 , 11.9, 1   , 0.6 , 0.8 , 2.3 , 2.1 , 0   , 11.7, 7.4 , 0.1 , 0.4 , 0   , 0   , 0   ],
 [45000 , 243  , 0   , 2.8 , 0.1 , 0.7 , 12.2, 3.8 , 4.2 , 5.2 , 1.5 , 0.8 , 0.1 , 0   , 0   , 33.7, 0   , 0   , 5.9 , 26  , 1.8 , 1.4 , 0   , 0   , 0   ],
 [45000 , 286  , 0   , 63.1, 5.1 , 4.1 , 0.6 , 7.2 , 8.1 , 0   , 0.6 , 7.8 , 0   , 0   , 0   , 2.5 , 0   , 0   , 0.8 , 0   , 0   , 0   , 0   , 0   , 0   ],
 [45000 , 62   , 10.2, 6.3 , 0.1 , 1.1 , 0   , 0   , 0.1 , 0   , 0   , 0.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 78  , 0   , 4.1 , 0   , 0   , 0   ]],
[[20000 , 3220 , 0   , 2.5 , 0.1 , 0.1 , 0.7 , 0.9 , 1.3 , 0   , 0.1 , 0.1 , 4.9 , 0   , 0.4 , 1.2 , 13.2, 7.7 , 7.7 , 55.7, 0.6 , 2.9 , 0   , 0   , 0   ]],
[[20000 , 3220 , 0   , 2.5 , 0.1 , 0.1 , 0.7 , 0.9 , 1.3 , 0   , 0.1 , 0.1 , 4.9 , 0   , 0.4 , 1.2 , 13.2, 7.7 , 7.7 , 55.7, 0.6 , 2.9 , 0   , 0   , 0   ]],
[[5000  , 4000 , 31  , 0   , 7.1 , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 23  , 8   , 0   , 0   , 0   , 0   , 0   , 5   , 0   , 25.9]],
[[5000  , 4000 , 31  , 0   , 7.1 , 0   , 0   , 0   , 0   , 0   , 0   , 14  , 3.8 , 0   , 0   , 0   , 13.2, 0   , 0   , 0   , 0   , 0   , 0   , 5   , 25.9]],
[[1000  , 3000 , 13.5, 0   , 48.7, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 9.8 , 3.3 , 0   , 0   , 0   , 0.4 , 0   , 24.4, 0   , 0   ]],
[[1000  , 3000 , 13.4, 0   , 48.7, 0   , 0   , 0   , 0   , 0   , 0   , 5.8 , 1.6 , 0   , 0   , 0   , 5.4 , 0   , 0   , 0   , 0   , 0.6 , 0   , 24.4, 0   ]],
[[5000  , 4000 , 40  , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 30  , 0   , 0   , 0   , 0   , 10  , 10  , 0   , 0   , 10  ]]]

neuron_number = [mat[0][0] for mat in matrix]
cortex_number = neuron_number[:17]
cortex_soma_pars = [pars[0] for pars in layers_pars[:17]]
# print(cortex_soma_pars)

connection_list = []
for mat in matrix[:17]:
    a = np.matrix(mat)[:,1].T
    b = np.matrix(mat)[:,2:]
    c = np.dot(a,b)
    d = c[:,:17]/np.array(neuron_number[:17])
    connection_list.append(d)
connection_percent = np.vstack(connection_list)
connection_matrix = connection_percent*0.01
(x,y) = connection_matrix.shape

neuron_part_number = [int(nn*0.001) for nn in neuron_number]
cortex_part_number = neuron_part_number[:17]
name_list = [cortex_soma_par["name"] for cortex_soma_par in cortex_soma_pars]
cortex_name_list = name_list[:17]
# print(len(cortex_number))
# for i in range(x):
#     for j in range(y):
#         if connection_matrix[i,j]>1:
#             print("wrong")

# print(connection_matrix)

# print([i*0.0001 for i in cortex_number])

