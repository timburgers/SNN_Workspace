import yaml

with open("configs/config_LIF_DEFAULT.yaml","r") as f:
    config = yaml.safe_load(f)

ind = 4
for l1_neur in [40]:
    for rec in [True,False]:
        # for w_2x2_cross_shared in [True,False]:
        for adapt in [True, False]:
            config["NEURONS"] = l1_neur
            config["LAYER_SETTING"]["l0"]["neurons"] = l1_neur
            config["LAYER_SETTING"]["l1"]["recurrent"] = rec
            # config["LAYER_SETTING"]["l1"]["shared_2x2_weight_cross"] = w_2x2_cross_shared             # for P,D,PD,PID
            # config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = w_2x2_cross_shared               #For I

            config["LAYER_SETTING"]["l1"]["adaptive"] = adapt



            file=open("configs/config_LIF_"+ str(ind) +".yaml","w")
            yaml.dump(config,file)
            file.close()
            ind = ind +1



# for l1_neur in [40]:
#     for w_diag in [True]:
#         for w_diag_2x2 in [False]:
#             for rec in [True,False]:
#                 for shared_w in [False]:
#                     for shared_i in [True,False]:
#                         for adapt in [True,False]:
#                             if adapt == True:
#                                 for adapt_input in [True,False]:
#                                     for adapt_2x2 in [True,False]:
#                                         if adapt_2x2 == True:
#                                             for adapt_shared in [True,False]:
#                                                 config["NEURONS"] = l1_neur
#                                                 config["LAYER_SETTING"]["l1"]["recurrent"] = rec
#                                                 config["LAYER_SETTING"]["l1"]["adaptive"] = adapt
                                                

#                                                 config["LAYER_SETTING"]["l1"]["adapt_2x2_connection"] = adapt_2x2
#                                                 config["LAYER_SETTING"]["l1"]["adapt_thres_input_spikes"] = adapt_input
#                                                 config["LAYER_SETTING"]["l1"]["adapt_share_add_t"] = adapt_shared
                                                
#                                                 config["LAYER_SETTING"]["l0"]["shared_weight_and_bias"] = shared_w
#                                                 config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = shared_w
#                                                 config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = shared_w

#                                                 config["LAYER_SETTING"]["l0"]["shared_leak_i"] = shared_i
#                                                 config["LAYER_SETTING"]["l1"]["shared_leak_i"] = shared_i

#                                                 config["LAYER_SETTING"]["l1"]["w_diagonal"] = w_diag
#                                                 config["LAYER_SETTING"]["l1"]["w_diagonal_2x2"] = w_diag_2x2
#                                                 if w_diag == False:
#                                                     config["LAYER_SETTING"]["l0"]["neurons"] = l1_neur

#                                                 file=open("configs/config_LIF_"+ str(ind) +".yaml","w")
#                                                 yaml.dump(config,file)
#                                                 file.close()
#                                                 ind = ind +1
#                                         else:
#                                             config["NEURONS"] = l1_neur
#                                             config["LAYER_SETTING"]["l1"]["recurrent"] = rec
#                                             config["LAYER_SETTING"]["l1"]["adaptive"] = adapt
                                            

#                                             config["LAYER_SETTING"]["l1"]["adapt_2x2_connection"] = adapt_2x2
#                                             config["LAYER_SETTING"]["l1"]["adapt_thres_input_spikes"] = adapt_input
#                                             config["LAYER_SETTING"]["l1"]["adapt_share_add_t"] = adapt_shared
                                            
#                                             config["LAYER_SETTING"]["l0"]["shared_weight_and_bias"] = shared_w
#                                             config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = shared_w
#                                             config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = shared_w

#                                             config["LAYER_SETTING"]["l0"]["shared_leak_i"] = shared_i
#                                             config["LAYER_SETTING"]["l1"]["shared_leak_i"] = shared_i

#                                             config["LAYER_SETTING"]["l1"]["w_diagonal"] = w_diag
#                                             config["LAYER_SETTING"]["l1"]["w_diagonal_2x2"] = w_diag_2x2
#                                             if w_diag == False:
#                                                 config["LAYER_SETTING"]["l0"]["neurons"] = l1_neur

#                                             file=open("configs/config_LIF_"+ str(ind) +".yaml","w")
#                                             yaml.dump(config,file)
#                                             file.close()
#                                             ind = ind +1
#                             else:
#                                 config["NEURONS"] = l1_neur
#                                 config["LAYER_SETTING"]["l1"]["recurrent"] = rec
#                                 config["LAYER_SETTING"]["l1"]["adaptive"] = adapt
                                

#                                 config["LAYER_SETTING"]["l1"]["adapt_2x2_connection"] = adapt_2x2
#                                 config["LAYER_SETTING"]["l1"]["adapt_thres_input_spikes"] = adapt_input
#                                 config["LAYER_SETTING"]["l1"]["adapt_share_add_t"] = adapt_shared
                                
#                                 config["LAYER_SETTING"]["l0"]["shared_weight_and_bias"] = shared_w
#                                 config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = shared_w
#                                 config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = shared_w

#                                 config["LAYER_SETTING"]["l0"]["shared_leak_i"] = shared_i
#                                 config["LAYER_SETTING"]["l1"]["shared_leak_i"] = shared_i

#                                 config["LAYER_SETTING"]["l1"]["w_diagonal"] = w_diag
#                                 config["LAYER_SETTING"]["l1"]["w_diagonal_2x2"] = w_diag_2x2
#                                 if w_diag == False:
#                                     config["LAYER_SETTING"]["l0"]["neurons"] = l1_neur

#                                 file=open("configs/config_LIF_"+ str(ind) +".yaml","w")
#                                 yaml.dump(config,file)
#                                 file.close()
#                                 ind = ind +1



# print(config)
# config["NEURONS"] = "test"

# file=open("configs\config_LIF_1.yaml","w")
# yaml.dump(config,file)
# file.close()