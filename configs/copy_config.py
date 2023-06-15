import yaml

with open("configs\config_LIF_DEFAULT.yaml","r") as f:
    config = yaml.safe_load(f)

ind = 0

for l1_neur in [30,40,50]:
    for rec in [True,False]:
        for adapt in [True,False]:
            for shared in [True,False]:
                for w_diag in [True, False]:
                    config["NEURONS"] = l1_neur
                    config["LAYER_SETTING"]["l1"]["recurrent"] = rec
                    config["LAYER_SETTING"]["l1"]["adaptive"] = adapt
                    
                    config["LAYER_SETTING"]["l0"]["shared_weight_and_bias"] = shared
                    config["LAYER_SETTING"]["l1"]["shared_weight_and_bias"] = shared
                    config["LAYER_SETTING"]["l2"]["shared_weight_and_bias"] = shared
                    config["LAYER_SETTING"]["l1"]["adapt_share_add_t"] = shared
                    config["LAYER_SETTING"]["l0"]["shared_leak_i"] = shared
                    config["LAYER_SETTING"]["l1"]["shared_leak_i"] = shared

                    config["LAYER_SETTING"]["l1"]["w_diagonal"] = w_diag
                    if w_diag == False:
                        config["LAYER_SETTING"]["l0"]["neurons"] = l1_neur

                    file=open("configs\config_LIF_"+ str(ind) +".yaml","w")
                    yaml.dump(config,file)
                    file.close()
                    ind = ind +1


# print(config)
# config["NEURONS"] = "test"

# file=open("configs\config_LIF_1.yaml","w")
# yaml.dump(config,file)
# file.close()