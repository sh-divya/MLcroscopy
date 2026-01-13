import torch
import pprint

def get_hp_vals(keys):
    hp_dict = {}
    for k in keys:
        section = k.split(".")
        print(section[1])
        if section[1] not in hp_dict:
            hp_dict[section[1]] = []
            if section[-2][:2] == "bn":
                print(section[-2])
            else:
                hp_dict[section[1]].append(section[3])

    return hp_dict
        

if __name__ == "__main__":
    x = torch.load(
        "/network/scratch/d/divya.sharma/runs/ada/ada/6bhivonc/checkpoints/tmp.ckpt",
        map_location=torch.device("cpu"),
    )
    pprint.pprint(list(x['state_dict'].keys()))
    # hp = get_hp_vals(x['state_dict'].keys())
    # print(hp)
