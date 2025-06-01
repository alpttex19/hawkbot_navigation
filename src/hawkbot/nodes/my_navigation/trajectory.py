import pickle


def load_V():
    with open("./G/G_goal00_Ndemo=6000.pkl", "rb") as g:  # 读取pkl文件数据
        gmm_parameters = pickle.load(g, encoding="bytes")
    with open("./V/V_goal00_Ndemo=6000.pkl", "rb") as v:
        Vxf = pickle.load(v, encoding="bytes")
    return gmm_parameters, Vxf
