import pickle
import os
import time
import torch
import matplotlib.pyplot as plt

from python.data import *
from python.algorithms import *

exp_file = "saved_experiments.pkl"

params = {"n": 100_000, "d": 2, "std": 5.0, "cl": 5, "eps": 0.05, "rounds": 10}

ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
ds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# ds = [12]
stds = [1., 5., 10., 15., 20.]
stds = [10.]
cls = [2, 4, 8, 12, 16]
# cls = [12]
epss = [0.01, 0.05, 0.1, 0.25, 0.5]

label_GPU_SynC = "GPU-SynC"
label_simple_GPU_SynC = "simple GPU-SynC"
label_SynC = "SynC"
label_FSynC = "FSynC"
algorithms = [
    (SynC, label_SynC),
    (FSynC, label_FSynC),
    # (simple_GPU_SynC, label_simple_GPU_SynC),
    (GPU_SynC, label_GPU_SynC),
]

label_3D_spatial_network = "3D_spatial_network"
label_CCPP = "CCPP"
label_banknote = "data_banknote_authentication"
label_Skin_NonSkin = "Skin_NonSkin"
label_eb = "Tamilnadu Electricity Board Hourly Readings"
label_Wilt = "Wilt"
label_Yeast = "Yeast"
datasets = [
    (load_banknote, label_banknote),
    (load_Yeast, label_Yeast),
    (load_Wilt, label_Wilt),
    (load_CCPP, label_CCPP),
    (load_eb, label_eb),
    (load_Skin_NonSkin, label_Skin_NonSkin),
    (load_D_spatial_network, label_3D_spatial_network),
]

dataset_style = {
    label_3D_spatial_network: {"shortname": "Roads"},
    label_CCPP: {"shortname": "CPP"},
    label_banknote: {"shortname": "Bank"},
    label_Skin_NonSkin: {"shortname": "Skin"},
    label_eb: {"shortname": "EB"},
    label_Wilt: {"shortname": "Wilt"},
    label_Yeast: {"shortname": "Yeast"},
}

style_map = {
    label_SynC: {"color": "#A29FCC", "marker": "x", "linestyle": "solid", "legend": label_SynC},
    label_FSynC: {"color": "#D95F02", "marker": "o", "linestyle": "solid", "legend": label_FSynC},
    label_GPU_SynC: {"color": "#147054", "marker": "*", "linestyle": "solid", "legend": "EGG-SynC   "},
    label_simple_GPU_SynC: {"color": "#e7298a", "marker": "+", "linestyle": "solid", "legend": label_simple_GPU_SynC},
}
y_max_speedup = 8000
running_time_max = 100000.
running_time_min = 0.01
figure_size = (2.3, 1.75)
font_size = 8
plt.rcParams.update({'font.family': "Times New Roman"})
plt.rcParams.update({'font.serif': "Times New Roman"})
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'mathtext.default': "sf"})
plt.rcParams['text.usetex'] = True


def get_legend(name, algorithms, figsize):
    if not os.path.exists('plots/'):
        os.makedirs('plots/')

    plt.figure(figsize=figsize)
    for _, algo_name in algorithms:
        plt.plot([], [], color=style_map[algo_name]["color"], marker=style_map[algo_name]["marker"],
                 linestyle=style_map[algo_name]["linestyle"], label=style_map[algo_name]["legend"], linewidth=1)

    plt.legend(loc='center right', fontsize=font_size)
    plt.tight_layout()
    plt.gca().set_axis_off()
    plt.savefig("plots/" + name + "_legend.pdf")
    plt.clf()


class Experiments:
    def __init__(self):
        super().__init__()
        self.experiments_dict = {}
        if os.path.exists(exp_file):
            with open(exp_file, 'rb') as f:
                self.experiments_dict = pickle.load(f)

    def save(self):
        with open(exp_file, 'wb') as f:
            pickle.dump(self.experiments_dict, f)

    def set(self, method, dataset=None, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"],
            eps=params["eps"], round=0, running_time=0.):
        if dataset is None:
            if method not in self.experiments_dict:
                self.experiments_dict[method] = {}
            if n not in self.experiments_dict[method]:
                self.experiments_dict[method][n] = {}
            if d not in self.experiments_dict[method][n]:
                self.experiments_dict[method][n][d] = {}
            if std not in self.experiments_dict[method][n][d]:
                self.experiments_dict[method][n][d][std] = {}
            if cl not in self.experiments_dict[method][n][d][std]:
                self.experiments_dict[method][n][d][std][cl] = {}
            if eps not in self.experiments_dict[method][n][d][std][cl]:
                self.experiments_dict[method][n][d][std][cl][eps] = {}
            if round not in self.experiments_dict[method][n][d][std][cl][eps]:
                self.experiments_dict[method][n][d][std][cl][eps][round] = {}

            self.experiments_dict[method][n][d][std][cl][eps][round]["time"] = running_time
        else:
            if method not in self.experiments_dict:
                self.experiments_dict[method] = {}
            if dataset not in self.experiments_dict[method]:
                self.experiments_dict[method][dataset] = {}
            if eps not in self.experiments_dict[method][dataset]:
                self.experiments_dict[method][dataset][eps] = {}
            if round not in self.experiments_dict[method][dataset][eps]:
                self.experiments_dict[method][dataset][eps][round] = {}

            self.experiments_dict[method][dataset][eps][round]["time"] = running_time

        self.save()

    def get(self, method, dataset=None, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"],
            eps=params["eps"], round=0):
        if dataset is None:
            return self.experiments_dict[method][n][d][std][cl][eps][round]
        else:
            return self.experiments_dict[method][dataset][eps][round]

    def get_time(self, method, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"], eps=params["eps"],
                 round=0):
        return self.experiments_dict[method][n][d][std][cl][eps][round]["time"]

    def clear(self, method, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"], eps=params["eps"],
              round=0):
        del self.experiments_dict[method][n][d][std][cl][eps][round]
        if len(self.experiments_dict[method][n][d][std][cl][eps]) == 0:
            del self.experiments_dict[method][n][d][std][cl][eps]
        if len(self.experiments_dict[method][n][d][std][cl]) == 0:
            del self.experiments_dict[method][n][d][std][cl]
        if len(self.experiments_dict[method][n][d][std]) == 0:
            del self.experiments_dict[method][n][d][std]
        if len(self.experiments_dict[method][n][d]) == 0:
            del self.experiments_dict[method][n][d]
        if len(self.experiments_dict[method][n]) == 0:
            del self.experiments_dict[method][n]
        if len(self.experiments_dict[method]) == 0:
            del self.experiments_dict[method]

        self.save()

    def print(self):

        print(self.experiments_dict)
        print("")

        for method in self.experiments_dict.keys():
            print(method)
            for n in self.experiments_dict[method].keys():
                print("- n:", n)
                for d in self.experiments_dict[method][n].keys():
                    print("-- d:", d)
                    for std in self.experiments_dict[method][n][d].keys():
                        print("--- std:", std)
                        for cl in self.experiments_dict[method][n][d][std].keys():
                            print("---- cl:", cl)
                            for eps in self.experiments_dict[method][n][d][std][cl].keys():
                                print("----- eps:", eps)
                                times = []
                                for round in self.experiments_dict[method][n][d][std][cl][eps].keys():
                                    times.append(self.experiments_dict[method][n][d][std][cl][eps][round]["time"])
                                print("------ times:", times)

    def get_data(self, n, d, std, cl, round):
        filename = "data/n" + str(n) + "d" + str(d) + "std" + str(std) + "cl" + str(cl) + "round" + str(round) + ".pkl"

        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)

        data = torch.from_numpy(
            min_max_normalize(load_synt_gauss_rnd(d=d, n=n, cl=cl, cl_d=d, std=std, noise=0.))).float()
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        return data

    def save_data_to_txt(self, n, d, std, cl, round):
        data = self.get_data(n, d, std, cl, round)
        filename = "data/n" + str(n) + "d" + str(d) + "std" + str(std) + "cl" + str(cl) + "round" + str(round) + ".csv"

        np.savetxt(filename, data, delimiter=' ')

    def run(self, algortihms, data=None, dataset=None, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"],
            eps=params["eps"],
            rounds=params["rounds"]):

        for algorithm, method in algortihms:
            print(method)
            for round in range(rounds):
                if dataset is None \
                        and method in self.experiments_dict \
                        and n in self.experiments_dict[method] \
                        and d in self.experiments_dict[method][n] \
                        and std in self.experiments_dict[method][n][d] \
                        and cl in self.experiments_dict[method][n][d][std] \
                        and eps in self.experiments_dict[method][n][d][std][cl] \
                        and round in self.experiments_dict[method][n][d][std][cl][eps] \
                        and method != label_GPU_SynC:
                    print("Round", round, "has been run!")
                elif dataset is not None \
                        and method in self.experiments_dict \
                        and dataset in self.experiments_dict[method] \
                        and eps in self.experiments_dict[method][dataset] \
                        and round in self.experiments_dict[method][dataset][eps] \
                        and method != label_GPU_SynC:
                    print("Round", round, "has been run!")
                else:
                    print("Round", round, "is running!")
                    if dataset is None:
                        data = self.get_data(n, d, std, cl, round)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    algorithm(data, eps)
                    t1 = time.time()
                    if dataset is None:
                        self.set(method, n=n, d=d, std=std, cl=cl, eps=eps, round=round, running_time=t1 - t0)
                    else:
                        self.set(method, dataset=dataset, eps=eps, round=round, running_time=t1 - t0)

    def plot(self, experiment_name, get_xs_and_ys, x_label, y_label="time in seconds", y_max=running_time_max,
             y_scale="linear"):

        if not os.path.exists('plots/'):
            os.makedirs('plots/')

        plt.figure(figsize=figure_size)

        for _, method in algorithms:
            xs, ys = get_xs_and_ys(method)

            plt.plot(xs, ys, color=style_map[method]["color"], marker=style_map[method]["marker"],
                     linestyle=style_map[method]["linestyle"], label=style_map[method]["legend"], linewidth=1)

            print(method)
            print(xs)
            print(ys)

        plt.gcf().subplots_adjust(left=0.14)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.yscale(y_scale)
        plt.grid(True, which="both", ls="-")
        if not y_max is None:
            plt.ylim(running_time_min, y_max)
        plt.tight_layout()
        plt.savefig("plots/" + experiment_name + ".pdf")
        plt.clf()

    def plot_speedup(self, experiment_name, get_xs_and_ys, x_label, y_label="time in seconds", y_max=running_time_max,
                     y_scale="linear"):
        plt.figure(figsize=figure_size)

        xs = None
        yss = []
        for _, method in algorithms:
            xs, ys = get_xs_and_ys(method)

            yss.append(ys)

        for i in reversed(range(len(yss))):
            for j in range(len(yss[i])):
                yss[i][j] = yss[0][j] / yss[i][j]

        i = 0
        for _, method in algorithms:
            ys = yss[i]
            plt.plot(xs, ys, color=style_map[method]["color"], marker=style_map[method]["marker"],
                     linestyle=style_map[method]["linestyle"], label=style_map[method]["legend"], linewidth=1)
            i += 1

            print(method)
            print(xs)
            print(ys)

        plt.gcf().subplots_adjust(left=0.14)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.yscale(y_scale)
        plt.grid(True, which="both", ls="-")
        if not y_max is None:
            plt.ylim(running_time_min, y_max)
        plt.tight_layout()
        plt.savefig("plots/" + experiment_name + "_speedup.pdf")
        plt.clf()

    def run_inc_n(self):
        for n in ns:
            print("n:", n)
            self.run(algorithms, n=n)

    def plot_inc_n(self):
        def get_xs_and_ys(method):
            ys = []
            for n in ns:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, n=n, round=round)
                ys.append(sum / params["rounds"])
            return ns, ys

        self.plot("inc_n", get_xs_and_ys, x_label="size of dataset", y_scale="log")

    def plot_inc_n_speedup(self):
        def get_xs_and_ys(method):
            ys = []
            for n in ns:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, n=n, round=round)
                ys.append(sum / params["rounds"])
            return ns, ys

        self.plot_speedup("inc_n", get_xs_and_ys, x_label="size of dataset", y_max=y_max_speedup, y_label="speedup")

    def run_inc_d(self):
        for d in ds:
            print("d:", d)
            self.run(algorithms, d=d)

    def plot_inc_d(self):
        def get_xs_and_ys(method):
            ys = []
            for d in ds:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, d=d, round=round)
                ys.append(sum / params["rounds"])
            return ds, ys

        self.plot("inc_d", get_xs_and_ys, x_label="number of dimensions", y_scale="log")

    def plot_inc_d_speedup(self):
        def get_xs_and_ys(method):
            ys = []
            for d in ds:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, d=d, round=round)
                ys.append(sum / params["rounds"])
            return ds, ys

        self.plot_speedup("inc_d", get_xs_and_ys, x_label="number of dimensions", y_max=y_max_speedup,
                          y_label="speedup")

    def run_inc_std(self):
        for std in stds:
            print("std:", std)
            self.run(algorithms, std=std)

    def plot_inc_std(self):
        def get_xs_and_ys(method):
            ys = []
            for std in stds:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, std=std, round=round)
                ys.append(sum / params["rounds"])
            return stds, ys

        self.plot("inc_std", get_xs_and_ys, x_label="standard deviation", y_scale="log")

    def plot_inc_std_speedup(self):
        def get_xs_and_ys(method):
            ys = []
            for std in stds:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, std=std, round=round)
                ys.append(sum / params["rounds"])
            return stds, ys

        self.plot_speedup("inc_std", get_xs_and_ys, x_label="standard deviation", y_max=y_max_speedup,
                          y_label="speedup")

    def run_inc_cl(self):
        for cl in cls:
            print("cl:", cl)
            self.run(algorithms, cl=cl)

    def plot_inc_cl(self):
        def get_xs_and_ys(method):
            ys = []
            for cl in cls:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, cl=cl, round=round)
                ys.append(sum / params["rounds"])
            return cls, ys

        self.plot("inc_cl", get_xs_and_ys, x_label="number of clusters", y_scale="log")

    def plot_inc_cl_speedup(self):
        def get_xs_and_ys(method):
            ys = []
            for cl in cls:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, cl=cl, round=round)
                ys.append(sum / params["rounds"])
            return cls, ys

        self.plot_speedup("inc_cl", get_xs_and_ys, x_label="number of clusters", y_max=y_max_speedup, y_label="speedup")

    def run_inc_eps(self):
        for eps in epss:
            print("eps:", eps)
            self.run(algorithms, eps=eps)

    def plot_inc_eps(self):
        def get_xs_and_ys(method):
            ys = []
            for eps in epss:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, eps=eps, round=round)
                ys.append(sum / params["rounds"])
            return epss, ys

        self.plot("inc_eps", get_xs_and_ys, x_label="neighborhood radius", y_scale="log")

    def plot_inc_eps_speedup(self):
        def get_xs_and_ys(method):
            ys = []
            for eps in epss:
                sum = 0.
                for round in range(params["rounds"]):
                    sum += self.get_time(method, eps=eps, round=round)
                ys.append(sum / params["rounds"])
            return epss, ys

        self.plot_speedup("inc_eps", get_xs_and_ys, x_label="neighborhood radius", y_max=y_max_speedup,
                          y_label="speedup")

    def run_real(self):
        for load_data, dataset in datasets:
            print(dataset)
            data = torch.from_numpy(min_max_normalize(load_data())).float()
            self.run(algorithms, data=data, dataset=dataset)

    def plot_real(self):
        fig, ax = plt.subplots(figsize=(figure_size[0] * 0.49 / 0.33, figure_size[1]))
        ra = np.arange(len(datasets))
        width = 1. / (len(algorithms) + 1.)
        labels = [dataset_style[label]["shortname"] for _, label in datasets]
        print(labels)

        for i, algorithm in enumerate(algorithms):
            offset = (i * 2 - len(algorithms) + 2)

            times = []
            for dataset in datasets:
                avg = 0
                for j in range(params["rounds"]):
                    avg += self.get(method=algorithm[1], dataset=dataset[1], round=j)["time"]
                times.append(avg / params["rounds"])

            print(width)

            style = style_map[
                algorithm[1]]  # {"color": "#1b9e77", "marker": "x", "linestyle": "solid", "legend": label_SynC},

            rects = ax.bar(ra - offset * width / 2, times, width=width, label=style["legend"], color=style["color"])
            # ax.bar_label(rects, padding=3)

        ax.set_xticks(ra)
        ax.set_xticklabels(labels)

        ax.set_ylabel('time in seconds')

        # ax.legend()
        plt.rc('font', size=8)
        plt.yscale("log")
        fig.tight_layout()
        plt.savefig("plots/real.pdf")
        # plt.show()


exp = Experiments()
# print(exp.get(label_GPU_SynC, cl=16, round=1))
# exp.clear(label_GPU_SynC, cl=16, round=1)
# exp.run(algorithms, cl=16)
# print(exp.get(label_GPU_SynC, cl=16, round=1))


# exp.run_inc_d()
# exp.run_inc_std()
# exp.run_inc_cl()
# exp.run_inc_eps()
# exp.run_inc_n()
# exp.run_real()

# exp.plot_inc_std()
# exp.plot_inc_cl()
# exp.plot_inc_eps()
# exp.plot_inc_n()
# exp.plot_inc_d()
# exp.plot_real()
#
# exp.plot_inc_n_speedup()
# exp.plot_inc_d_speedup()
# exp.plot_inc_std_speedup()
# exp.plot_inc_cl_speedup()
# exp.plot_inc_eps_speedup()

get_legend("all", algorithms, figure_size)

# data = exp.get_data(3001, 3, 10., 4, 0)
# FSynC(data, 0.1)

# data = torch.from_numpy(min_max_normalize(load_eb())).float()
# eps = params["eps"]
# data = exp.get_data(1000000, 3, 10., 10, 0)
# exp.save_data_to_txt(1000000, 3, 10., 10, 0)

# t0 = time.time()
# SynC(data, eps)
# t1 = time.time()
# print("SynC:", t1-t0)

# t0 = time.time()
# FSynC(data, eps, 1000)
# t1 = time.time()
# print("FSynC B=1000:", t1-t0)

# t0 = time.time()
# FSynC(data, eps, 100)
# t1 = time.time()
# print("FSynC B=100:", t1-t0)

# t0 = time.time()
# FSynC(data, eps, 50)
# t1 = time.time()
# print("FSynC B=50:", t1-t0)

# t0 = time.time()
# FSynC(data, eps, 20)
# t1 = time.time()
# print("FSynC B=20:", t1-t0)


# torch.cuda.synchronize()
#
# t0 = time.time()
# GPU_SynC(data, eps, version=5)
# t1 = time.time()
# print("GPU-SynC:", t1 - t0)

# torch.cuda.synchronize()
#
# t0 = time.time()
# simple_GPU_SynC(data, eps)
# t1 = time.time()
# print("simple GPU-SynC:", t1 - t0) #26400
