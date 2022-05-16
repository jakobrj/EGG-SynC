import pickle
import os
import time
import torch
import matplotlib.pyplot as plt

from python.data import *
from python.algorithms import *

exp_file = "saved_experiments_server.pkl"

params = {"n": 100_000, "d": 2, "std": 5.0, "cl": 5, "eps": 0.05, "rounds": 3}

ns = [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000]
ds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
stds = [1., 5., 10., 15., 20.]
cls = [2, 4, 8, 12, 16]
epss = [0.0125, 0.025, 0.050, 0.1, 0.2, 0.500]

stages = ["Allocating", "Build structure", "Update", "Extra check", "Clustering", "Free Memory"]

label_GPU_SynC = "GPU-SynC"
label_simple_GPU_SynC = "simple GPU-SynC"
label_SynC = "SynC"
label_SynC_parallel = "SynC parallel"
label_FSynC = "FSynC"
algorithms = [
    # (SynC, label_SynC),
    # (SynC_parallel, label_SynC_parallel),
    # (FSynC, label_FSynC),
    (simple_GPU_SynC, label_simple_GPU_SynC),
    (EGG_SynC, label_GPU_SynC),
]

stage_maxs = {
    label_SynC: 80,
    label_SynC_parallel: 80,
    label_FSynC: 80,
    label_simple_GPU_SynC: 80,
    label_GPU_SynC: 2,
}

label_3D_spatial_network = "3D_spatial_network"
label_CCPP = "CCPP"
label_banknote = "data_banknote_authentication"
label_Skin_NonSkin = "Skin_NonSkin"
label_eb = "Tamilnadu Electricity Board Hourly Readings"
label_Wilt = "Wilt"
label_Yeast = "Yeast"
label_EEG_Eye_State = "EEG Eye State"
label_Letter = "Letter"

datasets = [
    (load_banknote, label_banknote),
    (load_Yeast, label_Yeast),
    (load_Wilt, label_Wilt),
    (load_CCPP, label_CCPP),
    (load_eb, label_eb),
    (load_Skin_NonSkin, label_Skin_NonSkin),
    (load_D_spatial_network, label_3D_spatial_network),
    (load_EEG_Eye_State, label_EEG_Eye_State),
    (load_Letter, label_Letter),
]

dataset_style = {
    label_3D_spatial_network: {"shortname": "Roads"},
    label_CCPP: {"shortname": "CPP"},
    label_banknote: {"shortname": "Bank"},
    label_Skin_NonSkin: {"shortname": "Skin"},
    label_eb: {"shortname": "EB"},
    label_Wilt: {"shortname": "Wilt"},
    label_Yeast: {"shortname": "Yeast"},
    label_EEG_Eye_State: {"shortname": "EEG"},
    label_Letter: {"shortname": "Letter"},
}

style_map = {
    label_SynC: {"color": "#A29FCC", "marker": "x", "linestyle": "solid", "legend": label_SynC},
    label_SynC_parallel: {"color": "#645cd6", "marker": "x", "linestyle": "solid", "legend": "MP-SynC"},
    label_FSynC: {"color": "#D95F02", "marker": "o", "linestyle": "solid", "legend": label_FSynC},
    label_GPU_SynC: {"color": "#147054", "marker": "*", "linestyle": "solid", "legend": "EGG-SynC"},
    label_simple_GPU_SynC: {"color": "#e7298a", "marker": "+", "linestyle": "solid", "legend": "GPU-SynC"},
}
y_max_speedup = 20000
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


def plot_stack_column_chart(to_plot, number_of_groups, size_of_group, labels, group_labels, figsize, filename,
                            ylim=80.):
    fig, ax = plt.subplots(figsize=figsize)
    width = 1 / (size_of_group + 0.5)
    width_ = width - width / 10

    x = [i + width * j for i in range(number_of_groups) for j in range(size_of_group)]
    x_ = [i + (size_of_group - 1) * width / 2 + 0.01 for i in range(number_of_groups)]
    offset = [0.] * number_of_groups * size_of_group
    for measurements, label in to_plot:
        ax.bar(
            # tick_label=labels,
            x=x,
            height=measurements,
            width=width_,
            bottom=offset,
            label=label,
        )
        for i in range(len(measurements)):
            offset[i] += measurements[i]

    # ax.grid( 'off', axis='x' )
    # ax.grid( axis='x', which='minor')

    # ax.tick_params( axis='x', which='minor', direction='out', length=30 )
    ax.tick_params(axis='x', which='major', bottom=False, top=False)

    group_labels = ['\n' + l for l in group_labels]
    ax.set_xticks(x + x_)
    ax.set_xticklabels(labels + group_labels)
    ax.set_ylabel("time in seconds")
    ax.set_xlabel("size of dataset")
    ax.legend()
    plt.tight_layout()
    # plt.ylim([0., ylim])

    plt.savefig(filename)
    plt.show()


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
            eps=params["eps"], round=0, running_time=None, itr_times=None, stage_times=None, space=None):
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

            if running_time is not None:
                self.experiments_dict[method][n][d][std][cl][eps][round]["time"] = running_time

            if itr_times is not None:
                self.experiments_dict[method][n][d][std][cl][eps][round]["itr_times"] = itr_times

            if stage_times is not None:
                self.experiments_dict[method][n][d][std][cl][eps][round]["stage_times"] = stage_times

            if space is not None:
                self.experiments_dict[method][n][d][std][cl][eps][round]["space"] = space[0].item()


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

    def get_space(self, method, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"], eps=params["eps"],
                  round=0):
        print("space:", self.experiments_dict[method][n][d][std][cl][eps][round]["space"])
        return self.experiments_dict[method][n][d][std][cl][eps][round]["space"]

    def get_itr_times(self, method, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"], eps=params["eps"],
                      round=0):
        return self.experiments_dict[method][n][d][std][cl][eps][round]["itr_times"]

    def get_stage_times(self, method, n=params["n"], d=params["d"], std=params["std"], cl=params["cl"],
                        eps=params["eps"],
                        round=0):
        return self.experiments_dict[method][n][d][std][cl][eps][round]["stage_times"]

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
                        and round in self.experiments_dict[method][n][d][std][cl][eps]:  # \
                    # and method != label_GPU_SynC:
                    print("Round", round, "has been run!")
                elif dataset is not None \
                        and method in self.experiments_dict \
                        and dataset in self.experiments_dict[method] \
                        and eps in self.experiments_dict[method][dataset] \
                        and round in self.experiments_dict[method][dataset][eps]:  # \
                    # and method != label_GPU_SynC:
                    print("Round", round, "has been run!")
                else:
                    print("Round", round, "is running!")
                    if dataset is None:
                        data = self.get_data(n, d, std, cl, round)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    _, itr_times, stage_times, space = algorithm(data, eps)
                    t1 = time.time()
                    print(itr_times, stage_times, space)
                    if dataset is None:
                        self.set(method, n=n, d=d, std=std, cl=cl, eps=eps, round=round, running_time=t1 - t0,
                                 itr_times=itr_times, stage_times=stage_times, space=space)
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

            # print(width)

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


    def plot_real_v2(self):
        fig, ax = plt.subplots(figsize=(figure_size[0] * 1.6, figure_size[1]))
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

            # print(width)

            style = style_map[
                algorithm[1]]  # {"color": "#1b9e77", "marker": "x", "linestyle": "solid", "legend": label_SynC},

            rects = ax.bar(ra - offset * width / 2, times, width=width, label=style["legend"], color=style["color"])
            # ax.bar_label(rects, padding=3)

        ax.set_xticks(ra)
        ax.set_xticklabels(labels)

        ax.set_ylabel('time in seconds')

        ax.legend()
        plt.rc('font', size=8)
        plt.yscale("log")
        fig.tight_layout()
        plt.savefig("plots/real.pdf")
        # plt.show()

    def run_single(self):
        n = params["n"]
        d = params["d"]
        std = params["std"]
        cl = params["cl"]
        eps = params["eps"]

        for round in range(params["rounds"]):
            for algorithm, method in algorithms:
                self.run(algorithms, eps=eps)

    def plot_itr_time(self):
        plt.figure(figsize=figure_size)

        for _, method in algorithms:
            max_itr = 0
            for round in range(params["rounds"]):
                itr_times = self.get_itr_times(method, round=round)
                if len(itr_times) > max_itr:
                    max_itr = len(itr_times)

            xs = list(range(max_itr))
            ys = [0] * max_itr

            itr_timess = []
            for round in range(params["rounds"]):
                itr_times = self.get_itr_times(method, round=round)
                print(itr_times)
                itr_timess.append(itr_times)

            for itr in range(max_itr):
                count = 0
                for round in range(params["rounds"]):
                    if itr < len(itr_timess[round]):
                        count += 1
                        ys[itr] += itr_timess[round][itr]
                if count != 0:
                    ys[itr] /= count

            plt.plot(xs, ys, color=style_map[method]["color"], marker=style_map[method]["marker"],
                     linestyle=style_map[method]["linestyle"], label=style_map[method]["legend"], linewidth=1)

        plt.gcf().subplots_adjust(left=0.14)
        plt.ylabel('time in seconds')
        plt.xlabel('iterations')
        plt.tight_layout()
        plt.savefig("plots/itr_time.pdf")
        plt.show()

    def plot_stages_v1(self):
        plt.figure(figsize=(figure_size[0] * 3, figure_size[1]))

        max_itr = 0
        for _, method in algorithms:
            for round in range(params["rounds"]):
                itr_times = self.get_stage_times(method, round=round)
                if len(itr_times) > max_itr:
                    max_itr = len(itr_times)

        xs = list(range(max_itr))

        for _, method in algorithms:
            ys = [0] * max_itr

            itr_timess = []
            for round in range(params["rounds"]):
                itr_times = self.get_stage_times(method, round=round)
                print(itr_times)
                itr_timess.append(itr_times)

            for itr in range(max_itr):
                count = 0
                for round in range(params["rounds"]):
                    if itr < len(itr_timess[round]):
                        count += 1
                        ys[itr] += itr_timess[round][itr]
                if count != 0:
                    ys[itr] /= count

            plt.bar(xs, ys, color=style_map[method]["color"],  # marker=style_map[method]["marker"],
                    linestyle=style_map[method]["linestyle"], label=style_map[method]["legend"], linewidth=1)

        plt.gcf().subplots_adjust(left=0.14)
        plt.ylabel('time in seconds')
        # plt.xlabel('stages')
        plt.tight_layout()
        plt.savefig("plots/stage_time.pdf")
        plt.show()

    def plot_stages_v2(self):
        subset_ns = ns[-3:]
        exp = {}
        for _, method in algorithms:
            exp[method] = {}
            for n in subset_ns:
                number_of_stages = len(stages)
                avg_stages_times = [0.] * number_of_stages
                for round in range(params["rounds"]):
                    stages_times = self.get_stage_times(method, n=n, round=round)
                    for i in range(number_of_stages):
                        avg_stages_times[i] += stages_times[i]

                for i in range(number_of_stages):
                    avg_stages_times[i] /= params["rounds"]
                exp[method][n] = avg_stages_times

        size_of_group = len(algorithms)
        number_of_groups = len(subset_ns)
        labels = [style_map[method]["legend"] for n in subset_ns for _, method in algorithms]
        group_labels = ["n=" + str(n) for n in subset_ns]
        to_plot = [([exp[method][n][i] for n in subset_ns for _, method in algorithms], stages[i])
                   for i in range(number_of_stages)]

        plot_stack_column_chart(
            to_plot=to_plot,
            number_of_groups=number_of_groups,
            size_of_group=size_of_group,
            labels=labels,
            group_labels=group_labels,
            figsize=(figure_size[0] * 2, figure_size[1]),
            filename="plots/stage_time.pdf",
        )

    def plot_stages_v3(self):
        number_of_stages = len(stages)
        width = 0.75
        subset_ns = ns[-7:]  # [ns[i] for i in [-3, -2, -1]]
        exp = {}
        for _, method in algorithms:
            exp[method] = {}
            for n in subset_ns:
                avg_stages_times = [0.] * number_of_stages
                for round in range(params["rounds"]):
                    stages_times = self.get_stage_times(method, n=n, round=round)
                    for i in range(number_of_stages):
                        avg_stages_times[i] += stages_times[i]

                for i in range(number_of_stages):
                    avg_stages_times[i] /= params["rounds"]
                exp[method][n] = avg_stages_times

            len(subset_ns)
            labels = [str(n) for n in subset_ns]

            to_plot = [([exp[method][n][i] for n in subset_ns], stages[i])
                       for i in range(number_of_stages)]

            fig, ax = plt.subplots(figsize=(figure_size[0] * 1., figure_size[1]))

            x = [i for i in range(len(subset_ns))]
            offset = [0.] * len(subset_ns)
            for measurements, label in to_plot:
                ax.bar(
                    x=x,
                    height=measurements,
                    width=width,
                    bottom=offset,
                    label=label,
                )
                for i in range(len(measurements)):
                    offset[i] += measurements[i]

            ax.tick_params(axis='x', which='major', bottom=False, top=False)

            ax.set_xticks([x[i] for i in range(len(x)) if i % 2 == 0])
            ax.set_xticklabels([labels[i] for i in range(len(labels)) if i % 2 == 0])
            ax.set_ylabel("time in seconds")
            ax.set_xlabel("size of dataset")
            ax.legend()
            plt.tight_layout()
            # plt.ylim([0., stage_maxs[method]])

            plt.savefig("plots/" + method + "_stage_time.pdf")
            plt.show()

    def plot_stages_v4(self):
        subset_ns = [ns[i] for i in [-3, -2, -1]]
        exp = {}
        for _, method in algorithms:
            exp[method] = {}
            for n in subset_ns:
                number_of_stages = len(stages)
                avg_stages_times = [0.] * number_of_stages
                for round in range(params["rounds"]):
                    stages_times = self.get_stage_times(method, n=n, round=round)
                    for i in range(number_of_stages):
                        avg_stages_times[i] += stages_times[i]

                total = 0
                for i in range(number_of_stages):
                    avg_stages_times[i] /= params["rounds"]
                    total += avg_stages_times[i]

                for i in range(number_of_stages):
                    avg_stages_times[i] /= total

                exp[method][n] = avg_stages_times

        size_of_group = len(algorithms)
        number_of_groups = len(subset_ns)
        labels = [style_map[method]["legend"] for n in subset_ns for _, method in algorithms]
        group_labels = ["n=" + str(n) for n in subset_ns]
        to_plot = [([exp[method][n][i] for n in subset_ns for _, method in algorithms], stages[i])
                   for i in range(number_of_stages)]

        plot_stack_column_chart(plot_stack_column_chart(
            to_plot=to_plot,
            number_of_groups=number_of_groups,
            size_of_group=size_of_group,
            labels=labels,
            group_labels=group_labels,
            figsize=(figure_size[0] * 2, figure_size[1]),
            filename="plots/stage_time.pdf",
            ylim=1.
        ))

    def table_stages(self):
        number_of_stages = len(stages)

        latex_tabular = "\\begin{tabular}{ r r | "
        latex_tabular += "c " * number_of_stages
        latex_tabular += "}\n"

        latex_tabular += "\\textbf{size of dataset} & \\textbf{Method}"
        for i in range(number_of_stages):
            latex_tabular += " & \\textbf{" + stages[i] + "}"
        latex_tabular += "\\\\\n"

        for n in ns[-4:]:
            latex_tabular += "\\hline\n"
            latex_tabular += "\\multirow{" + str(len(algorithms)) + "}{*}{$" + str(n) + "$}"
            for _, method in algorithms:
                latex_tabular += " & " + style_map[method]["legend"]
                avg_stages_times = [0.] * number_of_stages
                for round in range(params["rounds"]):
                    stages_times = self.get_stage_times(method, n=n, round=round)
                    for i in range(number_of_stages):
                        avg_stages_times[i] += stages_times[i]

                for i in range(number_of_stages):
                    avg_stages_times[i] /= params["rounds"]
                    latex_tabular += " & %f" % avg_stages_times[i].item()
                latex_tabular += "\\\\\n"

        latex_tabular += "\\end{tabular}"

        with open("plots/table_stages.txt", "w") as text_file:
            text_file.write(latex_tabular)

    def plot_space(self):
        def get_xs_and_ys(method):
            ys = []
            for n in ns:
                sum = 0
                for round in range(params["rounds"]):
                    sum += self.get_space(method, n=n, round=round) / (1024 * 1024)
                ys.append(sum / params["rounds"])
            return ns, ys

        self.plot("space", get_xs_and_ys, x_label="size of dataset", y_label="memory usage (MB)", y_max=50)


exp = Experiments()
exp.run(algorithms)


# exp.run_real()
# exp.run_inc_d()
# exp.run_inc_std()
# exp.run_inc_cl()
# exp.run_inc_n()
# exp.run_inc_eps()

# exp.plot_itr_time()
# exp.table_stages()
# exp.plot_stages_v3()
# exp.plot_space()

# exp.plot_inc_std()
# exp.plot_inc_cl()
# exp.plot_inc_eps()
# exp.plot_inc_n()
# exp.plot_inc_d()
# exp.plot_real_v2()

# exp.plot_inc_n_speedup()
# exp.plot_inc_d_speedup()
# exp.plot_inc_std_speedup()
# exp.plot_inc_cl_speedup()
# exp.plot_inc_eps_speedup()

# get_legend("all", algorithms, figure_size)
