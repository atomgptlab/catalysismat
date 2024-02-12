import glob, os, jarvis_leaderboard
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
from jarvis.db.figshare import data
import csv
import os
import sys
import time
import json
import zipfile
from jarvis.db.figshare import data
from jarvis.tasks.queue_jobs import Queue
import pandas as pd
import os, glob
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from jarvis.db.jsonutils import loadjson
from alignn.graphs import Graph
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from jarvis.core.atoms import Atoms
import torch
from jarvis.db.figshare import data
from alignn.graphs import Graph
#wget https://raw.githubusercontent.com/usnistgov/alignn/main/alignn/examples/sample_data/config_example.json
conf = loadjson("config_example.json")

root_dir = str(jarvis_leaderboard.__path__[0])
for i in glob.glob(
    "jarvis_leaderboard/jarvis_leaderboard/contributions/alignn_model/*mae.csv.zip"
):
    if "AGRA" not in i and "tinnet" not in i and "dft_3d" in i:
        # if "AGRA" in i or "tinnet" in i:
        benchmark_file = i.split("/")[-1]
        method = benchmark_file.split("-")[0]
        task = benchmark_file.split("-")[1]
        prop = benchmark_file.split("-")[2]
        dataset = benchmark_file.split("-")[3]
        temp = dataset + "_" + prop + ".json.zip"
        temp2 = dataset + "_" + prop + ".json"
        fname = os.path.join(root_dir, "benchmarks", method, task, temp)
        print(benchmark_file)
        dat = data(dataset)
        if "jid" in dat[0]:
            id_tag = "jid"
        else:
            id_tag = "id"
        info = {}
        for i in dat:
            info[i[id_tag]] = Atoms.from_dict(i["atoms"])

        zp = zipfile.ZipFile(fname)
        train_val_test = json.loads(zp.read(temp2))
        train = train_val_test["train"]
        conf["n_train"] = len(train)
        val = {}
        if "val" in train_val_test:
            val = train_val_test["val"]
        test = train_val_test["test"]
        conf["n_test"] = len(test)
        conf["n_val"] = len(test)
        if conf["batch_size"] > len(test):
            conf["batch_size"] = len(test)
        print(train_val_test)
        data_dir = "DataDir_" + str(benchmark_file).split(".csv.zip")[0]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        fname = data_dir + "/id_prop.csv"
        f = open(fname, "w")
        for ii in dat:
            if ii[id_tag] in list(train.keys()):
                nm = ii[id_tag] + "," + str(ii[prop]) + "\n"
                f.write(nm)
                nm = data_dir + "/" + ii[id_tag]
                atoms = Atoms.from_dict(ii["atoms"])
                atoms.write_poscar(nm)
        for ii in dat:
            # if ii[id_tag] in test:
            if ii[id_tag] in list(test.keys()):
                nm = ii[id_tag] + "," + str(ii[prop]) + "\n"
                f.write(nm)
                # nm="DataDir/"+ii[id_tag]
                atoms = Atoms.from_dict(ii["atoms"])
                nm = data_dir + "/" + ii[id_tag]
                atoms.write_poscar(nm)
        for ii in dat:
            if ii[id_tag] in list(test.keys()):
                # nm = ii[id_tag] + "," + ii[prop] + "\n"
                nm = ii[id_tag] + "," + str(ii[prop]) + "\n"
                f.write(nm)
                # nm="DataDir/"+ii[id_tag]
                atoms = Atoms.from_dict(ii["atoms"])
                nm = data_dir + "/" + ii[id_tag]
                atoms.write_poscar(nm)
        f.close()
        config_file = (
            data_dir
            + "/config_"
            + str(benchmark_file).split(".csv.zip")[0]
            + ".json"
        )
        dumpjson(data=conf, filename=config_file)
        out_dir = "out_" + str(benchmark_file)
        cmd = (
            "train_folder_ff.py --root_dir "
            + data_dir
            + " --config "
            + config_file
            + " --output_dir="
            + out_dir.split(".csv.zip")[0]
        )

        print(cmd)

        cwd = os.getcwd()
        subname = str(benchmark_file).split(".csv.zip")[0] + "_sub_job"
        # subname = prop+"_"+dataset
        job_line = (
            "\n. ~/.bashrc\nconda activate /wrk/knc6/Software/mini_alignn\n"
            + cmd
        )
        jobout = "job.out." + str(benchmark_file).split(".csv.zip")[0]
        joberr = "job.err." + str(benchmark_file).split(".csv.zip")[0]
        Queue.slurm(
            filename=subname,
            job_line=job_line,
            jobname=str(benchmark_file).split(".csv.zip")[0],
            directory=os.getcwd(),
            pre_job_lines="#SBATCH --gres=gpu:1",
            memory="200G",
            queue="singlegpu,interactive",
            jobout=jobout,
            joberr=joberr,
            walltime="3-00:00:00",
            submit_cmd=["sbatch", subname],
        )

        # break


### Evaluate

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def load_model(
    dir_path="", config_filename="config.json", filename="best_model.pt"
):
    best_path = os.path.join(dir_path, filename)
    config = loadjson(os.path.join(dir_path, config_filename))
    # print("config",config)
    model = ALIGNN(ALIGNNConfig(**config["model"]))
    model.state_dict()
    model.load_state_dict(
        torch.load(os.path.join(dir_path, filename), map_location=device)[
            "model"
        ]
    )
    model.to(device)
    model.eval()
    return model


def evaluate(csv_file="", dir_path="", config_filename="config.json"):
    model = load_model(dir_path=dir_path)
    df = pd.read_csv(csv_file)
    tmp = csv_file.split("/")[0].split("-")
    dataset = tmp[3]
    prop = tmp[2]
    dat = data(dataset)
    config = loadjson(os.path.join(dir_path, config_filename))
    if "jid" in dat[0]:
        id_tag = "jid"
    else:
        id_tag = "id"
    targ = []
    pred = []
    # print('keys',dat[0].keys())
    # print('df',csv_file,df)
    for ii in dat:
        # if ii[id_tag] in test:
        if ii[id_tag] in list(df["id"].values):
            # nm="DataDir/"+ii[id_tag]
            atoms = Atoms.from_dict(ii["atoms"])
            g, lg = Graph.atom_dgl_multigraph(
                atoms,
                neighbor_strategy=config["neighbor_strategy"],
                cutoff=config["cutoff"],
                max_neighbors=config["max_neighbors"],
                atom_features=config["atom_features"],
                use_canonize=config["use_canonize"],
            )
            result = (
                model((g.to(device), lg.to(device))).cpu().detach().numpy()
            )
            # print("result", result, ii[prop])
            targ.append(ii[prop])
            pred.append(result)
    return pearsonr(targ, pred)


for i in glob.glob("out*/*test_set.csv"):
    if "AGRA" in i or "tinnet" in i:
        # if "test" in i:
        df = pd.read_csv(i)
        mae = mean_absolute_error(df["target"], df["prediction"])
        pr = pearsonr(df["target"], df["prediction"])
        for j in glob.glob("out*/*test_set.csv"):
            # if "test" in j:
            if "AGRA" in j or "tinnet" in j:
                pp = evaluate(csv_file=i, dir_path=j.split("/")[0])
                print(i, j, pp)
        # print(i, mae, pr[0])
