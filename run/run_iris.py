# %% run multiple instances of IRIS in parallel.
import iris
import iris_input
import write_text_output
import datetime, os
from multiprocessing import Pool

# resDir: output directory
# inputs: model input parameter dictionary from iris_input.py
# nEns: number of ensembles to generate
# nYperEns: number of synthetic years per ensemble member
# basins: list of basins to generate

nCores = 16  # max number of cores to use in multiprocessing
def run_iris_par(resDir, inputs, nEns, nYperEns,basins):
    irisArgs = []
    for basin in basins:
        print(basin)
        for i in range(nEns):
            print("dataset:", i)
            run_id = "{:04d}".format(i)
            results_path = resDir + "results-" + basin + "-" + run_id
            args = {
                "inputDat": inputs[basin],
                "result": results_path,
                "seed": i,
                "nY":nYperEns,
            }
            irisArgs.append(args)

    time_start = datetime.datetime.now()
    with Pool(nCores) as pool:
        _ = pool.map(iris.run_1arg, irisArgs)
    time_finish = datetime.datetime.now()

    print("Runtime: ", time_finish - time_start)


# define run parameters
basins = ["NA", "WP", "EP", "NI", "SI", "SP",] # basins to simulate
rStr = "r01" # run version string

# fast test, 100y. ~2 mins on 16 cores.
nEns = 10 # number of ensembles
nYperEns = 10 # years per ensemble

# # recommended 10,000 year settings
# nEns = 100 # number of ensembles
# nYperEns = 100 # years per ensemble

runDesc = str(nEns) + "e-" + str(nYperEns) + "y"
resDir = "../out/" + rStr + "-" + runDesc+'/'
if not os.path.exists(resDir):
    os.makedirs(resDir)

# get input data and parameters from module
inputDats = iris_input.inputs

# run iris
run_iris_par(resDir, inputDats,nEns,nYperEns,basins)

#%% write text files from model output
outDir = '../out_txt/' + rStr + "-" + runDesc + '/'
nYperBlock=50 # number of years per output block
write_text_output.write(resDir,outDir,basins,nEns,nYperEns,nYperBlock)

