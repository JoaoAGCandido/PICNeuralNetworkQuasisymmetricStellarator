import argparse
import numpy as np
import pandas as pd
from qsc import Qsc
import time
import datetime
import sys
import os
import logging


parser = argparse.ArgumentParser(
    description="Scans using pyQsc for 1st order with the following parameters: rc1 -> 0, 0.3; z1 -> 0, -0.3; eta -> -3, -0.01\nexample:\npython3 CreateDatabase/dataSetGeneratorGrid.py 10 1 8 scan.cs")
parser.add_argument(
    "num", help="Number of calculations for each parameter", type=int)
parser.add_argument(
    "nfpMin", help="Starting number of field periods (nfp) for the scan", type=int)
parser.add_argument(
    "nfpMax", help="Ending number of field periods (nfp) for the scan", type=int)
parser.add_argument(
    "fileName", help="Name of the file to be created")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Prints the duration in seconds and the progress, by showing the current number of field periods (nfp)")
args = parser.parse_args()


def estimatedTime():
    estimatedTime = (time.time() - startTime) * \
        input['num'] * (input['nfp'][1] - input['nfp'][0] + 1)
    if args.verbose:
        print("Estimated time: ", str(
            datetime.timedelta(seconds=int(estimatedTime))))


def saveData(out):
    df = pd.DataFrame(out)
    file_exists = os.path.isfile(args.fileName)
    if file_exists:
        df.to_csv(args.fileName, index=False, header=False, mode="a")
    else:
        df.to_csv(args.fileName, index=False)
    # clear out
    out = {
        'axLenght': [],
        'RotTrans': [],
        'nfp': [],
        'heli': [],
        'rc1': [],
        'zs1': [],
        'eta': [],
        'max_elong': [],
        'LgradB': [],
        'min_R0': []
    }
    return out


def saveFailedData(rc1, zs1, nfp, eta, message):
    dict = {
        'nfp': [nfp],
        'rc1': [rc1],
        'zs1': [zs1],
        'eta': [eta],
        'message': message
    }
    df = pd.DataFrame(dict)
    file_exists = os.path.isfile("failedStellarators.csv")
    if file_exists:
        df.to_csv("failedStellarators.csv",
                  index=False, header=False, mode="a")
    else:
        df.to_csv("failedStellarators.csv", index=False)


def newton_solution_fail_filter(record):
    '''If Newton solve doesn't converge, capture warning and save on another file'''
    global warningsCaptured
    warningsCaptured = True
    if args.verbose:
        print("\nRecording stellarator on ' failedStellarators.csv ' instead of '",
              args.fileName, "' beacause:")
    saveFailedData(rc1, zs1, nfp, eta, str(record))
    if args.verbose:
        return True


logger = logging.getLogger("qsc.newton")
logger.addFilter(newton_solution_fail_filter)


# part of the radial component of the axis
rc0 = 1

# store if estimated time was printed
timePrinted = False


# arrays to store input
input = {
    # number of measurements (nfp * num ^3)
    'num': args.num,
    'nfp': [args.nfpMin, args.nfpMax],
    'rc1': [0, 0.3],
    'zs1': [0, -0.3],
    'eta': [-3, -0.01],
}

# arrays to store output
out = {
    'axLenght': [],
    'RotTrans': [],
    'nfp': [],
    'heli': [],
    'rc1': [],
    'zs1': [],
    'eta': [],
    'max_elong': [],
    'LgradB': [],
    'min_R0': []
}


startTime = time.time()
# this for changes nfp - number of field Periods
for nfp in np.arange(input['nfp'][0], input['nfp'][1] + 1, 1):
    if args.verbose:
        print(nfp, " out of ", input['nfp'][1])
    for rc1 in np.linspace(input['rc1'][0], input['rc1'][1], input['num']):
        for zs1 in np.linspace(input['zs1'][0], input['zs1'][1], input['num']):
            for eta in np.linspace(input['eta'][0], input['eta'][1], input['num']):
                try:
                    warningsCaptured = False
                    # get the stellarator
                    stel = Qsc(rc=[rc0, rc1], zs=[0, zs1], nfp=nfp, etabar=eta)
                    if not warningsCaptured:
                        # get output
                        out['axLenght'].append(
                            stel.axis_length / 2 / np.pi / rc0)
                        out['RotTrans'].append(stel.iota)

                        # check if stel is QA <- heli=False or QH <- heli=True
                        out['heli'].append(
                            False if stel.helicity == 0 else True)

                        # save parameters
                        out['rc1'].append(rc1)
                        out['zs1'].append(zs1)
                        out['nfp'].append(nfp)
                        out['eta'].append(eta)
                        out['max_elong'].append(stel.max_elongation)
                        out['LgradB'].append(min(stel.L_grad_B))
                        out['min_R0'].append(stel.min_R0)
                except KeyboardInterrupt:
                    print('\nInterrupted')
                    try:
                        sys.exit(130)
                    except SystemExit:
                        os._exit(130)
                except:
                    if args.verbose:
                        print("\nAn exception has ocurred:")
                    saveFailedData(rc1, zs1, nfp, eta, "An exception ocurred")

        # backup data and clear out
        out = saveData(out)

        if (timePrinted == False):
            estimatedTime()
            timePrinted = True

saveData(out)


endTime = time.time() - startTime
if args.verbose:
    print("\nEnd Time: ", str(datetime.timedelta(seconds=int(endTime))))
