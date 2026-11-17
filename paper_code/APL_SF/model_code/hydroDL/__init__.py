import os
import socket
import collections
print('loading package hydroDL')


def initPath():
    """initial shortcut for some import paths
    """
    hostName = socket.gethostname()
    if hostName == 'smallLinux':
        dirDB = os.path.join(os.path.sep, 'mnt', 'sdc', 'rnnSMAP',
                             'Database_SMAPgrid')
        dirOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                              'Model_SMAPgrid')
        dirResult = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnSMAP',
                                 'Result_SMAPgrid')
    else:
        dirDB = '/'
        dirOut = '/'
        dirResult = '/'

    pathSMAP = collections.OrderedDict(
        DB_L3_Global=os.path.join(dirDB, 'Daily_L3'),
        DB_L3_NA=os.path.join(dirDB, 'Daily_L3_NA'),
        Out_L3_Global=os.path.join(dirOut, 'L3_Global'),
        Out_L3_NA=os.path.join(dirOut, 'L3_NA'),
        outTest=os.path.join(dirOut, 'Test'),
        dirResult=dirResult)

    # pathCamels = collections.OrderedDict(
    #     DB=os.path.join(os.path.sep, 'scratch', 'Camels'),
    #     Out=os.path.join(os.path.sep, 'data', 'rnnStreamflow'))
    # return pathSMAP, pathCamels

    # pathCamels = collections.OrderedDict(
    #     DB=os.path.join(os.path.sep, 'mnt', 'sdb', 'Data', 'Camels'),
    #     Out=os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnStreamflow'),
	#     Out_SQ=os.path.join(os.path.sep, 'mnt', 'sdc', 'streamflowDL_HYPERFACETS', 'SQ'))

    pathCamels = collections.OrderedDict(
        DB=os.path.join(os.path.sep, 'data', 'wxt42', 'raw_data', 'CAMELS'),
        DB_ext=os.path.join(os.path.sep, 'data', 'wxt42', 'raw_data', 'csvLst_new', 'CAMELs'),
        Out=os.path.join(os.path.sep, 'data', 'wxt42', 'camels_streamflow_ext'))

    pathCamels_fPL = collections.OrderedDict(
        DB=os.path.join(os.path.sep, 'data', 'wxt42', 'raw_data', 'CAMELS'),
        DB_fPL=os.path.join(os.path.sep, 'data', 'wxt42', 'raw_data', 'csvLst_new', 'CAMELs'),
        Out_fPL=os.path.join(os.path.sep, 'data', 'wxt42', 'Inv_streamflow_cases', 'stats_basin'))

    return pathSMAP, pathCamels, pathCamels_fPL

pathSMAP, pathCamels, pathCamels_fPL = initPath()

from . import utils
from . import data
from . import model
from . import post