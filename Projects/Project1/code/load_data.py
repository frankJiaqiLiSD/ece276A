import pickle
import sys

def read_file(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

def get_data(dataset_num):
  ds = int(dataset_num)
  cam_list = [1,2,8,9]
  if ds in cam_list:
    dataset_num = str(dataset_num)
    cfile = "../ECE276A_PR1/data/trainset/cam/cam" + dataset_num + ".p"
    ifile = "../ECE276A_PR1/data/trainset/imu/imuRaw" + dataset_num + ".p"
    vfile = "../ECE276A_PR1/data/trainset/vicon/viconRot" + dataset_num + ".p"

    camd = read_file(cfile)
    imud = read_file(ifile)
    vicd = read_file(vfile)

    return [camd, imud, vicd]
  else:
    ifile = "../ECE276A_PR1/data/trainset/imu/imuRaw" + dataset_num + ".p"
    vfile = "../ECE276A_PR1/data/trainset/vicon/viconRot" + dataset_num + ".p"

    camd = []
    imud = read_file(ifile)
    vicd = read_file(vfile)

    return [camd, imud, vicd]
