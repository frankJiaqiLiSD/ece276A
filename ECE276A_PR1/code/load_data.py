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
  cfile = "../ECE276A_PR1/data/trainset/cam/cam" + dataset_num + ".p"
  ifile = "../ECE276A_PR1/data/trainset/imu/imuRaw" + dataset_num + ".p"
  vfile = "../ECE276A_PR1/data/trainset/vicon/viconRot" + dataset_num + ".p"

  camd = read_file(cfile)
  imud = read_file(ifile)
  vicd = read_file(vfile)

  return [camd, imud, vicd]




