
import torch
import numpy as np
import os
import cv2 as cv

def resize_img(img,scale):
  scale_percent = scale # percent of original size
  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)

  # resize image
  resized = cv.resize(img, dim, interpolation = cv.INTER_AREA)
  return resized

def get_odg(img,u,v,E,hwf,z):
  #o = -E[:3,:3].matmul(E[:3,3])
  o = E[:3,3]
  #px_2_world(hwf[0]//2,hwf[1]//2, E, hwf, 0)
  #d = fp-o
  d = torch.zeros(3,1)
  d[0],d[1],d[2] = u-hwf[0]/2,v-hwf[1]/2,hwf[2]
  d = d.to(torch.float32)
  #d = E[:3,:3].transpose(0,1).matmul(d).reshape(-1)
  d = E[:3,:3].matmul(d).reshape(-1)

  #fp = px_2_world(u,v, E, hwf, z)
  d = -d/(torch.sqrt((d*d).sum(-1))+1e-10) + 1e-10

  g = img[u,v]/256
  return o,d,g

def sample_tt(basedir, nm_cams, to_test=5):
  #basedir = "/content/drive/MyDrive/github-project/My_Nerf/datasets/fufu"
  nm_testing = nm_cams//to_test
  nm_training = nm_cams - nm_testing

  poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))

  info = poses_arr[:, :-2].reshape([-1, 3, 5])#.transpose([1,2,0])
  #bds = poses_arr[:, -2:].transpose([1,0])
  z = 10 #int(torch.means(bds))

  imgs = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
          if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

  scale = 20
  hwf = torch.from_numpy(info[0,:3,-1])
  hwf[0], hwf[1] = hwf[0]*(scale/100), hwf[1]*(scale/100)
  print(int(hwf[0]))
  print(int(hwf[1]))


  poses = torch.from_numpy(info[:,:3,:4])

  Es = torch.zeros((poses.shape[0],1,4))
  Es[:,:,-1] = 1
  Es = torch.concat((poses, Es), axis = 1)

  ray_one_img = int(hwf[0]*hwf[1])

  nm_training *= ray_one_img
  nm_testing *= ray_one_img

  training = torch.ones([nm_training,9])
  testing = torch.ones([nm_testing,9])

  print(training.shape)
  print(testing.shape)
  
  off_training = 0
  off_testing = 0

  for i in range(nm_cams):
    print(i)
    
    E = Es[i].to(torch.float32)
    img = cv.imread(imgs[i])
    img = resize_img(img, scale)
    img = torch.from_numpy(cv.resize(img,(int(hwf[1]),int(hwf[0]))))

    off = 0
    for j in range(int(hwf[0])):
      for k in range(int(hwf[1])):
        u,v = j,k
        o,d,g = get_odg(img,u,v,E,hwf,z)

        if (i+1)%to_test != 0:
          # traning
          training[off_training*ray_one_img + off,:3] = o
          training[off_training*ray_one_img + off,3:6] = d
          training[off_training*ray_one_img + off,6:] = g

        else:
          # testing
          testing[off_testing*ray_one_img + off,:3] = o
          testing[off_testing*ray_one_img + off,3:6] = d
          testing[off_testing*ray_one_img + off,6:] = g

        off += 1
    if (i+1)%to_test != 0:
      off_training += 1
    else:
      off_testing += 1

  return training, testing