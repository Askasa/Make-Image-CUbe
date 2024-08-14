import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.get_configs import get_args
#from datasets import dataset_dict

from srcs.Nerf import NerfModel
from srcs.render import render_rays, compute_accumulated_transmittance

def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, nb_epochs=int(1e5),
           nb_bins=192, H=400, W=400, fine_stage=2):

  training_loss = []
  
  for _ in tqdm(range(nb_epochs)):
    for batch in data_loader:
      #for each training point in the batch it exist in form [origin(x,y,z), direction(a1,a2,a3), gt_pixel(r,g,b)]
      ray_origins = batch[:,:3].to(device) 
      ray_directions = batch[:,3:6].to(device)
      ground_truth_px_values = batch[:,6:].to(device)
      if _ < fine_stage:
        regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, stage='coarse')
      else:
        regenerated_px_values = render_rays(nerf_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins, stage='fine')
      loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      training_loss.append(loss.item())
    scheduler.step()
    for img_index in range(3):
      test(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)

  return training_loss

@torch.no_grad()
def test(hn, hf, dataset, chunk_size=5, img_index=0, nb_bins=192, H=400,W=400):
  ray_origins = dataset[img_index * H * W: (img_index+1)* H * W, :3]
  ray_directions = dataset[img_index * H * W: (img_index+1)* H * W, 3:6]

  data = []

  for i in range(int(np.ceil(H / chunk_size))):

    ray_origins_ = ray_origins[i * W * chunk_size : (i+1) * W * chunk_size].to(device)
    ray_directions_ = ray_directions[i * W * chunk_size : (i+1) * W * chunk_size].to(device)
    regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins = nb_bins)
    #print("rpv:",regenerated_px_values)
    data.append(regenerated_px_values)
  img = torch.cat(data).data.cpu().numpy().reshape(H,W,3)

  plt.figure()
  plt.imshow(img)
  plt.savefig(f'output/novel_views_2/img_{img_index}.png', bbox_inches='tight')
  plt.close()

if __name__ == '__main__':
  # get training argument
  args = get_args()
  device = args.device

  # get dataset
  # todo - load data to correct format [img.jpg + poses + ] -> [origin + direction + ground_truth] for training 
  training_dataset = torch.from_numpy(np.load('datasets/fufu/short_cut/training_data.pkl', allow_pickle=True))
  testing_dataset = torch.from_numpy(np.load('datasets/fufu/short_cut/training_data.pkl', allow_pickle=True))
  
  
  #idx = torch.randperm(training_dataset.size(0))
  #training_dataset = training_dataset[torch.sort(idx).indices,:]
  #num_data_use = int(int(idx.size(0))/10)

  #training_dataset = training_dataset[:num_data_use,:]
  #print(testing_dataset)
  """
  idx = torch.randperm(testing_dataset.size(0))
  testing_dataset = testing_dataset[torch.sort(idx).indices,:]
  num_data_use = int(int(idx.size(0)))

  testing_dataset = testing_dataset[:num_data_use,:]
  #print(testing_dataset)
  """

  # initialize NeRF
  model = NerfModel(hidden_dim=256).to(device)
  model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

  data_loader = DataLoader(training_dataset, batch_size=16, shuffle=True) # fit data to batches

  train(model, model_optimizer, scheduler, data_loader, nb_epochs=args.num_epochs, device=device, hn=2, hf=6, nb_bins=100, H=288, W=216, fine_stage = args.fine_stage)

