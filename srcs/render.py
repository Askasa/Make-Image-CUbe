import torch
# utils
###
def compute_accumulated_transmittance(alphas):
  accumulated_transmittance = torch.cumprod(alphas,1) #cumprod get the cumulative product of alphas
  return torch.cat((torch.ones((accumulated_transmittance.shape[0],1), device=alphas.device),
                   accumulated_transmittance[:,:-1]), dim = -1)

def PDF_reverse(sigma, hn, hf):
  # x_axis: sigma [nb_samples, nb_bins]
  device = sigma.device
  nb_samples = sigma.shape[0]
  nb_bins = sigma.shape[1]

  # normalization
  sigma = torch.cumsum(sigma,-1)
  n_sigma = sigma.div((1e-10)+sigma.sum(-1).unsqueeze(-1))


  t = torch.linspace(0, hf-hn, nb_bins, device=device).expand(nb_samples, nb_bins)
  off = torch.linspace(hn, hn, nb_bins, device=device).expand(nb_samples, nb_bins)


  t = t.mul(n_sigma)+off
  
  return t

def coarse_sample(ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, nb_samples=100):
  device = ray_origins.device

  t = torch.linspace(hn,hf,nb_bins, device=device).expand(nb_samples, nb_bins)
  mid = (t[:,:-1]+t[:,1:]) / 2
  lower = torch.cat((t[:,:1],mid), -1)
  upper = torch.cat((mid, t[:,-1:]),-1)
  u = torch.rand(t.shape, device=device)
  t = lower + (upper - lower) * u


  delta = torch.cat((t[:,1:] - t[:,:-1], torch.tensor([1e10], device=device).expand(nb_samples, 1)),-1)
  
  x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
  ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0,1)
  
  return x, ray_directions, delta

def fine_sample(sigma, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, nb_samples=100):
  device = ray_origins.device

  t = PDF_reverse(sigma, hn, hf)

  mid = (t[:,:-1]+t[:,1:]) / 2
  lower = torch.cat((t[:,:1],mid), -1)
  upper = torch.cat((mid, t[:,-1:]),-1)
  u = torch.rand(t.shape, device=device)
  t = lower + (upper - lower) * u

  delta = torch.cat((t[:,1:] - t[:,:-1], torch.tensor([1e10], device=device).expand(nb_samples, 1)),-1)

  x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)
  ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0,1)
  
  return x, ray_directions, delta

  
###
def render_rays(nerf_model, ray_origins, ray_directions, hn=0, hf=0.5, nb_bins=192, stage ='coarse'):
  device = ray_origins.device
  nb_samples = ray_origins.shape[0]

  # course sampling
  x_coarse, r_d_coarse, delta_coarse = coarse_sample(ray_origins, ray_directions, hn, hf, nb_bins, nb_samples)
  x, r_d, delta = x_coarse, r_d_coarse, delta_coarse

  if stage == 'fine':
    # fine sampling
    c, sigma = nerf_model(x.reshape(-1,3), r_d.reshape(-1,3))
    sigma = sigma.reshape(nb_samples,nb_bins)

    x_fine, r_d_fine, delta_fine = fine_sample(sigma, ray_origins, ray_directions, hn, hf, nb_bins, nb_samples)

    # cancating sampling
    
    x = torch.cat((x_fine, x_coarse),-1)
    r_d = torch.cat((r_d_fine, r_d_coarse),-1)
    delta = torch.cat((delta_fine, delta_coarse),-1)
    
    x, r_d, delta = x_fine, r_d_fine,delta_fine

  #print("delta",delta)
  # final rendering
  colors, sigma = nerf_model(x.reshape(-1,3), r_d.reshape(-1,3))
  
  colors = colors.reshape(x.shape)
  sigma = sigma.reshape(x.shape[:-1])
  #print("sigma",sigma)

  alpha = 1 - torch.exp(-sigma * delta)

  #print("alpha:",alpha)
  weights = compute_accumulated_transmittance(1-alpha).unsqueeze(2) * alpha.unsqueeze(2)
  c = (weights * colors).sum(dim=1)
  weight_sum = weights.sum(-1).sum(-1)
  return c + 1 - weight_sum.unsqueeze(-1)