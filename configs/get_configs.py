import argparse

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--from_configs', type=bool, default=False)
  parser.add_argument('--dataset_name', type=str, default='images')
  parser.add_argument('--img_wh', nargs='+', type=int, default=[256,256]) #The + value for nargs gathers all command-line arguments into a list.
  
  parser.add_argument('--device', type=str, default = 'cuda')

  parser.add_argument('--batch_size', type=int, default=1024)
  parser.add_argument('--num_epochs', type=int, default=8)

  parser.add_argument('--fine_stage', type=int, default=1)

  return parser.parse_args()