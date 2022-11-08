
import os
import torch
import logging
import pdb

from torch.utils.collect_env import get_pretty_env_info
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from distutils.version import LooseVersion
if LooseVersion(torch.__version__) >= LooseVersion("1.8.0"):
    # TODO: Use DistributedSampler in torch library, after
    #       this implementation is merged to release.
    #   https://github.com/pytorch/pytorch/pull/51841/
    #   'Fix DistributedSampler mem usage on large datasets'
    from torch.utils.data.distributed import DistributedSampler
    #  from uvoicespeech_denoise.dataset.data.distributed import DistributedSampler
else:
    # temporary solution to fix DistributedSampler's padding error in version 1.7.0 and below
    from dataset.sampler import UVDistributedSampler as DistributedSampler

def config_logging(args) -> None:
    # set stream output
    stream_handle = logging.StreamHandler()
    # ssa will dump a lot of DEBUG message
    stream_handle.setLevel(logging.INFO)
    # log to file only for rank 0
    if args.rank == 0:
        train_log_file = args.train_log_file
        #exp_id = os.path.basename(args.model_dir)

        if train_log_file:
            train_log_dir = os.path.dirname(train_log_file)
        else:
            train_log_dir = "logs" #os.path.join('logs', exp_id)
            train_log_name = "trainLog_%s" % (
                datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            train_log_file = os.path.join(train_log_dir, train_log_name)

        os.makedirs(train_log_dir, exist_ok=True)
        file_handle = logging.FileHandler(train_log_file)
        file_handle.setLevel(logging.WARNING)
        handlers = [file_handle, stream_handle]
    else:
        handlers = [stream_handle]

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s(line:%(lineno)d) [%(levelname)s] %(message)s',
                        handlers=handlers)

def create_summary_writer(configs):
    #rank = configs['rank']
    tensorboard_dir = configs['tensorboard_dir']

    #if rank == 0 and tensorboard_dir:
    #    return SummaryWriter(tensorboard_dir)
    #else:
    #    return None
    return SummaryWriter(tensorboard_dir)

def collect_environment_info(log_2_file):
    output = get_pretty_env_info()
    if log_2_file:
        logging.warning('{} {} {}'.format(
            '=' * 20, 'Environment Information', '=' * 20))
        info_lines = output.split('\n')
        for line in info_lines:
            logging.warning(line)
        logging.warning('-' * (20 + len('Environment Information') + 20))
    else:
        print(output)

#def load_configs(args):
#    #os.makedirs(args.output_dir, exist_ok=True)
#    print("Load configs .....")
#    configs = dict()
#    configs['model_store_path'] = args.output_dir+"/model/"
#    os.makedirs(configs['model_store_path'], exist_ok=True)
#    if not os.path.exist(configs['model_store_path']):
#        print("Can not create ",configs['model_store_path'])
#        sys.exit()
#    print(configs['model_store_path'],"created.")
#    sys.exit()
#
#    # save args parameters
#    for arg in vars(args):
#        configs[arg] = getattr(args, arg)
#
#    # setup environment before collecting environment info !!!
#    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
#
#    # save configs to model_dir/train.yaml for inference and export
#    collect_environment_info(log_2_file=True)
#
#    return configs

def backup_configs(configs):
    model_dir = configs['model_dir']
    saved_config_path = os.path.join(model_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)

