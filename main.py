from utils import *
from train import Train
from datetime import datetime
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import warnings
warnings.filterwarnings("ignore")


def main(args):
    log_file_name = 'experiment_log-%s' % (datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    logdirname = args.dataset + 'logs'
    if os.path.exists(logdirname) is False:
        os.makedirs(logdirname)
    log_path = './' + logdirname + '/{}.log'.format(log_file_name)
    logging.basicConfig(
        filename=os.path.join(log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info("Parameter settings:" + str(args))
    logger.info("training beginning...")


    output_file = open(f"{args.dataset}_result.csv", "a+")
    output_file.write('Settings: {}'.format(args))
    output_file.write('\n')
    
    setup_seed(args.seed)
    prev_time = datetime.now()
    acc = Train(args)
    logger.info("AVE_ACC:" + str(acc.item()))
    
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)

if __name__ == '__main__':
    args = build_args()
    args = load_best_configs(args, "configs.yml")
    main(args)