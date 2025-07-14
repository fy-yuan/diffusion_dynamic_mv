
import ml_collections
import torch

def get_default_configs():
    config = ml_collections.ConfigDict()

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.noise_removal = False 
    sampling.snr=0.16 # ratio in annealed Langivan sampling
    sampling.n_steps_each = 1
    sampling.probability_flow = False


    config.training = ml_collections.ConfigDict()


    config.modeling = modeling = ml_collections.ConfigDict()



    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
   
    return config


def get_ddpm_config():
    config = get_default_configs()

    config.weight_decay = None
    config.reduce_mean = True
    config.likelihood_weighting = False
    config.batch_size = 64
    config.epochs = 20

    modeling = config.modeling
    modeling.num_scales = 100
    modeling.beta_min = 0.01
    modeling.beta_max = 10
    modeling.md_type = 'vpsde'

    sampling = config.sampling
    sampling.method = 'pc'
    sampling.predictor = 'ancestral_sampling'
    sampling.corrector = 'none'

    training = config.training
    training.continuous = True
    training.seed = 123

    config.train = True
    config.save = True
    config.path = './model/ddpm_d.pkl'

    return config    