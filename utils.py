import math
from multiprocessing import Value

def check_stage(dataset, epoch, stage=2): # epoch start from 1
    if epoch == 0:
        raise ValueError(f"Epoch start from 1.")
    stg = stage
    change_pts = []
    if dataset == "Cifar10":
        change_pts = [70, 180]
    elif dataset == "Cifar100":
        change_pts = [95, 175]
    elif dataset == "ImageNet":
        change_pts = [95, 165]
    elif dataset in ["Fashion-MNIST", "adult", "higgs", "volkert"]:
        change_pts = [15]
        stg -= 1
        if stg - 1 > len(change_pts):
            # only support stage 3 in LCBench
            raise ValueError(f"Only support stage 3 in LCBench.")
    elif dataset in ["jasmine", "vehicle"]:
        change_pts = [20]
        stg -= 1
        if stg - 1 > len(change_pts):
            # only support stage 3 in LCBench
            raise ValueError(f"Only support stage 3 in LCBench.")
    else:
        raise ValueError(f"Unknown datasest {dataset}.")

    if stg >= 2 and stg <= 3:   # stage 2 and 3 all use validation
        return epoch > change_pts[stg-2]
    else:
        raise ValueError(f"Do not support stage {stage}.")

def get_seeds():
    return [777, 888, 999]

def get_window_size(dataset):
    if dataset in ["Cifar10", "Cifar100", "ImageNet"]:
        return 5
    elif dataset in ["Fashion-MNIST", "jasmine", "vehicle",
                     "adult", "higgs", "volkert"]:
        return 2
    else:
        raise ValueError(f"Unknown dataset {dataset}.")

def get_cta_dir(task):
    print(f"task = {task}")
    if task == 'acc_loss':
        criterias = ['train_accuracy', 'train_losses', 'valid_accuracy', 'valid_losses']
        direction = ['Max', 'Min', 'Max', 'Min']
    elif task == 'acc':
        criterias = ['train_accuracy', 'valid_accuracy']
        direction = ['Max', 'Max']
    elif task == 'loss':
        criterias = ['train_losses', 'valid_losses']
        direction = ['Min', 'Min']
    elif task == 'adp_stage_loss_nsb':
        criterias = ['stage_2_loss', 'stage_3_loss']    # change to val loss at stage 2 or stage 3
        direction = ['Min', 'Min']
    elif task == 'adp_stage_loss_lcb':
        criterias = ['stage_3_loss']    # change to val loss at stage 3
        direction = ['Min']
    elif task == "loss_seed":
        criterias = ['train_losses_seed', 'valid_losses_seed']
        direction = ['Min', 'Min']
    elif task == "win_loss":
        criterias = ['win_mu_train_loss', 'win_mu_valid_loss']
        direction = ['Min', 'Min']
    elif task == "stage_win_loss_nsb":
        criterias = ['stage_2_win_loss', 'stage_3_win_loss']
        direction = ['Min', 'Min']
    elif task == "stage_win_loss_lcb":
        criterias = ['stage_3_win_loss', 'stage_3_win_train_loss'] 
        # stage_3_win_loss means stage 3 use validtion loss to compute window smoothing metric
        # stage_3_win_train_loss means stage 3 use training loss to compute window smoothing metric
        direction = ['Min', 'Min']
    elif task == "stage_loss_seed_nsb":
        criterias = ['stage_2_loss_seed', 'stage_3_loss_seed']
        direction = ['Min', 'Min']
    elif task == "stage_loss_seed_lcb":
        criterias = ['stage_3_loss_seed']
        direction = ['Min']
    elif task == 'acc_loss_seed':
        criterias = ['train_accuracy_seed', 'train_losses_seed', 'valid_accuracy_seed', 'valid_losses_seed']
        direction = ['Max', 'Min', 'Max', 'Min']
    elif task == 'loss_change_rate':
        criterias = ['train_losses', 'train_loss_change_rate', 'valid_losses', 'valid_loss_change_rate']
        direction = ['Min', 'Max', 'Min', 'Max']
    elif task == 'loss_change_rate_min':
        criterias = ['train_losses', 'train_loss_change_rate', 'valid_losses', 'valid_loss_change_rate']
        direction = ['Min'] * 4
    elif task == 'win_5_train':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy',
                    'train_losses', 'win_5_mu_train_loss', 'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss']
        direction = ['Max'] * 4 + ['Min'] * 4
    elif task == 'win_5_valid':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy',
                    'valid_losses', 'win_5_mu_valid_loss', 'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss']
        direction = ['Max'] * 4 + ['Min'] * 4

    elif task == 'dyn_win_size':
        criterias = []
        direction = []
        for winsize in range(2, 7):
            criterias.append(f'win_{winsize}_mu_train_accuracy')
            criterias.append(f'win_{winsize}_mu-sig_train_accuracy')
            # criterias.append(f'win_{winsize}_mu+sig_train_accuracy')
            criterias.append(f'win_{winsize}_mu_train_loss')
            criterias.append(f'win_{winsize}_mu-sig_train_loss')
            # criterias.append(f'win_{winsize}_mu+sig_train_loss')
            criterias.append(f'win_{winsize}_mu_valid_accuracy')
            criterias.append(f'win_{winsize}_mu-sig_valid_accuracy')
            # criterias.append(f'win_{winsize}_mu+sig_valid_accuracy')
            criterias.append(f'win_{winsize}_mu_valid_loss')
            criterias.append(f'win_{winsize}_mu-sig_valid_loss')
            # criterias.append(f'win_{winsize}_mu+sig_valid_loss')
            # direction += (['Max', 'Max', 'Max', 'Min', 'Min', 'Min'] * 2)
            direction += (['Max', 'Max', 'Min', 'Min'] * 2)

    elif task == 'win_head_train':
        criterias = ['train_accuracy', 'win_head_mu_train_accuracy', 'win_head_mu+sig_train_accuracy', 'win_head_mu-sig_train_accuracy',
                     'train_losses', 'win_head_mu_train_loss', 'win_head_mu+sig_train_loss', 'win_head_mu-sig_train_loss']
        direction = ['Max'] * 4 + ['Min'] * 4
    elif task == 'win_head_valid':
        criterias = ['valid_accuracy', 'win_head_mu_valid_accuracy', 'win_head_mu+sig_valid_accuracy', 'win_head_mu-sig_valid_accuracy',
                     'valid_losses', 'win_head_mu_valid_loss', 'win_head_mu+sig_valid_loss', 'win_head_mu-sig_valid_loss']
        direction = ['Max'] * 4 + ['Min'] * 4

    elif task == 'wgh_valid_lcr_valid_loss':
        # fixed weight throughout the searching process
        # lcr is calculated by two consecutive epochs
        # wgh_1 * normalized(valid loss change rate) + wgh_2 * normalized(valid loss / valid accuracy)
        criterias = ['valid_loss_change_rate', 'valid_loss_change_rate_min', 'valid_losses']
        direction = ['Max', 'Min', 'Min']
        cta = 'wgh_valid_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_valid_loss'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_valid_loss'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_valid_lcr_valid_accuracy': 
        criterias = ['valid_loss_change_rate', 'valid_loss_change_rate_min', 'valid_accuracy']
        direction = ['Max', 'Min', 'Max']
        cta = 'wgh_valid_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_valid_accuracy'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_valid_accuracy'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_valid_lcr_train_loss': 
        criterias = ['valid_loss_change_rate', 'valid_loss_change_rate_min', 'train_losses']
        direction = ['Max', 'Min', 'Min']
        cta = 'wgh_valid_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_train_loss'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_train_losss'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_valid_lcr_train_accuracy': 
        criterias = ['valid_loss_change_rate', 'valid_loss_change_rate_min', 'train_accuracy']
        direction = ['Max', 'Min', 'Max']
        cta = 'wgh_valid_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_train_accuracy'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_train_accuracy'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_train_lcr_valid_loss': 
        criterias = ['train_loss_change_rate', 'train_loss_change_rate_min', 'valid_losses']
        direction = ['Max', 'Min', 'Min']
        cta = 'wgh_train_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_valid_loss'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_valid_loss'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_train_lcr_valid_accuracy': 
        criterias = ['train_loss_change_rate', 'train_loss_change_rate_min', 'valid_accuracy']
        direction = ['Max', 'Min', 'Max']
        cta = 'wgh_train_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_valid_accuracy'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_valid_accuracy'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_train_lcr_train_loss': 
        criterias = ['train_loss_change_rate', 'train_loss_change_rate_min', 'train_losses']
        direction = ['Max', 'Min', 'Min']
        cta = 'wgh_train_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_train_loss'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_train_loss'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2
    elif task == 'wgh_train_lcr_train_accuracy': 
        criterias = ['train_loss_change_rate', 'train_loss_change_rate_min', 'train_accuracy']
        direction = ['Max', 'Min', 'Max']
        cta = 'wgh_train_lcr'
        for i in range(1, 10):
            gen_cta = cta + f'_{int(i)}_{int(10-i)}_train_accuracy'
            gen_cta_min = cta + f'_min_{int(i)}_{int(10-i)}_train_accuracy'
            criterias += [gen_cta, gen_cta_min]
            direction += ['Min'] * 2

    elif task == 'dyn_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy',
                     'dyn_win_5_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy',
                     'dyn_win_5_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                     'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss',
                     'dyn_win_5_train_loss']
        direction = ['Min'] * 5
    elif task == 'dyn_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                     'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss',
                     'dyn_win_5_valid_loss']
        direction = ['Min'] * 5
    

    elif task == 'dyn_wgh_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy',
                     'dyn_wgh_win_5_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_wgh_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy',
                     'dyn_wgh_win_5_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_wgh_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                     'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss',
                     'dyn_wgh_win_5_train_loss']
        direction = ['Min'] * 5
    elif task == 'dyn_wgh_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                     'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss',
                     'dyn_wgh_win_5_valid_loss']
        direction = ['Min'] * 5
        
        # First 50 epochs with mu+-sig, then only mu
    
    elif task == 'dyn_win_size_dyn_sig_train_accuracy':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_sig_win_{win}_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_size_dyn_sig_valid_accuracy':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_sig_win_{win}_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_size_dyn_sig_train_loss':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_sig_win_{win}_train_loss']
        direction = ['Min'] * 5
    elif task == 'dyn_win_size_dyn_sig_valid_loss':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_sig_win_{win}_valid_loss']
        direction = ['Min'] * 5

    # Only the weight of sigma changes with budgets. Linear trend.
    elif task == 'dyn_sig_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy',
                     'dyn_sig_win_5_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_sig_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy',
                     'dyn_sig_win_5_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_sig_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                     'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss',
                     'dyn_sig_win_5_train_loss']
        direction = ['Min'] * 5
    elif task == 'dyn_sig_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                     'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss',
                     'dyn_sig_win_5_valid_loss']
        direction = ['Min'] * 5
    
    elif task == 'dyn_win_size_dyn_log_sig_train_accuracy':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_log_sig_win_{win}_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_size_dyn_log_sig_valid_accuracy':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_log_sig_win_{win}_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_win_size_dyn_log_sig_train_loss':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_log_sig_win_{win}_train_loss']
        direction = ['Min'] * 5
    elif task == 'dyn_win_size_dyn_log_sig_valid_loss':
        criterias = []
        for win in range(2, 7):
            criterias += [f'dyn_log_sig_win_{win}_valid_loss']
        direction = ['Min'] * 5

    # Only the weight of sigma changes with budgets. Logarithm trend.
    elif task == 'dyn_log_sig_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy',
                     'dyn_log_sig_win_5_train_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_log_sig_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy',
                     'dyn_log_sig_win_5_valid_accuracy']
        direction = ['Max'] * 5
    elif task == 'dyn_log_sig_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                    #  'win_5_mu+sig_train_loss', 
                     'win_5_mu-sig_train_loss',
                     'dyn_log_sig_win_5_train_loss']
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_log_sig_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                    #  'win_5_mu+sig_valid_loss', 
                     'win_5_mu-sig_valid_loss',
                     'dyn_log_sig_win_5_valid_loss']
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_win_size_dyn_exp_sig_train_accuracy':
        criterias = []
        for win in range(2, 7):
            for exp in range(5, 25, 5):
                criterias += [f'dyn_exp_{exp}_sig_win_{win}_train_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_dyn_exp_sig_valid_accuracy':
        criterias = []
        for win in range(2, 7):
            for exp in range(5, 25, 5):
                criterias += [f'dyn_exp_{exp}_sig_win_{win}_valid_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_dyn_exp_sig_train_loss':
        criterias = []
        for win in range(2, 7):
            for exp in range(5, 25, 5):
                criterias += [f'dyn_exp_{exp}_sig_win_{win}_train_loss']
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_win_size_dyn_exp_sig_valid_loss':
        criterias = []
        for win in range(2, 7):
            for exp in range(5, 25, 5):
                criterias += [f'dyn_exp_{exp}_sig_win_{win}_valid_loss']
        direction = ['Min'] * len(criterias)
    
    # Only the weight of sigma changes with budgets. Minus exponential trend.
    elif task == 'dyn_exp_sig_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy']
        for exp in range(5, 25, 5):
            criterias.append(f"dyn_exp_{exp}_sig_win_5_train_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_exp_sig_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy']
        for exp in range(5, 25, 5):
            criterias.append(f"dyn_exp_{exp}_sig_win_5_valid_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_exp_sig_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                    #  'win_5_mu+sig_train_loss', 
                     'win_5_mu-sig_train_loss']
        for exp in range(5, 25, 5):
            criterias.append(f"dyn_exp_{exp}_sig_win_5_train_loss")
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_exp_sig_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                    #  'win_5_mu+sig_valid_loss', 
                     'win_5_mu-sig_valid_loss']
        for exp in range(5, 25, 5):
            criterias.append(f"dyn_exp_{exp}_sig_win_5_valid_loss")
        direction = ['Min'] * len(criterias)
    

    elif task == 'dyn_win_size_dyn_sqrt_sig_train_accuracy':
        criterias = []
        for win in range(2, 7):
            for sqrt in range(1, 6):
                criterias += [f'dyn_sqrt_{sqrt}_sig_win_{win}_train_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_dyn_sqrt_sig_valid_accuracy':
        criterias = []
        for win in range(2, 7):
            for sqrt in range(1, 6):
                criterias += [f'dyn_sqrt_{sqrt}_sig_win_{win}_valid_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_dyn_sqrt_sig_train_loss':
        criterias = []
        for win in range(2, 7):
            for sqrt in range(1, 6):
                criterias += [f'dyn_sqrt_{sqrt}_sig_win_{win}_train_loss']
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_win_size_dyn_sqrt_sig_valid_loss':
        criterias = []
        for win in range(2, 7):
            for sqrt in range(1, 6):
                criterias += [f'dyn_sqrt_{sqrt}_sig_win_{win}_valid_loss']
        direction = ['Min'] * len(criterias)

    # Only the weight of sigma changes with budgets. Reverse sqrt trend.
    elif task == 'dyn_sqrt_sig_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy']
        for sqrt in range(1, 6):
            criterias.append(f"dyn_sqrt_{sqrt}_sig_win_5_train_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sqrt_sig_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy']
        for sqrt in range(1, 6):
            criterias.append(f"dyn_sqrt_{sqrt}_sig_win_5_valid_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sqrt_sig_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                     'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss']
        for sqrt in range(1, 6):
            criterias.append(f"dyn_sqrt_{sqrt}_sig_win_5_train_loss")
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_sqrt_sig_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                     'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss']
        for sqrt in range(1, 6):
            criterias.append(f"dyn_sqrt_{sqrt}_sig_win_5_valid_loss")
        direction = ['Min'] * len(criterias)
    
    elif task == 'mix_dyn_sig_win_5_train_accuracy':
        criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                     'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy', 'dyn_wgh_win_5_train_accuracy'
                     'dyn_sig_win_5_train_accuracy', 'dyn_exp_5_sig_win_5_train_accuracy', 
                     'dyn_log_sig_win_5_train_accuracy', 'dyn_sqrt_5_sig_win_5_valid_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'mix_dyn_sig_win_5_valid_accuracy':
        criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                     'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy', 'dyn_wgh_win_5_valid_accuracy',
                     'dyn_sig_win_5_valid_accuracy', 'dyn_exp_5_sig_win_5_valid_accuracy', 
                     'dyn_log_sig_win_5_valid_accuracy', 'dyn_sqrt_5_sig_win_5_valid_accuracy']
        direction = ['Max'] * len(criterias)
    elif task == 'mix_dyn_sig_win_5_train_loss':
        criterias = ['train_losses', 'win_5_mu_train_loss', 
                     'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss', 'dyn_wgh_win_5_train_loss',
                     'dyn_sig_win_5_train_loss', 'dyn_exp_5_sig_win_5_train_loss', 
                     'dyn_log_sig_win_5_train_loss', 'dyn_sqrt_5_sig_win_5_train_loss']
        direction = ['Min'] * len(criterias)
    elif task == 'mix_dyn_sig_win_5_valid_loss':
        criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                     'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss', 'dyn_wgh_win_5_valid_loss',
                     'dyn_sig_win_5_valid_loss', 'dyn_exp_5_sig_win_5_valid_loss', 
                     'dyn_log_sig_win_5_valid_loss', 'dyn_sqrt_5_sig_win_5_valid_loss']
        direction = ['Min'] * len(criterias)

    elif task.startswith('fix'):
        if 'train_accuracy' in task:
            criterias = ['train_accuracy', 'win_5_mu_train_accuracy', 
                        'win_5_mu+sig_train_accuracy', 'win_5_mu-sig_train_accuracy']
            for i in range(5, 30, 5):
                criterias.append(f'fix_{i}_win_5_train_accuracy')
            direction = ['Max'] * 9
        elif 'valid_accuracy' in task:
            criterias = ['valid_accuracy', 'win_5_mu_valid_accuracy', 
                        'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_valid_accuracy']
            for i in range(5, 30, 5):
                criterias.append(f'fix_{i}_win_5_valid_accuracy')
            direction = ['Max'] * 9
        elif 'train_loss' in task:
            criterias = ['train_losses', 'win_5_mu_train_loss', 
                        'win_5_mu+sig_train_loss', 'win_5_mu-sig_train_loss']
            for i in range(5, 30, 5):
                criterias.append(f'fix_{i}_win_5_train_loss')
            direction = ['Min'] * 9
        elif 'valid_loss' in task:
            criterias = ['valid_losses', 'win_5_mu_valid_loss', 
                        'win_5_mu+sig_valid_loss', 'win_5_mu-sig_valid_loss']
            for i in range(5, 30, 5):
                criterias.append(f'fix_{i}_win_5_valid_loss')
            direction = ['Min'] * 9

    elif task == 'dyn_win_size_train_accuracy':
        criterias = ['train_accuracy']
        for win in range(2, 6):
            criterias.append(f'win_{win}_mu_train_accuracy')
            criterias.append(f'win_{win}_mu+sig_train_accuracy')
            criterias.append(f'win_{win}_mu-sig_train_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_valid_accuracy':
        criterias = ['valid_accuracy']
        for win in range(2, 6):
            criterias.append(f'win_{win}_mu_valid_accuracy')
            criterias.append(f'win_{win}_mu+sig_valid_accuracy')
            criterias.append(f'win_{win}_mu-sig_valid_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_win_size_train_loss':
        criterias = ['train_losses']
        for win in range(2, 6):
            criterias.append(f'win_{win}_mu_train_loss')
            criterias.append(f'win_{win}_mu+sig_train_loss')
            criterias.append(f'win_{win}_mu-sig_valid_loss')
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_win_size_valid_loss':
        criterias = ['valid_losses']
        for win in range(2, 6):
            criterias.append(f'win_{win}_mu_valid_loss')
            criterias.append(f'win_{win}_mu+sig_valid_loss')
            criterias.append(f'win_{win}_mu-sig_valid_loss')
        direction = ['Min'] * len(criterias)
    
    elif task == 'dyn_wgh_dyn_win_train_accuracy':
        criterias = ['train_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_accuracy')
            criterias.append(f'win_{size}_mu-sig_train_accuracy')
            criterias.append(f'win_{size}_mu+sig_train_accuracy')
            criterias.append(f'dyn_wgh_win_{size}_train_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_wgh_dyn_win_valid_accuracy':
        criterias = ['valid_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_accuracy')
            criterias.append(f'win_{size}_mu-sig_valid_accuracy')
            criterias.append(f'win_{size}_mu+sig_valid_accuracy')
            criterias.append(f'dyn_wgh_win_{size}_valid_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_wgh_dyn_win_train_loss':
        criterias = ['train_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_loss')
            criterias.append(f'win_{size}_mu-sig_train_loss')
            criterias.append(f'win_{size}_mu+sig_train_loss')
            criterias.append(f'dyn_wgh_win_{size}_train_loss')
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_wgh_dyn_win_valid_loss':
        criterias = ['valid_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_loss')
            criterias.append(f'win_{size}_mu-sig_valid_loss')
            criterias.append(f'win_{size}_mu+sig_valid_loss')
            criterias.append(f'dyn_wgh_win_{size}_valid_loss')
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_sig_dyn_win_train_accuracy':
        criterias = ['train_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_accuracy')
            criterias.append(f'win_{size}_mu-sig_train_accuracy')
            criterias.append(f'win_{size}_mu+sig_train_accuracy')
            criterias.append(f'dyn_sig_win_{size}_train_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sig_dyn_win_valid_accuracy':
        criterias = ['valid_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_accuracy')
            criterias.append(f'win_{size}_mu-sig_valid_accuracy')
            criterias.append(f'win_{size}_mu+sig_valid_accuracy')
            criterias.append(f'dyn_sig_win_{size}_valid_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sig_dyn_win_train_loss':
        criterias = ['train_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_loss')
            criterias.append(f'win_{size}_mu-sig_train_loss')
            criterias.append(f'win_{size}_mu+sig_train_loss')
            criterias.append(f'dyn_sig_win_{size}_train_loss')
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_sig_dyn_win_valid_loss':
        criterias = ['valid_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_loss')
            criterias.append(f'win_{size}_mu-sig_valid_loss')
            criterias.append(f'win_{size}_mu+sig_valid_loss')
            criterias.append(f'dyn_sig_win_{size}_valid_loss')
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_log_sig_dyn_win_train_accuracy':
        criterias = ['train_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_accuracy')
            criterias.append(f'win_{size}_mu-sig_train_accuracy')
            criterias.append(f'win_{size}_mu+sig_train_accuracy')
            criterias.append(f'dyn_log_sig_win_{size}_train_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_log_sig_dyn_win_valid_accuracy':
        criterias = ['valid_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_accuracy')
            criterias.append(f'win_{size}_mu-sig_valid_accuracy')
            criterias.append(f'win_{size}_mu+sig_valid_accuracy')
            criterias.append(f'dyn_log_sig_win_{size}_valid_accuracy')
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_log_sig_dyn_win_train_loss':
        criterias = ['train_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_loss')
            criterias.append(f'win_{size}_mu-sig_train_loss')
            criterias.append(f'win_{size}_mu+sig_train_loss')
            criterias.append(f'dyn_log_sig_win_{size}_train_loss')
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_log_sig_dyn_win_valid_loss':
        criterias = ['valid_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_loss')
            criterias.append(f'win_{size}_mu-sig_valid_loss')
            criterias.append(f'win_{size}_mu+sig_valid_loss')
            criterias.append(f'dyn_log_sig_win_{size}_valid_loss')
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_exp_sig_dyn_win_train_accuracy':
        criterias = ['train_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_accuracy')
            criterias.append(f'win_{size}_mu-sig_train_accuracy')
            criterias.append(f'win_{size}_mu+sig_train_accuracy')
            for exp in range(5, 25, 5):
                criterias.append(f"dyn_exp_{exp}_sig_win_{size}_train_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_exp_sig_dyn_win_valid_accuracy':
        criterias = ['valid_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_accuracy')
            criterias.append(f'win_{size}_mu-sig_valid_accuracy')
            criterias.append(f'win_{size}_mu+sig_valid_accuracy')
            for exp in range(5, 25, 5):
                criterias.append(f"dyn_exp_{exp}_sig_win_{size}_valid_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_exp_sig_dyn_win_train_loss':
        criterias = ['train_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_loss')
            criterias.append(f'win_{size}_mu-sig_train_loss')
            criterias.append(f'win_{size}_mu+sig_train_loss')
            for exp in range(5, 25, 5):
                criterias.append(f"dyn_exp_{exp}_sig_win_{size}_train_loss")
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_exp_sig_dyn_win_valid_loss':
        criterias = ['valid_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_loss')
            criterias.append(f'win_{size}_mu-sig_valid_loss')
            criterias.append(f'win_{size}_mu+sig_valid_loss')
            for exp in range(5, 25, 5):
                criterias.append(f"dyn_exp_{exp}_sig_win_{size}_valid_loss")
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_sqrt_sig_dyn_win_train_accuracy':
        criterias = ['train_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_accuracy')
            criterias.append(f'win_{size}_mu-sig_train_accuracy')
            criterias.append(f'win_{size}_mu+sig_train_accuracy')
            for sqrt in range(1, 6):
                criterias.append(f"dyn_sqrt_{sqrt}_sig_win_{size}_train_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sqrt_sig_dyn_win_valid_accuracy':
        criterias = ['valid_accuracy']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_accuracy')
            criterias.append(f'win_{size}_mu-sig_valid_accuracy')
            criterias.append(f'win_{size}_mu+sig_valid_accuracy')
            for sqrt in range(1, 6):
                criterias.append(f"dyn_sqrt_{sqrt}_sig_win_{size}_valid_accuracy")
        direction = ['Max'] * len(criterias)
    elif task == 'dyn_sqrt_sig_dyn_win_train_loss':
        criterias = ['train_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_train_loss')
            criterias.append(f'win_{size}_mu-sig_train_loss')
            criterias.append(f'win_{size}_mu+sig_train_loss')
            for sqrt in range(1, 6):
                criterias.append(f"dyn_sqrt_{sqrt}_sig_win_{size}_train_loss")
        direction = ['Min'] * len(criterias)
    elif task == 'dyn_sqrt_sig_dyn_win_valid_loss':
        criterias = ['valid_losses']
        for size in range(2, 4):
            criterias.append(f'win_{size}_mu_valid_loss')
            criterias.append(f'win_{size}_mu-sig_valid_loss')
            criterias.append(f'win_{size}_mu+sig_valid_loss')
            for sqrt in range(1, 6):
                criterias.append(f"dyn_sqrt_{sqrt}_sig_win_{size}_valid_loss")
        direction = ['Min'] * len(criterias)

    elif task == 'dyn_lin_train_val':
        criterias = ['train_accuracy', 'train_losses', 'valid_accuracy', 'valid_losses', 'dyn_lin_train_val_accuracy', 'dyn_lin_train_val_loss']
        direction = ['Max', 'Min', 'Max', 'Min', 'Max', 'Min']
    elif task == 'dyn_train_val':
        criterias = ['train_accuracy', 'valid_accuracy', 'train_losses', 'valid_losses', 
                     'dyn_lin_train_val_accuracy', 'dyn_lin_train_val_loss', 
                     'dyn_log_train_val_accuracy', 'dyn_log_train_val_loss', 
                     'dyn_exp_train_val_accuracy', 'dyn_exp_train_val_loss',
                     'win_5_mu_train_accuracy', 'win_5_mu_valid_accuracy', 'win_5_mu_train_loss', 'win_5_mu_valid_loss',
                     'win_5_mu+sig_train_accuracy', 'win_5_mu+sig_valid_accuracy', 'win_5_mu-sig_train_loss', 'win_5_mu-sig_valid_loss', 
                     'dyn_wgh_win_5_train_accuracy', 'dyn_wgh_win_5_valid_accuracy', 'dyn_wgh_win_5_train_loss', 'dyn_wgh_win_5_valid_loss',
                     'dyn_sig_win_5_train_accuracy', 'dyn_sig_win_5_valid_accuracy', 'dyn_sig_win_5_train_loss', 'dyn_sig_win_5_valid_loss',
                     'dyn_exp_5_sig_win_5_train_accuracy', 'dyn_exp_5_sig_win_5_valid_accuracy', 'dyn_exp_5_sig_win_5_train_loss', 'dyn_exp_5_sig_win_5_valid_loss', 
                     'dyn_log_sig_win_5_train_accuracy', 'dyn_log_sig_win_5_valid_accuracy', 'dyn_log_sig_win_5_train_loss', 'dyn_log_sig_win_5_valid_loss', 
                     'dyn_sqrt_5_sig_win_5_train_accuracy', 'dyn_sqrt_5_sig_win_5_valid_accuracy', 'dyn_sqrt_5_sig_win_5_train_loss', 'dyn_sqrt_5_sig_win_5_valid_loss']
        direction = ['Max', 'Max', 'Min', 'Min', 'Max', 'Min', 'Max', 'Min', 'Max', 'Min']
        direction += ['Max', 'Max', 'Min', 'Min'] * 7
    return criterias, direction


def calculate_weight(epoch, total_epochs):
    # Define a function that changes the weight based on the current epoch
    early_weight = max(0, 1 - (epoch / (total_epochs * 0.8)))  # Weight decreases as epochs progress
    later_weight = 1 - early_weight  # Weight increases as epochs progress

    return early_weight, later_weight

def calculate_log_weight(epoch, total_epochs):
    # Define a function that changes the weight based on the current epoch
    early_weight = max(0, 1 - math.log(epoch + 1) / math.log(total_epochs + 1))  # Weight decreases as epochs progress

    later_weight = 1 - early_weight  # Weight increases as epochs progress

    return early_weight, later_weight


def dynamic_criteria(average_accuracy, std_dev_accuracy, epoch, total_epochs):
    early_weight, later_weight = calculate_weight(epoch, total_epochs)

    # Combine the average and standard deviation with dynamic weights
    combined_score = (later_weight * average_accuracy) + (early_weight * std_dev_accuracy)

    return combined_score

def dynamic_sigma_criteria(average_accuracy, std_dev_accuracy, epoch, total_epochs):
    early_weight, _ = calculate_weight(epoch, total_epochs)

    # Combine the average and standard deviation with dynamic weights
    combined_score = average_accuracy + (early_weight * std_dev_accuracy)

    return combined_score


def dynamic_log_sigma_criteria(average_accuracy, std_dev_accuracy, epoch, total_epochs):
    early_weight, _ = calculate_log_weight(epoch, total_epochs)

    # Combine the average and standard deviation with dynamic weights
    combined_score = average_accuracy + (early_weight * std_dev_accuracy)

    return combined_score

def dynamic_exp_sigma_criteria(average_accuracy, std_dev_accuracy, epoch, decay_rate):
    sigma_weight = math.exp(- decay_rate * epoch)
    combined_score = average_accuracy + (sigma_weight * std_dev_accuracy)
    return combined_score

def dynamic_sqrt_sigma_criteria(average_accuracy, std_dev_accuracy, epoch, decay_rate):
    sigma_weight = 1 / decay_rate * math.sqrt(epoch + 1)
    combined_score = average_accuracy + (sigma_weight * std_dev_accuracy)
    return combined_score

def dynamic_train_val(train_cta, val_cta, epoch, total_epoch):
    early_weight, later_weight = calculate_weight(epoch, total_epoch)
    return early_weight * train_cta + later_weight * val_cta

def dynamic_log_train_val(train_cta, val_cta, epoch, total_epochs):
    w1, w2 = calculate_log_weight(epoch, total_epochs)
    return w1 * train_cta + w2 * val_cta

def dynamic_exp_train_val(train_cta, val_cta, epoch, decay_rate=5):
    w2 = math.exp(- decay_rate * epoch)
    w1 = 1 - w2
    return w1 * train_cta + w2 * val_cta