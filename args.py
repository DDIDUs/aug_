import argparse

def build_parser():
    
    parser = argparse.ArgumentParser(description="Run single model")
    
    parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
    
    # Run Config
    parser.add_argument('-dataset', type=str, default='cifar10', choices=['mnist', 'fmnist', 'cifar10', 'cifar100'], help='Dataset')
    parser.add_argument('-augmentation', action='store_true', help='Boolean flag')
    parser.add_argument('-repeat_num', type=int, default=3, help="The number of times you want to run the experiment")
    parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
    parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
    parser.add_argument('-imageshowflag', type=bool, default=False, choices=[True, False], help='To check image')
    parser.add_argument('-w_base', type=int, default=32, help='shake-shake model width base')
    parser.set_defaults(outputs=True)
    
    # Device Configuration
    parser.add_argument('-gpu', type=int, default=0, help='Specify the gpu to use')
    parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
    parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
    parser.set_defaults(save_model=False)
    
    #Model Parameters
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('-epochs', type=int, default=120, help='Maximum # of training epochs')
    parser.add_argument('-patience', type=int, default=30, help='Early stop')
    parser.add_argument('-train_mode', type=int, default=5, choices=[1, 2, 3, 4, 5, 6], help='Select train mode')
    parser.add_argument('-train_model', type=str, default='vggnet', choices=['vggnet', 'resnet', 'densenet', 'pyramidnet', 'shakeshake'], help= 'Select neural network model')
    
    return parser
