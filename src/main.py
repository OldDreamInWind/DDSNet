import argparse
from train import train
from eval import eval

def get_args():
    parser = argparse.ArgumentParser(description="Parse command line arguments for model training.")
    
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or eval.')
    parser.add_argument('--input_channel', type=int, default=3, help='Number of input channels.')
    parser.add_argument('--output_channel', type=int, default=7, help='Number of output channels.')
    parser.add_argument('--device', type=str, default='cuda:0', help='String of device.')
    parser.add_argument('--dataset', type=str, default='Skin', help='Dataset to be used for training.')
    parser.add_argument('--dataset_path', type=str, default='./dataset/data', help='Folder path of dataset.')
    parser.add_argument('--image_size', type=int, default=224, help='Size of image.')
    parser.add_argument('--model', type=str, default='DDS', help='Model type to be used for training.')

    parser.add_argument('--epoch', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the model.')
    parser.add_argument('--result_path', type=str, default='result', help='Path to save results.')

    parser.add_argument('--pretrained', type=str, default='model/GC_DDS_0.0064.pth', help='Path of pretrained model.')
    parser.add_argument('--eval', type=str, default='tsne', help='Eval methods: tsne, confusion matrix, gcam or time.')
    parser.add_argument('--test_image', type=str, default='test.png', help='Path for test image.')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = get_args()

    print(f"Mode: {args.mode}")
    print(f"Input Channels: {args.input_channel}")
    print(f"Output Channels: {args.output_channel}")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset_path: {args.dataset_path}")
    print(f"Device: {args.device}")
    print(f"Image_size: {args.image_size}")
    print(f"Model: {args.model}")

    if args.mode == 'train':
        print(f"Epochs: {args.epoch}")
        print(f"Learning Rate: {args.learning_rate}")
        print(f"Result_path: {args.result_path}")

        train(args)

    elif args.mode == 'eval':
        print(f"Pretrained_path: {args.pretrained}")
        print(f"Eval_method: {args.eval}")
        if args.eval == 'gcam':
            print(f"Test_image: {args.test_image}")

        eval(args)


