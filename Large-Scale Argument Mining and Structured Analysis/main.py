from trainer import JointTrainer
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="accelerate.utils.torch_xla")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="论证挖掘系统训练/评估/推理")
    parser.add_argument("--mode", type=str,  default="train_eval",choices=["train", "evaluate", "train_eval", "save"],
                        help="运行模式：train（仅训练）、evaluate（仅评估）、train_eval（训练+评估）、save（保存模型）")
    return parser.parse_args()

def main():
    args = parse_args()
    joint_trainer = JointTrainer()
    
    if args.mode == "train":
        joint_trainer.train_all()
    elif args.mode == "evaluate":
        joint_trainer.evaluate_all()
    elif args.mode == "train_eval":
        joint_trainer.train_all()
        joint_trainer.evaluate_all()
        joint_trainer.save_all_models()
    elif args.mode == "save":
        joint_trainer.save_all_models()
    
    print("\n任务完成！")

if __name__ == "__main__":
    main()