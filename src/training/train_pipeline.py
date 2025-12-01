from src.training.train_base import train
from src.training.fine_tune_children import fine_tune_children


def main():
    print("Stage 1: training on full UTKFace")
    base_model, base_history, base_eval = train()

    print("\nStage 2: fine-tuning on children + replay")
    tuned_model, hist_child, hist_mixed, eval_children = fine_tune_children()


if __name__ == "__main__":
    main()
