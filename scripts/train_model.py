"""
Train the CNN-BiLSTM-Attention model.
"""

from training.trainer import Trainer


def main():

    trainer = Trainer()

    trainer.compile()

    trainer.train()


if __name__ == "__main__":

    main()