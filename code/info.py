# info.py
# some utilities for printing information about the network

import functools
import operator


class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'


def print_header_columns():
    print("""
 Epoch  |  Train Loss  |  Valid Loss  |  Train / Val  |  Valid Acc  |  Dur
--------|--------------|--------------|---------------|-------------|------\
""")


def print_layer_info(nn_layers):
    for layer in nn_layers:
        output_shape = layer.get_output_shape()
        print("  {:<18}\t{:<20}\tproduces {:>7} outputs".format(
            layer.__class__.__name__,
            str(output_shape),
            str(functools.reduce(operator.mul, output_shape[1:])),
        ))


def print_epoch_info(valid_loss, best_valid_loss, valid_accuracy, train_loss, best_train_loss, epoch, duration):
    best_train = train_loss == best_train_loss
    best_valid = valid_loss == best_valid_loss
    print(" {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  "
          "|  {:>11.6f}  |  {:>9}  |  {:>3.1f}s".format(
              epoch,
              ansi.BLUE if best_train else "",
              train_loss,
              ansi.ENDC if best_train else "",
              ansi.GREEN if best_valid else "",
              valid_loss,
              ansi.ENDC if best_valid else "",
              train_loss / valid_loss,
              "{:.2f}%".format(valid_accuracy * 100),
              duration,
          ))
