from sdr_utils import *


def main():
    bin_msg = binaryify("Hello World")
    train = pulse_train(bin_msg)
    filtered = raised_cos_filter(train, 0.35)
    print(filtered)
    print(len(filtered))
    

if __name__ == "__main__":
    main()