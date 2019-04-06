
# Training Dataset Paths
AS_TRAIN = "../resources/icwb2-data/training/as_training_simplified.utf8"
CITYU_TRAIN = "../resources/icwb2-data/training/cityu_training_simplified.utf8"
MSR_TRAIN = "../resources/icwb2-data/training/msr_training.utf8"
PKU_TRAIN = "../resources/icwb2-data/training/pku_training.utf8"

# Dev Dataset Paths
AS_DEV = "../resources/icwb2-data/gold/as_testing_gold_simplified.utf8"
CITYU_DEV = "../resources/icwb2-data/gold/cityu_test_gold_simplified.utf8"
MSR_DEV = "../resources/icwb2-data/gold/msr_test_gold.utf8"
PKU_DEV = "../resources/icwb2-data/gold/pku_test_gold.utf8"

TRAINING_SETS = [AS_TRAIN, CITYU_TRAIN, MSR_TRAIN, PKU_TRAIN]
DEV_SETS = [AS_DEV, CITYU_DEV, MSR_DEV, PKU_DEV]
NAMES = ["as", "cityu", "msr", "pku"]


def remove_spaces_from_string(s) -> str:
    return "".join(s.split())

def string_to_BIES(s) -> str:
    # Booleans to denote if previous character was a whitespace or not.
    # Prev_whitespace initialized to True because of logic.

    word_array = s.split()
    bies_array = []
    for word in word_array:
        word_length = len(word)
        if word_length == 1:
            bies_array.append('S')
        elif word_length == 2:
            bies_array.append('BE')
        else:
            bies = ['B']
            for i in range(1, len(word) - 1):
                bies.append('I')
            bies.append('E')
            bies_array.append("".join(bies))


    return "".join(bies_array)

def read_file(original_file_path, input_file_path, label_file_path):
    with open(original_file_path, 'r', encoding='utf-8') as original_file, open(input_file_path, 'a', encoding='utf-8') as input_file, open(label_file_path, 'a', encoding='utf-8') as label_file:
        for line in original_file:
            input_file.write(remove_spaces_from_string(line))
            label_file.write(string_to_BIES(line))
            input_file.write("\n")
            label_file.write("\n")



if __name__ == '__main__':
    for i in range(len(TRAINING_SETS)):
        read_file(TRAINING_SETS[i], "../train/input/" + NAMES[i] + ".txt", "../train/labels/" + NAMES[i] + ".txt")
    for i in range(len(DEV_SETS)):
        read_file(DEV_SETS[i], "../dev/input/" + NAMES[i] + ".txt", "../dev/labels/" + NAMES[i] + ".txt")


