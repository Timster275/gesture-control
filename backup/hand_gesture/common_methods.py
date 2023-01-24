def load_class_names(filename):
    class_names = []
    with open(filename, 'r') as f:
        class_names = f.read().split('\n')
    return class_names
