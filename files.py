def parse_dataset(path):
    """
    Используется для парсинга dataset config файла
    :param path:
    :return:
    """
    with open(path, 'r+', encoding='cp1251') as f:
        lines = [line.rstrip() for line in f]

        last_file, result = '', {}
        for line in lines:
            line = line.replace('.frame', '.bmp')
            if '\\' in line:
                result[line] = []
                last_file = line
            else:
                result[last_file].append(line.split(', '))

        return result


def parse_classes(path):
    """
    Используется для парсинга dataset classes config файла
    :param path:
    :return:
    """
    with open(path, 'r+', encoding='cp1251') as f:
        lines = [line.rstrip() for line in f]
        class_names, class_ids = [], []
        for line in lines:
            parts = line.split(', ')
            class_names.append(parts[len(parts) - 1])
            class_ids.append(parts[0].split()[1])

    return class_names, class_ids
