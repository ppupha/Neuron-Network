NUMBER_COUNT = 10 # количество букв


class Data:
    def __init__(self, inputs, result=None):
        self.inputs = inputs
        self.results = [-1] * NUMBER_COUNT
        if result is not None:
            self.results[result] = 1

# Обучающий датасет с русскими буквами (они тут не все, нет ц, щ, например)
values = {
    '111111000110001100011000110001100011000111111': 0, 
    '000010000100001000010000100001000010000100001': 1, 
    '111110000100001000011111110000100001000011111': 2,
    '111110000100001000011111100001000010000111111': 3,
    '100011000110001100011111100001000010000100001': 4,
    '111111000010000100001111100001000010000111111': 5,
    '111111000010000100001111110001100011000111111': 6,
    '111110000100001000010000100001000010000100001': 7,
    '111111000110001100011111110001100011000111111': 8,
    '111111000110001100011111100001000010000111111': 9,
}

values = {
    '111001001001001': 7,
    '111101111101111': 8,
    '001001001001001': 1,
    '111101101101111': 0,
    '111001111100111': 2,
    '101101111001001': 4,
    '111100111001111': 5,
    '111001011001111': 3,
    '111100111101111': 6,
    '111101111001111': 9
}



dataset = [
    Data([int(ch) for ch in string], result)
    for string, result in values.items()
]

# Датасет для тестирования
test_values = {
    '111111000110001100011000110001100011000111111': 0, 
    '111111000110001100011000110001100011000111111': 0,
    '000010000100001000010000100001000010000100001': 1, 
    '111110000100001000011111110000100001000011111': 2,
    '111110000100001000011111110000100001000011111': 2,
    '111110000100001000011111100001000010000111111': 3,
    '100011000110001100011111100001000010000100001': 4,
    '000010000100001000010000100001000010000100001': 1,
    '000010000100001000010000100001000010000100001': 1,
    '111111000010000100001111100001000010000111111': 5,
    '111111000110001100011000110001100011000111111': 0,
    '111111000010000100001111110001100011000111111': 6,
    '111110000100001000011111110000100001000011111': 2,
    '111111000110001100011000110001100011000111111': 0,
    '000010000100001000010000100001000010000100001': 1,
    '111110000100001000010000100001000010000100001': 7,
    '111111000110001100011111110001100011000111111': 8,
    '111110000100001000011111110000100001000011111': 2,
    '111111000110001100011111100001000010000111111': 9,
}

test_values = {
    '111111101101111': 0,
    '011001001001001': 1,
    '011001111100111': 2,
    '111001111000111': 2,
    '101101111001001': 4,
    '100101111001001': 4,
    '011100101001011': 5,
    '001001111100111': 2,
    '001001111000111': 2,
    '111001111001111': 3,
    '111001011001111': 3,
    '011101111001001': 4,
    '111100111001111': 5,
    '001001001001001': 1,
    '111001001001001': 1,
    '010100111001111': 5,
    '110100111001111': 5,
    '111001011001001': 7,
    '111001111001001': 7,
    '111001111010100': 7,
    '011001111010100': 7,
    '101010101010100': None,
    '010101010101001': None,
    '100100111101101': None,
    '010010010010010': None,
    '110100111101111': 6,
    '111100111101111': 6,
    '111111011101111': 8,
    '111101111101111': 8,
    '011111111001111': 9,
    '110101111001111': 9,
    '111101001001111': 9,
    '111101001001011': 9,
    '011101001001011': 9,
    '011101001001111': 9,
    '000000000000000': None,
    '101000011101111': None,
    '011111101110000': None,
    '010010011101001': None,
    '111101101111111': 0,
    '101101101101101': 0,
    '111111111111101': None,
    '011110111110111': None,
    '111111111111111': None,
    '111111111000000': None,
    '000000011111111': None,
    '010101010101010': None,
    '101010101010101': None
}




test_dataset = [
    Data([int(ch) for ch in string], result)
    for string, result in test_values.items()
]