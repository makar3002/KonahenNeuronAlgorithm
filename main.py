import math
import random

from matplotlib import pyplot as plt


# Класс для нейрона, использующегося в нейронной сети Конахена.
class KonahenNeuron:
	# Коэффициент альфа по-умолчанию, соответствует значению 50/100.
	default_speed = 0.5
	# Минимальное значение веса синапса.
	min_synapse_weight = 0.01
	# Максимальное значение веса синапса.
	max_synapse_weight = 0.25

	# Конструктор.
	def __init__(self, synapse_count):
		# Инициализация весов входных синапсов нейрона (случайные значения от 0 до 1).
		self.synapse_weight_list = list(
			self.min_synapse_weight + random.random() * (self.max_synapse_weight - self.min_synapse_weight)
			for x in range(0, synapse_count)
		)
		self.speed = self.default_speed


# Класс для нейронной сети Конахена.
class KonahenClusteringAlgorithm:
	# Конструктор.
	def __init__(self, layer_config, input_count):
		self.input_count = input_count
		# Инициализация слоев сети.
		self.__layer_list = self.__init_layer_list(layer_config, input_count)

	# Метод для обучения нейронов Конахена по переданной координате,
	# меняет значения весов только на последнем слое (слое Конахена).
	# Выбрасывает исключение, если конфигурация объекта не подходит для работы с координатами.
	def learn_by_coordinates(self, training_set):
		if self.input_count != 2:
			raise Exception()

		# Выполняет 10000 эпох.
		for x in range(0, 10000):
			for i in range(0, len(training_set)):
				random_coordinate_data = training_set[int(random.random() * len(training_set))]
				# -1 - это индекс последнего слоя.
				self.__learn_layer(random_coordinate_data, -1)

	# Определяет кластер координаты.
	# Выбрасывает исключение, если конфигурация объекта не подходит для работы с координатами.
	def get_coordinate_cluster(self, coordinate_data):
		if self.input_count != 2:
			raise Exception()

		# -1 - это индекс последнего слоя.
		neuron_index = self.__get_neuron_winner_index(coordinate_data, -1)
		return neuron_index

	# Обучает слой согласно переданным значениям и номеру слоя.
	# Обучение происходит по алгоритму Конахена (только выйгравший нейрон).
	def __learn_layer(self, value_list, layer_index):
		neuron_winner_index = self.__get_neuron_winner_index(value_list, layer_index)
		neuron = self.__layer_list[layer_index][neuron_winner_index]
		self.__layer_list[layer_index][neuron_winner_index] = self.__recalculate_neuron(neuron, value_list)

	# Возвращает индекс выйгравшего нейрона по переданным значениям и номеру слоя.
	# Сравнивает значения и весовые коэффициенты синапсов всех нейронов в слое.
	def __get_neuron_winner_index(self, value_list, layer_index):
		distance_list = list()
		for neuron_index in range(0, len(self.__layer_list[layer_index])):
			neuron = self.__layer_list[layer_index][neuron_index]
			distance = 0
			for weight_index in range(0, len(neuron.synapse_weight_list)):
				distance += (value_list[weight_index] - neuron.synapse_weight_list[weight_index]) ** 2
			distance_list.append(math.sqrt(distance))

		return self.__get_min_list_index(distance_list)

	# Перераспределяет весовые коэффициенты входных синапсов нейрона согласно переданным значениям
	# согласно алгоритму Конахена.
	def __recalculate_neuron(self, neuron, value_list):
		synapse_weight_list = neuron.synapse_weight_list
		for weight_index in range(0, len(synapse_weight_list)):
			weight = synapse_weight_list[weight_index]
			# Изменение весов согласно текущей скорости, установленной для нейрона (альфа)
			synapse_weight_list[weight_index] += neuron.speed * (value_list[weight_index] - weight)

		neuron.synapse_weight_list = synapse_weight_list
		# Изменение скорости нейрона согласно формуле a = (50 - i) / 100, где i - итерация изменения. 
		# Или -1/100 каждую итерацию при значении по умолчанию 50/100. Не может быть меньше 0.
		neuron.speed -= 0.01
		if neuron.speed < 0:
			neuron.speed = 0

		return neuron

	# Возвращает индекс наименьшего значения в списке.
	def __get_min_list_index(self, target_list):
		min_index = 0
		min_value = target_list[min_index]
		for index in range(0, len(target_list)):
			value = target_list[index]
			if min_value > value:
				min_index = index
				min_value = value

		return min_index

	# Инициализирует и возвращает массив слоев из нейронов согласно переданному конфигу и кол-ву входных нейронов.
	# Конфиг это массив, каждое значение которого декларирует кол-во нейронов на слое.
	def __init_layer_list(self, layer_config, input_count):
		layer_list = list()
		for layer_index in range(0, len(layer_config)):
			if layer_index == 0:
				# Если индекс слоя 0, то это первый слой перед входным нейроном, и у него кол-во синапсов = кол-ву входов.
				synapse_count = input_count
			else:
				# Иначе столько синапсов, сколько было нейронов на предыдущем слое. 
				synapse_count = layer_config[layer_index-1]
				
			# Инициализация массива нейронов для каждого слоя.
			neuron_list = list(KonahenNeuron(synapse_count) for x in range(0, layer_config[layer_index]))
			layer_list.append(neuron_list)

		return layer_list

	# Номрализует и возвращает массив с данными координат.
	def normalize_coordinates_data(self, coordinate_set):
		x_sum = 0
		y_sum = 0
		for coordinate_index in range(0, len(coordinate_set)):
			coordinate = coordinate_set[coordinate_index]
			x_sum += coordinate.x ** 2
			y_sum += coordinate.y ** 2

		x_normal_value = math.sqrt(x_sum)
		y_normal_value = math.sqrt(y_sum)

		data_list = list()
		for coordinate_index in range(0, len(coordinate_set)):
			coordinate = coordinate_set[coordinate_index]
			data_list.append([coordinate.x / x_normal_value, coordinate.y / y_normal_value])

		return data_list


# Класс для координаты.
class Coordinate:
	def __init__(self, x, y):
		self.x = x
		self.y = y


# Инициализация объекта для алгоритма нейронной сети. 
# Координаты (второй параметр декларирует кол-во входных параметров для двух состовляющих - х и у)
# будут поделены на два кластера (соответствуют конфигу с одним слоем (выходным) из двух нейронов).
neuron_algorithm = KonahenClusteringAlgorithm([2], 2)
# Инициализация координат, вокруг которых формируются координаты для тренировочного датасета.
dots_set = [Coordinate(10, 10), Coordinate(20, 20)]
training_set = list()

# Формирование тренировочного датасета.
for i in range(0, 30):
	for center_coordinate_index in range(0, len(dots_set)):
		# Случайный угол поворота вектора
		angle = random.random() * math.pi * 2
		# Случайной длины от 0 до 10
		length = random.random() * 10
		# С началом в одной из двух координат.
		center_coordinate = dots_set[center_coordinate_index]
		x = length * math.cos(angle) + center_coordinate.x
		y = length * math.sin(angle) + center_coordinate.y
		training_set.append(Coordinate(x, y))

normalized_training_set = neuron_algorithm.normalize_coordinates_data(training_set)
# Обучение нейронной сети на третировочном датасете.
neuron_algorithm.learn_by_coordinates(normalized_training_set)

# Изображение координат на плоскости с цветом, соответствующим номеру кластера.
for coordinate_index in range(0, len(normalized_training_set)):
	coordinate = training_set[coordinate_index]
	normalized_coordinate_data = normalized_training_set[coordinate_index]
	if neuron_algorithm.get_coordinate_cluster(normalized_coordinate_data):
		# Если кластер не 0, то координата будет зеленого цвета.
		settings = 'go'
	else:
		# Иначе синего.
		settings = 'bo'

	# Рисование координаты.
	plt.plot(coordinate.x, coordinate.y, settings)

plt.show()
