Напишите функцию create_vector_db. Она принимает датасет, модель для кодирования текстов, TextSplitter и гиперпараметры LSH, проходит по всем записям в корпусе, делит тексты на куски, векторизует куски и добавляет полученные точки в созданную LSH базу данных. Подумайте о том, какой именно текст вы будете векторизовать, а так же какую информацию вы будете сохранять в базе данных вместе с точкой. Обратите внимание, что embedding_model.encode по умолчанию возвращает np.array. Чтобы она возвращала тензор, надо установить соответствующий флажок.

<LSHash>
class LSHash:
    def __init__(self, hidden_size: int, hash_size: int, num_hashtables: int):
        """
        hidden_size: размерность векторов в базе данных
        hash_size: размер хеша, число случайных гиперплоскостей для каждой хеш-таблицы
        num_hashtables: число хеш-таблиц
        """
        self.hash_size = hash_size
        self.hidden_size = hidden_size
        self.num_hashtables = num_hashtables

        self.uniform_planes = [self._generate_random_planes() for _ in range(num_hashtables)]
        self.hash_tables = [defaultdict(list) for i in range(num_hashtables)]

        # По id точки хранит пару (точка, дополнительная информация)
        # Это позволит нам в хеш-таблицах хранить только id для экономии памяти
        self.data_points = dict(
            # id: (point, extra_data)
        )

    def _generate_random_planes(self):
        """
        Генерирует случайные векторы нормали для гиперплоскостей.
        Генерируем из нормального распределения, так как нам важно,
        чтобы углы векторов были распределены равномерно по пространству.
        """
        return torch.randn(self.hash_size, self.hidden_size)

    def _hash(self, planes: torch.Tensor, point: torch.Tensor) -> str:
        """
        Считает хеш для полученной точки (point) по векторам нормали (planes) одной хеш-таблицы.

        planes: Tensor[hash_size, hidden_size]
        point: Tensor[hidden_size]

        Возвращает строку из self.hash_size 0 и 1
        """

        products = planes @ point

        return "".join(["1" if i > 0 else "0" for i in products])

    def add_point(self, point_id: Union[int, str], point: torch.Tensor, extra_data=None):
        """
        Добавляет точку по id в self.data_points, а так же ее id в каждую из хеш-таблиц.

        point_id: идентификатор точки
        point: точка, которую мы добавим в базу
        extra_data: любые дополнительные данные для точки
        """
        self.data_points[point_id] = (point, extra_data)

        for i in range(len(self.hash_tables)):
            h = self._hash(self.uniform_planes[i], point)
            self.hash_tables[i][h].append(point_id)

    def query(self, query_point: torch.Tensor, limit: int = 5) -> dict:
        """
        Находит и возвращает limit штук ближайших к query_point точек из базы данных.
        Сначала в список кандидатов записываются все точки, у которых хеш совпадает с
        query_point хотя бы в одной таблице. Из кандидатов выбираются limit наиболее
        близких к query_point по *косинусу*.

        query_point: Tensor[hidden_size] – входная точка
        limit: int – максимальное число точек на выходе

        Возвращает словарь с полями
            - 'ids': список id найденных точек
            - 'points': Tensor[<= limit, hidden_size] – найденные точки
            - 'extra_data': список дополнительных данных для каждой найденной точки
        """

        candidate_ids = set()

        for i in range(len(self.hash_tables)):
            binary_hash = self._hash(self.uniform_planes[i], query_point)
            candidate_ids.update(self.hash_tables[i][binary_hash])

        candidates = [(id, cosine(query_point, self.data_points[id][0])) for id in candidate_ids]
        candidates = sorted(candidates, key=lambda x: x[1])[:limit]

        ids = []
        points = []
        extra_data = []
        for candidate in candidates:
            point_id = candidate[0]
            ids.append(point_id)
            points.append(self.data_points[point_id][0])
            extra_data.append(self.data_points[point_id][1])

        return {"ids": ids, "points": torch.vstack(points), "extra_data": extra_data}
</LSHash>

<TextSplitter>
class TextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def __call__(self, text: str) -> List[str]:
        """
        Разбивает текст на куски с фиксированным числом слов и возвращает список полученных кусков
        """
        # Разбиваем текст на слова
        words = text.split()
        
        # Если текст пустой или недостаточно слов
        if not words or len(words) <= self.chunk_size:
            return [text] if text else []
        
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        
        # Проверка корректности параметров
        if step <= 0:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        
        # Формируем куски текста
        for i in range(0, len(words), step):
            # Определяем конец текущего куска
            end_idx = min(i + self.chunk_size, len(words))
            # Формируем кусок текста
            chunk = ' '.join(words[i:end_idx])
            chunks.append(chunk)
            
            # Если достигли конца текста
            if end_idx == len(words):
                break
        
        return chunks
</TextSplitter>

<create_vector_db>
def create_vector_db(dataset, embedding_model, text_splitter, hash_size=10, num_hashtables=30) -> LSHash: 
    hidden_size = embedding_model[0].auto_model.config.hidden_size
    # ваш код здесь
</create_vector_db>
