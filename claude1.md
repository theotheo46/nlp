Адаптируйте класс ActiveLearning для сохранения индексов размеченных текстов. Класс должен поддерживать список self.labeled_idxs, в который на каждой итерации активного обучения будут добавляться индексы новых текстов. Этот класс будет использоваться в качестве базового для следующих заданий.

<ActiveLearning>
class ActiveLearning:
    def __init__(self, unlabeled_data, min_training_items=512, label_num=128):
        self.unlabeled_data = np.array(unlabeled_data)
        self.labeled_data = []

        # модель, которую мы будем обучать
        self.model = MultinomialNB(alpha=0.01)
        # число стартовых размеченных текстов
        self.min_training_items = min_training_items
        # число текстов для разметки на каждом шаге
        self.label_num = label_num

    def init_labeled_data(self):
        """
        Выбираем случайные стартовые тексты для разметки
        """
        np.random.seed(0)
        idxs = np.random.choice(
            np.arange(len(self.unlabeled_data)),
            size=self.min_training_items,
            replace=False
        )

        self.labeled_data = self.unlabeled_data[idxs]
        self.unlabeled_data = np.delete(self.unlabeled_data, idxs, 0)

    def get_probs(self):
        """
        Возвращает предсказания вероятностей для всех неразмеченных текстов
        """
        x = self.unlabeled_data[:, 0]
        x = vstack(x)
        probs = self.model.predict_proba(x)
        return probs

    def get_scores(self):
        """
        Возвращает оценки для всех неразмеченных текстов.
        На основе этих оценок происходит выбор текстов для разметки.
        """
        if len(self.unlabeled_data) <= self.label_num:
            return np.zeros(len(self.unlabeled_data))

        return np.random.rand(len(self.unlabeled_data))

    def label_additional_data(self):
        """
        Размечает порцию текстов в соответствии с их оценками
        """
        if len(self.labeled_data) == 0:
            self.init_labeled_data()
            return

        scores = self.get_scores()
        idxs = np.argsort(scores)[-self.label_num:]

        self.labeled_data = np.concatenate((self.labeled_data, self.unlabeled_data[idxs]))
        self.unlabeled_data = np.delete(self.unlabeled_data, idxs, 0)

    def train_model(self):
        """
        Обучает модель на всех размеченных текстах
        """
        assert len(self.labeled_data) > 0, "You don't have any labeled data"

        x = self.labeled_data[:, 0]
        x = vstack(x)
        y = self.labeled_data[:, 1].astype(int)
        self.model.fit(x, y)
</ActiveLearning>
