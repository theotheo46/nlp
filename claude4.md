Допишите класс-обертку `FactorizedEmbedding`, который в конструкторе принимает на вход слой эмбеддингов `embedding` и пониженный ранг матрицы `rank`. Создайте новый слой эмбеддингов, реализующий произведение двух матриц. Выберите значение `rank` равным 64. Oбе матрицы можно инициализировать с помощью SVD разложения, чтобы начальное приближение было хорошим


<FactorizedEmbedding>

class FactorizedEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, rank: int = 64):
        super().__init__()

        # ваш код здесь

    def forward(self, input_ids):
        # ваш код здесь
        pass



</FactorizedEmbedding>
