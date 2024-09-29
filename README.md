# RUTUBE-soc-dem-24
Предсказание социально-демографических характеристик пользователя

# Установка
Использовался Python 3.8.13, с torch версии 2.0.1+cu117
* Другие необходимые зависимости в [requirements](/requirements.txt)
* Скачивание языковой модели [download_model](/download_model.py)
* Определение разницы по времени между UTC и локальным временем [calc_time_diff](/calc_time_diff.ipynb).
Использовался GigaChat, для запросов нужен "gigachain==0.1.17", готовый результат [time_diffs](/data/time_diffs.parquet)

# Модели
* [cb-solution](/cb-solution.ipynb) — на основе градиентного бустинга
* [events-supervised](/events-supervised-age_class.ipynb) — на основе ptls
* [events-bert](/events-bert-age_class.ipynb) — на основе BertForSequenceClassification