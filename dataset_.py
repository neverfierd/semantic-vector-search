import json
import random


def add_noise(text: str) -> str:
    prefixes = ["   ", "\t", "!!! ", "___", "*** ", "... ", "=== "]
    suffixes = ["  ", "\t\t", " !!!", "___", " ***", " ...", " ==="]

    noisy = text
    # с вероятностью 60% добавляем мусор в начало
    if random.random() > 0.4:
        noisy = random.choice(prefixes) + noisy
    # с вероятностью 60% добавляем мусор в конец
    if random.random() > 0.4:
        noisy = noisy + random.choice(suffixes)

    # имитируем случайные двойные пробелы
    noisy = noisy.replace(" ", "  ", random.randint(1, 3))
    return noisy


def expand_dataset(input_file: str, output_file: str, target_size: int = 200) -> None:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented = list(data)
    max_id = max(item["id"] for item in data)

    # добиваем массив до 200 элементов случайными замусоренными копиями
    while len(augmented) < target_size:
        base_item = random.choice(data)
        max_id += 1
        augmented.append(
            {
                "id": max_id,
                "category": base_item.get("category", "mixed"),
                "text": add_noise(base_item["text"]),
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    print(f"Готово! Сгенерировано {len(augmented)} записей в файл {output_file}")


if __name__ == "__main__":
    expand_dataset("initial_data.json", "augmented_data.json", 200)
