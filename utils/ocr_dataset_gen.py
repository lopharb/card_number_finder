import os
import random
import pandas as pd
from tqdm import tqdm
from trdg.generators import GeneratorFromStrings


def generate_credit_card_numbers(n):
    """
    Generate `n` fake 16-digit credit card numbers (optionally grouped).
    """
    numbers = []
    for _ in range(n):
        digits = ''.join(random.choices("0123456789", k=16))
        formatted = ' '.join(digits[i:i+4] for i in range(0, 16, 4))  # e.g., '1234 5678 9012 3456'
        numbers.append(formatted)
    return numbers


def generate_dataset(
    output_dir,
    num_samples=1000,
    font_dir="fonts",
    image_size=640
):
    os.makedirs(output_dir, exist_ok=True)
    strings = generate_credit_card_numbers(num_samples)

    generator = GeneratorFromStrings(
        strings,
        count=num_samples,
        fonts=[os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith((".ttf", ".otf"))],
        size=image_size,
        skewing_angle=30,
        random_skew=True,
        blur=1,
        random_blur=True,
        background_type=0,
        output_mask=False
    )

    labels = {
        'filename': [],
        'label': []
    }

    print(f"Generating {num_samples} samples...")
    for idx, (image, label) in tqdm(enumerate(generator)):
        filename = f"{idx}.png"
        image.save(os.path.join(output_dir, f"{idx}.png"))
        labels['filename'].append(filename)
        labels['label'].append(label)

    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(output_dir, "../labels.csv"), index=False, sep='\t')
    print(f"Dataset saved to '{output_dir}'")


if __name__ == "__main__":
    generate_dataset(
        output_dir="data/synthetic_numbers/images",
        font_dir="data/fonts",
        image_size=640,
        num_samples=10
    )
