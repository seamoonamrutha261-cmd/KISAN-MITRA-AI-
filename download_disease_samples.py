
from pathlib import Path
import re
import time
import requests

DATASET = "Project-AgML/tomato_leaf_disease"
API = "https://datasets-server.huggingface.co"
SAMPLES_PER_CLASS = 30

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "disease_dataset"


def get_json(endpoint, params):
    url = f"{API}/{endpoint}"

    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 2:
                raise
            print("Dataset server is slow; retrying...")
            time.sleep(3 * (attempt + 1))


def safe_folder_name(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def main():
    print("Connecting to the tomato-leaf dataset...")

    split_data = get_json("splits", {"dataset": DATASET})

    train_splits = [
        item for item in split_data.get("splits", [])
        if item.get("split") == "train"
    ]

    if not train_splits:
        raise RuntimeError("No training split was found.")

    config = train_splits[0]["config"]
    split = train_splits[0]["split"]

    first_rows = get_json(
        "first-rows",
        {"dataset": DATASET, "config": config, "split": split}
    )

    features = first_rows["features"]

    label_feature = next(
        (f for f in features if f["name"] == "label"), None
    )
    image_feature = next(
        (f for f in features
         if f["type"].get("_type") == "Image"),
        None
    )

    if label_feature is None or image_feature is None:
        raise RuntimeError("Could not identify the image and label columns.")

    class_names = label_feature["type"].get("names")

    if not class_names:
        raise RuntimeError("Could not read the class names.")

    OUTPUT_DIR.mkdir(exist_ok=True)

    class_folders = {}
    counts = {}

    for class_name in class_names:
        folder = OUTPUT_DIR / safe_folder_name(class_name)
        folder.mkdir(parents=True, exist_ok=True)
        class_folders[class_name] = folder
        counts[class_name] = 0

    total_rows = 11000
    offset = 0
    page_size = 100
    total_downloaded = 0

    print(f"Found {len(class_names)} classes.")
    print(f"Collecting up to {SAMPLES_PER_CLASS} images per class...")

    while offset < total_rows:
        page = get_json(
            "rows",
            {
                "dataset": DATASET,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size
            }
        )

        rows = page.get("rows", [])

        if not rows:
            break

        total_rows = page.get("num_rows_total", total_rows)

        for item in rows:
            row = item.get("row", {})
            raw_label = row.get(label_feature["name"])

            if isinstance(raw_label, int) and 0 <= raw_label < len(class_names):
                class_name = class_names[raw_label]
            elif isinstance(raw_label, str) and raw_label in counts:
                class_name = raw_label
            else:
                continue

            if counts[class_name] >= SAMPLES_PER_CLASS:
                continue

            image_data = row.get(image_feature["name"], {})
            image_url = image_data.get("src")

            if not image_url:
                continue

            filename = class_folders[class_name] / (
                f"image_{counts[class_name] + 1:03}.jpg"
            )

            try:
                image_response = requests.get(image_url, timeout=45)
                image_response.raise_for_status()


                filename.write_bytes(image_response.content)
                counts[class_name] += 1
                total_downloaded += 1

            except requests.RequestException as error:
                print(f"Image download failed: {error}")

        offset += len(rows)
        print(f"Scanned {offset}/{total_rows} rows; downloaded {total_downloaded} images")

        if all(count >= SAMPLES_PER_CLASS for count in counts.values()):
            break

    print("\nDownload finished.")
    for class_name, count in counts.items():
        print(f"{class_name}: {count} images")

    print(f"Total images downloaded: {total_downloaded}")
    print(f"Saved in: {OUTPUT_DIR}")

    if total_downloaded == 0:
        print("No images downloaded. Check the internet connection and try again.")


if __name__ == "__main__":
    main()