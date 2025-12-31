from pathlib import Path
import yaml
import argparse

def make_data_yaml(root_dir, train_images, val_images, out, names = None):
    root_dir = Path(root_dir)

    if names is None:
        raise ValueError("Class names list must not be None")
    
    data = {
        'path':root_dir.resolve().as_posix(),
        'train':train_images,
        'val':val_images,
        'nc':len(names),
        'names':names
    }

    out_path = root_dir/out

    with open(out_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data,f ,sort_keys=False)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root-dir', required = True, type = str)
    parser.add_argument('--train-images', required= True, type = str)
    parser.add_argument('--val-images', required = True, type = str)
    parser.add_argument('--output', required = True, type = str)

    args  = parser.parse_args()

    names = [
        "Osteophytes",
        "Spondylolysthesis",
        "Disc space narrowing",
        "Vertebral collapse",
        "Foraminal stenosis",
        "Surgical implant",
        "Other lesions",
    ]

    make_data_yaml(
        root_dir = args.root_dir,
        train_images = args.train_images,
        val_images= args.val_images,
        out = args.output,
        names = names
    )

if __name__ == '__main__':
    main()


