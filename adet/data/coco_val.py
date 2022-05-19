import json


def main():
    img_dir = '../data/coco/val2017'
    coco_val = '../data/coco/annotations/instances_val2017.json'
    davis_val = '../data/DAVIS_2019_unsupervised_480/trainval/davis_val_coco_format.json'
    f1 = open(coco_val, )
    coco_data = json.load(f1)

    f2 = open(davis_val, )
    davis_data = json.load(f2)

    for i in range(len(coco_data['annotations'])):
        coco_data['annotations'][i]['category_id'] = 0

    for i in range(len(coco_data['images'])):
        coco_data['images'][i]['file_name'] = img_dir + '/' + coco_data['images'][i]['file_name']

    coco_data['categories'] = davis_data['categories']

    with open(coco_val, 'w') as outfile:
        json.dump(coco_data, outfile)


if __name__ == "__main__":
    main()