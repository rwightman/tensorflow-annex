# Tensorflow Annex
A module for importing data into Tensorflow records format.

Initially focused on large image datasets like Imagenet or MS Coco but plan to support useful audio, video, and mixed media datasets in the future.

Currently the basics of MS Coco dataset to tensorflow record appear to work. The output hasn't been validated yet...

## Usage

    python importer.py --dataset mscoco --train_dir /data/mscoco/train2014/ --annotation_file /data/mscoco/annotations/instances_train2014.json  --output_dir /data/output/
