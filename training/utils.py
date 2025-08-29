from tensorflow.keras.utils import Sequence
import numpy as np
import json
import os

from PIL import Image

class SimpleCOCODataGenerator(Sequence):
    """ A data generator which reads data on the MS COCO format and
        reprocesses it to simply output whether a certain class exists in
        a given image.
    """
    def __init__(self, samples_dir, annotation_path, width=480, height=270,
                 batch_size=2, shuffle=True, balance=True):
        """ Initializes the data generator.

        Args:
            samples_dir (str): The path to the directory in which the dataset
                images are located.
            annotation_path (str): The path to the annotations json-file
            data_shape (tuple of ints, optional): The resolution to output
                images on.
            batch_size (int, optional): The number of samples per batch.
            shuffle (bool, optional): Whether to shuffle the dataset sample
                order after each epoch.
            balance (bool, optional): Whether to oversample the data in order
                reduce the effect of imbalanced classes
        """
        print('Creating data generator..')
        self.samples_dir = samples_dir
        annotations = json.load(open(annotation_path, 'r'))
        self.annotations = self._reprocess_annotations(annotations)
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance = balance
        self.on_epoch_end()


    def on_epoch_end(self, weights=[1, 1]):
        print('Shuffling dataset..')

        self.indices = np.arange(len(self.annotations))
        #check self.indices
        if self.balance:
            indices_set = set(self.indices)
            not_has_phone = indices_set - self.sample_classes['has_phone']
            classes = [not_has_phone,
                       self.sample_classes['has_phone']]
            samples_per_class = np.max([len(c) for c in classes])
            self.indices = np.concatenate([np.random.choice(list(c),
               size=int(weights[idx] * samples_per_class)) for idx, c
               in enumerate(classes)])
        if self.shuffle is True:
            np.random.shuffle(self.indices)

        
    def _reprocess_annotations(self, annotations):
        print('Reprocessing annotations..')

        has_phone = set()

        for annotation in annotations['annotations']:

            if annotation['category_id'] == 1:
                has_phone.add(annotation['image_id'])
        
        #print(len(has_phone))

        self.sample_classes = {'has_phone': set()}
        
        processed_annotations = []

        for image in annotations['images']:
            sample = {'file_name': image['file_name'],
                      'id': image['id'],
                      'has_phone': image['id'] in has_phone}
            img_path = os.path.join(self.samples_dir, image['file_name'])
            file_exists = os.path.exists(img_path)

            if file_exists and Image.open(img_path).mode == 'RGB':
                sample_idx = len(processed_annotations)
                if image['id'] in has_phone:
                    self.sample_classes['has_phone'].add(sample_idx)
                processed_annotations.append(sample)

        # print(len(processed_annotations))
        return np.array(processed_annotations)


    def __len__(self):
        """ Returns the number of data batches available to this generator.
        """
        return int(len(self.annotations) / self.batch_size)

    def __getitem__(self, index):
        """Returns one batch of data."""
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_annotations = self.annotations[batch_indices]
        X, y = self._generate_batch(batch_annotations)
        return X, y

    def _generate_batch(self, batch_annotations):
        """Produces a batch (X, y) from processed annotations for phone presence."""
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y_phone = np.zeros((self.batch_size, 1), dtype=np.float32)

        for i, annotation in enumerate(batch_annotations):
            img_path = os.path.join(self.samples_dir, annotation['file_name'])
            img = Image.open(img_path).resize((self.width, self.height))

            # Horizontal flipping with p=0.5
            if np.random.random() >= 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            X[i, ] = np.array(img)
            y_phone[i, ] = float(annotation['has_phone'])

        X /= 255.
        return X, y_phone
