# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from detectron2.data.common import MapDataset, AspectRatioGroupedDataset


class MapDatasetTwoCrop(MapDataset):
    """
    Map a function over the elements in a dataset.

    This customized MapDataset transforms an image with two augmentations
    as two inputs (queue and key).

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            data = self._map_func(self._dataset[cur_idx])
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return data

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class AspectRatioGroupedDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: an iterable. Each element must be a dict with keys
                "width" and "height", which will be used to batch data.
            batch_size (int):
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]
        self._buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        for d in self.dataset:
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            w, h = d[0]["width"], d[0]["height"]
            bucket_id = 0 if w > h else 1

            # bucket = bucket for normal images
            bucket = self._buckets[bucket_id]
            bucket.append(d[0])

            # buckets_key = bucket for augmented images
            buckets_key = self._buckets_key[bucket_id]
            buckets_key.append(d[1])
            if len(bucket) == self.batch_size:
                yield (bucket[:], buckets_key[:])
                del bucket[:]
                del buckets_key[:]


class AspectRatioGroupedSemiSupDatasetTwoCrop(AspectRatioGroupedDataset):
    """
    Batch data that have similar aspect ratio together.
    In this implementation, images whose aspect ratio < (or >) 1 will
    be batched together.
    This improves training speed because the images then need less padding
    to form a batch.

    It assumes the underlying dataset produces dicts with "width" and "height" keys.
    It will then produce a list of original dicts with length = batch_size,
    all with similar aspect ratios.
    """

    def __init__(self, dataset, batch_size):
        """
        Args:
            dataset: a tuple containing two iterable generators. （labeled and unlabeled data)
               Each element must be a dict with keys "width" and "height", which will be used
               to batch data.
            batch_size (int):
        """

        self.L_dataset, self.U_dataset, self.M_dataset, self.O_dataset = dataset
        self.batch_size_L, self.batch_size_U, self.batch_size_M, self.batch_size_O = batch_size

        self._L_buckets = [[] for _ in range(2)]
        self._L_buckets_key = [[] for _ in range(2)]
        self._U_buckets = [[] for _ in range(2)]
        self._U_buckets_key = [[] for _ in range(2)]
        self._M_buckets = [[] for _ in range(2)]
        self._M_buckets_key = [[] for _ in range(2)]
        self._O_buckets = [[] for _ in range(2)]
        self._O_buckets_key = [[] for _ in range(2)]
        # Hard-coded two aspect ratio groups: w > h and w < h.
        # Can add support for more aspect ratio groups, but doesn't seem useful

    def __iter__(self):
        L_bucket, U_bucket, M_bucket, O_bucket = [], [], [], []
        for d_L, d_U, d_M, d_O in zip(self.L_dataset, self.U_dataset, self.M_dataset, self.O_dataset):
            # d is a tuple with len = 2
            # It's two images (same size) from the same image instance
            # d[0] is with strong augmentation, d[1] is with weak augmentation

            # because we are grouping images with their aspect ratio
            # label and unlabel buckets might not have the same number of data
            # i.e., one could reach batch_size, while the other is still not
            if len(L_bucket) != self.batch_size_L:
                w, h = d_L[0]["width"], d_L[0]["height"]
                L_bucket_id = 0 if w > h else 1
                L_bucket = self._L_buckets[L_bucket_id]
                L_bucket.append(d_L[0])
                L_buckets_key = self._L_buckets_key[L_bucket_id]
                L_buckets_key.append(d_L[1])

            if len(U_bucket) != self.batch_size_U:
                w, h = d_U[0]["width"], d_U[0]["height"]
                U_bucket_id = 0 if w > h else 1
                U_bucket = self._U_buckets[U_bucket_id]
                U_bucket.append(d_U[0])
                U_buckets_key = self._U_buckets_key[U_bucket_id]
                U_buckets_key.append(d_U[1])

            if len(M_bucket) != self.batch_size_M:
                w, h = d_M[0]["width"], d_M[0]["height"]
                M_bucket_id = 0 if w > h else 1
                M_bucket = self._M_buckets[M_bucket_id]
                M_bucket.append(d_M[0])
                M_buckets_key = self._M_buckets_key[M_bucket_id]
                M_buckets_key.append(d_M[1])

            if len(O_bucket) != self.batch_size_O:
                w, h = d_O[0]["width"], d_O[0]["height"]
                O_bucket_id = 0 if w > h else 1
                O_bucket = self._O_buckets[O_bucket_id]
                O_bucket.append(d_O[0])
                O_buckets_key = self._O_buckets_key[O_bucket_id]
                O_buckets_key.append(d_O[1])
            # yield the batch of data until all buckets are full
            if (
                len(L_bucket) == self.batch_size_L
                and len(U_bucket) == self.batch_size_U
                and len(M_bucket) == self.batch_size_M
                and len(O_bucket) == self.batch_size_O
            ):
                # label_strong, label_weak, unlabed_strong, unlabled_weak
                yield (
                    L_bucket[:],
                    L_buckets_key[:],
                    U_bucket[:],
                    U_buckets_key[:],
                    M_bucket[:],
                    M_buckets_key[:],
                    O_bucket[:],
                    O_buckets_key[:],
                )
                del L_bucket[:]
                del L_buckets_key[:]
                del U_bucket[:]
                del U_buckets_key[:]
                del M_bucket[:]
                del M_buckets_key[:]
                del O_bucket[:]
                del O_buckets_key[:]