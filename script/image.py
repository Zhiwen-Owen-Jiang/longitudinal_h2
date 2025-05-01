import os
import logging
import h5py
import numpy as np
import pandas as pd
import script.dataset as ds


class LongitudinalImageManager:
    """
    Image management class
    which can keep, remove, and read in batch for a single image HDF5 file

    """

    def __init__(self, image_file, voxels=None):
        """
        Parameters:
        ------------
        image_file: a image HDF5 file path
        voxels: a np.array of voxel indices to keep (0 based)

        """
        self.file = h5py.File(image_file, "r")
        self.images = self.file["images"]
        self.coord = self.file["coord"][:]
        self.time = self.file["time"][:] 
        ids = self.file["id"][:]
        self.ids = pd.MultiIndex.from_arrays(ids.astype(str).T, names=["FID", "IID"])
        self.n_images, self.n_voxels = self.images.shape
        self.dim = self.coord.shape[1]
        self.id_idxs = np.arange(len(self.ids))
        self.extracted_ids = self.ids
        self.logger = logging.getLogger(__name__)
        
        if voxels is not None:
            self.voxels = voxels
            self.n_voxels = len(self.voxels)
            self.coord = self.coord[self.voxels]
        else:
            self.voxels = np.arange(self.n_voxels)

        self.logger.info(
            f"{self.n_images} images and {self.n_voxels} voxels (vertices) in {image_file}"
        )

    def keep_and_remove(self, keep_idvs=None, remove_idvs=None, check_empty=True):
        """
        Keeping and removing subjects

        Parameters:
        ------------
        keep_idvs: subject indices in pd.MultiIndex to keep
        remove_idvs: subject indices in pd.MultiIndex to remove
        check_empty: if check the current image set is empty

        """
        if keep_idvs is not None:
            self.extracted_ids = ds.get_common_idxs(self.extracted_ids, keep_idvs)
        if remove_idvs is not None:
            self.extracted_ids = ds.remove_idxs(self.extracted_ids, remove_idvs)
        if check_empty and len(self.extracted_ids) == 0:
            raise ValueError("no subject remaining after --keep and/or --remove")

        self.n_images = (self.ids.isin(self.extracted_ids)).sum()
        self.id_idxs = np.arange(len(self.ids))[self.ids.isin(self.extracted_ids)]
        
    def select_time(self, time):
        """
        Selecting time points

        Parameters:
        ------------
        time: a list of time points to keep
        
        """
        time_idxs = np.where(np.isin(self.time, time))[0]
        self.id_idxs = np.intersect1d(self.id_idxs, time_idxs)
        self.n_images = len(self.id_idxs)

    def image_reader(self, batch_size=None):
        """
        Reading imaging data in chunks as a generator

        Parameters:
        ------------
        batch_size: an int of batch size

        """
        if batch_size is None:
            memory_use = (
                self.n_images * self.n_voxels * np.dtype(np.float32).itemsize / (1024**3)
            )
            if memory_use <= 5:
                batch_size = self.n_images
            else:
                batch_size = int(self.n_images / memory_use * 5)

        for i in range(0, self.n_images, batch_size):
            id_idx_chuck = self.id_idxs[i : i + batch_size]
            yield self.images[id_idx_chuck][:, self.voxels], self.ids[id_idx_chuck]

    def save(self, out_dir):
        """
        Saving the processed images to a new HDF5 file

        Parameters:
        ------------
        out_dir: directory of output

        """
        self.logger.info(f"{self.n_images} subjects in the output image file.")
        try:
            with h5py.File(out_dir, "w") as output:
                dset = output.create_dataset(
                    "images", shape=(self.n_images, self.n_voxels), dtype="float32"
                )
                output.create_dataset(
                    "id", data=self.extracted_ids.tolist(), dtype="S10"
                )
                output.create_dataset("coord", data=self.coord)

                start, end = 0, 0
                for images_, _ in self.image_reader():
                    start = end
                    end += images_.shape[0]
                    dset[start:end] = images_

        except Exception:
            if os.path.exists(out_dir):
                os.remove(out_dir)
            raise

    def close(self):
        self.file.close()