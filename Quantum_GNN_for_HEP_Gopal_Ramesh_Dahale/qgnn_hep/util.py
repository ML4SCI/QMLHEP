from urllib.request import urlretrieve

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        This function updates the progress of a download by calculating the number of blocks and block
        size and setting the total size if provided.

        Args:
          blocks: The number of blocks that have been transferred. Defaults to 1
          bsize: bsize stands for "block size" and represents the size of each block being downloaded or
        uploaded. It is usually measured in bytes. Defaults to 1
          tsize: tsize stands for "total size" and represents the total size of the file being
        downloaded. It is an optional parameter that can be passed to the function. If it is provided,
        it will update the total size of the file being downloaded.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """
    This function downloads a file from a given URL and displays the progress using the TqdmUpTo
    library.

    Args:
      url: The URL of the file to be downloaded.
      filename: The name of the file to be saved after downloading from the given URL.
    """
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # noqa: S310
