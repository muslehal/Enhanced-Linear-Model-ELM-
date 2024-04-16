import warnings

import torch
from torch.utils.data import DataLoader


class DataLoaders:
    #This function is the constructor for the DataLoaders class. 
    #It is automatically called when a new instance of the class is created. 
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = True,
        shuffle_val = False
    ):
            #These lines assign the input parameters to instance variables for future use.

        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val

    #These lines call methods that create DataLoaders for training, validation, and testing datasets.
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
  #These methods each return a DataLoader for the respective dataset split (train, validation, test).      
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train,drop_last=True )

    def val_dataloader(self):        
        #return self._make_dloader("val", shuffle=self.shuffle_val ,drop_last=True)
        return self._make_dloader("test", shuffle=False  ,drop_last=True)


    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False  ,drop_last=True)

    #This is a private method used by the previous three methods.
    # It creates and returns a DataLoader with the specified split and shuffle flag.
    def _make_dloader(self, split, shuffle=False , drop_last=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            drop_last=drop_last
        )
    #This decorator indicates that the next method is a class method, 
    #meaning it is bound to the class and not the instance of the class.
    @classmethod 
    #This method allows adding command line interface (CLI) arguments for batch size and number of workers.
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )


#This method provides a way to add a new DataLoader to the existing set of DataLoaders.
# It can accept either a DataLoader,a Dataset, or raw data, and it will appropriately create a new DataLoader.
    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data

    

